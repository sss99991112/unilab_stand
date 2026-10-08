"""Independent G1 supine-recovery task with optional fixed-stage assistance.

This is a task baseline inspired by HoST, not a port of its multi-critic PPO.
The actor retains the existing height-aware observation and action contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dr import DomainRandomizationManager
from unilab.envs.common.rotation import np_quat_mul
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.g1.joystick import (
    LEFT_FOOT_CONTACT_SENSORS,
    RIGHT_FOOT_CONTACT_SENSORS,
    CurriculumConfig,
    G1RewardConfig,
    G1WalkDomainRandomizationProvider,
    G1WalkEnv,
    G1WalkFlatCfg,
    compute_aggregated_foot_contact,
)

from .recovery_assistance import RecoveryAssistanceCfg, assistance_force_rows


@dataclass
class RecoveryConfig:
    reset_base_height: float = 0.30
    reset_joint_noise: float = 0.05
    progress_floor_height: float = 0.12
    standing_height: float = 0.65
    stable_tilt_deg: float = 10.0
    stable_linear_speed: float = 0.20
    stable_angular_speed: float = 0.30
    stable_seconds: float = 1.0
    max_joint_speed: float = 40.0
    max_linear_speed: float = 20.0

    def validate(self, target_height: float, ctrl_dt: float) -> None:
        values = np.asarray(list(vars(self).values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("recovery parameters must be finite")
        if self.reset_base_height <= 0 or self.reset_joint_noise < 0:
            raise ValueError("recovery reset height must be positive and joint noise non-negative")
        if not 0 <= self.progress_floor_height < self.standing_height <= target_height:
            raise ValueError("recovery height thresholds must satisfy floor < standing <= target")
        if not 0 < self.stable_tilt_deg < 90:
            raise ValueError("recovery stable_tilt_deg must be between 0 and 90")
        if (
            min(
                self.stable_seconds,
                self.stable_linear_speed,
                self.stable_angular_speed,
                self.max_joint_speed,
                self.max_linear_speed,
                ctrl_dt,
            )
            <= 0
        ):
            raise ValueError("recovery durations and velocity limits must be positive")


def _recovery_commands() -> Commands:
    return Commands(
        vel_limit=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        rel_standing_envs=1.0,
        observe_height_command=True,
        random_height_during_walking=False,
        height_range=[0.754, 0.754],
        default_height=0.754,
    )


@registry.envcfg("G1Recovery")
@dataclass
class G1RecoveryCfg(G1WalkFlatCfg):
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    assistance: RecoveryAssistanceCfg = field(default_factory=RecoveryAssistanceCfg)
    commands: Commands = field(default_factory=_recovery_commands)
    reward_config: G1RewardConfig | None = None
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    max_episode_seconds: float = 10.0
    reset_base_qvel_limit: float = 0.0
    standing_reset_base_qvel_limit: float = 0.0


class G1RecoveryResetProvider(G1WalkDomainRandomizationProvider):
    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        # W-first quaternion. Rotate upright local +X toward world +Z (face up).
        supine = np.asarray([np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0], dtype=plan.qpos.dtype)
        plan.qpos[:, 3:7] = np_quat_mul(plan.qpos[:, 3:7], supine)
        plan.qpos[:, 2] = env.cfg.recovery.reset_base_height
        noise = env.cfg.recovery.reset_joint_noise
        joints = plan.qpos[:, -env._num_action :]
        joints += np.random.uniform(-noise, noise, joints.shape)
        np.clip(
            joints, env._recovery_joint_limits[:, 0], env._recovery_joint_limits[:, 1], out=joints
        )
        plan.qvel.fill(0)
        count = len(env_ids)
        plan.info_updates.update(
            {
                "commands": np.zeros((count, 3), dtype=plan.qpos.dtype),
                "height_commands": np.full(
                    (count, 1), env.cfg.commands.default_height, dtype=plan.qpos.dtype
                ),
                "gait_phase": np.zeros((count, 2), dtype=plan.qpos.dtype),
                "gait_enabled": np.zeros(count, dtype=plan.qpos.dtype),
                "recovery_stable_steps": np.zeros(count, dtype=np.int32),
                "recovery_last_step": np.zeros(count, dtype=np.int64),
                "recovery_frame": np.zeros(count, dtype=np.int64),
                "recovery_succeeded": np.zeros(count, dtype=bool),
                "recovery_stable": np.zeros(count, dtype=bool),
                "recovery_assistance_force_z": np.zeros(count, dtype=np.float64),
            }
        )
        env._spawn.record_episode_start(env_ids, plan.qpos[:, :3])
        return plan


def recovery_stable_mask(
    height: np.ndarray,
    gravity: np.ndarray,
    linvel: np.ndarray,
    gyro: np.ndarray,
    feet_contact: np.ndarray,
    cfg: RecoveryConfig,
) -> np.ndarray:
    """Require height, signed upright orientation, quiet motion, and both feet."""
    return (
        (height >= cfg.standing_height)
        & (gravity[:, 2] >= np.cos(np.deg2rad(cfg.stable_tilt_deg)))
        & (np.linalg.norm(linvel, axis=1) <= cfg.stable_linear_speed)
        & (np.linalg.norm(gyro, axis=1) <= cfg.stable_angular_speed)
        & np.all(feet_contact, axis=1)
    )


def recovery_reward_terms(
    height: np.ndarray,
    gravity: np.ndarray,
    linvel: np.ndarray,
    gyro: np.ndarray,
    dof_vel: np.ndarray,
    actions: np.ndarray,
    previous_actions: np.ndarray,
    stable: np.ndarray,
    cfg: RecoveryConfig,
    target_height: float,
) -> dict[str, np.ndarray]:
    height_progress = np.clip(
        (height - cfg.progress_floor_height) / (target_height - cfg.progress_floor_height),
        0,
        1,
    )
    upright = np.clip((gravity[:, 2] + 1) / 2, 0, 1)
    standing = (height >= cfg.standing_height) & (gravity[:, 2] > 0.8)
    return {
        "recovery_progress": height_progress * (0.1 + 0.9 * upright),
        "recovery_upright": upright,
        "recovery_stability": stable.astype(height.dtype),
        "recovery_stand_height": np.exp(-20 * np.abs(height - target_height)) * standing,
        "recovery_stand_motion": np.exp(
            -5 * np.sum(linvel**2, axis=1) - 2 * np.sum(gyro**2, axis=1)
        )
        * standing,
        "recovery_action_rate": np.sum((actions - previous_actions) ** 2, axis=1),
        "recovery_joint_speed": np.sum(dof_vel**2, axis=1),
    }


def update_recovery_stability(info: dict, stable: np.ndarray, steps: np.ndarray) -> None:
    """Count real transitions once; repeated refresh_state calls do not add time."""
    fresh = info["recovery_last_step"] != steps
    count = info["recovery_stable_steps"]
    count[fresh] = np.where(stable[fresh], count[fresh] + 1, 0)
    info["recovery_last_step"][fresh] = steps[fresh]


@registry.env("G1Recovery", sim_backend="mujoco")
class G1RecoveryEnv(G1WalkEnv):
    def __init__(self, cfg: G1RecoveryCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        cfg.recovery.validate(float(cfg.commands.default_height), float(cfg.ctrl_dt))
        cfg.assistance.validate()
        if cfg.mode_observation or not cfg.commands.observe_height_command:
            raise ValueError("G1Recovery requires the 99-D height-aware observation profile")
        if cfg.curriculum.enabled or cfg.stand_action_authority:
            raise ValueError(
                "G1Recovery must use its own curriculum and full policy action authority"
            )
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if (
            cfg.commands.random_height_during_walking
            or cfg.commands.default_height != cfg.reward_config.base_height_target
        ):
            raise ValueError("G1Recovery requires one fixed, consistent target height")
        supported = {
            "recovery_progress",
            "recovery_upright",
            "recovery_stability",
            "recovery_stand_height",
            "recovery_stand_motion",
            "recovery_action_rate",
            "recovery_joint_speed",
        }
        unknown = set(cfg.reward_config.scales) - supported
        if unknown:
            raise ValueError(f"Unknown recovery reward terms: {sorted(unknown)}")
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        limits = self._backend.get_joint_range()
        if limits is None or limits.shape != (self._num_action, 2):
            raise ValueError("G1Recovery requires joint limits in the policy joint order")
        self._recovery_joint_limits = np.asarray(limits)
        kp, kd = (
            self._backend.get_actuator_gains()
            if cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd
            else (None, None)
        )
        self._dr_manager = DomainRandomizationManager(
            self,
            G1RecoveryResetProvider(base_kp=kp, base_kd=kd),
        )

        # Cache all model/asset metadata once. Zero assistance does not resolve force resources.
        self._assistance_force_z = 0.0
        self._assistance_body_ids = np.empty(0, dtype=np.int32)
        self._assistance_stage = cfg.assistance.stage
        self._assistance_start_steps = int(
            np.ceil(cfg.assistance.start_after_seconds / cfg.ctrl_dt)
        )
        self._assistance_min_up = cfg.assistance.min_upright_cos
        fraction = cfg.assistance.effective_fraction
        if fraction > 0:
            if not self._backend.get_dr_capabilities().supports_interval_body_force:
                raise NotImplementedError("G1Recovery assistance requires body-force support")
            gravity = self._backend.get_gravity()
            if (
                gravity.shape != (3,)
                or not np.all(np.isfinite(gravity))
                or gravity[2] >= 0
                or not np.allclose(gravity[:2], 0)
            ):
                raise ValueError("G1Recovery assistance requires finite downward world-Z gravity")
            robot_ids = self._backend.get_body_subtree_ids(
                self._backend.get_body_id(cfg.asset.base_name)
            )
            robot_mass = float(np.sum(self._backend.get_body_mass()[robot_ids]))
            if not np.isfinite(robot_mass) or robot_mass <= 0:
                raise ValueError("G1Recovery assistance requires positive finite robot mass")
            self._assistance_force_z = fraction * robot_mass * -float(gravity[2])
            self._assistance_body_ids = np.asarray(
                [self._backend.get_body_id(cfg.assistance.body_name)], dtype=np.int32
            )

    def _apply_recovery_assistance(self, state: NpEnvState) -> None:
        # Backend consumes this force for the next control step and then clears it.
        if self._assistance_force_z == 0:
            state.info["recovery_assistance_force_z"].fill(0)
            return
        up = self._backend.get_sensor_data(self.cfg.sensor.upvector)[:, 2]
        force = assistance_force_rows(
            up,
            state.info["steps"],
            force_z=self._assistance_force_z,
            start_steps=self._assistance_start_steps,
            min_upright_cos=self._assistance_min_up,
        )
        self._backend.apply_body_force(self._assistance_body_ids, force)
        state.info["recovery_assistance_force_z"][:] = force[:, 0, 2]

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        # Preserve the existing reference-angle action contract; no gait controller.
        self._apply_recovery_assistance(state)
        state.info["recovery_frame"] += 1
        state.info["last_actions"] = state.info["current_actions"].copy()
        state.info["current_actions"] = actions.copy()
        executed = (
            state.info["last_actions"]
            if self.cfg.control_config.simulate_action_latency
            else actions
        )
        state.info["executed_actions"] = executed.copy()
        return executed * self.cfg.control_config.action_scale + self.default_angles

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel, gyro = self.get_local_linvel(), self.get_gyro()
        gravity = self._backend.get_sensor_data(self.cfg.sensor.upvector)
        dof_pos, dof_vel = self.get_dof_pos(), self.get_dof_vel()
        height = self._terrain_relative_base_height()
        feet = (
            np.column_stack(
                (
                    compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS),
                    compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS),
                )
            )
            > 0
        )
        if not all(
            np.all(np.isfinite(value))
            for value in (linvel, gyro, gravity, dof_pos, dof_vel, height)
        ):
            raise FloatingPointError("Non-finite G1Recovery physics state")
        cfg = self.cfg.recovery
        stable = recovery_stable_mask(height, gravity, linvel, gyro, feet, cfg)
        update_recovery_stability(state.info, stable, state.info["recovery_frame"])
        required = int(np.ceil(cfg.stable_seconds / self.cfg.ctrl_dt))
        state.info["recovery_succeeded"] |= state.info["recovery_stable_steps"] >= required
        terminated = (np.max(np.abs(dof_vel), axis=1) > cfg.max_joint_speed) | (
            np.linalg.norm(linvel, axis=1) > cfg.max_linear_speed
        )
        terms = recovery_reward_terms(
            height,
            gravity,
            linvel,
            gyro,
            dof_vel,
            state.info["current_actions"],
            state.info["last_actions"],
            stable,
            cfg,
            self.cfg.commands.default_height,
        )
        reward = np.zeros(self.num_envs, dtype=height.dtype)
        log = {}
        for name, scale in self._reward_cfg.scales.items():
            contribution = terms[name] * scale
            reward += contribution
            log[f"reward/{name}"] = float(np.mean(contribution))
        log.update(
            {
                "recovery/stable_fraction": float(np.mean(stable)),
                "recovery/success_fraction": float(np.mean(state.info["recovery_succeeded"])),
                "recovery/base_height": float(np.mean(height)),
                "recovery/assistance_force_z": float(
                    np.mean(state.info["recovery_assistance_force_z"])
                ),
                "recovery/assistance_stage": float(self._assistance_stage),
            }
        )
        state.info["log"] = log
        state.info["recovery_stable"] = stable
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward * self.cfg.ctrl_dt, terminated=terminated)
