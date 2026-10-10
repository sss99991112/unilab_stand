"""Opt-in physical recovery routing for the existing 99-D G1 locomotion interface."""

from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dr import DomainRandomizationManager

from .joystick import (
    LEFT_FOOT_CONTACT_SENSORS,
    RIGHT_FOOT_CONTACT_SENSORS,
    G1WalkDomainRandomizationProvider,
    G1WalkEnv,
    G1WalkHeightCfg,
    compute_aggregated_foot_contact,
)
from .recovery import G1RecoveryResetProvider, RecoveryConfig, recovery_stable_mask


@dataclass
class RecoveryRoutingConfig:
    enter_height: float = 0.40
    enter_tilt_deg: float = 60.0
    nominal_settle_steps: int = 100

    def validate(self, recovery: RecoveryConfig) -> None:
        if not 0 < self.enter_height < recovery.standing_height:
            raise ValueError("recovery entry height must be below standing height")
        if not recovery.stable_tilt_deg < self.enter_tilt_deg < 90:
            raise ValueError("recovery entry tilt must exceed stable tilt")
        if self.nominal_settle_steps < 0:
            raise ValueError("nominal settling steps must be non-negative")


def update_recovery_route(
    info, *, height, gravity, linvel, gyro, feet, steps, recovery, routing, ctrl_dt
):
    """Latch recovery until a full stable window; count each control frame once."""
    routing.validate(recovery)
    if not all(np.isfinite(value).all() for value in (height, gravity, linvel, gyro)):
        raise FloatingPointError("non-finite recovery routing input")
    n = len(height)
    if "recovery_active" not in info:
        info["recovery_active"] = np.zeros(n, dtype=bool)
        info["recovery_route_stable_steps"] = np.zeros(n, dtype=np.int32)
        info["recovery_route_last_step"] = np.full(n, -1, dtype=np.int64)
        info["recovery_settle_remaining"] = np.zeros(n, dtype=np.int32)
    fresh = np.asarray(steps) != info["recovery_route_last_step"]
    fallen = (height < routing.enter_height) | (
        gravity[:, 2] < np.cos(np.deg2rad(routing.enter_tilt_deg))
    )
    active = info["recovery_active"]
    active[fresh & fallen] = True
    stable = recovery_stable_mask(height, gravity, linvel, gyro, feet, recovery)
    count = info["recovery_route_stable_steps"]
    count[fresh] = np.where(active[fresh] & stable[fresh], count[fresh] + 1, 0)
    release = fresh & active & (count >= int(np.ceil(recovery.stable_seconds / ctrl_dt)))
    remaining = info["recovery_settle_remaining"]
    remaining[fresh] = np.maximum(remaining[fresh] - 1, 0)
    remaining[release] = routing.nominal_settle_steps
    active[release] = False
    info["recovery_route_last_step"][fresh] = np.asarray(steps)[fresh]
    return active


@registry.envcfg("G1RecoveryCombined")
@dataclass
class G1RecoveryCombinedCfg(G1WalkHeightCfg):
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    recovery_routing: RecoveryRoutingConfig = field(default_factory=RecoveryRoutingConfig)
    recovery_reset_supine: bool = False


class _CombinedResetProvider(G1RecoveryResetProvider):
    def build_reset_plan(self, env, env_ids):
        if env.cfg.recovery_reset_supine:
            plan = super().build_reset_plan(env, env_ids)
        else:
            plan = G1WalkDomainRandomizationProvider.build_reset_plan(self, env, env_ids)
        n = len(env_ids)
        plan.info_updates.update(
            {
                "recovery_active": np.full(n, env.cfg.recovery_reset_supine, dtype=bool),
                "recovery_route_stable_steps": np.zeros(n, dtype=np.int32),
                "recovery_route_last_step": np.full(n, -1, dtype=np.int64),
                "recovery_settle_remaining": np.zeros(n, dtype=np.int32),
            }
        )
        return plan


@registry.env("G1RecoveryCombined", sim_backend="mujoco")
class G1RecoveryCombinedEnv(G1WalkEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        cfg.recovery.validate(float(cfg.commands.default_height), float(cfg.ctrl_dt))
        cfg.recovery_routing.validate(cfg.recovery)
        if not cfg.commands.observe_height_command or cfg.mode_observation:
            raise ValueError("combined recovery requires the common 99-D actor interface")
        if cfg.stand_action_authority or cfg.curriculum.enabled:
            raise ValueError("combined recovery requires full policy authority and no curriculum")
        if cfg.reward_config.min_base_height != 0 or cfg.reward_config.max_tilt_deg != 180:
            raise ValueError("combined task must keep fallen states alive for recovery")
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._recovery_joint_limits = np.asarray(self._backend.get_joint_range())
        kp, kd = (
            self._backend.get_actuator_gains()
            if (cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd)
            else (None, None)
        )
        self._dr_manager = DomainRandomizationManager(
            self, _CombinedResetProvider(base_kp=kp, base_kd=kd)
        )

    def _update_commands(self, info):
        super()._update_commands(info)
        active = update_recovery_route(
            info,
            height=self._terrain_relative_base_height(),
            gravity=self._backend.get_sensor_data(self.cfg.sensor.upvector),
            linvel=self.get_local_linvel(),
            gyro=self.get_gyro(),
            feet=np.column_stack(
                [
                    compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS),
                    compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS),
                ]
            ),
            steps=info["steps"],
            recovery=self.cfg.recovery,
            routing=self.cfg.recovery_routing,
            ctrl_dt=self.cfg.ctrl_dt,
        )
        info["gait_phase"][active] = 0
        info["gait_enabled"][active] = 0

    def _effective_command_info(self, info):
        # Keep the latest request in the ordinary command fields. The guard changes
        # only the actor's effective input, so a stop/height cancellation is never lost.
        blocked = info["recovery_active"] | (info["recovery_settle_remaining"] > 0)
        commands = info["commands"].copy()
        height_commands = info["height_commands"].copy()
        commands[blocked] = 0
        height_commands[blocked] = self.cfg.commands.default_height
        info["recovery_effective_commands"] = commands
        info["recovery_effective_height_commands"] = height_commands
        return {**info, "commands": commands, "height_commands": height_commands}

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        return super()._compute_obs(
            self._effective_command_info(info), linvel, gyro, gravity, dof_pos, dof_vel
        )

    def _compute_reward(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        effective = self._effective_command_info(info)
        reward = super()._compute_reward(effective, linvel, gyro, gravity, dof_pos, dof_vel)
        if "log" in effective:
            info["log"] = effective["log"]
        return reward

    def update_state(self, state: NpEnvState):
        state = super().update_state(state)
        linvel, dof_vel = self.get_local_linvel(), self.get_dof_vel()
        if not np.isfinite(linvel).all() or not np.isfinite(dof_vel).all():
            raise FloatingPointError("non-finite combined recovery physics")
        return state.replace(
            terminated=(
                (np.linalg.norm(linvel, axis=1) > self.cfg.recovery.max_linear_speed)
                | (np.max(np.abs(dof_vel), axis=1) > self.cfg.recovery.max_joint_speed)
            )
        )
