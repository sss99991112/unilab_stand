"""Opt-in frozen-expert standing assistance for simulation evaluation/playback.

Default playback, collectors and checkpoints retain their original route.
Acceptance still uses the original full stable window and nominal settling time.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from unilab.envs.locomotion.g1.recovery_combined import update_recovery_route

from .recovery_integration import recovery_expert_indices, validate_recovery_env


@dataclass(frozen=True)
class StandingAssistConfig:
    enabled: bool = False

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("standing assist enabled must be boolean")


class StandingAssistController:
    """One-env evaluation controller; no learned router, optimizer or new weights.

    At most one assistance attempt per reset. Ten stable samples start a 20-action
    crossfade. The full 50-sample success window restarts after the crossfade;
    all velocity/height/tilt/contact thresholds remain owned by the source env.
    """

    ENTRY_FRAMES = 10
    BLEND_FRAMES = 20
    TIMEOUT_FRAMES = 100

    def __init__(self, env, student, config=StandingAssistConfig(), *, actuation=None):
        validate_recovery_env(env)
        if env.num_envs != 1:
            raise ValueError("standing-assist experiment requires exactly one simulation env")
        self.env, self.student, self.config = env, student, config
        self.actuation = None
        if config.enabled:
            if actuation is None:
                raise ValueError("enabled standing assist requires the physical actuation contract")
            expected = {"kp": (29,), "kd": (29,), "joint_ranges": (29, 2), "force_limits": (29, 2)}
            self.actuation = {}
            for key, shape in expected.items():
                value = np.asarray(actuation[key], dtype=np.float64).copy()
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid actuation contract {key}")
                self.actuation[key] = value
        self.reset()

    def reset(self):
        self.info = {
            "commands": np.zeros((1, 3), np.float32),
            "height_commands": np.full((1, 1), 0.754, np.float32),
            "gait_phase": np.asarray([[0, np.pi]], np.float32),
            "current_actions": np.zeros((1, 29), np.float32),
        }
        self.assisting = False
        self.attempted = False
        self.assist_age = 0
        self.last_frame = -1
        self.event = ""

    def _route(self, frame, state):
        if frame == self.last_frame:
            return False
        if frame != self.last_frame + 1:
            raise ValueError("simulation frames must be contiguous; reset before a new episode")
        self.event = ""
        was_assisting = self.assisting
        update_recovery_route(
            self.info,
            height=np.asarray([state["height"]], np.float32),
            gravity=np.asarray(state["upvector"], np.float32)[None],
            linvel=np.asarray(state["linvel"], np.float32)[None],
            gyro=np.asarray(state["gyro"], np.float32)[None],
            feet=np.asarray(state["feet"], bool)[None],
            steps=np.asarray([frame]),
            recovery=self.env.cfg.recovery,
            routing=self.env.cfg.recovery_routing,
            ctrl_dt=self.env.cfg.ctrl_dt,
        )
        active = bool(self.info["recovery_active"][0])
        if self.config.enabled:
            cfg = self.env.cfg.recovery_routing
            fallen = state["height"] < cfg.enter_height or state["upvector"][2] < np.cos(
                np.deg2rad(cfg.enter_tilt_deg)
            )
            if was_assisting:
                self.assist_age += 1
                if fallen or self.assist_age >= self.TIMEOUT_FRAMES:
                    self.assisting = False
                    self.info["recovery_active"][:] = True
                    self.info["recovery_route_stable_steps"][:] = 0
                    self.info["recovery_settle_remaining"][:] = 0
                    self.event = "fallback_refall" if fallen else "fallback_timeout"
                elif not active:
                    self.assisting = False
                    self.event = "release"
                elif self.assist_age < self.BLEND_FRAMES:
                    # Samples accrued before/during crossfade are never success.
                    self.info["recovery_route_stable_steps"][:] = 0
            elif (
                active
                and not self.attempted
                and self.info["recovery_route_stable_steps"][0] >= self.ENTRY_FRAMES
            ):
                self.assisting = True
                self.attempted = True
                self.assist_age = 0
                self.info["recovery_route_stable_steps"][:] = 0
                self.event = "assist_start"
        if self.info["recovery_active"][0] and not self.assisting:
            self.info["gait_phase"][:] = 0
        self.last_frame = frame
        return True

    @property
    def alpha(self):
        if self.assisting:
            return min((self.assist_age + 1) / self.BLEND_FRAMES, 1.0)
        return 0.0 if self.info.get("recovery_active", [False])[0] else 1.0

    def select_action(self, frame, state, *, commands=(0, 0, 0), height=0.754):
        """Select a network action; never apply it or advance action history/phase.

        Both heads see actual applied action history. Recovery retains its zero
        gait phase, while the standing head uses the advancing nominal phase.
        """
        if frame == self.last_frame:
            raise ValueError("frame already selected; use a detached preview for repeated reads")
        for name, size in (
            ("upvector", 3),
            ("linvel", 3),
            ("gyro", 3),
            ("q", 29),
            ("dq", 29),
            ("feet", 2),
        ):
            value = np.asarray(state[name])
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"invalid physical {name}")
        if not np.isin(state["feet"], [0, 1]).all():
            raise ValueError("physical foot contacts must be boolean")
        commands = np.asarray(commands, np.float32)
        if commands.shape != (3,) or not np.isfinite(commands).all() or not np.isfinite(height):
            raise ValueError("invalid requested command")
        self.info["commands"][:] = commands
        self.info["height_commands"][:] = height
        self._route(frame, state)
        obs = self.env._compute_obs(
            self.info,
            np.asarray(state["linvel"], np.float32)[None],
            np.asarray(state["gyro"], np.float32)[None],
            np.asarray(state["upvector"], np.float32)[None],
            np.asarray(state["q"], np.float32)[None],
            np.asarray(state["dq"], np.float32)[None],
        )["obs"]
        recovery_obs = obs.copy()
        recovery_obs[:, 97:99] = 0
        index = int(recovery_expert_indices(self.info)[0])
        with torch.inference_mode():
            if self.assisting:
                recovery_action = self.student.experts[2](torch.from_numpy(recovery_obs)).numpy()
                standing_action = self.student.experts[1](torch.from_numpy(obs)).numpy()
                alpha = np.float32(self.alpha)
                action = (np.float32(1) - alpha) * recovery_action + alpha * standing_action
                index = 1 if self.alpha == 1 else -1
            else:
                action = self.student.experts[index](torch.from_numpy(obs)).numpy()
                recovery_action = standing_action = action.copy()
        proposed_target = action[0] + np.asarray(self.env.default_angles, np.float32).reshape(29)
        rejected = False
        if self.assisting:
            contract = self.actuation
            default = np.asarray(self.env.default_angles, np.float32).reshape(29)
            recovery_target = recovery_action[0] + default

            def excess(value, limits):
                return np.maximum(np.maximum(limits[:, 0] - value, value - limits[:, 1]), 0)

            joint_worse = excess(proposed_target, contract["joint_ranges"]) > (
                excess(recovery_target, contract["joint_ranges"]) + 1e-7
            )
            raw = contract["kp"] * (proposed_target - state["q"]) - contract["kd"] * state["dq"]
            recovery_raw = (
                contract["kp"] * (recovery_target - state["q"]) - contract["kd"] * state["dq"]
            )
            force_worse = excess(raw, contract["force_limits"]) > (
                excess(recovery_raw, contract["force_limits"]) + 1e-6
            )
            rejected = bool(joint_worse.any() or force_worse.any())
            if rejected:
                # Reject the entire candidate. No joint clipping and no phantom
                # action/phase history from a proposal that was never executed.
                self.assisting = False
                self.info["recovery_active"][:] = True
                self.info["recovery_route_stable_steps"][:] = 0
                self.info["recovery_settle_remaining"][:] = 0
                self.info["gait_phase"][:] = 0
                self.event = "fallback_load"
                action, obs, index = recovery_action, recovery_obs, 2
        if action.shape != (1, 29) or not np.isfinite(action).all():
            raise FloatingPointError("invalid frozen expert action")
        return action, {
            "obs": obs[0].copy(),
            "recovery_obs": recovery_obs[0].copy(),
            "proposed_target": proposed_target.copy(),
            "load_rejected": rejected,
            "action": action[0].copy(),
            "recovery_action": recovery_action[0].copy(),
            "standing_action": standing_action[0].copy(),
            "expert": index,
            "alpha": self.alpha,
            "assisting": self.assisting,
            "attempted": self.attempted,
            "assist_age": self.assist_age,
            "event": self.event,
            "active": bool(self.info["recovery_active"][0]),
            "stable_count": int(self.info["recovery_route_stable_steps"][0]),
            "settle": int(self.info["recovery_settle_remaining"][0]),
        }

    def decide(self, frame, state, *, commands=(0, 0, 0), height=0.754):
        """Legacy independent evaluator: select and commit exactly once.

        Formal playback uses select_action and lets env.step commit instead.
        """
        action, extra = self.select_action(frame, state, commands=commands, height=height)
        target = self.env.apply_action(action, SimpleNamespace(info=self.info))[0]
        return target, extra
