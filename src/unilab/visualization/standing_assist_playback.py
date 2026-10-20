"""Opt-in formal playback adapter; env.step is the sole actuation owner."""

from __future__ import annotations

import copy

import numpy as np
import torch

from unilab.algos.torch.distill.standing_assist import (
    StandingAssistConfig,
    StandingAssistController,
)
from unilab.envs.locomotion.g1.recovery_combined import G1RecoveryCombinedEnv
from unilab.visualization.interactive_playback import RslRlPlaybackSession


class StandingAssistPlaybackSession(RslRlPlaybackSession):
    """One CPU simulation, frozen experts, explicit selection/commit boundary.

    Inference previews operate on detached controller state and preserve noise
    RNG. Only step_once consumes a control frame; env.step applies its action.
    """

    def __init__(self, *, student, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.env, G1RecoveryCombinedEnv):
            raise ValueError("standing assist requires G1RecoveryCombined simulation")
        cfg = self.env.cfg
        if (
            self.device != "cpu"
            or self.num_envs != 1
            or self.action_mode != "policy"
            or cfg.control_config.action_scale != 1.0
            or cfg.control_config.simulate_action_latency
            or self.env._gait_constraint_cfg().enabled
        ):
            raise ValueError(
                "standing assist requires CPU policy playback, one env, unit action scale and no latency/gait constraint"
            )
        self.student = student
        self.controller = StandingAssistController(
            self.env, student, StandingAssistConfig(enabled=True), actuation=self._actuation()
        )
        self.last_decision = None
        self.env.defer_playback_recovery_routing(True)
        self.env.set_autoreset(False)
        self.controller.info = self.env.state.info

    def _actuation(self):
        # Cold path: capture actual reset-randomized gains, never nominal gains.
        model = self.env.get_playback_model(0)
        joints = model.actuator_trnid[:, 0]
        if model.nu != 29 or len(set(joints.tolist())) != 29:
            raise ValueError("standing assist requires 29 distinct joint actuators")
        np.testing.assert_array_equal(model.actuator_gear[:, 0], np.ones(29))
        np.testing.assert_array_equal(model.actuator_biasprm[:, 1], -model.actuator_gainprm[:, 0])
        return dict(
            kp=model.actuator_gainprm[:, 0],
            kd=-model.actuator_biasprm[:, 2],
            joint_ranges=model.jnt_range[joints],
            force_limits=model.actuator_forcerange,
        )

    def reset(self):
        super().reset()
        self.controller.reset()
        # Preserve the real reset's phase/commands/noise/DR; clear only route clock.
        self.controller.info = self.env.state.info
        self.controller.info["recovery_route_last_step"].fill(-1)
        self.controller.actuation = {k: np.asarray(v).copy() for k, v in self._actuation().items()}
        self.last_decision = None
        return self.obs

    def _select(self, *, preview):
        if self.obs is None:
            raise RuntimeError("Playback session must be reset before stepping")
        controller = self.controller
        rng = None
        if preview:
            controller = copy.copy(controller)
            controller.info = copy.deepcopy(self.env.state.info)
            rng = np.random.get_state()
        else:
            controller.info = self.env.state.info
        info = controller.info
        try:
            action, extra = controller.select_action(
                self.step_count,
                self.env.recovery_playback_state(),
                commands=info["commands"][0].copy(),
                height=float(info["height_commands"][0, 0]),
            )
        finally:
            if rng is not None:
                np.random.set_state(rng)
        return torch.from_numpy(action), extra

    def _build_actions(self):
        """Read-only inference/dry-run, including routing and observation-noise RNG."""
        return self._select(preview=True)[0]

    def step_once(self):
        actions, extra = self._select(preview=False)
        self.actions = actions
        self.action_obs = torch.from_numpy(extra["obs"][None])
        # No call to apply_action here. The original wrapper/env own execution.
        self.obs, _reward, _done, _info = self.wrapped_env.step(actions)
        self.step_count += 1
        self.last_decision = extra
        self.controller.info = self.env.state.info
        setattr(self.policy, "_unilab_standing_assist_last_decision", extra)
        setattr(self.policy, "_unilab_distill_last_selected_experts", (extra["expert"],))
        setattr(self.policy, "_unilab_distill_command_routing_applied", True)
        return self.obs
