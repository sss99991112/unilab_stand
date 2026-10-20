"""Bounded verification through the formal playback factory and native env.step.

Freezes nominal/no-noise conditions explicitly for comparison; no training,
hardware or policy edits. Physics loads are sampled by checked 150-Hz replay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from evaluate_g1_standing_assist import STUDENT_SHA, Measurements
from omegaconf import OmegaConf

from unilab.algos.torch.distill.standing_assist import (
    StandingAssistConfig,
    StandingAssistController,
)
from unilab.envs.locomotion.g1.recovery import recovery_stable_mask
from unilab.visualization.interactive_playback import (
    RslRlPlaybackConfig,
    create_distill_playback_session,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(case, seed, mode, output, checkpoint, *, steps_override=None):
    np.random.seed(19)
    torch.manual_seed(19)
    cfg = OmegaConf.load(ROOT / "conf/distill/config.yaml")
    del cfg["defaults"]
    cfg = OmegaConf.merge(
        cfg, OmegaConf.load(ROOT / "conf/distill/task/g1_recovery_combined/mujoco.yaml")
    )
    if mode == "missing":
        del cfg.interactive["standing_assist"]
    else:
        cfg.interactive.standing_assist = mode == "on"
    cfg.env.noise_config.level = 0.0
    cfg.env.domain_rand = {"randomize_kp": False, "randomize_kd": False}
    # This is a declared comparison workload, not a change to playback defaults.
    session, _, _ = create_distill_playback_session(
        playback_cfg=RslRlPlaybackConfig(
            task="g1_recovery_combined",
            load_run="-1",
            checkpoint=None,
            checkpoint_path=str(checkpoint),
            action_mode="policy",
            policy_obs_mode="actor",
            algo_log_name="distill",
            log_root=None,
        ),
        cfg=cfg,
        root_dir=ROOT,
        device="cpu",
        log=lambda message: None,
    )
    env = session.env
    try:
        env.set_autoreset(False)
        session.reset()
        model = env.get_playback_model(0)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        if case == "supine":
            data.qpos[2] = 0.30
            data.qpos[3:7] = [np.sqrt(0.5), 0, -np.sqrt(0.5), 0]
            data.qpos[7:] += np.random.default_rng(seed).uniform(-0.05, 0.05, 29)
            data.qpos[7:] = np.clip(data.qpos[7:], model.jnt_range[1:, 0], model.jnt_range[1:, 1])
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        env._backend.set_state(np.array([0], np.int32), data.qpos[None], data.qvel[None])
        info = env.state.info
        info["steps"].fill(0)
        info["gait_phase"][:] = [[0, np.pi]]
        info["commands"][:] = 0
        info["height_commands"][:] = 0.754
        info["current_actions"][:] = 0
        info["recovery_active"][:] = False
        info["recovery_route_stable_steps"][:] = 0
        info["recovery_settle_remaining"][:] = 0
        info["recovery_route_last_step"][:] = -1
        env.refresh_state()
        session.refresh_observation()
        # OFF's normal route refresh and ON's deferred route start on the same frame.
        if mode == "on":
            session.controller.info = info
        measurements = Measurements(model, mujoco.MjData(model))
        student = getattr(session.policy, "_unilab_distill_student_policy")
        shadow_env = copy.copy(env)
        shadow = StandingAssistController(
            shadow_env,
            student,
            StandingAssistConfig(enabled=mode == "on"),
            actuation=dict(
                kp=measurements.kp,
                kd=measurements.kd,
                joint_ranges=measurements.limits,
                force_limits=measurements.force_limits,
            ),
        )
        body = model.body("pelvis").id
        force = np.zeros((1, 1, 3))
        env.set_external_body_force_provider(lambda: (np.array([body], np.int32), force.copy()))
        budgets = {"stand": 150, "low_stand": 150, "walk": 150, "supine": 750, "push": 1000}
        budget = steps_override or budgets[case]
        trace = []
        actions = []
        targets = []
        observations = []
        phases = []
        events = []
        errors = np.zeros(3)
        histories = []
        physics_states = []
        stable_count = longest = 0
        route_rows, releases, settled = [], [], []
        previous_active, previous_settle = False, 0
        request_height = 0.754
        replay_error = 0.0
        with patch.object(env, "apply_action", wraps=env.apply_action) as apply:
            for frame in range(budget):
                force[:] = 0
                if case == "push" and 100 <= frame < 110:
                    force[0, 0, 0] = -600
                command = np.array([0.4, 0, 0] if case == "walk" else [0, 0, 0], np.float32)
                if case == "low_stand" and frame < 11:
                    request_height = max(0.65, request_height - 0.01)
                session.set_external_command(command)
                session.set_external_height(request_height)
                physical = env.recovery_playback_state()
                if mode == "on":
                    ref_target, ref = shadow.decide(
                        frame, physical, commands=command, height=request_height
                    )
                else:
                    # The old native route evaluates post-step physics on its
                    # own clock. Compare OFF to that actual legacy policy, not
                    # to the standalone evaluator's pre-step routing clock.
                    from unilab.algos.torch.distill.recovery_integration import (
                        recovery_expert_indices,
                    )
                    from unilab.visualization.interactive_playback import (
                        _distill_student_obs_tensor,
                    )

                    old_obs = _distill_student_obs_tensor(session.obs, device="cpu").numpy()[0]
                    old_action = session.policy(session.obs).numpy()[0]
                    old_index = int(recovery_expert_indices(env.state.info)[0])
                    ref = dict(
                        obs=old_obs,
                        action=old_action,
                        expert=old_index,
                        settle=int(env.state.info["recovery_settle_remaining"][0]),
                        alpha=0.0 if old_index == 2 else 1.0,
                        event="",
                        active=bool(env.state.info["recovery_active"][0]),
                        stable_count=int(env.state.info["recovery_route_stable_steps"][0]),
                    )
                    ref_target = old_action + np.asarray(env.default_angles, np.float32).reshape(29)
                before = env.get_physics_state_snapshot()[0].astype(np.float64)
                napply = apply.call_count
                session.step_once()
                assert apply.call_count == napply + 1, (
                    "formal control frame committed more than once"
                )
                actual_action = env.state.info["executed_actions"][0]
                target = (
                    env.state.info["executed_actions"] * env.cfg.control_config.action_scale
                    + env.default_angles
                )[0]
                if mode == "on":
                    extra = session.last_decision
                    actual_obs = extra["obs"]
                    for key in (
                        "expert",
                        "alpha",
                        "assisting",
                        "attempted",
                        "assist_age",
                        "event",
                        "active",
                        "stable_count",
                        "settle",
                        "load_rejected",
                    ):
                        assert extra[key] == ref[key], f"route/phase boundary {key} at {frame}"
                else:
                    extra = ref
                    # Normal session action_obs is a TensorDict containing canonical policy obs.
                    from unilab.visualization.interactive_playback import (
                        _distill_student_obs_tensor,
                    )

                    actual_obs = _distill_student_obs_tensor(
                        session.action_obs, device="cpu"
                    ).numpy()[0]
                if mode == "on":
                    np.testing.assert_array_equal(
                        env.state.info["gait_phase"], shadow.info["gait_phase"]
                    )
                    np.testing.assert_array_equal(
                        env.state.info["current_actions"], shadow.info["current_actions"]
                    )
                values = [
                    (actual_obs, ref["obs"]),
                    (actual_action, ref["action"]),
                    (target, ref_target),
                ]
                for j, (a, b) in enumerate(values):
                    errors[j] = max(errors[j], float(np.max(np.abs(a - b))))
                    np.testing.assert_array_equal(
                        a, b, err_msg=f"decision parity frame {frame} axis {j}"
                    )
                # Replay exactly the native control interval to measure actual substep loads.
                d = measurements.data
                mujoco.mj_resetData(model, d)
                mujoco.mj_setState(model, d, before, SPEC)
                d.xfrc_applied[body, :3] = force[0, 0]
                measurements.integrate(target)
                after_replay = np.empty(mujoco.mj_stateSize(model, SPEC))
                mujoco.mj_getState(model, d, after_replay, SPEC)
                error = float(
                    np.max(
                        np.abs(
                            after_replay.astype(np.float32) - env.get_physics_state_snapshot()[0]
                        )
                    )
                )
                replay_error = max(replay_error, error)
                assert error <= 1e-5, f"invalid physics replay boundary {frame}: {error}"
                state = env.recovery_playback_state()
                stable = bool(
                    recovery_stable_mask(
                        np.array([state["height"]]),
                        state["upvector"][None],
                        state["linvel"][None],
                        state["gyro"][None],
                        state["feet"][None],
                        env.cfg.recovery,
                    )[0]
                )
                stable_count = stable_count + 1 if stable else 0
                longest = max(longest, stable_count)
                tilt = float(np.rad2deg(np.arccos(np.clip(state["upvector"][2], -1, 1))))
                trace.append(
                    [
                        frame,
                        state["height"],
                        tilt,
                        np.linalg.norm(state["linvel"]),
                        np.linalg.norm(state["gyro"]),
                        int(stable),
                        extra["expert"],
                        extra["settle"],
                    ]
                )
                actions.append(actual_action.copy())
                targets.append(target.copy())
                observations.append(actual_obs.copy())
                phases.append(env.state.info["gait_phase"][0].copy())
                histories.append(env.state.info["current_actions"][0].copy())
                physics_states.append(env.get_physics_state_snapshot()[0].copy())
                active = bool(extra["active"])
                remaining = int(extra["settle"])
                if previous_active and not active:
                    releases.append(frame)
                if previous_settle > 0 and remaining == 0:
                    settled.append(frame)
                previous_active, previous_settle = active, remaining
                route_rows.append(
                    [
                        active,
                        extra["stable_count"],
                        remaining,
                        extra["alpha"],
                        extra.get("assisting", False),
                        extra.get("load_rejected", False),
                    ]
                )
                if extra["event"]:
                    events.append(
                        dict(
                            frame=frame,
                            time_s=frame * 0.02,
                            event=extra["event"],
                            alpha=extra["alpha"],
                        )
                    )
                if env.state.terminated[0]:
                    raise RuntimeError("source termination reached within fixed budget")
        output.mkdir()
        arrays = dict(
            trace=np.array(trace),
            route_state=np.asarray(route_rows),
            action=np.array(actions),
            target=np.array(targets),
            obs=np.array(observations),
            phase=np.array(phases),
            history=np.array(histories),
            physics_state=np.array(physics_states),
            joint_names=np.array(measurements.names),
            joint_ranges=measurements.limits,
            force_limits=measurements.force_limits,
        )
        arrays.update({"physics_" + k: np.array(v) for k, v in measurements.physics.items()})
        np.savez_compressed(output / "trajectory.npz", **arrays)
        (output / "collisions.json").write_text(json.dumps(measurements.collisions) + "\n")
        result = dict(
            case=case,
            seed=seed,
            mode=mode,
            steps=len(trace),
            events=events,
            release_frames=releases,
            release_times_s=[f * 0.02 for f in releases],
            settle_complete_times_s=[f * 0.02 for f in settled],
            stable_at_end=stable_count >= 50,
            longest_stable_s=longest * 0.02,
            final_height_m=float(state["height"]),
            final_active=bool(env.state.info["recovery_active"][0]),
            final_settle=int(env.state.info["recovery_settle_remaining"][0]),
            decision_errors_obs_action_target=errors.tolist(),
            action_apply_calls=apply.call_count,
            replay_max_error=replay_error,
            physics_substeps=len(measurements.physics["time"]),
            pd_formula_error=measurements.pd_error,
        )
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return result
    finally:
        if env._backend._pool is not None:
            env._backend._pool.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "dagger_iteration_2.pt")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert digest(args.checkpoint) == STUDENT_SHA
    torch.set_num_threads(1)
    cases = (
        [("stand", 0, "on")]
        if args.smoke
        else [("supine", 2, "missing"), ("supine", 2, "off")]
        + [(c, 0, "off") for c in ("stand", "low_stand", "walk")]
        + [
            (c, s, "on")
            for c, s in [
                ("stand", 0),
                ("low_stand", 0),
                ("walk", 0),
                ("supine", 0),
                ("supine", 1),
                ("supine", 2),
                ("push", 0),
            ]
        ]
    )
    results = []
    for case, seed, mode in cases:
        results.append(
            run_case(
                case,
                seed,
                mode,
                args.output / f"{mode}_{case}_{seed}",
                args.checkpoint,
                steps_override=20 if args.smoke else None,
            )
        )
    off_equal = None
    if not args.smoke:
        a = np.load(args.output / "missing_supine_2/trajectory.npz")
        b = np.load(args.output / "off_supine_2/trajectory.npz")
        off_equal = all(np.array_equal(a[k], b[k]) for k in a.files)
        assert off_equal, "explicit OFF changed old default native playback"
        for case in ("stand", "low_stand", "walk"):
            old = np.load(args.output / f"off_{case}_0/trajectory.npz")
            new = np.load(args.output / f"on_{case}_0/trajectory.npz")
            for key in ("trace", "action", "target", "obs", "phase", "history", "physics_state"):
                np.testing.assert_array_equal(
                    old[key], new[key], err_msg=f"old capability regression: {case} {key}"
                )
    assert digest(args.checkpoint) == STUDENT_SHA
    report = dict(
        checkpoint_sha256=STUDENT_SHA,
        off_default_exact=off_equal,
        cases=results,
        scope="formal factory + original wrapper + native env.step, nominal/no-noise comparison; unchanged production defaults",
        note="Independent legacy evaluator is float64 between controls; native backend persists float32 FULLPHYSICS. Same-state decision parity is checked separately from closed-loop outcome.",
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
