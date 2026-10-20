"""Bounded one-env, frozen-policy handover experiment; no hardware or training.

Uses UniLab's nominal playback model, canonical observation builder, route and
PD-target execution owner. The new controller is enabled only by --standing-assist.
Raw traces remain local. Output must be a fresh directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.distill import load_distillation_student_policy
from unilab.algos.torch.distill.recovery_integration import validate_recovery_student
from unilab.algos.torch.distill.standing_assist import (
    StandingAssistConfig,
    StandingAssistController,
)
from unilab.envs.locomotion.g1.recovery import recovery_stable_mask
from unilab.training import BackendAdapter, create_env, ensure_registries

STUDENT_SHA = "27d06b8d73734fb2a17c730aa63ff70b2ba4ffbef7d3d71373df43974ab9bf18"
EXPORT_SHA = "1435ecffdb457eb8d2cc41445c492c87322f4e318e2ed5186276d66c363e384d"
MODEL_FIELDS = (
    "body_mass",
    "body_inertia",
    "dof_armature",
    "dof_damping",
    "jnt_range",
    "geom_friction",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "actuator_ctrlrange",
    "actuator_ctrllimited",
    "actuator_forcelimited",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_hash(model):
    return {key: hashlib.sha256(getattr(model, key).tobytes()).hexdigest() for key in MODEL_FIELDS}


def source_env(root):
    ensure_registries()
    cfg = OmegaConf.load(root / "conf/distill/config.yaml")
    del cfg["defaults"]
    cfg = OmegaConf.merge(
        cfg, OmegaConf.load(root / "conf/distill/task/g1_recovery_combined/mujoco.yaml")
    )
    cfg.env.noise_config.level = 0.0
    return create_env(
        cfg,
        num_envs=1,
        sim_backend="mujoco",
        task_name="G1RecoveryCombined",
        env_cfg_override=BackendAdapter(
            cfg, root_dir=root, algo_name="distill"
        ).build_task_env_cfg_override(),
    )


class Measurements:
    def __init__(self, model, data):
        self.model, self.data = model, data
        joints = model.actuator_trnid[:, 0]
        self.qadr, self.vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.names = [model.joint(int(j)).name for j in joints]
        self.limits = model.jnt_range[joints].copy()
        self.kp, self.kd = model.actuator_gainprm[:, 0].copy(), -model.actuator_biasprm[:, 2].copy()
        self.force_limits = model.actuator_forcerange.copy()
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.sensors = {}
        for name in ["torso_gyro", "torso_upvector", "pelvis_local_linvel"] + [
            f"{side}_foot_contact_{i}" for side in ("left", "right") for i in range(4)
        ]:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if index < 0:
                raise ValueError(f"missing physical sensor {name}")
            self.sensors[name] = slice(
                int(model.sensor_adr[index]), int(model.sensor_adr[index] + model.sensor_dim[index])
            )
        self.physics, self.collisions = defaultdict(list), []
        self.pd_error = 0.0

    def read(self, name):
        return self.data.sensordata[self.sensors[name]].astype(np.float32)

    def state(self):
        feet_raw = np.array(
            [
                self.read(f"{side}_foot_contact_{i}")[0]
                for side in ("left", "right")
                for i in range(4)
            ],
            np.float32,
        )
        return dict(
            height=np.float32(self.data.qpos[2] - self.model.geom_pos[self.floor, 2]),
            upvector=self.read("torso_upvector"),
            gyro=self.read("torso_gyro"),
            linvel=self.read("pelvis_local_linvel"),
            feet=(feet_raw.reshape(2, 4) > 0.5).any(axis=1),
            foot_sensors=feet_raw,
            q=self.data.qpos[self.qadr].astype(np.float32),
            dq=self.data.qvel[self.vadr].astype(np.float32),
        )

    def integrate(self, target):
        model, data = self.model, self.data
        for _ in range(3):
            q, dq = data.qpos[self.qadr].copy(), data.qvel[self.vadr].copy()
            time = float(data.time)
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
            raw = self.kp * (target - q) - self.kd * dq
            tau = data.actuator_force.copy()
            self.pd_error = max(
                self.pd_error,
                float(
                    np.max(
                        np.abs(tau - np.clip(raw, self.force_limits[:, 0], self.force_limits[:, 1]))
                    )
                ),
            )
            assert self.pd_error < 1e-6
            ground = np.zeros(2)
            pairs = defaultdict(float)
            wrench = np.empty(6)
            for i in range(data.ncon):
                contact = data.contact[i]
                a, b = int(contact.geom1), int(contact.geom2)
                mujoco.mj_contactForce(model, data, i, wrench)
                force = abs(float(wrench[0]))
                if self.floor in (a, b):
                    name = model.geom(b if a == self.floor else a).name or ""
                    if name.startswith("left_foot"):
                        ground[0] += force
                    elif name.startswith("right_foot"):
                        ground[1] += force
                elif force > 0:
                    ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
                    if ba and bb and ba != bb:
                        names = sorted([model.body(ba).name, model.body(bb).name])
                        pairs[" <-> ".join(names)] += force
            if pairs:
                self.collisions.append(dict(time_s=time, pairs_normal_N=dict(pairs)))
            for key, value in dict(
                time=time,
                q=q,
                dq=dq,
                target=target,
                tau=tau,
                raw_tau=raw,
                foot_normal_force=ground,
                max_self_contact=max(pairs.values(), default=0.0),
            ).items():
                self.physics[key].append(np.asarray(value).copy())


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    assert sha(checkpoint) == STUDENT_SHA
    assert sha(args.exported_policy) == EXPORT_SHA
    torch.set_num_threads(1)
    loaded = load_distillation_student_policy(checkpoint, device="cpu")
    validate_recovery_student(loaded.policy, loaded.distill_runtime_cfg)
    exported = torch.jit.load(str(args.exported_policy)).eval()
    env = source_env(root)
    try:
        model = env.get_playback_model()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        if args.case == "supine":
            data.qpos[2] = 0.3
            data.qpos[3:7] = [np.sqrt(0.5), 0, -np.sqrt(0.5), 0]
            data.qpos[7:] += np.random.default_rng(args.seed).uniform(-0.05, 0.05, 29)
            data.qpos[7:] = np.clip(data.qpos[7:], model.jnt_range[1:, 0], model.jnt_range[1:, 1])
        data.time = 0.0
        data.qvel[:] = 0.0
        data.qfrc_applied[:] = data.xfrc_applied[:] = data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(model, data)
        measurement = Measurements(model, data)
        before = model_hash(model)
        initial = dict(initial_qpos=data.qpos.copy(), initial_qvel=data.qvel.copy())
        controller = StandingAssistController(
            env,
            loaded.policy,
            StandingAssistConfig(enabled=args.standing_assist),
            actuation=dict(
                kp=measurement.kp,
                kd=measurement.kd,
                joint_ranges=measurement.limits,
                force_limits=measurement.force_limits,
            ),
        )
        decisions, states = defaultdict(list), defaultdict(list)
        trace, events, preflights = [], [], []
        budget = (
            150
            if args.case in ("stand", "walk", "low_stand")
            else (1000 if args.case == "push" else 750)
        )
        longest = stable_run = 0
        final_error = None
        max_export_error = 0.0
        height_request = 0.754
        for frame in range(budget):
            if args.case == "push":
                data.xfrc_applied[model.body("pelvis").id, 0] = (
                    -600.0 if 100 <= frame < 110 else 0.0
                )
            state = measurement.state()
            commands = (0.4, 0, 0) if args.case == "walk" else (0, 0, 0)
            if args.case == "low_stand" and frame < 11:
                height_request = max(0.65, height_request - 0.01)
            height = height_request
            target, extra = controller.decide(frame, state, commands=commands, height=height)
            # Verify both frozen heads through the real export before applying anything.
            if extra["assisting"]:
                pairs = [
                    (extra["recovery_obs"], 2, extra["recovery_action"]),
                    (extra["obs"], 1, extra["standing_action"]),
                ]
            else:
                pairs = [(extra["obs"], extra["expert"], extra["action"])]
            for obs, index, expected in pairs:
                with torch.inference_mode():
                    actual = exported(torch.from_numpy(obs)[None], torch.tensor([index])).numpy()[0]
                error = float(np.max(np.abs(actual - expected)))
                max_export_error = max(max_export_error, error)
                np.testing.assert_array_equal(actual, expected)
            if extra["event"]:
                event = dict(frame=frame, time_s=float(data.time), event=extra["event"])
                events.append(event)
                print(json.dumps(event), flush=True)
            if extra["event"] == "assist_start":
                # First blended action must add no joint-range/force-limit violation
                # relative to the recovery head at this exact physical state.
                raw = measurement.kp * (target - state["q"]) - measurement.kd * state["dq"]
                recovery_target = extra["recovery_action"] + np.asarray(
                    env.default_angles, np.float32
                ).reshape(29)
                raw_reference = (
                    measurement.kp * (recovery_target - state["q"]) - measurement.kd * state["dq"]
                )
                limits = measurement.force_limits
                new_saturation = ((raw < limits[:, 0]) | (raw > limits[:, 1])) & ~(
                    (raw_reference < limits[:, 0]) | (raw_reference > limits[:, 1])
                )
                ranges = measurement.limits
                excess = np.maximum(np.maximum(ranges[:, 0] - target, target - ranges[:, 1]), 0)
                ref_excess = np.maximum(
                    np.maximum(ranges[:, 0] - recovery_target, recovery_target - ranges[:, 1]), 0
                )
                gate = dict(
                    frame=frame,
                    time_s=float(data.time),
                    new_saturated_joints=np.asarray(measurement.names)[new_saturation].tolist(),
                    max_target_excess_rad=float(excess.max()),
                    recovery_target_excess_rad=float(ref_excess.max()),
                    max_target_jump_rad=float(np.max(np.abs(target - decisions["target"][-1]))),
                    max_raw_PD_Nm=float(np.max(np.abs(raw))),
                )
                preflights.append(gate)
                if new_saturation.any() or (excess > ref_excess + 1e-7).any():
                    final_error = "first blended action failed same-state preflight; not applied"
                    break
            for key, value in {
                **state,
                **extra,
                "target": target,
                "time": float(data.time),
            }.items():
                decisions[key].append(np.asarray(value).copy())
            measurement.integrate(target)
            after = measurement.state()
            if not all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.sensordata)):
                final_error = "nonfinite physics"
                break
            if np.linalg.norm(after["linvel"]) > 20 or np.max(np.abs(after["dq"])) > 40:
                final_error = "original velocity termination boundary"
                break
            stable = bool(
                recovery_stable_mask(
                    np.array([after["height"]]),
                    after["upvector"][None],
                    after["linvel"][None],
                    after["gyro"][None],
                    after["feet"][None],
                    env.cfg.recovery,
                )[0]
            )
            stable_run = stable_run + 1 if stable else 0
            longest = max(longest, stable_run)
            for key, value in {**after, "time": float(data.time), "stable": stable}.items():
                states[key].append(np.asarray(value).copy())
            tilt = float(np.rad2deg(np.arccos(np.clip(after["upvector"][2], -1, 1))))
            trace.append(
                [
                    frame,
                    after["height"],
                    tilt,
                    np.linalg.norm(after["linvel"]),
                    np.linalg.norm(after["gyro"]),
                    int(stable),
                    extra["expert"],
                    extra["settle"],
                ]
            )
        assert before == model_hash(model)
        assert sha(checkpoint) == STUDENT_SHA and sha(args.exported_policy) == EXPORT_SHA
        arrays = {f"decision_{k}": np.asarray(v) for k, v in decisions.items()}
        arrays.update({f"state_{k}": np.asarray(v) for k, v in states.items()})
        arrays.update({f"physics_{k}": np.asarray(v) for k, v in measurement.physics.items()})
        arrays.update(
            initial,
            joint_names=np.asarray(measurement.names),
            joint_ranges=measurement.limits,
            kp=measurement.kp,
            kd=measurement.kd,
            force_limits=measurement.force_limits,
            trace=np.asarray(trace),
        )
        np.savez_compressed(args.output / "trajectory.npz", **arrays)
        parity = None
        if args.reference_trace is not None:
            reference = np.load(args.reference_trace)["trace"]
            parity = bool(np.array_equal(reference, arrays["trace"]))
            if not parity:
                final_error = "OFF source playback differs from historical RoboJudo trace"
        result = dict(
            case=args.case,
            seed=args.seed,
            standing_assist=args.standing_assist,
            completed_steps=len(trace),
            limit_steps=budget,
            physical_substeps=len(measurement.physics["time"]),
            longest_stable_s=longest * 0.02,
            stable_at_end=stable_run >= 50,
            active=bool(controller.info["recovery_active"][0]),
            settle_remaining=int(controller.info["recovery_settle_remaining"][0]),
            events=events,
            first_blend_preflights=preflights,
            stop_reason=final_error,
            historical_trace_exact=parity,
            max_source_export_action_error=max_export_error,
            pd_formula_max_error_Nm=measurement.pd_error,
            physics_hashes=before,
            python=platform.python_version(),
            numpy=np.__version__,
            torch=torch.__version__,
            mujoco=mujoco.__version__,
            checkpoint_sha256=STUDENT_SHA,
            export_sha256=EXPORT_SHA,
            scope="nominal one-env simulation, no observation noise, no training or hardware",
            code_sha256={
                str(p.relative_to(root)): sha(p)
                for p in [
                    Path(__file__),
                    root / "src/unilab/algos/torch/distill/standing_assist.py",
                ]
            },
        )
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output / "collision_events.json").write_text(
            json.dumps(measurement.collisions, indent=2) + "\n"
        )
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("physics_hashes", "code_sha256")},
                indent=2,
            ),
            flush=True,
        )
        if final_error:
            raise RuntimeError(final_error)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exported-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--standing-assist", action="store_true")
    parser.add_argument(
        "--case", choices=("stand", "walk", "low_stand", "supine", "push"), default="supine"
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--reference-trace", type=Path)
    args = parser.parse_args()
    if args.standing_assist and args.reference_trace:
        parser.error(
            "historical parity is an OFF-only assertion; candidate behavior intentionally differs"
        )
    run(args)


if __name__ == "__main__":
    main()
