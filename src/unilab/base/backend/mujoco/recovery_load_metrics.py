"""Exact opt-in substep load replay beside the untouched native control step.

Native BatchEnvPool exposes only terminal state/sensors. This observer pays
an additional physics replay to retain 150-Hz loads without splitting or
changing native integration. A replay mismatch stops, rather than fabricating
a reward. Independently owned reset-randomized models are refreshed by owner.
"""

import mujoco
import numpy as np


class RecoveryLoadSampler:
    def __init__(self, models, body_pairs):
        self.body_pairs = tuple(tuple(p) for p in body_pairs)
        self.models, self.data, self.indices, self.geom_pairs = [], [], [], []
        self.metrics = None
        for i, model in enumerate(models):
            self.replace_model(i, model)

    def replace_model(self, i, model):
        if i < len(self.models):
            # Reset randomizes model values, not names/topology/joint ordering.
            # Keep init-cached metadata; do not resolve assets on reset.
            self.models[i] = model
            self.data[i] = mujoco.MjData(model)
            if self.metrics is not None:
                self.metrics["valid"][i] = False
            return
        j = model.actuator_trnid[:, 0]
        if model.nu != 29 or len(set(j.tolist())) != 29:
            raise ValueError("recovery loads require 29 distinct scalar position actuators")
        if (
            np.any(model.actuator_ctrllimited)
            or np.any(model.actuator_dyntype)
            or np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT)
            or np.any(model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
            or np.any(model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE)
            or not np.all(model.actuator_forcelimited)
            or not np.all(model.jnt_limited[j])
        ):
            raise ValueError("unsupported physical position-actuator load contract")
        np.testing.assert_array_equal(model.actuator_gear[:, 0], np.ones(29))
        np.testing.assert_array_equal(model.actuator_gear[:, 1:], np.zeros((29, 5)))
        np.testing.assert_array_equal(model.actuator_biasprm[:, 0], np.zeros(29))
        np.testing.assert_array_equal(model.actuator_biasprm[:, 1], -model.actuator_gainprm[:, 0])
        if (
            np.any(model.actuator_gainprm[:, 0] <= 0)
            or np.any(model.actuator_biasprm[:, 2] > 0)
            or np.any(model.actuator_forcerange[:, 0] >= 0)
            or np.any(model.actuator_forcerange[:, 1] <= 0)
        ):
            raise ValueError("positive PD gains and signed nonzero force ranges required")
        pair_ids = []
        for a, b in self.body_pairs:
            a_id, b_id = model.body(a).id, model.body(b).id
            if not a_id or not b_id or a_id == b_id:
                raise ValueError("load pairs must name distinct robot bodies, never world")
            pair_ids.append(frozenset((a_id, b_id)))
        values = (
            model,
            mujoco.MjData(model),
            (model.jnt_qposadr[j], model.jnt_dofadr[j]),
            pair_ids,
        )
        for arr, value in zip((self.models, self.data, self.indices, self.geom_pairs), values):
            if i == len(arr):
                arr.append(value)
            else:
                arr[i] = value
        if self.metrics is not None:
            self.metrics["valid"][i] = False

    def capture(self, before, control, xfrc, after, nsteps):
        if nsteps != 3 or any(not np.isclose(m.opt.timestep * nsteps, 0.02) for m in self.models):
            raise ValueError("recovery metric contract requires 50Hz control / 150Hz physics")
        q, dq, raw, forces, ranges, caps = [], [], [], [], [], []
        error = 0.0
        for i, (m, d) in enumerate(zip(self.models, self.data)):
            mujoco.mj_resetData(m, d)
            mujoco.mj_setState(
                m, d, before[i].astype(np.float64), mujoco.mjtState.mjSTATE_FULLPHYSICS
            )
            d.ctrl[:] = control[i]
            d.xfrc_applied[:] = xfrc[i].reshape(m.nbody, 6)
            qa, va = self.indices[i]
            qi, vi, ri, fi = [], [], [], []
            for _ in range(nsteps):
                request = (
                    m.actuator_gainprm[:, 0] * (control[i] - d.qpos[qa])
                    + m.actuator_biasprm[:, 2] * d.qvel[va]
                )
                mujoco.mj_step(m, d)
                np.testing.assert_allclose(
                    d.actuator_force,
                    np.clip(request, m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1]),
                    atol=1e-6,
                    rtol=0,
                )
                pair_force = np.zeros(len(self.body_pairs))
                wrench = np.empty(6)
                for k in range(d.ncon):
                    c = d.contact[k]
                    bodies = frozenset((int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])))
                    for pair, ids in enumerate(self.geom_pairs[i]):
                        if bodies == ids:
                            mujoco.mj_contactForce(m, d, k, wrench)
                            pair_force[pair] += abs(wrench[0])
                qi.append(d.qpos[qa].copy())
                vi.append(d.qvel[va].copy())
                ri.append(request.copy())
                fi.append(pair_force)
            terminal = np.empty(len(before[i]))
            mujoco.mj_getState(m, d, terminal, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            err = float(np.max(np.abs(terminal.astype(after.dtype) - after[i])))
            error = max(error, err)
            if err != 0:
                raise RuntimeError(
                    f"recovery load replay differs from native physics: env={i}, error={err}"
                )
            q.append(qi)
            dq.append(vi)
            raw.append(ri)
            forces.append(fi)
            j = m.actuator_trnid[:, 0]
            ranges.append(m.jnt_range[j].copy())
            caps.append(m.actuator_forcerange.copy())
        self.metrics = dict(
            q=np.array(q),
            dq=np.array(dq),
            raw_pd=np.array(raw),
            pair_normal_force=np.array(forces),
            joint_ranges=np.array(ranges),
            force_ranges=np.array(caps),
            valid=np.ones(len(q), bool),
            replay_error=error,
        )

    def read(self):
        return (
            None
            if self.metrics is None
            else {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.metrics.items()}
        )
