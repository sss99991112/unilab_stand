"""Playback diagnostics must use the live model after per-env reset randomization."""

import numpy as np
import pytest

from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg
from unilab.dr.types import ResetRandomizationPayload


def test_playback_model_contains_per_env_reset_parameters_and_preserves_nominal_model(tmp_path):
    source = tmp_path / "model.xml"
    source.write_text(
        '<mujoco><worldbody><body name="root" pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size=".1" mass="1"/><body name="arm" pos=".2 0 0">'
        '<joint name="hinge" range="-90 90"/><geom type="sphere" size=".1" mass="1"/>'
        '</body></body></worldbody><actuator><position joint="hinge" kp="10" kv="1"/>'
        "</actuator></mujoco>"
    )
    backend = MuJoCoBackend(SceneCfg(model_file=str(source)), 2, 0.002, base_name="root")
    try:
        assert backend.get_playback_model(0).actuator_gainprm[0, 0] == 10
        backend.materialize()
        qpos = backend.model.qpos0[None, :].copy()
        qvel = np.zeros((1, backend.model.nv))
        fields = {
            name: getattr(backend.model, name).copy()
            for name in (
                "body_mass",
                "body_inertia",
                "body_ipos",
                "body_iquat",
                "geom_friction",
                "dof_armature",
            )
        }
        fields["body_mass"][1:] *= 1.1
        fields["body_inertia"][1:] *= 1.1
        fields["body_ipos"][1, 0] += 0.01
        fields["body_iquat"][1] = [np.cos(0.1), np.sin(0.1), 0, 0]
        fields["geom_friction"] *= 0.8
        fields["dof_armature"][6] = 0.05
        backend.set_state(
            np.array([0], dtype=np.int32),
            qpos,
            qvel,
            randomization=ResetRandomizationPayload(
                **{name: value[None, ...] for name, value in fields.items()},
                gravity=np.array([[0.0, 0.0, -9.0]]),
                kp=np.array([[20.0]]),
                kd=np.array([[2.0]]),
            ),
        )
        model0, model1 = backend.get_playback_model(0), backend.get_playback_model(1)
        for name, value in fields.items():
            np.testing.assert_allclose(getattr(model0, name), value)
            np.testing.assert_array_equal(getattr(model1, name), getattr(backend.model, name))
        np.testing.assert_array_equal(model0.opt.gravity, [0.0, 0.0, -9.0])
        np.testing.assert_array_equal(model1.opt.gravity, backend.model.opt.gravity)
        assert backend.get_playback_model() is backend.model
        assert model0.actuator_gainprm[0, 0] == 20
        np.testing.assert_array_equal(model0.actuator_biasprm[0, 1:3], [-20, -2])
        assert model1.actuator_gainprm[0, 0] == 10
        assert backend.model.actuator_gainprm[0, 0] == 10
        with pytest.raises(IndexError):
            backend.get_playback_model(-1)
        with pytest.raises(IndexError):
            backend.get_playback_model(2)
    finally:
        if backend._pool is not None:
            backend._pool.close()
        backend.cleanup_scene_assets()


def test_playback_snapshot_keeps_motor_actuator_semantics(tmp_path):
    source = tmp_path / "motor.xml"
    source.write_text(
        '<mujoco><worldbody><body name="root" pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size=".1" mass="1"/><body name="arm" pos=".2 0 0">'
        '<joint name="hinge"/><geom type="sphere" size=".1" mass="1"/>'
        '</body></body></worldbody><actuator><motor joint="hinge"/>'
        "</actuator></mujoco>"
    )
    backend = MuJoCoBackend(SceneCfg(model_file=str(source)), 1, 0.002, base_name="root")
    try:
        backend.materialize()
        snapshot = backend.get_playback_model(0)
        np.testing.assert_array_equal(snapshot.actuator_gainprm, backend.model.actuator_gainprm)
        np.testing.assert_array_equal(snapshot.actuator_biasprm, backend.model.actuator_biasprm)
        # A diagnostic snapshot must never mutate the native simulation pool.
        snapshot.body_mass[1] *= 2
        assert backend._pool.get_field(0, "body_mass")[1] == backend.model.body_mass[1]
    finally:
        if backend._pool is not None:
            backend._pool.close()
        backend.cleanup_scene_assets()
