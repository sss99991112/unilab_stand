from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.training.backend_adapter import BackendAdapter
from unilab.training.g1_recovery import validate_recovery_training_config

ROOT = Path(__file__).resolve().parents[2]


def _cfg(profile="mujoco", overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=["algo=sac", f"task=sac/g1_recovery/{profile}", *overrides],
        )


def _parent(tmp_path):
    parent = _cfg("mujoco_assist40")
    checkpoint = tmp_path / "model_500.pt"
    checkpoint.write_bytes(b"fixture; no policy weights loaded")
    (tmp_path / "run_config.json").write_text(
        json.dumps({"config": OmegaConf.to_container(parent, resolve=True)})
    )
    return checkpoint


def _continued(tmp_path):
    checkpoint = _parent(tmp_path)
    return _cfg(
        "mujoco_assist20",
        [
            f"algo.actor_warm_start_checkpoint={checkpoint}",
            "algo.actor_warm_start_adapter=g1_height_actor_obs_99_to_99_v1",
        ],
    )


@pytest.mark.parametrize(
    "profile,fraction", [("mujoco_assist40", 0.4), ("mujoco_assist20", 0.2), ("mujoco", 0)]
)
def test_owner_profiles_compose_fixed_stages(profile, fraction):
    cfg = _cfg(profile)
    validate_recovery_training_config(cfg)
    assert cfg.env.assistance.force_fractions[cfg.env.assistance.stage] == fraction
    assert cfg.env.control_config.action_scale == 1


def test_playback_forces_assistance_off_even_if_override_requests_it():
    cfg = _cfg("mujoco_assist40", ["training.play_only=true", "env.assistance.evaluation=false"])
    overrides = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    assert overrides["assistance"]["evaluation"] is True


def test_continuation_accepts_same_or_next_stage_with_fixed_interface(tmp_path):
    cfg = _continued(tmp_path)
    validate_recovery_training_config(cfg)
    cfg.env.assistance.stage = 0
    validate_recovery_training_config(cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("env.assistance.stage", 2),
        ("env.control_config.action_scale", 0.5),
        ("algo.actor_warm_start_adapter", "g1_height_actor_obs_98_to_99_v1"),
        ("algo.load_run", "old-run"),
        ("env.assistance.evaluation", True),
        ("algo.obs_normalization", False),
        ("env.assistance.body_name", "pelvis"),
    ],
)
def test_incompatible_continuation_fails_closed(tmp_path, field, value):
    cfg = _continued(tmp_path)
    OmegaConf.update(cfg, field, value)
    with pytest.raises(ValueError):
        validate_recovery_training_config(cfg)


def test_continuation_requires_sidecar_and_new_output_directory(tmp_path):
    cfg = _continued(tmp_path)
    cfg.training.log_dir = str(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        validate_recovery_training_config(cfg)
    cfg.training.log_dir = None
    (tmp_path / "run_config.json").unlink()
    with pytest.raises(ValueError, match="stage identity"):
        validate_recovery_training_config(cfg)


def test_other_tasks_are_untouched():
    validate_recovery_training_config(
        OmegaConf.create({"training": {"task_name": "G1StandHeight"}})
    )
