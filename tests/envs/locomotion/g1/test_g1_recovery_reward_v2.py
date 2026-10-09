"""Semantic reward and owner-config checks; no policy convergence claims."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.cli import build_route
from unilab.envs.locomotion.g1.recovery import G1RecoveryCfg, RecoveryConfig, recovery_reward_terms
from unilab.training.backend_adapter import BackendAdapter
from unilab.training.g1_recovery import validate_recovery_training_config

ROOT = Path(__file__).resolve().parents[4]


def _terms(heights, up, cfg):
    heights = np.asarray(heights, dtype=float)
    gravity = np.zeros((len(heights), 3))
    gravity[:, 2] = up
    return recovery_reward_terms(
        heights,
        gravity,
        np.zeros_like(gravity),
        np.zeros_like(gravity),
        np.zeros((len(heights), 29)),
        np.zeros((len(heights), 29)),
        np.zeros((len(heights), 29)),
        np.zeros(len(heights), dtype=bool),
        cfg,
        0.754,
    )


def test_legacy_reward_is_unchanged_when_new_gate_is_off():
    heights = np.asarray([0.07, 0.12, 0.3, 0.65, 0.754])
    up = np.asarray([0.0, 0.4, 0.6, 0.9, 1.0])
    cfg = RecoveryConfig()
    assert cfg.height_gated_upright is False
    terms = _terms(heights, up, cfg)
    expected = np.clip((up + 1) / 2, 0, 1)
    np.testing.assert_array_equal(terms["recovery_upright"], expected)
    progress = np.clip((heights - 0.12) / (0.754 - 0.12), 0, 1)
    np.testing.assert_array_equal(terms["recovery_progress"], progress * (0.1 + 0.9 * expected))


def test_v2_suppresses_low_body_pose_reward_and_retains_dense_lifting_signal():
    heights = np.asarray([0.06, 0.07, 0.08, 0.12, 0.3, 0.65, 0.754])
    cfg = RecoveryConfig(height_gated_upright=True, progress_floor_height=0.06)
    terms = _terms(heights, np.ones(len(heights)), cfg)
    assert terms["recovery_upright"][0] == 0
    assert terms["recovery_upright"][1] < 0.02
    assert np.all(np.diff(terms["recovery_upright"]) > 0)
    assert np.all(np.diff(terms["recovery_progress"]) > 0)
    assert terms["recovery_upright"][-1] == 1
    assert terms["recovery_progress"][-1] == 1
    horizontal = _terms([0.3], [0.0], cfg)
    assert horizontal["recovery_progress"][0] > 0  # No hard upright gate before lifting.


def test_v2_full_height_upright_is_better_than_inverted_and_low_sitting():
    cfg = RecoveryConfig(height_gated_upright=True, progress_floor_height=0.06)
    terms = _terms([0.072, 0.754, 0.754], [0.36, 1.0, -1.0], cfg)
    positive = 5 * terms["recovery_progress"] + terms["recovery_upright"]
    assert positive[1] > positive[2] > positive[0]
    assert positive[0] < 0.1


def test_v2_profile_connects_cli_env_and_training_with_assistance_off():
    registry.ensure_registries()
    route = build_route("sac", "g1_recovery", "mujoco", profile="unassisted_v2")
    assert route.owner_task == "sac/g1_recovery/mujoco_unassisted_v2.yaml"
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algo=sac",
                "task=sac/g1_recovery/mujoco_unassisted_v2",
            ],
        )
    validate_recovery_training_config(cfg)
    assert cfg.algo.actor_warm_start_checkpoint is None
    assert str(cfg.algo.load_run) == "-1"
    assert cfg.algo.runtime_impl is None and cfg.algo.runtime_resolver is None
    assert cfg.algo.num_envs == 128 and cfg.algo.max_iterations == 50000
    overrides = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    resolved = G1RecoveryCfg()
    registry.apply_cfg_overrides(resolved, overrides)
    resolved.recovery.validate(resolved.commands.default_height, resolved.ctrl_dt)
    assert resolved.assistance.effective_fraction == 0
    assert resolved.recovery.height_gated_upright
    assert resolved.recovery.progress_floor_height == 0.06
    assert resolved.control_config.action_scale == 1.0
    play = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    play.training.play_only = True
    play_override = BackendAdapter(play, root_dir=ROOT).build_task_env_cfg_override()
    assert play_override["recovery"] == overrides["recovery"]
    assert play_override["reward_config"] == overrides["reward_config"]
    assert play_override["assistance"]["evaluation"] is True


def test_height_gate_requires_boolean():
    with pytest.raises(ValueError, match="boolean"):
        RecoveryConfig(height_gated_upright=1).validate(0.754, 0.02)
