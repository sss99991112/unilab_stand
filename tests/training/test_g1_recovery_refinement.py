"""New reward initialization must preserve the actor, never learner state."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.fast_sac.learner import FastSACLearner
from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.training.g1_recovery import validate_recovery_training_config
from unilab.training.g1_recovery_refinement import (
    load_refinement_actor,
    validate_fresh_refinement_output,
)
from unilab.training.offpolicy_resume import validate_configured_learner_resume

ROOT = Path(__file__).resolve().parents[2]


def cfg(profile="mujoco_constrained_v1"):
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[f"task=sac/g1_recovery/{profile}", "algo.actor_hidden_dim=32"],
        )


def learner():
    return FastSACLearner(
        obs_dim=99,
        action_dim=29,
        critic_obs_dim=102,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        num_atoms=11,
        device="cpu",
        use_compile=False,
    )


@pytest.fixture
def prepared(tmp_path):
    old = cfg("mujoco_unassisted_v2")
    src = learner()
    with torch.no_grad():
        for p in src.actor.parameters():
            p.normal_(0, 0.1)
        for p in src.qnet.parameters():
            p.fill_(0.7)
    checkpoint = tmp_path / "model_12000.pt"
    torch.save(src.get_state_dict(), checkpoint)
    (tmp_path / "run_config.json").write_text(
        json.dumps({"config": OmegaConf.to_container(old, resolve=True)})
    )
    current = cfg()
    current.env.recovery_constraints.enabled = True
    current.algo.actor_warm_start_checkpoint = str(checkpoint)
    current.algo.actor_warm_start_expected_sha256 = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    current.training.log_dir = str(tmp_path / "new_run")
    return current, src, checkpoint


def test_defaults_off_and_all_costs_configured_together():
    c = cfg()
    assert not c.env.recovery_constraints.enabled
    assert c.algo.actor_warm_start_checkpoint is None
    assert c.algo.resume_checkpoint is None
    assert c.algo.learning_starts <= c.algo.replay_buffer_n
    validate_recovery_training_config(c)


def test_real_cli_initializes_only_actor_and_keeps_critic_target_alpha_optimizers_cold(prepared):
    from scripts.train_offpolicy import apply_configured_actor_warm_start

    c, source, path = prepared
    validate_recovery_training_config(c)
    validate_fresh_refinement_output(c)
    dst = learner()
    before = copy.deepcopy(dst.get_state_dict())
    metadata = apply_configured_actor_warm_start("sac", c, SimpleNamespace(learner=dst))
    observations = torch.linspace(-1, 1, 99 * 7).reshape(7, 99)
    with torch.no_grad():
        for a, b in zip(source.actor(observations), dst.actor(observations)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for name in ("qnet", "qnet_target"):
        for k, v in before[name].items():
            torch.testing.assert_close(v, getattr(dst, name).state_dict()[k], rtol=0, atol=0)
    torch.testing.assert_close(dst.log_alpha, before["log_alpha"], rtol=0, atol=0)
    assert dst.update_count == 0
    assert all(not opt.state for opt in (dst.actor_optimizer, dst.q_optimizer, dst.alpha_optimizer))
    assert metadata["action_contract"] == "g1_joint_offset_default_angles_scale1_v1"
    assert dst.get_state_dict()["actor_warm_start"]["replay_initialization"] == "cold"


@pytest.mark.parametrize(
    "field,value",
    [
        ("algo.resume_checkpoint", "old.pt"),
        ("algo.actor_warm_start_adapter", "g1_height_actor_obs_99_to_99_v1"),
        ("algo.actor_warm_start_expected_sha256", "0" * 64),
        ("env.control_config.action_scale", 0.5),
        ("env.recovery.stable_seconds", 0.5),
        ("reward.scales.recovery_progress", 9.0),
        ("reward.scales.recovery_pd_excess", float("nan")),
        ("algo.learning_starts", 513),
    ],
)
def test_mixed_or_incompatible_initialization_rejected(prepared, field, value):
    c, _, _ = prepared
    OmegaConf.update(c, field, value)
    with pytest.raises(ValueError):
        validate_recovery_training_config(c)


def test_old_full_resume_still_rejects_changed_reward(prepared):
    c, _, p = prepared
    c.algo.actor_warm_start_checkpoint = None
    c.algo.actor_warm_start_adapter = None
    c.algo.resume_checkpoint = str(p)
    with pytest.raises(ValueError, match="config mismatch"):
        validate_configured_learner_resume(c)


def test_existing_run_never_overwritten(prepared):
    c, _, _ = prepared
    Path(c.training.log_dir).mkdir()
    with pytest.raises(ValueError, match="already exists"):
        validate_fresh_refinement_output(c)


@pytest.mark.parametrize("key", ["obs_normalizer", "action_scale"])
def test_changed_normalizer_or_action_mapping_requires_another_version(prepared, key):
    _, _, p = prepared
    payload = torch.load(p, weights_only=True)
    if key == "obs_normalizer":
        payload[key] = {"fake": torch.ones(99)}
    else:
        payload["actor"][key].fill_(0.5)
    altered = p.parent / "altered.pt"
    torch.save(payload, altered)
    with pytest.raises(ValueError):
        load_refinement_actor(learner(), altered)


def test_cold_replay_one_batch_receives_changed_reward_and_can_update(prepared):
    _, _, p = prepared
    dst = learner()
    load_refinement_actor(dst, p)
    replay = ReplayBuffer(capacity=16, obs_dim=99, action_dim=29, critic_dim=102, device="cpu")
    assert int(replay.size[0]) == 0
    replay.add(
        torch.zeros(8, 99),
        torch.zeros(8, 29),
        torch.full((8,), -0.3),
        torch.zeros(8, 99),
        torch.zeros(8),
        torch.zeros(8),
        critic=torch.zeros(8, 102),
        next_critic=torch.zeros(8, 102),
    )
    batch = replay.sample(8)
    torch.testing.assert_close(batch["rewards"], torch.full((8,), -0.3))
    # One throwaway unit-test update, no runner, checkpoint save or training run.
    values = dst.update_critic(batch) | dst.update_actor(batch)
    assert all(np.isfinite(v) for v in values.values())
    assert dst.q_optimizer.state and dst.actor_optimizer.state
