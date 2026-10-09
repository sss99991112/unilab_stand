"""Same-task restore must retain critic, target, optimizer moments and counters."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.fast_sac.learner import FastSACLearner
from unilab.training.g1_recovery import validate_recovery_training_config
from unilab.training.offpolicy_resume import (
    apply_configured_learner_resume,
    validate_configured_learner_resume,
)

ROOT = Path(__file__).resolve().parents[2]


def _cfg():
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "algo=sac",
                "task=sac/g1_recovery/mujoco_unassisted_v2",
                "algo.actor_hidden_dim=32",
                "algo.critic_hidden_dim=32",
                "algo.num_atoms=11",
            ],
        )


def _learner():
    return FastSACLearner(
        obs_dim=99,
        critic_obs_dim=102,
        action_dim=29,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        num_atoms=11,
        device="cpu",
        use_compile=False,
        use_amp=False,
    )


def _source(tmp_path):
    cfg = _cfg()
    learner = _learner()
    # Populate actual optimizer moments without a simulator or RL training loop.
    for optimizer in (learner.actor_optimizer, learner.q_optimizer, learner.alpha_optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.full_like(parameter, 0.03)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for parameter in learner.qnet_target.parameters():
            parameter.add_(0.1)
    learner.update_count = 15000
    source = copy.deepcopy(learner.get_state_dict())
    checkpoint = tmp_path / "model_15000.pt"
    torch.save(source, checkpoint)
    (tmp_path / "run_config.json").write_text(
        json.dumps({"config": OmegaConf.to_container(cfg, resolve=True)})
    )
    cfg.algo.resume_checkpoint = str(checkpoint)
    cfg.algo.max_iterations = 15000
    cfg.algo.learning_starts = 500
    return cfg, source, checkpoint


def _assert_nested_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_nested_equal(x, y)
    else:
        assert a == b


def test_all_saved_learner_state_restores_exactly_and_source_is_unchanged(tmp_path):
    cfg, source, checkpoint = _source(tmp_path)
    original_bytes = checkpoint.read_bytes()
    validate_recovery_training_config(cfg)
    target = _learner()
    assert not target.q_optimizer.state_dict()["state"]
    result = apply_configured_learner_resume("sac", cfg, SimpleNamespace(learner=target))
    _assert_nested_equal(source, target.get_state_dict())
    assert result["restored_update_count"] == 15000
    assert result["replay_restored"] is False
    assert checkpoint.read_bytes() == original_bytes


def test_resume_off_does_not_access_runner_or_checkpoint():
    cfg = _cfg()
    assert apply_configured_learner_resume("sac", cfg, object()) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("algo.actor_warm_start_checkpoint", "actor.pt"),
        ("algo.actor_warm_start_adapter", "g1_height_actor_obs_99_to_99_v1"),
        ("algo.load_run", "old-run"),
        ("env.assistance.stage", 1),
        ("env.recovery.height_gated_upright", False),
        ("reward.scales.recovery_progress", 10),
        ("algo.gamma", 0.99),
        ("algo.actor_lr", 1e-4),
        ("algo.critic_hidden_dim", 64),
    ],
)
def test_resume_rejects_mixed_initializers_or_changed_task_math(tmp_path, field, value):
    cfg, _, _ = _source(tmp_path)
    OmegaConf.update(cfg, field, value)
    with pytest.raises(ValueError):
        validate_configured_learner_resume(cfg)


def test_resume_rejects_overwriting_parent_and_missing_sidecar(tmp_path):
    cfg, _, _ = _source(tmp_path)
    cfg.training.log_dir = str(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        validate_configured_learner_resume(cfg)
    cfg.training.log_dir = None
    (tmp_path / "run_config.json").unlink()
    with pytest.raises(ValueError, match="run_config"):
        validate_configured_learner_resume(cfg)


def test_incomplete_payload_does_not_mutate_fresh_learner(tmp_path):
    cfg, source, checkpoint = _source(tmp_path)
    del source["q_optimizer"]
    torch.save(source, checkpoint)
    target = _learner()
    before = copy.deepcopy(target.get_state_dict())
    with pytest.raises(ValueError, match="incomplete"):
        apply_configured_learner_resume("sac", cfg, SimpleNamespace(learner=target))
    _assert_nested_equal(before, target.get_state_dict())


def test_training_entry_restores_learner_before_calling_learn(tmp_path, monkeypatch):
    import scripts.train_offpolicy as entry

    cfg, source, _ = _source(tmp_path)
    cfg.training.device = "cpu"
    cfg.training.no_play = True
    cfg.training.log_dir = str(tmp_path / "new_run")
    target = _learner()
    events = []

    def fake_learn(**kwargs):
        _assert_nested_equal(source, target.get_state_dict())
        assert kwargs["max_iterations"] == 15000
        events.append("learn_called_after_restore")

    runner = SimpleNamespace(
        learner=target,
        learn=fake_learn,
        close=lambda: events.append("closed"),
        last_run_summary={"status": "completed", "completed_iterations": 0},
    )
    monkeypatch.setattr(entry, "build_runner", lambda algo, composed: runner)
    entry.main.__wrapped__(cfg)
    assert events == ["learn_called_after_restore", "closed"]
    summary = json.loads((tmp_path / "new_run" / "run_summary.json").read_text())
    assert summary["restored_update_count"] == 15000
    assert summary["replay_restored"] is False
    assert summary["cumulative_learner_update_count"] == 15000
