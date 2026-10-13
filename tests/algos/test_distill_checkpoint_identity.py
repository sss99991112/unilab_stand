"""A policy's weights and provenance must come from one checkpoint snapshot."""

import pytest
import torch

from unilab.algos.torch.distill import (
    MLPStudentPolicy,
    MoEStudentPolicy,
    load_distillation_student_policy,
    save_distillation_checkpoint,
)


@pytest.mark.parametrize("model_type", ["mlp", "moe"])
def test_playback_load_keeps_weights_and_provenance_from_same_snapshot(
    tmp_path, monkeypatch, model_type
):
    cfg = {
        "student_model_type": model_type,
        "student_obs_dim": 5,
        "student_action_dim": 3,
        "student_hidden_dims": [8],
        "student_num_experts": 2,
        "student_expert_hidden_dims": [8],
        "student_router_hidden_dims": [],
        "source_version": "first",
    }

    def policy():
        if model_type == "mlp":
            return MLPStudentPolicy(obs_dim=5, action_dim=3, hidden_dims=(8,))
        return MoEStudentPolicy(obs_dim=5, action_dim=3, num_experts=2, expert_hidden_dims=(8,))

    first, second = policy(), policy()
    checkpoint_path = tmp_path / "published.pt"
    save_distillation_checkpoint(
        checkpoint_path,
        student=first,
        agent_steps=10,
        teacher_metadata={"source": "first"},
        distill_runtime_cfg=cfg,
    )
    original_load = torch.load
    loads = []

    def load_then_publish(*args, **kwargs):
        checkpoint = original_load(*args, **kwargs)
        loads.append(args[0])
        if len(loads) == 1:
            save_distillation_checkpoint(
                checkpoint_path,
                student=second,
                agent_steps=20,
                teacher_metadata={"source": "second"},
                distill_runtime_cfg={**cfg, "source_version": "second"},
            )
        return checkpoint

    monkeypatch.setattr(torch, "load", load_then_publish)
    loaded = load_distillation_student_policy(checkpoint_path)
    assert loaded.agent_steps == 10
    assert loaded.teacher_metadata == {"source": "first"}
    assert loaded.distill_runtime_cfg["source_version"] == "first"
    for name, weight in first.state_dict().items():
        torch.testing.assert_close(loaded.policy.state_dict()[name], weight, rtol=0, atol=0)
    obs = torch.randn(4, 5)
    torch.testing.assert_close(loaded.policy(obs), first(obs), rtol=0, atol=0)
    assert len(loads) == 1
