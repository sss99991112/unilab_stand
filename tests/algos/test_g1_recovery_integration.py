from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill import (
    BehaviorDistillationTrainer,
    DistillationBatch,
    MoEStudentPolicy,
    annotate_distillation_dataset_scenario,
    build_distillation_dataset,
    build_multitask_distillation_dataset,
    load_distillation_student_policy,
    save_distillation_checkpoint,
)
from unilab.algos.torch.distill.recovery_integration import (
    RECOVERY_ROLES,
    RecoveryRoutedPolicy,
    collect_recovery_handover,
    extend_recovery_student_checkpoint,
    recovery_expert_indices,
)
from unilab.envs.locomotion.g1.recovery import RecoveryConfig
from unilab.envs.locomotion.g1.recovery_combined import (
    RecoveryRoutingConfig,
    update_recovery_route,
)


def _checkpoint(path, obs_dim=99):
    policy = MoEStudentPolicy(
        obs_dim=obs_dim, action_dim=29, num_experts=2, expert_hidden_dims=(8,), routing_mode="hard"
    )
    cfg = {
        "student_model_type": "moe",
        "student_obs_dim": obs_dim,
        "student_action_dim": 29,
        "student_num_experts": 2,
        "student_expert_hidden_dims": [8],
        "student_router_hidden_dims": [],
        "student_activation": "elu",
        "student_squash_action": True,
        "student_routing_mode": "hard",
        "role_expert_targets": {"walk": 0, "stand_height": 1},
    }
    save_distillation_checkpoint(path, student=policy, agent_steps=123, distill_runtime_cfg=cfg)
    return policy


def test_migration_preserves_old_experts_exactly_and_strict_reload(tmp_path):
    source, output = tmp_path / "source.pt", tmp_path / "three.pt"
    old = _checkpoint(source)
    original = source.read_bytes()
    metadata = extend_recovery_student_checkpoint(source, output)
    loaded = load_distillation_student_policy(output)
    obs = torch.randn(17, 99)
    for index in (0, 1):
        assert torch.equal(old.experts[index](obs), loaded.policy.experts[index](obs))
    assert loaded.policy.num_experts == 3
    assert loaded.agent_steps == 123
    assert loaded.distill_runtime_cfg["role_expert_targets"] == RECOVERY_ROLES
    assert metadata["new_expert_trained"] is False
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        extend_recovery_student_checkpoint(source, output)


def test_migration_rejects_legacy_98d(tmp_path):
    source = tmp_path / "old98.pt"
    _checkpoint(source, obs_dim=98)
    with pytest.raises(ValueError, match="99/29"):
        extend_recovery_student_checkpoint(source, tmp_path / "new.pt")
    assert not (tmp_path / "new.pt").exists()


def _route(info, step, *, height=0.75, up=1.0, speed=0.0, feet=True, n=2):
    return update_recovery_route(
        info,
        height=np.full(n, height),
        gravity=np.tile([0.0, 0.0, up], (n, 1)),
        linvel=np.tile([speed, 0.0, 0.0], (n, 1)),
        gyro=np.zeros((n, 3)),
        feet=np.full((n, 2), feet),
        steps=np.full(n, step),
        recovery=RecoveryConfig(),
        routing=RecoveryRoutingConfig(),
        ctrl_dt=0.02,
    )


def test_recovery_latches_and_requires_full_stable_window_without_refresh_double_count():
    info = {}
    assert _route(info, 0, height=0.3, up=0.0).all()
    # Height alone cannot release the recovery expert; large speed resets the window.
    for step in range(1, 31):
        assert _route(info, step).all()
        before = info["recovery_route_stable_steps"].copy()
        _route(info, step)
        np.testing.assert_array_equal(before, info["recovery_route_stable_steps"])
    assert _route(info, 31, speed=0.3).all()
    for step in range(32, 81):
        assert _route(info, step).all()
    assert not _route(info, 81).any()
    np.testing.assert_array_equal(info["recovery_settle_remaining"], [100, 100])
    assert _route(info, 82, up=0.0).all()


def test_routing_recovery_has_priority_over_walking_and_old_expert_outputs_match():
    student = MoEStudentPolicy(obs_dim=99, action_dim=29, num_experts=3, expert_hidden_dims=(8,))
    info = {
        "recovery_active": np.array([True, False, False]),
        "commands": np.array([[0.4, 0, 0], [0.4, 0, 0], [0, 0, 0]]),
    }
    info["recovery_effective_commands"] = info["commands"].copy()
    np.testing.assert_array_equal(recovery_expert_indices(info), [2, 0, 1])
    env = SimpleNamespace(state=SimpleNamespace(info=info))
    obs = torch.randn(3, 99)
    action = RecoveryRoutedPolicy(student, env)(obs)
    for row, index in enumerate([2, 0, 1]):
        torch.testing.assert_close(
            action[row], student.experts[index](obs[row : row + 1])[0], rtol=0, atol=0
        )


def test_recovery_role_updates_only_third_expert_despite_inactive_command():
    student = MoEStudentPolicy(obs_dim=99, action_dim=29, num_experts=3, expert_hidden_dims=(8,))
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=torch.nn.Identity(),
        optimizer=optimizer,
        role_loss_coef=0.25,
        role_expert_targets=RECOVERY_ROLES,
        expert_behavior_loss_source="role",
        command_intent_expert_targets={"active": 0, "inactive": 1},
    )
    before = [{k: v.clone() for k, v in e.state_dict().items()} for e in student.experts]
    stats = trainer.update(
        DistillationBatch(
            student_obs=torch.randn(4, 99),
            teacher_obs=torch.empty(4, 0),
            teacher_actions=torch.zeros(4, 29),
            role_labels=("recovery",) * 4,
            command_intents=("inactive",) * 4,
        )
    )
    assert stats.behavior_action_source == "role_expert"
    for index in (0, 1):
        assert all(
            torch.equal(v, before[index][k]) for k, v in student.experts[index].state_dict().items()
        )
        assert all(p not in optimizer.state for p in student.experts[index].parameters())
    assert any(not torch.equal(v, before[2][k]) for k, v in student.experts[2].state_dict().items())


class _ConstantPolicy(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, obs):
        return torch.full((len(obs), 29), self.value, device=obs.device)


class _HandoverEnv:
    num_envs = 1
    action_space = SimpleNamespace(shape=(29,))

    def __init__(self):
        self.state = None
        self.applied = []

    def reset(self, ids):
        self.tick = 0
        info = {
            "commands": np.zeros((1, 3)),
            "height_commands": np.full((1, 1), 0.754),
            "recovery_active": np.ones(1, dtype=bool),
        }
        obs = np.zeros((1, 99), dtype=np.float32)
        obs[:, 96] = 0.754
        obs[:, 0] = 0.25
        obs[:, 35] = 0.1
        self.state = SimpleNamespace(
            info=info,
            obs={"obs": obs},
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
        )
        info["recovery_effective_commands"] = info["commands"].copy()
        info["recovery_effective_height_commands"] = info["height_commands"].copy()
        return self.state.obs, self.state.info

    def step(self, action):
        self.applied.append(action.copy())
        self.tick += 1
        self.state.info["recovery_active"][:] = self.tick < 2
        return self.state


def test_handover_targets_follow_pre_action_route_and_keep_transition_schema(tmp_path):
    env = _HandoverEnv()
    data = collect_recovery_handover(
        env,
        recovery_teacher=_ConstantPolicy(0.2),
        standing_teacher=_ConstantPolicy(0.7),
        num_samples=4,
    )
    assert data.role_labels == ("recovery", "recovery", "stand_height", "stand_height")
    torch.testing.assert_close(data.teacher_actions[:, 0], torch.tensor([0.2, 0.2, 0.7, 0.7]))
    assert data.transition_ages.tolist() == [-1, -1, 0, 1]
    torch.testing.assert_close(data.teacher_obs[:, 0], torch.tensor([1.0, 1.0, 0.25, 0.25]))
    torch.testing.assert_close(data.teacher_obs[:, 35], torch.tensor([2.0, 2.0, 0.1, 0.1]))
    assert data.metadata["teacher_observation_units"] == "role_native"
    assert data.scenario_labels == ("recovery_to_stand",) * 4
    assert data.command_intents == ("inactive",) * 4
    from unilab.algos.torch.distill import load_distillation_dataset, save_distillation_dataset

    path = tmp_path / "transition.pt"
    save_distillation_dataset(path, data)
    loaded = load_distillation_dataset(path)
    assert loaded.role_labels == data.role_labels
    assert loaded.transition_ages.tolist() == data.transition_ages.tolist()


def test_supine_scenario_annotations_are_valid_with_existing_scenarios():
    data = build_distillation_dataset(
        torch.zeros(2, 99),
        torch.zeros(2, 99),
        teacher_actions=torch.zeros(2, 29),
        commands=torch.zeros(2, 3),
        target_height=torch.ones(2, 1),
        command_intents=("inactive",) * 2,
        role_labels=("recovery",) * 2,
    )
    annotated = annotate_distillation_dataset_scenario(data, "supine_recovery")
    assert annotated.role_labels == ("recovery",) * 2
    assert annotated.transition_ages.tolist() == [-1, -1]


def test_recovery_observation_units_roundtrip_and_teacher_action_parity():
    from unilab.algos.torch.distill.g1_observation_adapter import (
        RecoveryTeacherObservationAdapter,
        recovery_to_walk_obs,
        walk_to_recovery_obs,
    )

    original = torch.arange(198, dtype=torch.float32).reshape(2, 99) / 10
    canonical = recovery_to_walk_obs(original)
    assert torch.equal(canonical[:, 3:35], original[:, 3:35])
    assert torch.equal(canonical[:, 64:], original[:, 64:])
    torch.testing.assert_close(walk_to_recovery_obs(canonical), original)
    teacher = torch.nn.Linear(99, 29)
    torch.testing.assert_close(
        RecoveryTeacherObservationAdapter(teacher)(canonical), teacher(original)
    )
    from unilab.algos.torch.distill.collector import project_student_obs

    np.testing.assert_allclose(
        project_student_obs(
            original.numpy(), projection="g1_recovery_to_walk_99_v1", expected_student_obs_dim=99
        ),
        canonical.numpy(),
    )
    with pytest.raises(ValueError, match="99-D"):
        recovery_to_walk_obs(torch.zeros(1, 98))


def test_handover_workflow_returns_rich_result_and_preserves_labels(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    import unilab.algos.torch.distill.recovery_workflow as owner
    from unilab.algos.torch.distill import load_distillation_dataset
    from unilab.algos.torch.distill.performance import LEGACY_REQUEST_STAGE_NAMES
    from unilab.algos.torch.distill.recovery_integration import RECOVERY_ROUTING_CONTRACT

    cfg = OmegaConf.create(
        {
            "training": {
                "device": "cpu",
                "workflow": {"collect_num_envs": 1, "dagger_samples_per_role": 4},
            }
        }
    )
    role_cfg = OmegaConf.create(
        {
            "teacher": {
                "actor_hidden_dim": 512,
                "use_layer_norm": True,
                "obs_normalization": True,
                "checkpoint_path": "/tmp/unused.pt",
            }
        }
    )
    student = MoEStudentPolicy(obs_dim=99, action_dim=29, num_experts=3, expert_hidden_dims=(8,))
    monkeypatch.setattr(owner, "load_sac_teacher_policy", lambda *a, **k: _ConstantPolicy(0.2))
    monkeypatch.setattr(
        owner,
        "load_distillation_student_policy",
        lambda *a, **k: SimpleNamespace(
            policy=student,
            distill_runtime_cfg={
                "role_expert_targets": RECOVERY_ROLES,
                "expert_behavior_loss_source": "role",
                "recovery_routing_contract": RECOVERY_ROUTING_CONTRACT,
            },
        ),
    )
    monkeypatch.setattr(owner, "ensure_registries", lambda: None)
    monkeypatch.setattr(
        owner,
        "BackendAdapter",
        lambda *a, **k: SimpleNamespace(build_task_env_cfg_override=lambda: {}),
    )
    env = _HandoverEnv()
    env.close = lambda: None
    monkeypatch.setattr(owner, "create_env", lambda *a, **k: env)
    ticks = iter(range(1000))
    result = owner.collect_recovery_workflow_scenario(
        cfg,
        {"recovery": role_cfg, "stand_height": role_cfg},
        tmp_path / "student.pt",
        tmp_path / "data.pt",
        performance_clock=lambda: float(next(ticks)),
    )
    assert result.num_samples == 4
    assert (
        tuple(o.stage for o in result.performance_stage_observations) == LEGACY_REQUEST_STAGE_NAMES
    )
    loaded = load_distillation_dataset(tmp_path / "data.pt")
    assert loaded.role_labels == ("recovery", "recovery", "stand_height", "stand_height")


def test_latest_stop_and_height_request_survive_recovery_guard():
    from unilab.envs.locomotion.g1.recovery_combined import G1RecoveryCombinedEnv

    env = SimpleNamespace(cfg=SimpleNamespace(commands=SimpleNamespace(default_height=0.754)))
    info = {
        "commands": np.array([[0.4, 0.0, 0.0]]),
        "height_commands": np.array([[0.65]]),
        "recovery_active": np.array([True]),
        "recovery_settle_remaining": np.array([0]),
    }
    effective = G1RecoveryCombinedEnv._effective_command_info(env, info)
    np.testing.assert_array_equal(effective["commands"], [[0, 0, 0]])
    np.testing.assert_array_equal(info["commands"], [[0.4, 0, 0]])
    assert effective["height_commands"][0, 0] == 0.754
    # A newer stop / nominal-height request arrives while recovering.
    info["commands"][:] = 0
    info["height_commands"][:] = 0.754
    info["recovery_active"][:] = False
    info["recovery_settle_remaining"][:] = 0
    effective = G1RecoveryCombinedEnv._effective_command_info(env, info)
    np.testing.assert_array_equal(effective["commands"], [[0, 0, 0]])
    assert effective["height_commands"][0, 0] == 0.754
