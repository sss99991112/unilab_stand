from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.train_distill as train_distill
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def _compose(monkeypatch, workflow):
    for name in [
        "UNILAB_G1_WALK_HEIGHT_TEACHER",
        "UNILAB_G1_STAND_HEIGHT_TEACHER",
        "UNILAB_G1_RECOVERY_TEACHER",
        "UNILAB_G1_RECOVERY_MOE_INIT",
    ]:
        monkeypatch.setenv(name, "/tmp/source.pt")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf/distill"), version_base="1.3"):
        return compose(
            "config",
            overrides=["task=g1_walk_height_nominal/mujoco", "workflow=" + workflow],
        )


def test_three_expert_profile_is_opt_in_and_old_profile_unchanged(monkeypatch):
    new = _compose(monkeypatch, "g1_stand_height_walk_recovery")
    old = _compose(monkeypatch, "g1_stand_height_walk")
    assert new.student.num_experts == 3 and old.student.num_experts == 2
    assert new.training.recovery_integration and not old.training.recovery_integration
    assert new.algo.expert_behavior_loss_source == "role"
    assert new.algo.command_intent_loss_coef == 0
    assert new.algo.role_expert_targets.recovery == 2
    assert new.training.workflow.schema_version == 3
    assert new.training.offline_resume_optimizer is False
    assert new.training.offline_repeat_dataset is True
    assert new.training.offline_balance_key == "role"
    assert list(new.training.offline_balanced_labels) == ["walk", "stand_height", "recovery"]
    assert all(entry.dataset_path == "" for entry in new.training.workflow.roles)
    assert len(new.training.workflow.scenarios) == 5
    assert sum(s.quota for s in new.training.workflow.scenarios) == pytest.approx(1)
    assert old.training.workflow.transition_nominal_settle_steps == 100
    runtime = train_distill._distill_runtime_cfg(new, distill_source="test")
    assert runtime["recovery_integration"] is True


def test_recovery_role_owner_uses_exact_unassisted_v2_semantics(monkeypatch):
    cfg = _compose(monkeypatch, "g1_stand_height_walk_recovery")
    entry = next(e for e in train_distill._workflow_role_entries(cfg) if e["role"] == "recovery")
    role = train_distill._workflow_role_cfg(cfg, entry)
    train_distill._require_teacher_policy_collection_route(role)
    assert role.training.task_name == "G1Recovery"
    assert role.training.recovery_integration is True
    assert role.teacher.task_name == "G1Recovery"
    assert role.env.assistance.stage == 2
    assert role.env.recovery.height_gated_upright
    assert role.env.recovery.progress_floor_height == 0.06
    assert role.env.commands.default_height == 0.754
    assert role.env.control_config.action_scale == 1
    from unilab.base.registry import apply_cfg_overrides
    from unilab.envs.locomotion.g1.recovery import G1RecoveryCfg
    from unilab.training import BackendAdapter

    env_cfg = G1RecoveryCfg()
    apply_cfg_overrides(
        env_cfg,
        BackendAdapter(role, root_dir=ROOT, algo_name="distill").build_task_env_cfg_override(),
    )
    assert env_cfg.reward_config.gait_frequency == 0
    assert env_cfg.reward_config.feet_phase_tracking_sigma == 0.04


def test_recovery_workflow_rejects_unimplemented_persistent_route_before_io(monkeypatch):
    cfg = _compose(monkeypatch, "g1_stand_height_walk_recovery")
    cfg.training.workflow.execution_mode = "persistent_async"
    with pytest.raises(ValueError, match="execution_mode=legacy"):
        train_distill.run_single_entry_workflow(cfg)


def test_partial_recovery_activation_is_rejected_before_artifact_io(monkeypatch):
    cfg = _compose(monkeypatch, "g1_stand_height_walk_recovery")
    cfg.training.recovery_integration = False
    with pytest.raises(ValueError, match="activated together"):
        train_distill.run_single_entry_workflow(cfg)


def test_recovery_handover_owner_accepts_locked_cli_config_without_mutating_it(monkeypatch):
    from unilab.algos.torch.distill import recovery_workflow
    from unilab.algos.torch.distill.moe_student import MoEStudentPolicy
    from unilab.algos.torch.distill.recovery_integration import (
        RECOVERY_ROLES,
        RECOVERY_ROUTING_CONTRACT,
        validate_recovery_env,
    )
    from unilab.base.registry import apply_cfg_overrides
    from unilab.envs.locomotion.g1.recovery_combined import G1RecoveryCombinedCfg

    cfg = _compose(monkeypatch, "g1_stand_height_walk_recovery")
    before = OmegaConf.to_container(cfg, resolve=False)
    assert OmegaConf.is_struct(cfg.env)
    assert "max_episode_seconds" not in cfg.env
    role_cfgs = {
        entry["role"]: train_distill._workflow_role_cfg(cfg, entry)
        for entry in train_distill._workflow_role_entries(cfg)
    }
    policy = MoEStudentPolicy(
        obs_dim=99,
        action_dim=29,
        num_experts=3,
        expert_hidden_dims=[8],
        router_hidden_dims=[8],
        routing_mode="hard",
    )
    loaded = SimpleNamespace(
        policy=policy,
        distill_runtime_cfg={
            "role_expert_targets": RECOVERY_ROLES,
            "recovery_routing_contract": RECOVERY_ROUTING_CONTRACT,
            "expert_behavior_loss_source": "role",
        },
    )
    monkeypatch.setattr(
        recovery_workflow, "load_sac_teacher_policy", lambda *a, **kw: torch.nn.Identity()
    )
    monkeypatch.setattr(
        recovery_workflow, "load_distillation_student_policy", lambda *a, **kw: loaded
    )
    monkeypatch.setattr(recovery_workflow, "ensure_registries", lambda: None)

    class EnvBoundaryReachedError(Exception):
        pass

    def verify_config(scenario_cfg, *, env_cfg_override, task_name, sim_backend, num_envs):
        assert task_name == "G1RecoveryCombined" and sim_backend == "mujoco"
        assert num_envs == cfg.training.workflow.collect_num_envs
        assert scenario_cfg.student.num_experts == 3
        assert scenario_cfg.algo.role_expert_targets == cfg.algo.role_expert_targets
        assert scenario_cfg.training.workflow.run_dir == cfg.training.workflow.run_dir
        env_cfg = G1RecoveryCombinedCfg()
        apply_cfg_overrides(env_cfg, env_cfg_override)
        assert env_cfg.max_episode_seconds == 15.0
        assert env_cfg.recovery_reset_supine is True
        assert env_cfg.commands.rel_standing_envs == 1.0
        assert all(value == 0 for bound in env_cfg.commands.vel_limit for value in bound)
        assert env_cfg.curriculum.enabled is False
        assert env_cfg.reward_config.min_base_height == 0.0
        assert env_cfg.reward_config.max_tilt_deg == 180.0
        validate_recovery_env(SimpleNamespace(cfg=env_cfg))
        raise EnvBoundaryReachedError

    monkeypatch.setattr(recovery_workflow, "create_env", verify_config)
    with pytest.raises(EnvBoundaryReachedError):
        recovery_workflow.collect_recovery_workflow_scenario(
            cfg,
            role_cfgs,
            Path("/tmp/student.pt"),
            Path("/tmp/data.pt"),
            performance_clock=lambda: 0.0,
        )
    assert OmegaConf.to_container(cfg, resolve=False) == before
    assert OmegaConf.is_struct(cfg) and OmegaConf.is_struct(cfg.env)
