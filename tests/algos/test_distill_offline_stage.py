from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest
import scripts.train_distill as train_distill
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.distill import (
    MoEStudentPolicy,
    RoleArtifactSpec,
    WorkflowScenarioSpec,
    build_distillation_dataset,
    create_role_artifact_manifest,
    file_sha256,
    fork_workflow_run,
    load_distillation_dataset,
    load_distillation_student_policy,
    run_bootstrap_workflow,
    run_multirole_dagger_workflow,
    save_distillation_checkpoint,
    save_distillation_dataset,
    write_role_artifact_manifest,
)
from unilab.algos.torch.distill.offline_stage import run_offline_stage_process
from unilab.algos.torch.fast_sac.learner import SACActor

ROOT = Path(__file__).resolve().parents[2]
ROLES = {"walk": 0, "stand_height": 1, "recovery": 2}
SCENARIOS = (
    WorkflowScenarioSpec("walk_flat", "role", ("walk",), 0.35),
    WorkflowScenarioSpec("static_stand", "role", ("stand_height",), 0.20),
    WorkflowScenarioSpec("walk_to_stop", "transition", ("walk", "stand_height"), 0.15),
    WorkflowScenarioSpec("supine_recovery", "role", ("recovery",), 0.20),
    WorkflowScenarioSpec("recovery_to_stand", "transition", ("recovery", "stand_height"), 0.10),
)


def _dataset(path, scenarios=SCENARIOS):
    labels, scenario_labels, ages, commands = [], [], [], []
    for scenario in scenarios:
        labels.extend(
            scenario.source_roles if scenario.kind == "transition" else scenario.source_roles * 2
        )
        scenario_labels.extend([scenario.name] * 2)
        ages.extend([-1, 0] if scenario.kind == "transition" else [-1, -1])
        commands.extend(
            [[0.4, 0.0, 0.0] if role == "walk" else [0.0, 0.0, 0.0] for role in labels[-2:]]
        )
    rows = len(labels)
    torch.manual_seed(19)
    data = build_distillation_dataset(
        torch.randn(rows, 5),
        torch.randn(rows, 7),
        teacher_actions=torch.full((rows, 3), 0.25),
        role_labels=labels,
        commands=torch.tensor(commands),
        command_intents=["active" if role == "walk" else "inactive" for role in labels],
        scenario_labels=scenario_labels,
        transition_ages=torch.tensor(ages),
        command_before=torch.tensor(commands),
        command_after=torch.tensor(commands),
    )
    save_distillation_dataset(path, data)
    return data


def _fixture(tmp_path):
    teacher = tmp_path / "teacher.pt"
    actor = SACActor(obs_dim=7, action_dim=3, hidden_dim=8, use_layer_norm=False, device="cpu")
    torch.save({"actor": actor.state_dict()}, teacher)
    cfg = OmegaConf.load(ROOT / "conf/distill/config.yaml")
    del cfg["defaults"]
    cfg.teacher.obs_dim, cfg.teacher.action_dim = 7, 3
    cfg.teacher.actor_hidden_dim, cfg.teacher.use_layer_norm = 8, False
    cfg.teacher.obs_normalization = False
    cfg.student.obs_dim, cfg.student.action_dim = 5, 3
    cfg.student.model_type, cfg.student.num_experts = "moe", 3
    cfg.student.expert_hidden_dims, cfg.student.routing_mode = [8], "hard"
    cfg.algo.role_loss_coef = 0.25
    cfg.algo.role_expert_targets = ROLES
    cfg.algo.expert_behavior_loss_source = "role"
    cfg.training.offline_resume_optimizer = False
    cfg.training.offline_save_optimizer = False
    cfg.training.offline_repeat_dataset = True
    cfg.training.offline_shuffle = True
    cfg.training.offline_balance_key = "scenario"
    cfg.training.offline_balanced_labels = [s.name for s in SCENARIOS]
    cfg.training.offline_balance_quotas = {s.name: s.quota for s in SCENARIOS}
    cfg.training.offline_min_balanced_replay_passes = 8
    cfg.training.offline_min_balanced_replay_labels = ["walk_to_stop", "recovery_to_stand"]
    specs = []
    for scenario in (SCENARIOS[0], SCENARIOS[1], SCENARIOS[3]):
        role = scenario.source_roles[0]
        dataset_path = tmp_path / "roles" / (role + ".pt")
        _dataset(dataset_path, (scenario,))
        spec = RoleArtifactSpec(
            role=role,
            task=role,
            teacher_checkpoint_path=teacher,
            dataset_path=dataset_path,
            schema_version=3,
            student_obs_dim=5,
            teacher_obs_dim=7,
            teacher_action_dim=3,
            teacher_obs_key="obs",
            teacher_projection="identity",
            student_projection="identity",
            student_drop_index=None,
            command_sample_filter="all",
            command_info_key="commands",
            command_xy_threshold=0.05,
            command_yaw_threshold=0.05,
            owner_config={},
        )
        write_role_artifact_manifest(
            spec.manifest_path, create_role_artifact_manifest(spec, num_samples=2)
        )
        specs.append(spec)
    policy = MoEStudentPolicy(
        obs_dim=5, action_dim=3, num_experts=3, expert_hidden_dims=[8], routing_mode="hard"
    )
    runtime = train_distill._distill_runtime_cfg(cfg, distill_source="test")
    parent = tmp_path / "parent"

    def bootstrap_update(dataset, checkpoint):
        save_distillation_checkpoint(
            checkpoint, student=policy, agent_steps=20, distill_runtime_cfg=runtime
        )
        return 1

    boot = run_bootstrap_workflow(
        run_dir=parent,
        role_specs=specs,
        scenario_specs=SCENARIOS,
        collect_role=lambda _: pytest.fail("bootstrap must reuse role artifacts"),
        assemble_roles=lambda paths, output: _dataset(output).num_samples,
        update_student=bootstrap_update,
    )
    seed_data = tmp_path / "saved_aggregate.pt"
    _dataset(seed_data)
    seed = tmp_path / "saved_student.pt"
    save_distillation_checkpoint(
        seed,
        student=policy,
        agent_steps=40,
        distill_runtime_cfg={
            **runtime,
            "dataset_path": str(seed_data),
            "student_init_checkpoint_sha256": file_sha256(boot.checkpoint_path),
        },
    )
    return cfg, teacher, tuple(specs), parent, seed, seed_data


def test_saved_update_fork_keeps_parent_and_preserves_mixed_roles(tmp_path):
    cfg, _, specs, parent, seed, data = _fixture(tmp_path)
    original = (parent / "run_manifest.json").read_bytes()
    path = fork_workflow_run(
        parent_run_dir=parent,
        run_dir=tmp_path / "child",
        checkpoint_override=seed,
        dataset_override=data,
    )
    manifest = json.loads(path.read_text())
    assert manifest["bootstrap_checkpoint_path"] == str(seed.resolve())
    assert manifest["completed_dagger_iterations"] == 0
    assert manifest["bootstrap_updates"] == 0
    assert manifest["bootstrap_num_samples"] == 10
    assert manifest["bootstrap_sources"][0]["preserve_row_role_labels"] is True
    assert (parent / "run_manifest.json").read_bytes() == original
    cfg.training.multitask_sources = manifest["bootstrap_sources"]
    output = tmp_path / "aggregate.pt"
    train_distill.run_multitask_dataset_assembly(cfg, dataset_path=output)
    assert (
        load_distillation_dataset(output).role_labels == load_distillation_dataset(data).role_labels
    )
    assert {spec.role for spec in specs} == set(ROLES)


@pytest.mark.parametrize(
    "failure", ["checkpoint_only", "dataset_only", "wrong_dataset", "wrong_parent"]
)
def test_saved_update_fork_rejects_unmatched_seed_before_manifest_write(tmp_path, failure):
    _, _, _, parent, seed, data = _fixture(tmp_path)
    if failure == "wrong_parent":
        payload = torch.load(seed, weights_only=False)
        payload["distill_runtime_cfg"]["student_init_checkpoint_sha256"] = "a" * 64
        torch.save(payload, seed)
    if failure == "wrong_dataset":
        data = tmp_path / "unrelated.pt"
    child = tmp_path / "child"
    with pytest.raises(ValueError, match="together|supplied dataset|descend"):
        fork_workflow_run(
            parent_run_dir=parent,
            run_dir=child,
            checkpoint_override=None if failure == "dataset_only" else seed,
            dataset_override=None if failure == "checkpoint_only" else data,
        )
    assert not (child / "run_manifest.json").exists()


def test_eight_rounds_use_previous_student_and_fresh_offline_processes(tmp_path):
    cfg, teacher, specs, parent, seed, data = _fixture(tmp_path)
    before = (parent / "run_manifest.json").read_bytes()
    run_dir = tmp_path / "eight_rounds"
    fork_workflow_run(
        parent_run_dir=parent, run_dir=run_dir, checkpoint_override=seed, dataset_override=data
    )
    observed = []

    def collect(scenario, checkpoint, iteration, output):
        observed.append((iteration, scenario.name, checkpoint.resolve()))
        return _dataset(output, (scenario,)).num_samples

    def aggregate(sources, output):
        config = OmegaConf.to_container(cfg, resolve=True)
        config["training"]["multitask_sources"] = [
            {
                "path": str(s.path),
                "role": s.role,
                "preserve_row_role_labels": s.preserve_row_role_labels,
                **({"scenario": s.scenario} if s.scenario else {}),
            }
            for s in sources
        ]
        result = run_offline_stage_process(
            entrypoint=ROOT / "scripts/train_distill.py",
            config=config,
            operation="aggregate",
            arguments={"dataset_path": str(output)},
            output_path=output,
        )
        return result["dataset_num_samples"]

    def update(dataset, input_checkpoint, output):
        config = OmegaConf.to_container(cfg, resolve=True)
        config["training"]["offline_init_checkpoint"] = str(input_checkpoint)
        result = run_offline_stage_process(
            entrypoint=ROOT / "scripts/train_distill.py",
            config=config,
            operation="update",
            arguments={
                "teacher_checkpoint": str(teacher),
                "dataset_path": str(dataset),
                "batch_size": 20,
                "max_updates": 1,
                "checkpoint_path": str(output),
                "device": "cpu",
                "auto_expand_replay_budget": True,
            },
            output_path=output,
        )
        return result["update_count"]

    result = run_multirole_dagger_workflow(
        run_dir=run_dir,
        role_specs=specs,
        scenario_specs=SCENARIOS,
        target_iterations=8,
        collect_role=lambda *a: pytest.fail("scenario owner must collect"),
        collect_scenario=collect,
        aggregate_datasets=aggregate,
        update_student=update,
    )
    assert result.completed_iterations == 8 and result.cumulative_num_samples == 90
    assert len(observed) == 40
    for iteration, _, checkpoint in observed:
        expected = (
            seed
            if iteration == 1
            else run_dir / "checkpoints" / f"dagger_iteration_{iteration - 1}.pt"
        )
        assert checkpoint == expected.resolve()
    manifest = json.loads(result.manifest_path.read_text())
    assert [r["updates"] for r in manifest["dagger_iterations"]] == [
        8 * (i + 1) for i in range(1, 9)
    ]
    final_data = load_distillation_dataset(
        manifest["dagger_iterations"][-1]["aggregate_dataset_path"]
    )
    assert Counter(final_data.scenario_labels) == {s.name: 18 for s in SCENARIOS}
    assert set(final_data.role_labels) == set(ROLES)
    final = load_distillation_student_policy(result.checkpoint_path)
    initial = load_distillation_student_policy(seed)
    assert any(
        not torch.equal(v, initial.policy.state_dict()[k])
        for k, v in final.policy.state_dict().items()
    )
    assert final.distill_runtime_cfg["student_init_checkpoint_sha256"] == file_sha256(
        run_dir / "checkpoints/dagger_iteration_7.pt"
    )
    for result_path in run_dir.rglob("*.offline-result.json"):
        assert json.loads(result_path.read_text())["worker_pid"] != os.getpid()
    assert (parent / "run_manifest.json").read_bytes() == before


def test_offline_worker_failure_stops_without_success_result(tmp_path):
    cfg = OmegaConf.load(ROOT / "conf/distill/config.yaml")
    del cfg["defaults"]
    cfg.training.multitask_sources = [{"path": str(tmp_path / "missing.pt"), "role": "walk"}]
    output = tmp_path / "failed_aggregate.pt"
    with pytest.raises(subprocess.CalledProcessError):
        run_offline_stage_process(
            entrypoint=ROOT / "scripts/train_distill.py",
            config=OmegaConf.to_container(cfg, resolve=True),
            operation="aggregate",
            arguments={"dataset_path": str(output)},
            output_path=output,
        )
    assert not output.exists()
    assert not output.with_name(output.name + ".offline-result.json").exists()
