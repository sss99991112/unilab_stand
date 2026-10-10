"""Explicit two-to-three expert migration and shared recovery-first action selection."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .checkpoint import save_distillation_checkpoint
from .collector import collect_distillation_dataset_from_env, command_active_mask
from .data import build_distillation_dataset
from .g1_observation_adapter import RecoveryTeacherObservationAdapter, walk_to_recovery_obs
from .moe_student import MoEStudentPolicy
from .playback import load_distillation_student_policy

RECOVERY_ROLES = {"walk": 0, "stand_height": 1, "recovery": 2}
RECOVERY_ROUTING_CONTRACT = {
    "enter_height": 0.40,
    "enter_tilt_deg": 60.0,
    "nominal_settle_steps": 100,
    "standing_height": 0.65,
    "stable_tilt_deg": 10.0,
    "stable_linear_speed": 0.20,
    "stable_angular_speed": 0.30,
    "stable_seconds": 1.0,
    "target_height": 0.754,
    "ctrl_dt": 0.02,
    "state_source": "mujoco_physical_sensors",
    "student_observation_layout": "g1_walk_scaled_99_v1",
    "recovery_teacher_adapter": "g1_recovery_to_walk_99_v1",
}


def validate_recovery_env(env):
    for key in ("enter_height", "enter_tilt_deg", "nominal_settle_steps"):
        if getattr(env.cfg.recovery_routing, key) != RECOVERY_ROUTING_CONTRACT[key]:
            raise ValueError(f"recovery routing config mismatch: {key}")
    for key in (
        "standing_height",
        "stable_tilt_deg",
        "stable_linear_speed",
        "stable_angular_speed",
        "stable_seconds",
    ):
        if getattr(env.cfg.recovery, key) != RECOVERY_ROUTING_CONTRACT[key]:
            raise ValueError(f"recovery stable config mismatch: {key}")
    if env.cfg.ctrl_dt != 0.02 or env.cfg.commands.default_height != 0.754:
        raise ValueError("recovery control period or nominal height mismatch")


def validate_recovery_student(policy, runtime_cfg):
    if not isinstance(policy, MoEStudentPolicy) or (
        policy.obs_dim,
        policy.action_dim,
        policy.num_experts,
    ) != (99, 29, 3):
        raise ValueError("recovery integration requires a 99/29 three-expert MoE")
    if dict(runtime_cfg.get("role_expert_targets") or {}) != RECOVERY_ROLES:
        raise ValueError("recovery role mapping must be walk=0, stand_height=1, recovery=2")
    if dict(runtime_cfg.get("recovery_routing_contract") or {}) != RECOVERY_ROUTING_CONTRACT:
        raise ValueError("recovery routing checkpoint contract mismatch")
    if runtime_cfg.get("expert_behavior_loss_source") != "role":
        raise ValueError("recovery integration requires role-supervised expert behavior")


def extend_recovery_student_checkpoint(source, output, *, seed=1):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output == source:
        raise FileExistsError("migration must write a new checkpoint; source remains immutable")
    loaded = load_distillation_student_policy(source, device="cpu")
    old = loaded.policy
    cfg = dict(loaded.distill_runtime_cfg)
    if not isinstance(old, MoEStudentPolicy) or (old.obs_dim, old.action_dim, old.num_experts) != (
        99,
        29,
        2,
    ):
        raise ValueError("migration requires the existing 99/29 two-expert student")
    if dict(cfg.get("role_expert_targets") or {}) != {"walk": 0, "stand_height": 1}:
        raise ValueError("source role semantics are incompatible")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        target = MoEStudentPolicy(
            obs_dim=99,
            action_dim=29,
            num_experts=3,
            expert_hidden_dims=cfg["student_expert_hidden_dims"],
            router_hidden_dims=cfg["student_router_hidden_dims"],
            activation=cfg["student_activation"],
            squash_action=cfg["student_squash_action"],
            routing_mode="hard",
            router_temperature=cfg.get("student_router_temperature", 1.0),
        )
    for index in (0, 1):
        target.experts[index].load_state_dict(old.experts[index].state_dict(), strict=True)
    old_router, new_router = list(old.router.children()), list(target.router.children())
    for old_layer, new_layer in zip(old_router[:-1], new_router[:-1], strict=True):
        new_layer.load_state_dict(old_layer.state_dict(), strict=True)
    with torch.no_grad():
        new_router[-1].weight[:2].copy_(old_router[-1].weight)
        new_router[-1].bias[:2].copy_(old_router[-1].bias)
        new_router[-1].weight[2].zero_()
        new_router[-1].bias[2].fill_(-10)
    cfg.update(
        student_num_experts=3,
        student_routing_mode="hard",
        role_expert_targets=RECOVERY_ROLES,
        expert_behavior_loss_source="role",
        command_intent_loss_coef=0.0,
        recovery_integration=True,
        recovery_routing_contract=RECOVERY_ROUTING_CONTRACT,
    )
    cfg["recovery_migration"] = {
        "adapter": "g1_moe_2_to_3_v1",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preserved_experts": [0, 1],
        "new_expert": 2,
        "new_expert_trained": False,
        "optimizer_restored": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_distillation_checkpoint(
        output,
        student=target,
        agent_steps=loaded.agent_steps,
        teacher_metadata=loaded.teacher_metadata,
        distill_runtime_cfg=cfg,
    )
    return cfg["recovery_migration"]


def recovery_role_labels(info):
    active = np.asarray(info["recovery_active"], dtype=bool)
    commands = np.asarray(info["recovery_effective_commands"])
    walking = command_active_mask(commands, xy_threshold=0.05, yaw_threshold=0.05)
    return tuple(
        "recovery" if r else "walk" if w else "stand_height"
        for r, w in zip(active, walking, strict=True)
    )


def recovery_expert_indices(info):
    return np.asarray([RECOVERY_ROLES[role] for role in recovery_role_labels(info)], dtype=np.int64)


class RoleExpertPolicy(torch.nn.Module):
    def __init__(self, student, role):
        super().__init__()
        self.student = student
        self.index = RECOVERY_ROLES[role]

    def forward(self, obs):
        return self.student.experts[self.index](obs)


class RecoveryRoutedPolicy(torch.nn.Module):
    """The same env-owned physical latch selects DAgger and playback experts."""

    def __init__(self, policy, env):
        super().__init__()
        self.policy = policy
        self.env = env

    def forward(self, obs):
        roles = recovery_role_labels(self.env.state.info)
        action = torch.empty((len(roles), 29), dtype=obs.dtype, device=obs.device)
        for role, index in RECOVERY_ROLES.items():
            rows = torch.as_tensor([r == role for r in roles], device=obs.device)
            if rows.any():
                if isinstance(self.policy, MoEStudentPolicy):
                    action[rows] = self.policy.experts[index](obs[rows])
                else:
                    action[rows] = self.policy[role](obs[rows])
        return action


def collect_recovery_handover(
    env,
    *,
    recovery_teacher,
    standing_teacher,
    num_samples,
    student=None,
    metadata=None,
    performance_clock=None,
):
    teachers = torch.nn.ModuleDict(
        {
            "recovery": RecoveryTeacherObservationAdapter(recovery_teacher),
            "stand_height": standing_teacher,
        }
    )
    dataset = collect_distillation_dataset_from_env(
        env,
        num_samples=num_samples,
        expected_student_obs_dim=99,
        expected_teacher_obs_dim=99,
        action_mode="teacher_policy" if student is None else "student_policy",
        teacher_policy=RecoveryRoutedPolicy(teachers, env),
        rollout_policy=None if student is None else RecoveryRoutedPolicy(student, env),
        command_sample_filter="inactive",
        command_info_key="recovery_effective_commands",
        target_height_info_key="recovery_effective_height_commands",
        role_selector=recovery_role_labels,
        metadata=metadata,
        performance_clock=performance_clock,
    )

    if any(role not in {"recovery", "stand_height"} for role in dataset.role_labels):
        raise ValueError("recovery-to-stand scenario requires zero commanded velocity")
    teacher_native_obs = dataset.teacher_obs.clone()
    recovery_rows = torch.as_tensor([role == "recovery" for role in dataset.role_labels])
    teacher_native_obs[recovery_rows] = walk_to_recovery_obs(teacher_native_obs[recovery_rows])
    ages = np.full(dataset.num_samples, -1, dtype=np.int64)
    counters = np.zeros(env.num_envs, dtype=np.int64)
    for i, role in enumerate(dataset.role_labels):
        row = i % env.num_envs
        if role == "recovery":
            counters[row] = 0
        else:
            ages[i] = counters[row]
            counters[row] += 1
    return build_distillation_dataset(
        dataset.student_obs,
        teacher_native_obs,
        expected_student_obs_dim=99,
        expected_teacher_obs_dim=99,
        expected_teacher_action_dim=29,
        role_labels=dataset.role_labels,
        teacher_actions=dataset.teacher_actions,
        commands=dataset.commands,
        target_height=dataset.target_height,
        command_intents=dataset.command_intents,
        scenario_labels=("recovery_to_stand",) * dataset.num_samples,
        transition_ages=torch.as_tensor(ages),
        command_before=dataset.commands.clone(),
        command_after=dataset.commands.clone(),
        metadata={
            **dataset.metadata,
            "student_observation_layout": "g1_walk_scaled_99_v1",
            "recovery_teacher_adapter": "g1_recovery_to_walk_99_v1",
            "teacher_observation_units": "role_native",
        },
    )
