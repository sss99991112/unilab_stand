"""Versioned actor-only refinement contract; old learner state never transfers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.envs.locomotion.g1.recovery_constraints import CONSTRAINT_TERMS

REFINEMENT_ADAPTER = "g1_recovery_load_actor_99_v1"
ACTION_CONTRACT = "g1_joint_offset_default_angles_scale1_v1"


def validate_refinement_config(cfg):
    if OmegaConf.select(cfg, "training.play_only", default=False):
        return
    if OmegaConf.select(cfg, "algo.resume_checkpoint") not in (None, ""):
        raise ValueError(
            "new recovery rewards require actor-only initialization, never learner resume"
        )
    if str(OmegaConf.select(cfg, "algo.load_run", default="-1")) != "-1":
        raise ValueError("refinement cannot use load_run")
    if OmegaConf.select(cfg, "algo.actor_warm_start_adapter") != REFINEMENT_ADAPTER:
        raise ValueError("refinement requires its versioned actor-only adapter")
    if (
        OmegaConf.select(cfg, "algo.algo") != "sac"
        or OmegaConf.select(cfg, "training.sim_backend") != "mujoco"
    ):
        raise ValueError("refinement requires UniLab SAC / MuJoCo")
    if (
        OmegaConf.select(cfg, "env.ctrl_dt", default=0.02) != 0.02
        or OmegaConf.select(cfg, "env.sim_substeps", default=3) != 3
    ):
        raise ValueError("refinement v1 requires original 50Hz / 3-substep timing")
    p = OmegaConf.select(cfg, "algo.actor_warm_start_checkpoint")
    if not p:
        raise ValueError("refinement requires the frozen parent checkpoint")
    parent = Path(str(p)).expanduser().resolve()
    expected = OmegaConf.select(cfg, "algo.actor_warm_start_expected_sha256")
    if not parent.is_file() or hashlib.sha256(parent.read_bytes()).hexdigest() != expected:
        raise ValueError("refinement parent checkpoint SHA mismatch")
    previous = OmegaConf.create(
        json.loads((parent.parent / "run_config.json").read_text())["config"]
    )
    if (
        OmegaConf.select(previous, "training.task_name") != "G1Recovery"
        or OmegaConf.select(previous, "algo.algo") != "sac"
    ):
        raise ValueError("refinement parent must be the recorded G1Recovery SAC task")
    for key in (
        "env.control_config.action_scale",
        "env.commands",
        "env.assistance",
        "env.recovery",
        "algo.actor_hidden_dim",
        "algo.use_layer_norm",
        "algo.obs_normalization",
    ):
        old, new = OmegaConf.select(previous, key), OmegaConf.select(cfg, key)
        if OmegaConf.is_config(old):
            old = OmegaConf.to_container(old, resolve=True)
        if OmegaConf.is_config(new):
            new = OmegaConf.to_container(new, resolve=True)
        if old != new:
            raise ValueError(f"refinement actor/physics contract mismatch: {key}")
    old_env = OmegaConf.to_container(previous.env, resolve=True)
    new_env = OmegaConf.to_container(cfg.env, resolve=True)
    new_env.pop("recovery_constraints", None)
    if old_env != new_env:
        raise ValueError("refinement must preserve the complete baseline env/physics configuration")
    for key in ("algo.actor", "algo.runtime_impl", "algo.runtime_resolver", "algo.use_symmetry"):
        old, new = OmegaConf.select(previous, key), OmegaConf.select(cfg, key)
        if OmegaConf.is_config(old):
            old = OmegaConf.to_container(old, resolve=True)
        if OmegaConf.is_config(new):
            new = OmegaConf.to_container(new, resolve=True)
        if old != new:
            raise ValueError(f"refinement actor runtime contract mismatch: {key}")
    if OmegaConf.select(cfg, "env.control_config.simulate_action_latency", default=False):
        raise ValueError("refinement requires original direct-action semantics")
    for key, scale in previous.reward.scales.items():
        if OmegaConf.select(cfg, "reward.scales." + key) != scale:
            raise ValueError("refinement must preserve all original recovery reward weights")
    for key in CONSTRAINT_TERMS:
        scale = OmegaConf.select(cfg, "reward.scales." + key)
        if not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale >= 0:
            raise ValueError("refinement requires all four finite negative cost weights")
    from unilab.algos.torch.offpolicy.runner import compute_train_start_threshold

    if (
        compute_train_start_threshold(
            cfg.algo.batch_size, cfg.algo.learning_starts, cfg.algo.num_envs
        )
        > cfg.algo.replay_buffer_n * cfg.algo.num_envs
    ):
        raise ValueError("replay can never reach learning_starts; refusing a stalled run")
    destination = OmegaConf.select(cfg, "training.log_dir")
    if not destination or Path(str(destination)).expanduser().resolve() == parent.parent:
        raise ValueError("refinement requires a distinct explicit new run directory")


def validate_fresh_refinement_output(cfg):
    if (
        not OmegaConf.select(cfg, "env.recovery_constraints.enabled", default=False)
        or cfg.training.play_only
    ):
        return
    destination = Path(str(cfg.training.log_dir)).expanduser()
    if destination.exists():
        raise ValueError("refinement output already exists; do not overwrite or silently resume")


def load_refinement_actor(learner, path):
    from unilab.algos.torch.offpolicy.checkpoint_adapter import (
        load_g1_height_actor_continuation_warm_start,
    )

    if learner.update_count != 0 or any(
        opt.state for opt in (learner.actor_optimizer, learner.q_optimizer, learner.alpha_optimizer)
    ):
        raise ValueError("refinement requires fresh critic/optimizers and update_count zero")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("obs_normalizer") is not None
        or getattr(learner, "obs_normalizer", None) is not None
    ):
        raise ValueError(
            "v1 refinement uses the observed absent external normalizer; statistics cannot be invented"
        )
    state = checkpoint["actor"]
    if not torch.equal(state["action_scale"], torch.ones(29)) or not torch.equal(
        state["action_bias"], torch.zeros(29)
    ):
        raise ValueError("action mapping changed; a different versioned adapter is required")
    metadata = load_g1_height_actor_continuation_warm_start(learner, path)
    metadata.update(
        adapter_id=REFINEMENT_ADAPTER,
        action_contract=ACTION_CONTRACT,
        external_normalizer="absent_identity_v1",
        critic_initialization="fresh",
        optimizer_initialization="fresh",
        replay_initialization="cold",
    )
    learner.actor_warm_start_metadata = metadata
    return metadata
