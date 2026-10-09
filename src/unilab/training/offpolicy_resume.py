"""Explicit same-task SAC learner restoration; replay and simulator restart fresh."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

REQUIRED_LEARNER_KEYS = frozenset(
    {
        "actor",
        "qnet",
        "qnet_target",
        "log_alpha",
        "actor_optimizer",
        "q_optimizer",
        "alpha_optimizer",
        "update_count",
    }
)


def validate_configured_learner_resume(cfg: DictConfig) -> Path | None:
    value = OmegaConf.select(cfg, "algo.resume_checkpoint")
    if value in (None, ""):
        return None
    if (OmegaConf.select(cfg, "training.task_name"), OmegaConf.select(cfg, "algo.algo")) != (
        "G1Recovery",
        "sac",
    ):
        raise ValueError("learner resume currently supports only G1Recovery SAC")
    if bool(OmegaConf.select(cfg, "training.play_only", default=False)):
        raise ValueError("learner resume is a training option, not a playback model selector")
    if OmegaConf.select(cfg, "algo.actor_warm_start_checkpoint") not in (None, ""):
        raise ValueError("learner resume and actor-only warm start are mutually exclusive")
    if OmegaConf.select(cfg, "algo.actor_warm_start_adapter") not in (None, ""):
        raise ValueError("learner resume must not select an actor-only adapter")
    if str(OmegaConf.select(cfg, "algo.load_run", default="-1")) != "-1":
        raise ValueError("use resume_checkpoint without load_run")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    sidecar = path.parent / "run_config.json"
    if not sidecar.is_file():
        raise ValueError("learner resume requires the original run_config.json")
    previous = OmegaConf.create(json.loads(sidecar.read_text())["config"])
    # Budget, logging, batch size, replay filling and device may change. Reward,
    # physics and learner mathematics must match; optimizer load restores its LR.
    fields = (
        "training.task_name",
        "training.sim_backend",
        "training.use_amp",
        "env",
        "reward",
        "algo.algo",
        "algo.gamma",
        "algo.tau",
        "algo.actor_lr",
        "algo.critic_lr",
        "algo.actor_hidden_dim",
        "algo.critic_hidden_dim",
        "algo.num_atoms",
        "algo.use_layer_norm",
        "algo.obs_normalization",
        "algo.use_symmetry",
        "algo.runtime_impl",
        "algo.runtime_resolver",
        "algo.algo_params.alpha_lr",
        "algo.algo_params.alpha_init",
        "algo.algo_params.target_entropy_ratio",
        "algo.algo_params.max_grad_norm",
        "algo.algo_params.amp_dtype",
    )
    for field in fields:
        old, new = OmegaConf.select(previous, field), OmegaConf.select(cfg, field)
        if OmegaConf.is_config(old):
            old = OmegaConf.to_container(old, resolve=True)
        if OmegaConf.is_config(new):
            new = OmegaConf.to_container(new, resolve=True)
        if old != new:
            raise ValueError(f"learner resume config mismatch: {field}")
    log_dir = OmegaConf.select(cfg, "training.log_dir")
    if log_dir is not None and Path(str(log_dir)).expanduser().resolve() == path.parent:
        raise ValueError("learner resume must not overwrite the source run")
    return path


def apply_configured_learner_resume(
    algo_name: str, cfg: DictConfig, runner: Any
) -> dict[str, Any] | None:
    path = validate_configured_learner_resume(cfg)
    if path is None:
        return None
    import torch

    from unilab.algos.torch.fast_sac.learner import FastSACLearner

    if algo_name != "sac" or not isinstance(runner.learner, FastSACLearner):
        raise ValueError("learner resume requires UniLab FastSACLearner")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("resume checkpoint must contain a learner state mapping")
    missing = REQUIRED_LEARNER_KEYS - checkpoint.keys()
    if missing:
        raise ValueError(f"incomplete SAC learner checkpoint: {sorted(missing)}")
    count = checkpoint["update_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("resume update_count must be a nonnegative integer")
    # Validate shapes and optimizer structure before mutating the fresh learner.
    for name in ("actor", "qnet", "qnet_target"):
        saved = checkpoint[name]
        expected = getattr(runner.learner, name).state_dict()
        if not isinstance(saved, Mapping) or set(saved) != set(expected):
            raise ValueError(f"resume {name} state keys mismatch")
        for key, expected_tensor in expected.items():
            value = saved[key]
            if not torch.is_tensor(value) or value.shape != expected_tensor.shape:
                raise ValueError(f"resume {name}.{key} shape mismatch")
            if not torch.isfinite(value).all():
                raise ValueError(f"resume {name}.{key} contains non-finite values")
    alpha = checkpoint["log_alpha"]
    if (
        not torch.is_tensor(alpha)
        or alpha.shape != runner.learner.log_alpha.shape
        or not torch.isfinite(alpha).all()
    ):
        raise ValueError("invalid resume log_alpha")
    for name in ("actor_optimizer", "q_optimizer", "alpha_optimizer"):
        saved = checkpoint[name]
        expected = getattr(runner.learner, name).state_dict()
        if not isinstance(saved, Mapping) or not isinstance(saved.get("state"), Mapping):
            raise ValueError(f"invalid resume {name} state")
        groups = saved.get("param_groups")
        if not isinstance(groups, list) or len(groups) != len(expected["param_groups"]):
            raise ValueError(f"resume {name} parameter groups mismatch")
        if any(
            len(a.get("params", [])) != len(b["params"])
            for a, b in zip(groups, expected["param_groups"])
        ):
            raise ValueError(f"resume {name} parameter count mismatch")
    if checkpoint.get("obs_normalizer") is not None:
        raise ValueError("external obs_normalizer is unsupported by standard FastSACLearner resume")
    runner.learner.load_state_dict(dict(checkpoint))
    metadata = {
        "resume_kind": "sac_learner_state_with_fresh_replay",
        "resume_checkpoint": str(path),
        "resume_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "restored_update_count": runner.learner.update_count,
        "replay_restored": False,
        "simulator_and_rng_restored": False,
    }
    print(
        f"[LearnerResume] restored actor, critic, target critic, all optimizers and alpha; "
        f"update_count={runner.learner.update_count}; replay=fresh",
        flush=True,
    )
    return metadata
