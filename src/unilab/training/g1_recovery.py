"""Recovery stage continuation checks; no training, simulation, or checkpoint mutation."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from unilab.envs.locomotion.g1.recovery_assistance import RecoveryAssistanceCfg

RECOVERY_CONTINUATION_ADAPTER = "g1_height_actor_obs_99_to_99_v1"


def _assistance(cfg: DictConfig) -> RecoveryAssistanceCfg:
    value = OmegaConf.select(cfg, "env.assistance")
    if value is None:
        raise ValueError("recovery stage identity requires env.assistance")
    result = RecoveryAssistanceCfg(**OmegaConf.to_container(value, resolve=True))
    result.validate()
    return result


def validate_recovery_training_config(cfg: DictConfig) -> None:
    """Cross-stage actor transfer or explicit same-task learner restoration."""
    if OmegaConf.select(cfg, "training.task_name") != "G1Recovery":
        return
    current = _assistance(cfg)
    if bool(OmegaConf.select(cfg, "training.play_only", default=False)):
        return
    if current.evaluation and current.force_fractions[current.stage] > 0:
        raise ValueError("an assisted training stage cannot run in evaluation mode")
    if str(OmegaConf.select(cfg, "algo.algo")) != "sac":
        raise ValueError("G1Recovery staged training currently requires SAC")
    if str(OmegaConf.select(cfg, "algo.load_run", default="-1")) != "-1":
        raise ValueError(
            "recovery continuation uses actor_warm_start_checkpoint or resume_checkpoint, not load_run"
        )
    if OmegaConf.select(cfg, "algo.resume_checkpoint") not in (None, ""):
        from unilab.training.offpolicy_resume import validate_configured_learner_resume

        validate_configured_learner_resume(cfg)
        return
    parent = OmegaConf.select(cfg, "algo.actor_warm_start_checkpoint")
    if parent in (None, ""):
        return
    if OmegaConf.select(cfg, "algo.actor_warm_start_adapter") != RECOVERY_CONTINUATION_ADAPTER:
        raise ValueError("recovery continuation requires the strict 99-to-99 actor adapter")
    checkpoint = Path(str(parent)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"recovery parent checkpoint not found: {checkpoint}")
    sidecar = checkpoint.parent / "run_config.json"
    if not sidecar.is_file():
        raise ValueError("recovery parent requires its run_config.json stage identity")
    previous = OmegaConf.create(json.loads(sidecar.read_text())["config"])
    if (
        OmegaConf.select(previous, "training.task_name") != "G1Recovery"
        or OmegaConf.select(previous, "algo.algo") != "sac"
    ):
        raise ValueError("recovery parent must be a G1Recovery SAC run")
    prior = _assistance(previous)
    if prior.evaluation:
        raise ValueError("recovery parent must record a training-stage identity")
    if current.force_fractions != prior.force_fractions or current.stage not in (
        prior.stage,
        prior.stage + 1,
    ):
        raise ValueError(
            "recovery continuation must retain the schedule and use the same or next stage"
        )
    fields = (
        "env.control_config.action_scale",
        "env.commands.default_height",
        "env.commands.observe_height_command",
        "algo.obs_normalization",
        "algo.actor_hidden_dim",
        "algo.use_layer_norm",
        "env.assistance.body_name",
        "env.assistance.start_after_seconds",
        "env.assistance.min_upright_cos",
    )
    for field in fields:
        old, new = OmegaConf.select(previous, field), OmegaConf.select(cfg, field)
        if old is None or new is None or old != new:
            raise ValueError(f"recovery continuation contract mismatch: {field}")
    log_dir = OmegaConf.select(cfg, "training.log_dir")
    if log_dir is not None and Path(str(log_dir)).expanduser().resolve() == checkpoint.parent:
        raise ValueError("recovery continuation must not overwrite its parent run")
