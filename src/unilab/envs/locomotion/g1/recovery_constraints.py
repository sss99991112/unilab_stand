"""Opt-in privileged control-quality rewards; actor observations stay unchanged."""

from dataclasses import dataclass

import numpy as np

CONSTRAINT_TERMS = (
    "recovery_joint_boundary",
    "recovery_outward_speed",
    "recovery_pd_excess",
    "recovery_self_collision",
)


@dataclass
class RecoveryConstraintsCfg:
    enabled: bool = False
    version: str = "g1_recovery_load_reward_v1"
    boundary_margin_rad: float = 0.05
    outward_speed_scale: float = 5.0
    contact_force_scale_n: float = 100.0
    body_pairs: tuple = (
        ("right_hip_yaw_link", "right_wrist_pitch_link"),
        ("left_elbow_link", "torso_link"),
    )

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("recovery constraints enabled must be boolean")
        if self.version != "g1_recovery_load_reward_v1":
            raise ValueError("unsupported recovery load reward version")
        values = [self.boundary_margin_rad, self.outward_speed_scale, self.contact_force_scale_n]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("constraint normalization scales must be finite and positive")
        pairs = [tuple(p) for p in self.body_pairs]
        if not pairs or any(len(p) != 2 or p[0] == p[1] for p in pairs):
            raise ValueError("explicit distinct self-body pairs are required")
        if len({tuple(sorted(p)) for p in pairs}) != len(pairs):
            raise ValueError("duplicate self-body pairs")


def constraint_reward_terms(metrics, cfg, count):
    """Mean substep costs; only specified robot self-pairs, never ground contacts.

    Joint costs use post-integration measured q/dq. Raw PD uses the pre-step
    state, before actuator clipping. No policy target or observation is changed.
    """
    zero = {k: np.zeros(count, np.float64) for k in CONSTRAINT_TERMS}
    if not cfg.enabled or metrics is None:
        return zero, {}
    q, dq, raw = (metrics[k] for k in ("q", "dq", "raw_pd"))
    lo, hi = metrics["joint_ranges"][:, None, :, 0], metrics["joint_ranges"][:, None, :, 1]
    f_lo, f_hi = metrics["force_ranges"][:, None, :, 0], metrics["force_ranges"][:, None, :, 1]
    force = metrics["pair_normal_force"]
    if not all(np.isfinite(x).all() for x in (q, dq, raw, force)):
        raise FloatingPointError("non-finite privileged recovery measurements")
    m = cfg.boundary_margin_rad
    near_lo = np.maximum(1 - (q - lo) / m, 0)
    near_hi = np.maximum(1 - (hi - q) / m, 0)
    actual_excess = np.maximum(np.maximum(lo - q, q - hi), 0)
    pd_excess = raw - np.clip(raw, f_lo, f_hi)
    normalized_pd = pd_excess / np.maximum(np.abs(f_lo), np.abs(f_hi))
    outward = np.maximum(-dq, 0) * np.minimum(near_lo, 1) + np.maximum(dq, 0) * np.minimum(
        near_hi, 1
    )
    costs = (
        np.sum(near_lo**2 + near_hi**2, axis=2),
        np.sum((outward / cfg.outward_speed_scale) ** 2, axis=2),
        np.sum(normalized_pd**2, axis=2),
        np.sum((force / cfg.contact_force_scale_n) ** 2, axis=2),
    )
    valid = metrics["valid"]
    terms = {k: np.where(valid, v.mean(axis=1), 0) for k, v in zip(CONSTRAINT_TERMS, costs)}
    valid3 = valid[:, None, None]
    log = {
        "constraint/max_joint_excess_rad": float(np.max(np.where(valid3, actual_excess, 0))),
        "constraint/max_outward_speed_rad_s": float(np.max(np.where(valid3, outward, 0))),
        "constraint/max_pd_excess_nm": float(np.max(np.where(valid3, np.abs(pd_excess), 0))),
        "constraint/max_pair_normal_force_n": float(np.max(np.where(valid3, force, 0))),
        "constraint/replay_error": float(metrics["replay_error"]),
        "constraint/substeps_per_control": int(q.shape[1]),
    }
    return terms, log
