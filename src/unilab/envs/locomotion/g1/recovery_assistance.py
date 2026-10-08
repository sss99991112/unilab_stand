"""Fixed-per-run assistance stages for the independent G1 recovery task."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RecoveryAssistanceCfg:
    # Fractions of nominal robot weight; initial values, not trained/tuned results.
    force_fractions: list[float] = field(default_factory=lambda: [0.4, 0.2, 0.0])
    stage: int = 2
    body_name: str = "torso_link"
    start_after_seconds: float = 0.6
    min_upright_cos: float = 0.8
    evaluation: bool = False

    def validate(self) -> None:
        fractions = np.asarray(self.force_fractions, dtype=float)
        if fractions.ndim != 1 or len(fractions) == 0 or not np.all(np.isfinite(fractions)):
            raise ValueError("assistance force_fractions must be a finite non-empty list")
        if np.any(fractions < 0) or np.any(fractions >= 1) or fractions[-1] != 0:
            raise ValueError("assistance fractions must be in [0, 1) and end at zero")
        if np.any(np.diff(fractions) >= 0):
            raise ValueError("assistance fractions must strictly decrease to zero")
        if (
            isinstance(self.stage, bool)
            or not isinstance(self.stage, int)
            or not 0 <= self.stage < len(fractions)
        ):
            raise ValueError("assistance stage must index force_fractions")
        if (
            not self.body_name
            or not np.isfinite(self.start_after_seconds)
            or self.start_after_seconds < 0
        ):
            raise ValueError("assistance requires a body name and non-negative start time")
        if not np.isfinite(self.min_upright_cos) or not 0 <= self.min_upright_cos <= 1:
            raise ValueError("assistance min_upright_cos must be in [0, 1]")
        if not isinstance(self.evaluation, bool):
            raise ValueError("assistance evaluation must be boolean")

    @property
    def effective_fraction(self) -> float:
        return 0.0 if self.evaluation else float(self.force_fractions[self.stage])


def assistance_force_rows(
    up_z: np.ndarray,
    episode_steps: np.ndarray,
    *,
    force_z: float,
    start_steps: int,
    min_upright_cos: float,
) -> np.ndarray:
    """One world-Z force per environment; never pull during initial fall or upside down."""
    active = (episode_steps >= start_steps) & (up_z > min_upright_cos)
    force = np.zeros((len(up_z), 1, 3), dtype=np.float64)
    force[:, 0, 2] = float(force_z) * active
    return force
