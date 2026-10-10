"""Exact G1 actor-unit adapter; dimensions alone do not establish observation parity."""

import numpy as np
import torch

RECOVERY_TO_WALK_ADAPTER = "g1_recovery_to_walk_99_v1"


def recovery_to_walk_obs(obs):
    if obs.ndim != 2 or obs.shape[1] != 99:
        raise ValueError("G1 unit adapter requires rank-2 99-D actor observations")
    result = obs.copy() if isinstance(obs, np.ndarray) else obs.clone()
    result[:, 0:3] *= 0.25
    result[:, 35:64] *= 0.05
    return result


def walk_to_recovery_obs(obs):
    if obs.ndim != 2 or obs.shape[1] != 99:
        raise ValueError("G1 unit adapter requires rank-2 99-D actor observations")
    result = obs.copy() if isinstance(obs, np.ndarray) else obs.clone()
    result[:, 0:3] *= 4.0
    result[:, 35:64] *= 20.0
    return result


class RecoveryTeacherObservationAdapter(torch.nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.teacher = teacher

    def forward(self, obs):
        return self.teacher(walk_to_recovery_obs(obs))
