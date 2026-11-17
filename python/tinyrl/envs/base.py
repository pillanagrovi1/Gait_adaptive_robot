"""Minimal environment interface shared by every task.

Deliberately NOT gymnasium: fewer dependencies, and the learner only ever
needs discrete actions plus box bounds for its observation.

    obs = env.reset(seed)                       -> np.ndarray, shape (obs_dim,)
    obs, r, terminated, truncated = env.step(a) -> a is an int in [0, n_actions)
    env.set_shift(level)                        -> change the physics (adaptation tests)

`obs_lo` / `obs_hi` are the bounds the learner normalises with. Values outside
are clipped by the C code, so pick bounds that cover what actually happens.
"""
from __future__ import annotations

import numpy as np


class Env:
    name: str = "env"
    obs_lo: np.ndarray
    obs_hi: np.ndarray
    n_actions: int
    max_steps: int
    reward_scale: float = 1.0     # multiply rewards by this before the learner sees them

    @property
    def obs_dim(self) -> int:
        return int(self.obs_lo.shape[0])

    def reset(self, seed: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def step(self, a: int):
        raise NotImplementedError

    def set_shift(self, level: float) -> None:
        """level = 0 is the nominal task; >0 perturbs it. Default: no-op."""


def wrap_angle(x: float) -> float:
    return (x + np.pi) % (2 * np.pi) - np.pi
