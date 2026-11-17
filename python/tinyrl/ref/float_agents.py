"""Floating-point mirrors of the C learners (same algorithm, float64 maths).

Purpose: when a C learner misbehaves, run the float twin with the same config.
  - float learns, C does not  -> a fixed-point problem (range, rounding, underflow)
  - neither learns            -> an algorithm or hyperparameter problem
Use learner kind "mlp_float" / "tabular_float" / "linear_float" in the harness.
Not bit-exact with C (that is what the corridor gate is for).
"""
from __future__ import annotations

import numpy as np

from ..cbind import LEARNER_DEFAULTS


class FloatAgent:
    def __init__(self, learner: dict, obs_lo, obs_hi, n_actions: int, seed: int):
        p = {**LEARNER_DEFAULTS, **learner}
        self.p = p
        self.kind = p["kind"].replace("_float", "")
        self.lo, self.hi = np.asarray(obs_lo, float), np.asarray(obs_hi, float)
        self.d, self.A = len(obs_lo), n_actions
        self.rng = np.random.default_rng(seed)
        self.alpha, self.gamma = p["alpha"], p["gamma"]
        self.clip = p["td_clip"]
        self.n_updates = 0
        if self.kind == "tabular":
            self.bins = p["bins"]
            self.table = np.zeros((self.bins ** self.d, self.A))
            n_params = self.table.size
        elif self.kind == "linear":
            self.T, self.N, self.M = p["tilings"], p["tiles"], p["features"]
            self.w = np.zeros((self.M, self.A))
            n_params = self.w.size
        elif self.kind == "mlp":
            H = p["hidden"]
            s1, s2 = 1 / np.sqrt(self.d), 1 / np.sqrt(H)
            self.W1 = self.rng.uniform(-s1, s1, (H, self.d)); self.b1 = np.zeros(H)
            self.W2 = self.rng.uniform(-s2, s2, (self.A, H)); self.b2 = np.zeros(self.A)
            self.period = p["target_period"]
            self._sync()
            n_params = self.W1.size + H + self.W2.size + self.A
        else:
            raise ValueError(self.kind)
        self.params = n_params
        self.mem_bytes = 4 * n_params          # rough: float32 equivalent, scratch ignored
        self.host_agent_bytes = 0
        self.id = f"float-{self.kind}"
        cap = p["replay_cap"]
        self.cap, self.batch = cap, p["batch"]
        self.buf = []

    # ---------- representation ----------
    def _u(self, s):
        return np.clip((np.asarray(s, float) - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def _cell(self, s):
        b = np.minimum((self._u(s) * self.bins).astype(int), self.bins - 1)
        return int(np.sum(b * self.bins ** np.arange(self.d)))

    def _tiles(self, s):
        u = self._u(s)
        idx = []
        for t in range(self.T):
            off = ((t * (2 * np.arange(self.d) + 1)) % self.T) / self.T
            coords = np.floor(u * self.N + off).astype(int)
            idx.append(hash((t, *coords.tolist())) % self.M)
        return idx

    def _sync(self):
        self.tW1, self.tb1, self.tW2, self.tb2 = (self.W1.copy(), self.b1.copy(),
                                                  self.W2.copy(), self.b2.copy())

    def _fwd(self, s, target=False):
        x = 2 * self._u(s) - 1
        W1, b1, W2, b2 = ((self.tW1, self.tb1, self.tW2, self.tb2)
                          if target and self.period > 0 else (self.W1, self.b1, self.W2, self.b2))
        z = W1 @ x + b1
        h = np.maximum(z, 0)
        return W2 @ h + b2, x, z, h

    def qvalues(self, s, target=False):
        if self.kind == "tabular":
            return self.table[self._cell(s)].copy()
        if self.kind == "linear":
            return self.w[self._tiles(s)].sum(0)
        return self._fwd(s, target)[0]

    # ---------- acting ----------
    def greedy(self, s):
        q = self.qvalues(s)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def act(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.A))
        return self.greedy(s)

    # ---------- learning ----------
    def _clipd(self, d):
        return float(np.clip(d, -self.clip, self.clip)) if self.clip > 0 else d

    def _learn(self, s, a, r, s2, done):
        y = r if done else r + self.gamma * self.qvalues(s2, target=True).max()
        if self.kind == "tabular":
            c = self._cell(s)
            self.table[c, a] += self.alpha * self._clipd(y - self.table[c, a])
        elif self.kind == "linear":
            idx = self._tiles(s)
            d = self._clipd(y - self.w[idx, a].sum())
            for i in idx:
                self.w[i, a] += self.alpha / self.T * d
        else:
            q, x, z, h = self._fwd(s)
            d = self._clipd(y - q[a])
            g = d * self.W2[a] * (z > 0)
            self.W2[a] += self.alpha * d * h
            self.b2[a] += self.alpha * d
            self.W1 += self.alpha * np.outer(g, x)
            self.b1 += self.alpha * g
        self.n_updates += 1
        if self.kind == "mlp" and self.period > 0 and self.n_updates % self.period == 0:
            self._sync()

    def observe(self, s, a, r, s2, done):
        if self.cap == 0:
            self._learn(s, a, r, s2, done)
            return 1
        self.buf.append((np.array(s), a, r, np.array(s2), done))
        if len(self.buf) > self.cap:
            self.buf.pop(0)
        if len(self.buf) < self.batch:
            return 0
        for j in self.rng.integers(len(self.buf), size=self.batch):
            self._learn(*self.buf[j])
        return self.batch
