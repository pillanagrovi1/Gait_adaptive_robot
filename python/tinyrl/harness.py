"""Run one (task, learner, seed) experiment and write its logs.

Per run the harness writes, under results/raw/<experiment>/<run_id>/:
    seed<k>.csv   one row per training episode:
                  episode, train_return, steps, epsilon, updates, eval_return (blank if no eval)
    meta.json     the full config, C config id, memory numbers, wall time

Training loop (per episode):
    eps = linear schedule from eps_start to eps_end over decay_episodes
    act epsilon-greedily, call observe() on every transition
    done passed to the learner = TERMINATED only. A time-limit truncation is
    not a real terminal state, so the target still bootstraps there.

Evaluation: every `eval_every` episodes, run `eval_episodes` greedy episodes
on fixed seeds (no learning, no exploration). Their mean return is the
"learning performance" used in every plot.

Adaptation: if `shift_at` is set, env.set_shift(shift_level) is applied
from that training episode on (evaluation uses the shifted task too).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .cbind import Agent
from .envs import make_env

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "raw"

RUN_DEFAULTS = {
    "episodes": 300,
    "eps_start": 1.0, "eps_end": 0.05, "decay_episodes": 150,
    "eval_every": 10, "eval_episodes": 5,
    "shift_at": None, "shift_level": 0.0,
}


class BaselineAgent:
    """Non-learning reference policies. Every learner must beat these.
    kind "baseline_const": always action `action` (e.g. the nominal gait)
    kind "baseline_random": uniform random actions
    """

    def __init__(self, learner: dict, n_actions: int, seed: int):
        self.kind = learner["kind"]
        self.action = int(learner.get("action", 0))
        self.n = n_actions
        self.rng = np.random.default_rng(seed)
        self.mem_bytes, self.params, self.host_agent_bytes = 0, 0, 0
        self.id = f"{self.kind}-{self.action}"

    def greedy(self, obs):
        if self.kind == "baseline_random":
            return int(self.rng.integers(self.n))
        return self.action

    act = lambda self, obs, eps: self.greedy(obs)  # noqa: E731

    def observe(self, *args):
        return 0


def epsilon(ep: int, r: dict) -> float:
    if ep >= r["decay_episodes"]:
        return r["eps_end"]
    f = ep / max(1, r["decay_episodes"])
    return r["eps_start"] + f * (r["eps_end"] - r["eps_start"])


def evaluate(env, agent: Agent, n: int, base_seed: int = 10_000) -> float:
    total = 0.0
    for k in range(n):
        o = env.reset(seed=base_seed + k)
        done = False
        while not done:
            o, r, term, trunc = env.step(agent.greedy(o))
            total += r
            done = term or trunc
    return total / n


def run(task: dict, learner: dict, seed: int, run_cfg: dict | None = None,
        out_dir: Path | None = None, verbose: bool = False) -> dict:
    r = {**RUN_DEFAULTS, **(run_cfg or {})}
    env = make_env(task)
    eval_env = make_env(task)
    if learner["kind"].startswith("baseline"):
        agent = BaselineAgent(learner, env.n_actions, seed=seed + 1)
    elif learner["kind"].endswith("_float"):
        from .ref.float_agents import FloatAgent
        agent = FloatAgent(learner, env.obs_lo, env.obs_hi, env.n_actions, seed=seed + 1)
    else:
        agent = Agent(learner, env.obs_lo, env.obs_hi, env.n_actions, seed=seed + 1)
    scale = env.reward_scale
    rows = []
    t0 = time.time()
    updates = 0
    for ep in range(r["episodes"]):
        if r["shift_at"] is not None and ep == r["shift_at"]:
            env.set_shift(r["shift_level"])
            eval_env.set_shift(r["shift_level"])
        eps = epsilon(ep, r)
        o = env.reset(seed=seed * 100_003 + ep)
        ret, steps, done = 0.0, 0, False
        while not done:
            a = agent.act(o, eps)
            o2, rew, term, trunc = env.step(a)
            updates += agent.observe(o, a, rew * scale, o2, term)
            ret += rew
            steps += 1
            o = o2
            done = term or trunc
        ev = ""
        if (ep + 1) % r["eval_every"] == 0 or ep == r["episodes"] - 1:
            ev = evaluate(eval_env, agent, r["eval_episodes"])
        rows.append({"episode": ep, "train_return": ret, "steps": steps,
                     "epsilon": round(eps, 4), "updates": updates, "eval_return": ev})
        if verbose and ev != "":
            print(f"  ep {ep+1:4d}  eps {eps:.2f}  train {ret:9.2f}  eval {ev:9.2f}")
    meta = {
        "task": task, "learner": learner, "run": r, "seed": seed,
        "cfg_id": agent.id, "arena_bytes": agent.mem_bytes, "params": agent.params,
        "host_agent_bytes": agent.host_agent_bytes, "obs_dim": env.obs_dim,
        "n_actions": env.n_actions, "wall_s": round(time.time() - t0, 2),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"seed{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    return {"meta": meta, "rows": rows}
