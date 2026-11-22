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

Adaptation: if `shift_at` (or `shift_at_step`) is set, env.set_shift(shift_level)
is applied from that point on (evaluation uses the shifted task too).
A -> B -> A (D57, step budgets only): `shift_back_at_step` restores the first task. The return
on it right then, before any further learning, is meta["eval_after_shift_back"].

Eligibility traces are cleared at the start of every episode (agent.episode_start()).
meta.json also records how many learning-update products rounded to zero
(`underflow_frac`) and how many were clipped (`saturation_frac`): the Q16.16 diagnostics.
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

def _git_commit() -> str:
    """Commit of the code that produced a run (provenance for docs/results.md)."""
    import subprocess
    try:
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "c", "python/tinyrl"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return c.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


GIT_COMMIT = _git_commit()

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


def epsilon(t: int, horizon: int, r: dict) -> float:
    """Linear decay from eps_start to eps_end over `horizon` units (episodes or steps)."""
    if t >= horizon:
        return r["eps_end"]
    f = t / max(1, horizon)
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
    """Two budget modes:
      episode mode (default): `episodes`, `decay_episodes`, `eval_every` (episodes), `shift_at` (episode)
      step mode (if `steps` is set; recommended, see decision D02 and docs/research_review.md):
        `steps` env steps in total, `decay_steps`, `eval_every_steps`, `shift_at_step`.
      Either mode: `eval_base_seed` (default 10000) seeds the evaluation start states.
        Evaluation happens exactly at each checkpoint, mid-episode if necessary,
        on a separate environment, so every learner is evaluated after the same
        amount of experience. Evaluations are written to seed<k>_eval.csv.
    """
    r = {**RUN_DEFAULTS, **(run_cfg or {})}
    step_mode = r.get("steps") is not None
    env = make_env(task)
    eval_env = make_env(task)
    if learner.get("pil"):
        from .pil import PilAgent
        agent = PilAgent(learner, env.obs_lo, env.obs_hi, env.n_actions, seed=seed + 1)
    elif learner["kind"].startswith("baseline"):
        agent = BaselineAgent(learner, env.n_actions, seed=seed + 1)
    elif learner["kind"].endswith("_float"):
        from .ref.float_agents import FloatAgent
        agent = FloatAgent(learner, env.obs_lo, env.obs_hi, env.n_actions, seed=seed + 1)
    else:
        agent = Agent(learner, env.obs_lo, env.obs_hi, env.n_actions, seed=seed + 1)
    back_at = r.get("shift_back_at_step")
    if back_at is not None:
        first = r.get("shift_at_step")
        if not step_mode or first is None or not first < back_at < r["steps"]:
            raise ValueError("shift_back_at_step needs a step budget and shift_at_step < shift_back_at_step < steps")
    scale = env.reward_scale
    rows, evals = [], []
    t0 = time.time()
    updates, t_global, ep = 0, 0, 0
    shifted = False
    after_back = None
    if step_mode:
        budget, horizon = int(r["steps"]), int(r.get("decay_steps", r["steps"] // 2))
        every = int(r.get("eval_every_steps", max(1, budget // 30)))
        shift_at = r.get("shift_at_step")
    else:
        budget, horizon, every, shift_at = int(r["episodes"]), int(r["decay_episodes"]), int(r["eval_every"]), r["shift_at"]

    def do_shift():
        env.set_shift(r["shift_level"])
        eval_env.set_shift(r["shift_level"])

    if step_mode and shift_at == 0:          # from-scratch reference runs: shifted task throughout
        do_shift()
        shifted = True

    # evaluation start states (Go1: terrains) are seeds eval_base, eval_base + 1, ...; tuning uses its own (D66, D70)
    eval_base = int(r.get("eval_base_seed", 10_000))

    def do_eval(ev_step):
        v = evaluate(eval_env, agent, r["eval_episodes"], eval_base)
        evals.append({"step": ev_step, "episode": ep, "eval_return": v})
        if verbose:
            print(f"  step {ev_step:7d}  ep {ep:5d}  eval {v:9.2f}")
        return v

    while (t_global < budget) if step_mode else (ep < budget):
        if not step_mode and shift_at is not None and ep == shift_at:
            do_shift()
        eps_ep = epsilon(ep, horizon, r)
        o = env.reset(seed=seed * 100_003 + ep)
        if hasattr(agent, "episode_start"):
            agent.episode_start()
        ret, steps, done = 0.0, 0, False
        while not done:
            eps = epsilon(t_global, horizon, r) if step_mode else eps_ep
            a = agent.act(o, eps)
            o2, rew, term, trunc = env.step(a)
            updates += agent.observe(o, a, rew * scale, o2, term)
            ret += rew
            steps += 1
            t_global += 1
            o = o2
            done = term or trunc
            if step_mode:
                if t_global % every == 0:
                    do_eval(t_global)
                if shift_at is not None and not shifted and t_global >= shift_at:
                    do_shift()                   # after the eval: the checkpoint at shift_at is pre-shift (D58)
                    shifted = True
                if back_at is not None and after_back is None and t_global >= back_at:
                    env.set_shift(0.0)
                    eval_env.set_shift(0.0)
                    after_back = evaluate(eval_env, agent, r["eval_episodes"], eval_base)
                if t_global >= budget:
                    break
        ev = ""
        if not step_mode and ((ep + 1) % every == 0 or ep == budget - 1):
            ev = do_eval(t_global)
        rows.append({"episode": ep, "env_steps": t_global, "train_return": ret, "steps": steps,
                     "epsilon": round(eps, 4), "updates": updates, "eval_return": ev})
        ep += 1
    under = agent.underflow() if hasattr(agent, "underflow") else (0, 0)
    saturated = agent.saturation() if hasattr(agent, "saturation") else None
    pil = None
    if hasattr(agent, "stats"):
        cl = agent.cycle_log
        pil = {**agent.stats(), "cpu_hz": agent.cpu_hz,
               "mean_act_cycles": float(np.mean(cl["act"])) if cl["act"] else None,
               "mean_observe_cycles": float(np.mean(cl["observe"])) if cl["observe"] else None}
        agent.close()
    meta = {
        "task": task, "learner": learner, "run": r, "seed": seed, "budget": "steps" if step_mode else "episodes",
        "cfg_id": agent.id, "arena_bytes": agent.mem_bytes, "params": agent.params,
        "host_agent_bytes": agent.host_agent_bytes, "obs_dim": env.obs_dim,
        "n_actions": env.n_actions, "env_steps": t_global, "episodes": ep, "updates": updates,
        "underflow": under[0], "update_terms": under[1],
        "underflow_frac": round(under[0] / under[1], 4) if under[1] else None,
        "saturated": saturated,
        "saturation_frac": saturated / under[1] if saturated is not None and under[1] else None,
        "pil": pil, "eval_after_shift_back": after_back,
        "wall_s": round(time.time() - t0, 2), "git_commit": GIT_COMMIT,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"seed{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if step_mode:
            with open(out_dir / f"seed{seed}_eval.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["step", "episode", "eval_return"])
                w.writeheader()
                w.writerows(evals)
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        with open(out_dir / f"seed{seed}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    return {"meta": meta, "rows": rows, "evals": evals}
