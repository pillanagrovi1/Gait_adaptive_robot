#!/usr/bin/env python3
"""Go1 baseline B5: episodic (1+1) evolution strategy over the reflex gains.

The cheapest serious competitor to a per-step learner (docs/research_review_2.md):
memory = two parameter vectors (here 4 numbers each), one update per EPISODE.
Kohl & Stone (2004) tuned 12 AIBO gait parameters this way (policy-gradient
flavour) in ~1000 runs; Calandra et al. reached good biped gaits with Bayesian
optimisation in < 30-80 trials. If a per-step Q-learner cannot beat this on the
same number of episodes, that is a finding.

Parameters searched (in [0, 1], mapped to physical ranges):
    posture_gain   0 .. 1    IMU posture reflex gain (rad/rad)
    tegotae_sigma  0 .. 2    load-based phase feedback
    a_calf         0.25 .. 0.6 rad   swing foot clearance
    stumble        off / on  (on if > 0.5)
(1+1)-ES with the 1/5th success rule: sigma *= 1.5 after a success, *= 0.9 otherwise
(about the 1/5th rule's balance point). Each candidate is scored on `--trials` episodes
with fresh terrain seeds; the parent is re-scored every time, to avoid lucky parents.

    python3 scripts/go1_es.py --evals 60 --roughness 0.10
Output: results/figures/go1_es.csv (every evaluation) and go1_es.png
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tinyrl.envs.go1 import Go1Gait  # noqa: E402

OUT = ROOT / "results" / "figures"
NAMES = ("posture_gain", "tegotae_sigma", "a_calf", "stumble")


def to_kwargs(x):
    x = np.clip(x, 0.0, 1.0)
    return {"posture_gain": float(x[0]), "tegotae_sigma": float(2.0 * x[1]),
            "a_calf": float(0.25 + 0.35 * x[2]), "stumble": bool(x[3] > 0.5)}


def score(x, roughness, seeds):
    env = Go1Gait(roughness=roughness, phase_lock=True, obs_mode="imu4p", **to_kwargs(x))
    total = 0.0
    for s in seeds:
        env.reset(seed=int(s))
        done = False
        while not done:
            _, r, term, trunc = env.step(0)
            total += r
            done = term or trunc
    return total / len(seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=60, help="candidate evaluations (episodes = evals * trials * 2)")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--roughness", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    x = np.array([0.0, 0.0, (0.35 - 0.25) / 0.35, 0.0])    # start at B0: the open-loop trot
    sigma = 0.25
    rows = []
    episode_seed = 70_000
    for k in range(a.evals):
        seeds = np.arange(episode_seed, episode_seed + a.trials)
        episode_seed += a.trials
        cand = np.clip(x + sigma * rng.standard_normal(4), 0.0, 1.0)
        fp, fc = score(x, a.roughness, seeds), score(cand, a.roughness, seeds)   # same terrain for both
        if fc > fp:
            x, sigma = cand, min(sigma * 1.5, 0.5)
        else:
            sigma = max(sigma * 0.9, 0.02)
        rows.append({"eval": k, "episodes_used": (k + 1) * 2 * a.trials, "parent_return": fp,
                     "candidate_return": fc, "sigma": sigma, **{n: v for n, v in to_kwargs(x).items()}})
        print(f"{k:3d}  parent {fp:7.2f}  cand {fc:7.2f}  sigma {sigma:.3f}  {to_kwargs(x)}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "go1_es.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ep = [r["episodes_used"] for r in rows]
    ax.plot(ep, [max(r["parent_return"], r["candidate_return"]) if r["candidate_return"] > r["parent_return"]
                 else r["parent_return"] for r in rows], color="#6250d6", linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("episodes used", color="#52514e", fontsize=9)
    ax.set_ylabel("return of the kept gains (same terrain pair)", color="#52514e", fontsize=9)
    ax.set_title(f"B5: (1+1)-ES over reflex gains, roughness {a.roughness} (one run: a lead, not a result)",
                 color="#0b0b0b", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "go1_es.png", dpi=150, facecolor="#fcfcfb")
    print("final gains:", to_kwargs(x))
    print("wrote", (OUT / "go1_es.csv").relative_to(ROOT), "and go1_es.png")


if __name__ == "__main__":
    main()
