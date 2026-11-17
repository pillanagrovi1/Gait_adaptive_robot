#!/usr/bin/env python3
"""How hard is the Go1 terrain for the OPEN-LOOP trot? Run this before any
learning experiment on the Go1 (Stage 7).

Why: a learner can only beat the open-loop trot where the open-loop trot
fails. D2-GMBC (Rahme et al. 2020) evaluated where open loop fell in all
1000 trials; ETG-RL (Shi et al. 2022) found no gain from structure on easy
terrain where the robot rarely fell. Lee et al. (2020) train on terrain whose
traversability lies in [0.5, 0.9]. See docs/research_review.md.

For each roughness level, runs N episodes of the fixed nominal trot (action 0)
and reports the success rate (no fall AND at least --min-distance walked:
at high roughness the trot tends to get stuck rather than fall) and the mean
distance walked. Levels whose success rate lies inside --band are marked as
candidate training/evaluation difficulties.

    python3 scripts/go1_difficulty.py
    python3 scripts/go1_difficulty.py --levels 0.04 0.06 0.08 0.10 0.12 --episodes 30 --phase-lock
    python3 scripts/go1_difficulty.py --friction 0.4 1.0
Output: results/figures/go1_difficulty.csv and go1_difficulty.png
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tinyrl.envs.go1 import Go1Gait  # noqa: E402

OUT = ROOT / "results" / "figures"


def run_level(args):
    rough, n, phase_lock, friction, seed0, min_dist = args
    env = Go1Gait(roughness=rough, phase_lock=phase_lock,
                  friction_range=tuple(friction) if friction else None,
                  obs_mode="imu4p" if phase_lock else "imu4")
    falls, fails, dist = 0, 0, []
    for k in range(n):
        env.reset(seed=seed0 + k)
        x0 = float(env.data.qpos[0])
        done = term = False
        while not done:
            _, _, term, trunc = env.step(0)
            done = term or trunc
        falls += int(term)
        dist.append(float(env.data.qpos[0]) - x0)
        fails += int(term or dist[-1] < min_dist)       # fell, or got stuck
    return {"roughness": rough, "episodes": n, "success_rate": 1 - fails / n, "fall_rate": falls / n,
            "mean_distance_m": float(np.mean(dist)), "std_distance_m": float(np.std(dist))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", default=[0.03, 0.05, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--phase-lock", action="store_true")
    ap.add_argument("--friction", type=float, nargs=2, metavar=("LO", "HI"))
    ap.add_argument("--band", type=float, nargs=2, default=[0.4, 0.8],
                    help="success-rate band that leaves room for learning (default 0.4-0.8)")
    ap.add_argument("--min-distance", type=float, default=2.0,
                    help="metres an episode must cover to count as a success (flat ground: ~4 m in 10 s)")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()

    jobs = [(r, a.episodes, a.phase_lock, a.friction, 50_000, a.min_distance) for r in a.levels]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(run_level, jobs))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "go1_difficulty.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lo, hi = a.band
    print(f"{'roughness':>9}  {'success':>7}  {'falls':>5}  {'distance (m)':>12}")
    for r in rows:
        mark = "  <- candidate" if lo <= r["success_rate"] <= hi else ""
        print(f"{r['roughness']:9.3f}  {r['success_rate']:7.2f}  {r['fall_rate']:5.2f}  {r['mean_distance_m']:8.2f} ± {r['std_distance_m']:.2f}{mark}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [r["roughness"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhspan(lo, hi, color="#2a78d6", alpha=0.08, linewidth=0)
    ax.text(xs[0], hi, " room to learn", color="#52514e", fontsize=7, va="bottom")
    ax.plot(xs, [r["success_rate"] for r in rows], marker="o", color="#2a78d6", linewidth=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("terrain roughness (height-field scale, m)", color="#52514e", fontsize=9)
    ax.set_ylabel(f"open-loop success rate (n={a.episodes})", color="#52514e", fontsize=9)
    ax.set_title("Go1 open-loop trot: where is there something to learn?", color="#0b0b0b", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "go1_difficulty.png", dpi=150, facecolor="#fcfcfb")
    print("wrote", (OUT / "go1_difficulty.csv").relative_to(ROOT), "and go1_difficulty.png")


if __name__ == "__main__":
    main()
