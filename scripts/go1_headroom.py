#!/usr/bin/env python3
"""Is there anything for a Go1 gait learner to find? Headroom check before any learner runs (D71).

The learner nudges three persistent parameters (clearance, posture-reflex gain, body height) on a fixed grid of
6 x 6 x 5 = 180 reachable settings. If no FIXED setting beats the stumble-reflex gait B3 (the default setting) on
terrain it was not chosen on, a learner that finds the best setting gains nothing either. So, at each roughness:

  1. select:  score every setting and B3 on the selection terrains; take the setting with the highest mean return.
  2. confirm: score that one setting and B3 on separate confirmation terrains; paired by terrain, Student-t 95%.
     Headroom exists at that roughness only if the interval excludes 0 on the positive side.

The return is the evaluation return the harness uses (sum of raw rewards, action 0 = keep the setting). The task
settings come from the experiment config (default go1_gait_v3), so the episodes are the learner's evaluation
episodes with the parameters held fixed.

    python3 scripts/go1_headroom.py --levels 0.06 0.08 0.10 --jobs 2
Output: results/go1_headroom/<config>_r<level>_select.csv, ..._confirm.csv and a summary on stdout.
"""
import argparse
import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tinyrl import analysis, sweep  # noqa: E402
from tinyrl.envs import make_env  # noqa: E402
from tinyrl.envs.go1 import INC_PARAMS  # noqa: E402

OUT = ROOT / "results" / "go1_headroom"


def grid():
    axes = [np.round(np.arange(lo, hi + step / 2, step), 4) for _, lo, hi, step, _ in INC_PARAMS]
    names = [n for n, *_ in INC_PARAMS]
    return [dict(zip(names, map(float, v))) for v in itertools.product(*axes)]


DEFAULT = {n: float(init) for n, _, _, _, init in INC_PARAMS}      # = B3 when the stumble reflex is on


def returns(args):
    """-> one return per terrain seed for one fixed setting."""
    task, setting, seeds = args
    env = make_env({**task, "init_params": setting})
    out = []
    for s in seeds:
        env.reset(seed=s)
        total, done = 0.0, False
        while not done:
            _, r, term, trunc = env.step(0)
            total += r
            done = term or trunc
        out.append(total)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="go1_gait_v3")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.06, 0.08, 0.10])
    ap.add_argument("--select", type=int, nargs=2, default=[30000, 20], metavar=("FIRST_SEED", "N"))
    ap.add_argument("--confirm", type=int, nargs=2, default=[40000, 30], metavar=("FIRST_SEED", "N"))
    ap.add_argument("--settings", type=int, help="only the first N settings (for tests)")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    sel = list(range(a.select[0], a.select[0] + a.select[1]))
    con = list(range(a.confirm[0], a.confirm[0] + a.confirm[1]))
    if set(sel) & set(con):
        sys.exit("selection and confirmation terrains must differ")
    task = {**sweep.load(a.config)["task"]}
    settings = grid()[: a.settings] if a.settings else grid()
    if DEFAULT not in settings:
        settings.append(DEFAULT)
    OUT.mkdir(parents=True, exist_ok=True)
    print("| roughness | best setting (on selection terrains) | its mean − B3 there | "
          "confirmation: best − B3, 95% CI (terrains) | headroom |")
    print("|---|---|---|---|---|")
    for level in a.levels:
        t = {**task, "roughness": level}
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            R = list(pool.map(returns, [(t, s, sel) for s in settings]))
        means = [float(np.mean(r)) for r in R]
        tag = f"{a.config}_r{level:g}"
        with open(OUT / f"{tag}_select.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([*DEFAULT, "mean_return", "returns"])
            for s, r, m in zip(settings, R, means):
                w.writerow([*s.values(), round(m, 4), " ".join(f"{x:.4f}" for x in r)])
        best = settings[int(np.argmax(means))]
        b3_sel = means[settings.index(DEFAULT)]
        with ProcessPoolExecutor(max_workers=2) as pool:
            rb, r3 = pool.map(returns, [(t, best, con), (t, DEFAULT, con)])
        d = np.array(rb) - np.array(r3)
        m, h = float(d.mean()), analysis.t95(d)
        with open(OUT / f"{tag}_confirm.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "best", "B3"])
            w.writerows(zip(con, rb, r3))
        name = ", ".join(f"{k} {v:g}" for k, v in best.items())
        print(f"| {level:g} | {name} | {max(means) - b3_sel:+.2f} | {m:+.2f} [{m - h:+.2f}, {m + h:+.2f}] "
              f"({len(con)}) | {'yes' if m - h > 0 else 'no'} |")


if __name__ == "__main__":
    main()
