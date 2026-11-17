#!/usr/bin/env python3
"""Run every (configuration, seed) of an experiment, in parallel, resumable.

    python3 scripts/run_sweep.py pendulum_main            # configs/pendulum_main.json
    python3 scripts/run_sweep.py pendulum_main --jobs 6
    python3 scripts/run_sweep.py pendulum_main --only table   # one learner label
    python3 scripts/run_sweep.py pendulum_main --dry-run

A (run, seed) whose CSV already exists is skipped, so you can stop and restart.
Delete results/raw/<experiment>/ to start over.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tinyrl import harness, sweep  # noqa: E402


def _one(args):
    exp_name, spec, seed = args
    out = harness.RESULTS / exp_name / spec["run_id"]
    t = time.time()
    res = harness.run(spec["task"], spec["learner"], seed, spec["run"], out_dir=out)
    evals = [r["eval_return"] for r in res["rows"] if r["eval_return"] != ""]
    return spec["run_id"], seed, evals[-1] if evals else None, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--only", help="run only this learner label")
    ap.add_argument("--seeds", type=int, nargs="*", help="override seed list")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    exp = sweep.load(a.experiment)
    specs = sweep.expand(exp)
    if a.only:
        specs = [s for s in specs if s["label"] == a.only]
    seeds = a.seeds if a.seeds is not None else exp["seeds"]
    todo = []
    for s in specs:
        for seed in seeds:
            if not (harness.RESULTS / exp["name"] / s["run_id"] / f"seed{seed}.csv").exists():
                todo.append((exp["name"], s, seed))
    print(f"{exp['name']}: {len(specs)} configs x {len(seeds)} seeds, {len(todo)} to run, {a.jobs} jobs")
    if a.dry_run:
        for _, s, seed in todo:
            print("  ", s["run_id"], "seed", seed)
        return
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futs = [pool.submit(_one, t) for t in todo]
        for k, f in enumerate(as_completed(futs), 1):
            rid, seed, last, dt = f.result()
            last_s = f"{last:9.2f}" if last is not None else "      n/a"
            print(f"[{k}/{len(todo)}] {rid:40s} seed {seed}  last eval {last_s}  ({dt:5.1f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
