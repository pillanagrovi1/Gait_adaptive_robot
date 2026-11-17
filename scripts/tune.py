#!/usr/bin/env python3
"""Equal-budget hyperparameter tuning, on seeds that are NOT used for reporting.

Why (docs/research_review.md; Patterson et al. 2024, Jordan et al. 2020):
hand-picked hyperparameters favour whichever learner got more attention. Every
learner gets the same number K of sampled configurations, scored on separate
tuning seeds; the winner is then run on the reporting seeds by run_sweep.py.

The experiment config gets a "tune" section:

  "tune": {
    "k": 20,                              # configurations per learner (same for all)
    "seeds": [1000, 1001, 1002],          # tuning seeds, disjoint from "seeds"
    "score": "auc",                       # "auc" (mean eval over the run) or "final"
    "run": {"steps": 100000},             # optional cheaper budget for tuning
    "space": {
      "table":  {"alpha": {"log_uniform": [0.05, 1.0]}},
      "net":    {"alpha": {"log_uniform": [0.003, 0.1]}, "target_period": {"choice": [50, 100, 500]}}
    }
  }

Each learner is tuned once, at the MIDDLE value of its memory sweep (one
reference budget), and the chosen values are used for the whole sweep. Say so
in the methods; the alternative (re-tune at every budget) costs K x sweep size.

    python3 scripts/tune.py pendulum_main            # writes configs/pendulum_main_tuned.json
    python3 scripts/tune.py pendulum_main --only net
Output: configs/<name>_tuned.json and results/tuning/<name>.csv (every trial).
"""
import argparse
import copy
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tinyrl import harness, sweep  # noqa: E402


def sample(space: dict, rng) -> dict:
    out = {}
    for k, spec in space.items():
        if "log_uniform" in spec:
            lo, hi = spec["log_uniform"]
            out[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif "uniform" in spec:
            out[k] = float(rng.uniform(*spec["uniform"]))
        elif "choice" in spec:
            out[k] = spec["choice"][int(rng.integers(len(spec["choice"])))]
        else:
            raise ValueError(f"bad search spec for {k}: {spec}")
    return out


def score_run(args):
    task, learner, seed, run_cfg, score = args
    res = harness.run(task, learner, seed, run_cfg)
    ev = [e["eval_return"] for e in res["evals"]] or \
         [r["eval_return"] for r in res["rows"] if r["eval_return"] != ""]
    if not ev:
        return float("nan")
    return float(np.mean(ev)) if score == "auc" else float(np.mean(ev[-3:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--only")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    exp = sweep.load(a.experiment)
    t = exp.get("tune")
    if not t:
        sys.exit("this experiment has no 'tune' section")
    if set(t["seeds"]) & set(exp["seeds"]):
        sys.exit("tuning seeds must differ from reporting seeds")
    rng = np.random.default_rng(12345)
    run_cfg = {**exp.get("run", {}), **t.get("run", {})}
    tuned = copy.deepcopy(exp)
    tuned["name"] = exp["name"] + "_tuned"
    tuned["_note"] = (exp.get("_note", "") + f" Hyperparameters tuned by scripts/tune.py: K={t['k']} per learner, "
                      f"tuning seeds {t['seeds']}, score {t.get('score', 'auc')}.").strip()
    trials = []
    for L in tuned["learners"]:
        if L["base"]["kind"].startswith("baseline") or L["label"] not in t["space"]:
            continue
        if a.only and L["label"] != a.only:
            continue
        mid = {k: v[len(v) // 2] for k, v in L.get("sweep", {}).items()}
        task = {**exp["task"], **L.get("task", {})}
        for k, v in exp.get("task_sweep", {}).items():
            task[k] = v[0]
        cands = [sample(t["space"][L["label"]], rng) for _ in range(t["k"])]
        jobs = [(task, {**L["base"], **mid, **c}, s, run_cfg, t.get("score", "auc"))
                for c in cands for s in t["seeds"]]
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            scores = list(pool.map(score_run, jobs))
        S = np.array(scores).reshape(len(cands), len(t["seeds"]))
        means = np.nanmean(S, 1)
        best = int(np.nanargmax(means))
        for c, m, row in zip(cands, means, S):
            trials.append({"label": L["label"], **{f"hp_{k}": v for k, v in c.items()},
                           "score_mean": m, "scores": json.dumps(row.round(3).tolist())})
        print(f"{L['label']:>12}: best {cands[best]}  score {means[best]:.2f}  "
              f"(median of K={t['k']}: {np.nanmedian(means):.2f})")
        L["base"].update(cands[best])
    out = ROOT / "configs" / f"{exp['name']}_tuned.json"
    tuned.pop("tune", None)
    out.write_text(json.dumps(tuned, indent=2))
    tdir = ROOT / "results" / "tuning"
    tdir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in trials for k in r})
    with open(tdir / f"{exp['name']}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(trials)
    print("wrote", out.relative_to(ROOT), "and", (tdir / f"{exp['name']}.csv").relative_to(ROOT))
    print("next: python3 scripts/run_sweep.py", out.stem)


if __name__ == "__main__":
    main()
