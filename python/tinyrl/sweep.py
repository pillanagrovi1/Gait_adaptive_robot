"""Expand an experiment config (configs/*.json) into concrete runs.

Config schema:
{
  "name": "pendulum_main",
  "task": {"name": "pendulum", "obs_mode": "angle"},
  "task_sweep": {"obs_mode": ["angle", "trig"]},          # optional
  "run": {"episodes": 1500, ...},                          # harness RUN_DEFAULTS overrides
  "seeds": [0, 1, 2, 3, 4],
  "learners": [
    {"label": "table", "base": {"kind": "tabular", ...},
     "sweep": {"bins": [7, 11, 15, 21]},                   # cartesian product
     "task": {"obs_mode": "angle"}}                        # optional per-learner task override
  ]
}
Both the host harness (scripts/run_sweep.py) and the M4 measurement
(scripts/measure_m4.py) call expand(), so they always agree on what runs exist.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path) -> dict:
    p = Path(path)
    if not p.exists():
        p = ROOT / "configs" / f"{path}.json"
    with open(p) as f:
        return json.load(f)


def _grid(d: dict):
    if not d:
        yield {}
        return
    keys = list(d)
    for vals in itertools.product(*(d[k] for k in keys)):
        yield dict(zip(keys, vals))


def _tag(d: dict) -> str:
    return "-".join(f"{k}{v}" for k, v in d.items())


def expand(exp: dict) -> list[dict]:
    runs = []
    for L in exp["learners"]:
        for tvar in _grid(exp.get("task_sweep", {})):
            task = {**exp["task"], **L.get("task", {}), **tvar}
            for lvar in _grid(L.get("sweep", {})):
                learner = {**L["base"], **lvar}
                parts = [L["label"]] + [p for p in (_tag(tvar), _tag(lvar)) if p]
                runs.append({
                    "run_id": "_".join(parts),
                    "label": L["label"],
                    "task": task,
                    "learner": learner,
                    "run": {**exp.get("run", {}), **L.get("run", {})},
                    "sweep_values": {**tvar, **lvar},
                })
    ids = [r["run_id"] for r in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run ids: make learner labels unique")
    return runs


if __name__ == "__main__":
    import sys
    exp = load(sys.argv[1])
    for r in expand(exp):
        print(r["run_id"], json.dumps(r["learner"]), json.dumps(r["task"]))
