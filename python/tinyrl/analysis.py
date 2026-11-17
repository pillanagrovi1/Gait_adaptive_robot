"""Turn raw run logs into one summary row per configuration.

Definitions (write these into your methods section; change them deliberately):

  final performance   per seed: mean greedy eval return over the last K eval
                      points (K = 3) before any shift. Reported: mean over seeds
                      and a 95% t-interval.
  learning speed      per seed: mean eval return over ALL pre-shift eval points
                      (area under the learning curve / number of points). Higher
                      = learned sooner. If the experiment sets metrics.threshold,
                      also the first episode whose eval return >= threshold.
  adaptation          only if run.shift_at is set. Per seed:
                        drop     = first post-shift eval - final pre-shift performance
                        recovery = episodes after the shift until the eval return
                                   gets 90% of the way from its post-shift minimum
                                   to its post-shift final level (last K points)
  peak SRAM           from results/m4/<experiment>.csv (QEMU, Cortex-M4):
                      sizeof(agent) + arena + learner stack. If a configuration
                      was not measured, `peak_source` says "host-estimate" and the
                      value is arena + 220 B, which plots draw hollow.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import sweep

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
K_LAST = 3


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def load_run(exp_name: str, run_id: str):
    d = RAW / exp_name / run_id
    if not d.exists():
        return None, None
    meta = json.loads((d / "meta.json").read_text())
    frames = []
    for f in sorted(d.glob("seed*.csv")):
        df = pd.read_csv(f)
        df["seed"] = int(f.stem[4:])
        frames.append(df)
    return (pd.concat(frames, ignore_index=True) if frames else None), meta


def eval_matrix(df: pd.DataFrame):
    """-> (episodes array, seeds x points array) of greedy eval returns."""
    ev = df.dropna(subset=["eval_return"])
    piv = ev.pivot_table(index="seed", columns="episode", values="eval_return")
    return piv.columns.to_numpy() + 1, piv.to_numpy()


def load_m4(exp_name: str) -> dict:
    p = ROOT / "results" / "m4" / f"{exp_name}.csv"
    if not p.exists():
        return {}
    with open(p) as f:
        return {r["run_id"]: r for r in csv.DictReader(f)}


def _per_seed_metrics(eps, M, shift_at, threshold):
    out = []
    pre = eps <= (shift_at if shift_at is not None else np.inf)
    for row in M:
        m = {}
        r_pre = row[pre]
        m["final"] = np.mean(r_pre[-K_LAST:])
        m["auc"] = np.mean(r_pre)
        if threshold is not None:
            hit = np.flatnonzero(r_pre >= threshold)
            m["episodes_to_threshold"] = float(eps[pre][hit[0]]) if len(hit) else np.nan
        if shift_at is not None and (~pre).sum() >= K_LAST + 1:
            post, post_eps = row[~pre], eps[~pre]
            m["drop"] = post[0] - m["final"]
            lo, fin = post.min(), np.mean(post[-K_LAST:])
            m["post_final"] = fin
            target = lo + 0.9 * (fin - lo)
            imin = int(np.argmin(post))
            hit = np.flatnonzero(post[imin:] >= target)
            m["recovery_episodes"] = float(post_eps[imin + hit[0]] - shift_at) if len(hit) else np.nan
        out.append(m)
    return pd.DataFrame(out)


def summarize(exp_name_or_path) -> pd.DataFrame:
    exp = sweep.load(exp_name_or_path)
    specs = sweep.expand(exp)
    m4 = load_m4(exp["name"])
    shift_at = exp.get("run", {}).get("shift_at")
    threshold = exp.get("metrics", {}).get("threshold")
    rows = []
    for s in specs:
        df, meta = load_run(exp["name"], s["run_id"])
        if df is None:
            continue
        eps, M = eval_matrix(df)
        ps = _per_seed_metrics(eps, M, shift_at, threshold)
        row = {"run_id": s["run_id"], "label": s["label"], "kind": s["learner"]["kind"],
               **{f"sweep_{k}": v for k, v in s["sweep_values"].items()},
               "n_seeds": len(ps), "params": meta["params"], "arena_bytes": meta["arena_bytes"],
               "obs_dim": meta["obs_dim"], "cfg_id": meta["cfg_id"]}
        for col in ps.columns:
            row[f"{col}_mean"] = float(ps[col].mean())
            row[f"{col}_ci95"] = ci95(ps[col])
        mm = m4.get(s["run_id"])
        if mm and mm.get("peak_bytes") and mm.get("cfg_id") == meta["cfg_id"]:
            row["peak_bytes"] = int(mm["peak_bytes"])
            row["peak_source"] = "m4-qemu"
            row["fits_f446"] = int(mm["fits"])
            for k in ("insns_learn_step", "insns_greedy_step", "stack_learner", "agent_bytes"):
                row[k] = float(mm[k]) if mm.get(k) else np.nan
        else:
            row["peak_bytes"] = meta["arena_bytes"] + 220
            row["peak_source"] = "host-estimate"
            row["fits_f446"] = 0 if (mm and mm.get("fits") == "0") else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
