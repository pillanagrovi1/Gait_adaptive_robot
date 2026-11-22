"""Turn raw run logs into one summary row per configuration.

Definitions (write these into your methods section; change them deliberately).
Revised 17 Nov 2026 after the evaluation-methodology review (docs/research_review.md,
Patterson et al. 2024, Colas et al. 2019, Agarwal et al. 2021).

  x axis              environment steps in step-budget mode (recommended), else episodes.
  final performance   per seed: mean greedy eval return over the last K = 3 eval points
                      before any shift.
  learning speed      per seed: mean eval return over ALL pre-shift eval points
                      (area under the learning curve / number of points). Higher
                      = learned sooner. If the experiment sets metrics.threshold,
                      also the first x at which the eval return >= threshold.
  online return       per seed: mean TRAINING (exploratory) return over the last 20% of
                      pre-shift episodes. This is what a robot experiences while it learns.
  intervals           per configuration, n < 20 seeds: Student-t 95% interval (check the seed dots for
                      bimodality before trusting it); n >= 20: percentile bootstrap.
                      `ci_method` says which. Compare two learners with paired_diff()
                      (Student-t at every n, D52), not by checking whether their intervals overlap.
  adaptation          only if a shift is configured. Per seed:
                        P         = final pre-shift performance
                        target    = P - 0.1 * |P|          (fixed, anchored to P, not to the noisy minimum)
                        recovery  = x after the shift at which the 3-point moving average of the
                                    eval return first reaches the target; missing if never
                                    (censored: counted in recovered_frac, never dropped silently)
                        post_auc  = mean eval return over all post-shift points (fixed window)
                        drop      = first post-shift eval - P
                      With a switch back (A -> B -> A, D57) post_auc and recovery use the B
                      window only, and:
                        back      = return on A right after the switch back, before more learning
                        a2_auc    = mean eval return over the second A window
                        forgetting        = paired_diff(e, run, run, "back", metric_b="final")
                        forward transfer  = paired_diff(e, run, run, "post_auc", exp_b=scratch, metric_b="auc")
  underflow_frac      share of learning-update products that were non-zero before
                      rounding and exactly zero after (Q16.16 underflow), from the host run.
  saturation_frac     share of those update terms whose product or weight write was clipped
                      to the int32 range (D55), from the host run.
  peak SRAM           from results/m4/<experiment>.csv (QEMU, Cortex-M4):
                      sizeof(agent) + arena + learner stack. If a configuration
                      was not measured, `peak_source` says "host-estimate" and the
                      value is arena + 220 B, which plots draw hollow.
  flash_bytes         text + data of that benchmark firmware (learner, harness and startup),
                      not of the learner alone (D56).
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
BOOT_MIN_N = 20


def ci95(x):
    """Half-width of a 95% interval: t for n < 20, percentile bootstrap otherwise."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    if len(x) >= BOOT_MIN_N:
        rng = np.random.default_rng(0)
        means = rng.choice(x, size=(10_000, len(x)), replace=True).mean(1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return float((hi - lo) / 2)
    return t95(x)


def t95(x):
    """Half-width of a Student-t 95% interval, at any n."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def ci_method(n: int) -> str:
    return "bootstrap" if n >= BOOT_MIN_N else "t"


def load_run(exp_name: str, run_id: str):
    """-> (episode rows, eval rows or None, meta, per-seed metas)."""
    d = RAW / exp_name / run_id
    if not d.exists():
        return None, None, None, {}
    meta = json.loads((d / "meta.json").read_text())
    frames, evframes, metas = [], [], {}
    for f in sorted(d.glob("seed*.csv")):
        if f.stem.endswith("_eval"):
            continue
        seed = int(f.stem[4:])
        df = pd.read_csv(f)
        df["seed"] = seed
        frames.append(df)
        ef = d / f"seed{seed}_eval.csv"
        if ef.exists():
            e = pd.read_csv(ef)
            e["seed"] = seed
            evframes.append(e)
        mf = d / f"seed{seed}_meta.json"
        if mf.exists():
            metas[seed] = json.loads(mf.read_text())
    if not frames:
        return None, None, meta, metas
    df = pd.concat(frames, ignore_index=True)
    ev = pd.concat(evframes, ignore_index=True) if evframes else None
    return df, ev, meta, metas


def eval_matrix(df: pd.DataFrame, ev: pd.DataFrame | None = None):
    """-> (x array, seeds x points array, x label) of greedy eval returns."""
    if ev is not None:
        piv = ev.pivot_table(index="seed", columns="step", values="eval_return")
        return piv.columns.to_numpy(), piv.to_numpy(), "environment steps"
    e = df.dropna(subset=["eval_return"])
    e = e[e["eval_return"] != ""]
    piv = e.pivot_table(index="seed", columns="episode", values="eval_return")
    return piv.columns.to_numpy() + 1, piv.to_numpy(), "training episode"


def load_m4(exp_name: str) -> dict:
    p = ROOT / "results" / "m4" / f"{exp_name}.csv"
    if not p.exists():
        return {}
    with open(p) as f:
        return {r["run_id"]: r for r in csv.DictReader(f)}


def _shift_point(exp: dict):
    """Where the physics changes; None if never, or if from the start (scratch references)."""
    r = exp.get("run", {})
    x = r.get("shift_at_step") if r.get("steps") is not None else r.get("shift_at")
    return None if not x else x


def _per_seed_metrics(xs, M, shift, threshold, df, back=None, metas=None):
    out = []
    pre = xs <= (shift if shift is not None else np.inf)
    in_b = ~pre & (xs <= (back if back is not None else np.inf))
    xcol = "env_steps" if "env_steps" in df and df.attrs.get("step_mode") else "episode"
    for k, row in enumerate(M):
        m = {}
        r_pre = row[pre]
        m["final"] = np.nanmean(r_pre[-K_LAST:])
        m["auc"] = np.nanmean(r_pre)
        if threshold is not None:
            hit = np.flatnonzero(r_pre >= threshold)
            m["x_to_threshold"] = float(xs[pre][hit[0]]) if len(hit) else np.nan
        seed = df["seed"].unique()[k] if k < df["seed"].nunique() else None
        if seed is not None:
            tr = df[df["seed"] == seed]
            if shift is not None:
                tr = tr[tr[xcol] <= shift]
            tail = tr["train_return"].to_numpy()
            if len(tail):
                m["online"] = float(np.mean(tail[-max(1, len(tail) // 5):]))
        if shift is not None and back is not None:
            m["a2_auc"] = float(np.nanmean(row[xs > back]))
            v = (metas or {}).get(seed, {}).get("eval_after_shift_back")
            m["back"] = np.nan if v is None else v
        if shift is not None and in_b.sum() >= K_LAST:
            post, post_x = row[in_b], xs[in_b]
            P = m["final"]
            target = P - 0.1 * abs(P)
            smooth = np.convolve(post, np.ones(3) / 3, mode="valid")
            m["drop"] = post[0] - P
            m["post_auc"] = float(np.nanmean(post))
            hit = np.flatnonzero(smooth >= target)
            m["recovered"] = float(len(hit) > 0)
            m["recovery"] = float(post_x[hit[0] + 2] - shift) if len(hit) else np.nan
        out.append(m)
    return pd.DataFrame(out)


def summarize(exp_name_or_path) -> pd.DataFrame:
    exp = sweep.load(exp_name_or_path)
    specs = sweep.expand(exp)
    m4 = load_m4(exp["name"])
    shift = _shift_point(exp)
    back = exp.get("run", {}).get("shift_back_at_step")
    threshold = exp.get("metrics", {}).get("threshold")
    rows = []
    for s in specs:
        df, ev, meta, metas = load_run(exp["name"], s["run_id"])
        if df is None:
            continue
        df.attrs["step_mode"] = ev is not None
        seeds = sorted(df["seed"].unique())
        df = df.set_index("seed").loc[seeds].reset_index()
        xs, M, _ = eval_matrix(df, ev)
        ps = _per_seed_metrics(xs, M, shift, threshold, df, back, metas)
        row = {"run_id": s["run_id"], "label": s["label"], "kind": s["learner"]["kind"],
               **{f"sweep_{k}": v for k, v in s["sweep_values"].items()},
               "n_seeds": len(ps), "ci_method": ci_method(len(ps)),
               "round_mode": int(s["learner"].get("round_mode", 0)),
               "params": meta["params"], "arena_bytes": meta["arena_bytes"],
               "obs_dim": meta["obs_dim"], "cfg_id": meta["cfg_id"]}
        for col in ps.columns:
            row[f"{col}_mean"] = float(ps[col].mean())
            row[f"{col}_ci95"] = ci95(ps[col]) if col != "recovered" else float("nan")
        if metas:
            uf = [m.get("underflow_frac") for m in metas.values() if m.get("underflow_frac") is not None]
            row["underflow_frac"] = float(np.mean(uf)) if uf else np.nan
            sat = [m.get("saturation_frac") for m in metas.values() if m.get("saturation_frac") is not None]
            row["saturation_frac"] = float(np.mean(sat)) if sat else np.nan
            row["updates_per_step"] = float(np.mean([m["updates"] / max(1, m["env_steps"])
                                                     for m in metas.values() if "env_steps" in m]))
        mm = m4.get(s["run_id"])
        if mm and mm.get("peak_bytes") and mm.get("cfg_id") == meta["cfg_id"]:
            row["peak_bytes"] = int(mm["peak_bytes"])
            row["peak_source"] = "m4-qemu"
            row["fits_f446"] = int(mm["fits"])
            for k in ("insns_learn_step", "insns_greedy_step", "stack_learner", "agent_bytes"):
                row[k] = float(mm[k]) if mm.get(k) else np.nan
            text, data = mm.get("flash_text"), mm.get("ram_data")
            row["flash_bytes"] = int(text) + int(data) if text and data else np.nan
        else:
            row["peak_bytes"] = meta["arena_bytes"] + 220
            row["peak_source"] = "host-estimate"
            row["fits_f446"] = 0 if (mm and mm.get("fits") == "0") else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def per_seed(exp_name_or_path, run_id: str, metric: str = "final") -> pd.Series:
    """Per-seed values of one metric for one configuration, indexed by seed."""
    exp = sweep.load(exp_name_or_path)
    df, ev, _, metas = load_run(exp["name"], run_id)
    if df is None:
        raise FileNotFoundError(run_id)
    df.attrs["step_mode"] = ev is not None
    seeds = sorted(df["seed"].unique())
    df = df.set_index("seed").loc[seeds].reset_index()
    xs, M, _ = eval_matrix(df, ev)
    ps = _per_seed_metrics(xs, M, _shift_point(exp), exp.get("metrics", {}).get("threshold"), df,
                           exp.get("run", {}).get("shift_back_at_step"), metas)
    ps.index = seeds
    return ps[metric]


def paired_diff(exp_name_or_path, run_a: str, run_b: str, metric: str = "final",
                exp_b=None, metric_b: str | None = None) -> dict:
    """A - B on the seeds both share (training seeds are common random numbers:
    seed k gives both learners the same episode start states). Interval on the
    per-seed DIFFERENCES, as recommended by Patterson et al. 2024.
    B may come from another experiment (exp_b) or be another metric (metric_b)."""
    a = per_seed(exp_name_or_path, run_a, metric)
    b = per_seed(exp_b or exp_name_or_path, run_b, metric_b or metric)
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy(float)
    return {"metric": metric, "metric_b": metric_b or metric, "a": run_a, "b": run_b, "n": int(len(d)),
            "mean_diff": float(np.nanmean(d)) if len(d) else float("nan"),
            "ci95": t95(d), "ci_method": "t",
            "a_better_frac": float(np.mean(d > 0)) if len(d) else float("nan")}
