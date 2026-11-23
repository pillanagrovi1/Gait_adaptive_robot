#!/usr/bin/env python3
"""Print a markdown results table for one experiment, ready for docs/results.md.

Reads results/figures/<experiment>_summary.csv (written by make_plots.py) and adds the
provenance the results log requires: git commit, config file, command, number of seeds.
It exits 1, printing no table, if that provenance does not hold: no raw files, seed counts that
differ between raw files, summary and config, a raw run missing from the summary, runs made with
another config, host != M4, more than one commit, or code with uncommitted changes (+dirty).

    python3 scripts/results_table.py stage2_pendulum_table_tuned
    python3 scripts/results_table.py replay_ratio --command "<the command that was really run>"
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tinyrl import cbind, sweep  # noqa: E402


def refuse(*problems):
    print("results_table: no table printed.", *problems, sep="\n  ", file=sys.stderr)
    raise SystemExit(1)


def fmt(m, ci):
    if pd.isna(m):
        return "—"
    digits = 1 if abs(m) < 100 else 0       # Go1 returns are ~20: whole numbers would hide the interval
    return f"{m:.{digits}f} ± {ci:.{digits}f}" if not pd.isna(ci) else f"{m:.{digits}f}"


def hash_load(learner: dict, obs_dim: int) -> float:
    """Tile-coding cells per hashed feature (D70); above 1, different cells must share features."""
    p = {**cbind.LEARNER_DEFAULTS, **learner}
    return p["tilings"] * (p["tiles"] + 1) ** int(obs_dim) / p["features"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("--command", help="the command that was really run; required if only some arms were run")
    a = ap.parse_args()
    name = a.experiment
    p = ROOT / "results" / "figures" / f"{name}_summary.csv"
    d = pd.read_csv(p)
    raw = ROOT / "results" / "raw" / name
    metas = sorted(raw.glob("*/seed*_meta.json"))
    if not metas:
        refuse(f"no raw files in results/raw/{name}/. From the repo root, extract them with:",
               f"mkdir -p results/raw && tar -xzf results/raw_archive/{name}.tar.gz -C results/raw")
    counts = Counter(mf.parent.name for mf in metas)
    loaded = [(mf.parent.name, json.loads(mf.read_text())) for mf in metas]
    commits = {m.get("git_commit") for _, m in loaded} - {None, "", "?", "unknown"}
    exp = sweep.load(ROOT / "configs" / f"{name}.json")
    n_config = len(exp["seeds"])

    problems = []
    for run, n in zip(d["run_id"], d["n_seeds"]):
        if counts[run] != n:
            problems.append(f"{run}: {counts[run]} seed metas, summary says {n}")
        if n != n_config:
            problems.append(f"{run}: summary has {n} seeds, config says {n_config}")
    in_raw = {p.name for p in raw.iterdir() if p.is_dir()}
    for run in sorted(in_raw - set(d["run_id"])):
        problems.append(f"{run}: in results/raw/ but not in the summary (run make_plots.py again)")
    if len(commits) > 1:
        problems.append(f"more than one commit: {', '.join(sorted(commits))}")
    if any(c.endswith("+dirty") for c in commits):
        problems.append(f"runs from uncommitted code (+dirty): {', '.join(sorted(commits))}")
    specs = {s["run_id"]: s for s in sweep.expand(exp)}
    other = sorted({run for run, m in loaded if run in specs and not sweep.ran_as(specs[run], m)})
    if other:
        problems.append(f"runs differ from configs/{name}.json: {', '.join(other)}")
    if "host_match" in d:
        for run in d.loc[d["host_match"] == 0, "run_id"]:
            problems.append(f"{run}: host != M4, so its measurement cannot be used")
    not_run = set(specs) - set(d["run_id"])
    if not_run and not a.command:
        problems.append(f"{len(not_run)} of the config's runs are not in the summary: "
                        'give the command that was really run with --command "..."')
    if problems:
        refuse(*problems)

    commit = f"commit `{commits.pop()}`" if commits else "commit not recorded (runs predate commit tracking)"
    command = a.command or f"bash scripts/run_experiment.sh {name.removesuffix('_tuned')}"
    print(f"**{name}** · {commit} · config `configs/{name}.json` · "
          f"command `{command}` · "
          f"seeds per config: {int(d['n_seeds'].min())}–{int(d['n_seeds'].max())} · intervals: {d['ci_method'].iloc[0]} 95%\n")
    if any(L.get("sweep_zip") for L in exp["learners"] if L["base"].get("kind") == "linear"):   # D70
        d["hash_load"] = [hash_load(specs[run]["learner"], dim) if kind == "linear" else float("nan")
                          for run, kind, dim in zip(d["run_id"], d["kind"], d["obs_dim"])]
    extra = [(col, head, show) for col, head, show in (
        ("flash_bytes", "flash, bench firmware with all three learners (B)", lambda r, v: f"{v:.0f}"),  # D56
        ("saturation_frac", "update saturation", lambda r, v: f"{100 * v:.2g}%"),                      # D55
        ("post_auc_mean", "post-shift AUC (B window)", lambda r, v: fmt(v, r.get("post_auc_ci95"))),
        ("back_mean", "return on A right after switch back", lambda r, v: fmt(v, r.get("back_ci95"))),
        ("a2_auc_mean", "AUC, second A window", lambda r, v: fmt(v, r.get("a2_auc_ci95"))),
        ("recovered_mean", "recovered", lambda r, v: f"{100 * v:.0f}%"),
        ("heldout_mean", "held-out return (after training)", lambda r, v: fmt(v, r.get("heldout_ci95"))),   # D71
        ("hash_load", "hash load, tilings × (tiles+1)^d / features", lambda r, v: f"{v:.2f}"),
    ) if col in d and d[col].notna().any()]
    print("| config | final eval return | mean eval (AUC) | online return | peak SRAM (B) | instr./step (M4) | update underflow |"
          + "".join(f" {head} |" for _, head, _ in extra))
    print("|---|---|---|---|---|---|---|" + "---|" * len(extra))
    for _, r in d.iterrows():
        peak = f"{int(r['peak_bytes'])}" + ("" if r.get("peak_source") == "m4-qemu" else " (est.)")
        if r.get("fits_f446") == 0:
            peak += ", does not fit"
        if str(r["kind"]).startswith("baseline"):      # no learner, so no working memory to report
            peak = "—"
        ins = f"{r['insns_learn_step']:.0f}" if "insns_learn_step" in r and not pd.isna(r.get("insns_learn_step")) else "—"
        uf = f"{100 * r['underflow_frac']:.1f}%" if "underflow_frac" in r and not pd.isna(r.get("underflow_frac")) else "—"
        print(f"| {r['run_id']} | {fmt(r['final_mean'], r['final_ci95'])} | {fmt(r['auc_mean'], r['auc_ci95'])} | "
              f"{fmt(r.get('online_mean'), r.get('online_ci95'))} | {peak} | {ins} | {uf} |"
              + "".join(f" {'—' if pd.isna(r[col]) else show(r, r[col])} |" for col, _, show in extra))


if __name__ == "__main__":
    main()
