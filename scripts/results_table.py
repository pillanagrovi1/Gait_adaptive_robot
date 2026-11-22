#!/usr/bin/env python3
"""Print a markdown results table for one experiment, ready for docs/results.md.

Reads results/figures/<experiment>_summary.csv (written by make_plots.py) and adds the
provenance the results log requires: git commit, config file, command, number of seeds.
It exits 1, printing no table, if that provenance does not hold: no raw files, seed counts that
differ between raw files, summary and config, a raw run missing from the summary, more than one
commit, or code with uncommitted changes (+dirty).

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

from tinyrl import sweep  # noqa: E402


def refuse(*problems):
    print("results_table: no table printed.", *problems, sep="\n  ", file=sys.stderr)
    raise SystemExit(1)


def fmt(m, ci, digits=0):
    if pd.isna(m):
        return "—"
    return f"{m:.{digits}f} ± {ci:.{digits}f}" if not pd.isna(ci) else f"{m:.{digits}f}"


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
    commits = {json.loads(mf.read_text()).get("git_commit") for mf in metas} - {None, "", "?", "unknown"}
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
    not_run = {s["run_id"] for s in sweep.expand(exp)} - set(d["run_id"])
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
    print("| config | final eval return | mean eval (AUC) | online return | peak SRAM (B) | instr./step (M4) | update underflow |")
    print("|---|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        peak = f"{int(r['peak_bytes'])}" + ("" if r.get("peak_source") == "m4-qemu" else " (est.)")
        if str(r["kind"]).startswith("baseline"):      # no learner, so no working memory to report
            peak = "—"
        ins = f"{r['insns_learn_step']:.0f}" if "insns_learn_step" in r and not pd.isna(r.get("insns_learn_step")) else "—"
        uf = f"{100 * r['underflow_frac']:.1f}%" if "underflow_frac" in r and not pd.isna(r.get("underflow_frac")) else "—"
        print(f"| {r['run_id']} | {fmt(r['final_mean'], r['final_ci95'])} | {fmt(r['auc_mean'], r['auc_ci95'])} | "
              f"{fmt(r.get('online_mean'), r.get('online_ci95'))} | {peak} | {ins} | {uf} |")


if __name__ == "__main__":
    main()
