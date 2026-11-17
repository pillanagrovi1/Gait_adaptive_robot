#!/usr/bin/env python3
"""Print a markdown results table for one experiment, ready for docs/results.md.

Reads results/figures/<experiment>_summary.csv (written by make_plots.py) and adds the
provenance the results log requires: git commit, config file, command, number of seeds.

    python3 scripts/results_table.py stage2_pendulum_table_tuned
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fmt(m, ci, digits=0):
    if pd.isna(m):
        return "—"
    return f"{m:.{digits}f} ± {ci:.{digits}f}" if not pd.isna(ci) else f"{m:.{digits}f}"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    name = sys.argv[1]
    p = ROOT / "results" / "figures" / f"{name}_summary.csv"
    d = pd.read_csv(p)
    import json
    commits = set()
    for mf in (ROOT / "results" / "raw" / name).glob("*/seed*_meta.json"):
        commits.add(json.loads(mf.read_text()).get("git_commit", "?"))
    commits.discard("?")
    commit = ", ".join(sorted(commits)) if commits else (
        "not recorded (runs predate commit tracking; see docs/results.md) — HEAD now " +
        subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    print(f"**{name}** · commit `{commit}` · config `configs/{name}.json` · "
          f"command `bash scripts/run_experiment.sh {name.removesuffix('_tuned')}` · "
          f"seeds per config: {int(d['n_seeds'].min())}–{int(d['n_seeds'].max())} · intervals: {d['ci_method'].iloc[0]} 95%\n")
    cols = ["run_id", "final", "auc", "online", "peak_bytes", "peak_source", "insns_learn_step", "underflow_frac"]
    print("| config | final eval return | mean eval (AUC) | online return | peak SRAM (B) | instr./step (M4) | update underflow |")
    print("|---|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        peak = f"{int(r['peak_bytes'])}" + ("" if r.get("peak_source") == "m4-qemu" else " (est.)")
        ins = f"{r['insns_learn_step']:.0f}" if "insns_learn_step" in r and not pd.isna(r.get("insns_learn_step")) else "—"
        uf = f"{100 * r['underflow_frac']:.1f}%" if "underflow_frac" in r and not pd.isna(r.get("underflow_frac")) else "—"
        print(f"| {r['run_id']} | {fmt(r['final_mean'], r['final_ci95'])} | {fmt(r['auc_mean'], r['auc_ci95'])} | "
              f"{fmt(r.get('online_mean'), r.get('online_ci95'))} | {peak} | {ins} | {uf} |")
    _ = cols


if __name__ == "__main__":
    main()
