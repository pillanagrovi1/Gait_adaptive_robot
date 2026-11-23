#!/usr/bin/env python3
"""Check that the documents still match the archived data. Exit 1 on any mismatch.

For every archive in results/raw_archive/: extract it if needed, recompute the summary and compare it with the
committed results/figures/<experiment>_summary.csv, then require every line that scripts/results_table.py and
scripts/compare.py print to appear verbatim in docs/results.md. Also checks the links and `results/...` paths in
the documents, and that every interval in report.md and README.md appears in results.md.

    python3 scripts/check_docs.py
"""
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tinyrl import analysis  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from compare import COMPARISONS, MATCHED  # noqa: E402

DOCS = ["README.md", "docs/report.md", "docs/results.md", "docs/study_guide.md"]
BY_SRAM = {"pendulum_main_tuned", "cartpole_main_tuned"}


def run(*args) -> str:
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(args)} failed: {r.stderr.strip()[-300:]}")
    return r.stdout


def summary_problems(name: str) -> list[str]:
    new = analysis.summarize(name)
    new = pd.read_csv(pd.io.common.StringIO(new.to_csv(index=False)))
    old = pd.read_csv(ROOT / "results" / "figures" / f"{name}_summary.csv")
    out = []
    for c in old:
        if c not in new:
            out.append(f"{name}: summary column {c} is no longer computed")
        elif old[c].dtype.kind in "fi" and new[c].dtype.kind in "fi":
            if not np.allclose(old[c], new[c], equal_nan=True, rtol=1e-9, atol=1e-9):
                out.append(f"{name}: summary column {c} differs from the committed CSV")
        elif not old[c].astype(str).equals(new[c].astype(str)):
            out.append(f"{name}: summary column {c} differs from the committed CSV")
    return out


def main() -> int:
    results = (ROOT / "docs/results.md").read_text()
    lines = set(results.split("\n"))
    problems = []
    archives = sorted((ROOT / "results" / "raw_archive").glob("*.tar.gz"))
    for archive in archives:            # all first: some comparisons span two experiments
        if not (ROOT / "results" / "raw" / archive.name.removesuffix(".tar.gz")).exists():
            (ROOT / "results" / "raw").mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as t:
                t.extractall(ROOT / "results" / "raw", filter="data")
    for archive in archives:
        name = archive.name.removesuffix(".tar.gz")
        problems += summary_problems(name)
        header = re.search(rf"^\*\*{re.escape(name)}\*\* · .*command `([^`]*)`", results, re.M)
        printed = []
        if header:
            printed += run("scripts/results_table.py", name, "--command", header.group(1)).split("\n")
        if name in COMPARISONS:
            printed += run("scripts/compare.py", name).split("\n")
        if name in BY_SRAM:
            printed += run("scripts/compare.py", name, "--by-sram").split("\n")
        if name in MATCHED:
            printed += run("scripts/compare.py", name, "--matched").split("\n")
        missing = [p for p in printed if p.strip() and p not in lines]
        problems += [f"{name}: not in results.md: {p[:120]}" for p in missing]
        print(f"{name:32s} table {'yes' if header else 'no '}  lines checked {len([p for p in printed if p.strip()]):3d}"
              f"  missing {len(missing)}")
    for doc in DOCS:
        text = (ROOT / doc).read_text()
        for link in re.findall(r"\]\(([^)#]+)", text):
            if "://" not in link and not (ROOT / doc).parent.joinpath(link).exists():
                problems.append(f"{doc}: broken link {link}")
        for path in re.findall(r"`(results/[\w./-]+\.(?:csv|log|gz|png|json|versions))`", text):
            if not (ROOT / path).exists():
                problems.append(f"{doc}: missing file {path}")
    plain = results.replace("−", "-")
    for doc in ("README.md", "docs/report.md"):
        text = (ROOT / doc).read_text().replace("−", "-")
        for iv in sorted(set(re.findall(r"\[[+-][\d.]+, [+-][\d.]+\]", text))):
            if iv not in plain:
                problems.append(f"{doc}: interval {iv} is not in results.md")
    for p in problems:
        print("PROBLEM:", p)
    print("documents match the data" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
