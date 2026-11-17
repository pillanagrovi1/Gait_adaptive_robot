#!/usr/bin/env python3
"""Make the figures and summary table for one or more experiments.

    python3 scripts/make_plots.py pendulum_main
    python3 scripts/make_plots.py pendulum_main cartpole_main
Output: results/figures/<experiment>_{curves,headline,compute,adaptation}.png + _summary.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tinyrl import plots  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for name in sys.argv[1:]:
        for f in plots.make_all(name):
            print("wrote", Path(f).relative_to(ROOT))
