#!/usr/bin/env python3
"""Print an experiment's paired differences as a markdown table for docs/results.md.

The comparisons are the ones in docs/decisions.md (pre-registration of 22 Nov 2026), so that
no difference in results.md is typed by hand. Intervals: Student-t 95% on per-seed differences (D52).

    python3 scripts/compare.py cartpole_main_tuned
"""
import argparse
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tinyrl import analysis  # noqa: E402


class C(NamedTuple):
    a: str
    b: str
    metric: str
    status: str = "pre-registered"
    metric_b: str | None = None
    exp_b: str | None = None
    name: str | None = None

    @property
    def primary(self) -> bool:
        return self.status == "pre-registered"


POST_HOC = "not pre-registered"
ADAPT = ("table", "linear", "net", "net-stream", "net-stream-cbp")
GO1 = ("table", "linear_features1024", "linear_features4096", "net-stream_hidden16", "net-stream_hidden32")

COMPARISONS = {
    "pendulum_main_tuned": [
        C(a, b, m, POST_HOC)
        for a, b in (("net_obs_modetrig_hidden64", "linear_obs_modetrig_features4096"),
                     ("net-stream_obs_modeangle_hidden32", "linear_obs_modeangle_features256"))
        for m in ("final", "auc")
    ],
    "replay_ratio": [
        C("table_batch5", "table_batch1", "auc"),
        C("table_batch20", "table_batch1", "auc", "exploratory"),
        C("table_batch5", "table_batch1", "final", "exploratory"),
        C("table_batch20", "table_batch1", "final", "exploratory"),
    ],
    "cartpole_main_tuned": [
        C(a, b, m)
        for a, b in (("linear_features1024", "table_bins8"),
                     ("net_hidden32", "linear_features1024"),
                     ("net-stream_hidden32", "linear_features1024"))
        for m in ("final", "auc")
    ],
    "adaptation_pendulum": [
        C("net-stream", "linear", "post_auc"),
        C("net-stream-cbp", "net-stream", "post_auc"),
        *[C(r, r, "post_auc", metric_b="auc", exp_b="adaptation_pendulum_scratch",
            name=f"forward transfer, {r} (after the shift − from scratch)") for r in ADAPT],
        *[C(r, r, "back", metric_b="final", name=f"forgetting, {r} (right after switch back − P)") for r in ADAPT],
        *[C(r, r, "a2_auc", "pre-registered, secondary", metric_b="final",
            name=f"second A window, {r} (mean − P)") for r in ADAPT],
    ],
    "stream_ablation": [
        C(a, b, m)
        for a, b in (("table-obgd-lam-sr", "table-obgd-lam"),
                     ("linear-obgd-lam-sr", "linear-obgd-lam"),
                     ("net-obgd-lam-ln-sr", "net-obgd-lam-ln"),
                     ("table-sgd", "table-bandit"),
                     ("linear-sgd", "linear-bandit"))
        for m in ("final", "auc")
    ],
    "go1_gait_v3": [C(a, b, "final") for a in GO1 for b in ("B3-stumble", "B4-all-reflexes")],
}


def verdict(mean: float, half: float) -> str:
    if mean - half > 0:
        return "A higher"
    if mean + half < 0:
        return "A lower"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=sorted(COMPARISONS))
    a = ap.parse_args()
    print("| A − B | metric | status | pairs | mean | 95% CI | verdict |")
    print("|---|---|---|---|---|---|---|")
    for c in COMPARISONS[a.experiment]:
        r = analysis.paired_diff(a.experiment, c.a, c.b, c.metric, exp_b=c.exp_b, metric_b=c.metric_b)
        m, h = r["mean_diff"], r["ci95"]
        digits = 2 if abs(m) < 10 else 1
        pairs = f"{r['n']}" + (f" ({r['n_missing']} missing)" if r["n_missing"] else "")
        metric = c.metric if c.metric_b is None else f"{c.metric} − {c.metric_b}"
        print(f"| {c.name or f'{c.a} − {c.b}'} | {metric} | {c.status} | {pairs} | {m:+.{digits}f} | "
              f"[{m - h:+.{digits}f}, {m + h:+.{digits}f}] | {verdict(m, h)} |")


if __name__ == "__main__":
    main()
