#!/usr/bin/env python3
"""Print an experiment's paired differences as a markdown table for docs/results.md.

The comparisons are the ones in docs/decisions.md (pre-registration of 22 Nov 2026), so that
no difference in results.md is typed by hand. Intervals: Student-t 95% on per-seed differences (D52).

    python3 scripts/compare.py cartpole_main_tuned
    python3 scripts/compare.py cartpole_main_tuned --by-sram   # exploratory: every pair at similar or 3-10x SRAM
    python3 scripts/compare.py cartpole_main_v2_tuned --matched   # D70: best network vs best linear per budget
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
    ] + [
        *[C(a, b, m, "exploratory, added after the results (D69)")
          for a, b in (("table-obgd-lam", "table-obgd"),
                       ("linear-obgd-lam", "linear-obgd"),
                       ("net-obgd-lam", "net-obgd"),
                       ("net-sgd", "net-obgd-lam-ln-sr"))
          for m in ("final", "auc")],
    ],
    "cartpole_main_v2_tuned": [         # D70 comparison 1: the v1 pairs, now pre-registered
        C(a, b, m)
        for a, b in (("linear_features1024-tiles2", "table_bins8"),
                     ("net_hidden32", "linear_features1024-tiles2"),
                     ("net-stream_hidden32", "linear_features1024-tiles2"))
        for m in ("final", "auc")
    ],
    "pendulum_main_v2_tuned": [
        C(a, b, m)
        for a, b in (("net-trig_hidden64", "linear-trig_features4096-tiles7"),
                     ("net-stream-angle_hidden32", "linear-angle_features256-tiles5"))
        for m in ("final", "auc")
    ],
    "go1_gait_v3": [
        *[C(a, b, "final") for a in GO1 for b in ("B3-stumble", "B4-all-reflexes")],
        *[C(a, "random", "final", "exploratory, added after the results (D65)") for a in GO1],
    ],
}


def verdict(mean: float, half: float) -> str:
    if mean - half > 0:
        return "A higher"
    if mean + half < 0:
        return "A lower"
    return "inconclusive"


def memory_band(ratio: float) -> str | None:
    """linear SRAM / streaming-net SRAM -> the band it is compared in (D63), or None."""
    if 1 / 1.5 <= ratio <= 1.5:
        return "about equal (within 1.5x)"
    if 3 <= ratio <= 10:
        return "3-10x less for the streaming net"
    return None


def by_sram(exp: str) -> None:
    """Every streaming-net / linear pair on the same task variant that falls in a band: none hand-picked."""
    d = analysis.summarize(exp)
    nothing = (f"compare: no streaming-net / linear pairs for {exp}. If its raw files are missing, extract them: "
               f"mkdir -p results/raw && tar -xzf results/raw_archive/{exp}.tar.gz -C results/raw")
    if d.empty:
        raise SystemExit(nothing)
    variant = [c for c in d if c == "sweep_obs_mode"]
    rows = []
    for _, s in d[d["label"] == "net-stream"].iterrows():
        for _, lin in d[d["label"] == "linear"].iterrows():
            ratio = lin["peak_bytes"] / s["peak_bytes"]
            band = memory_band(ratio)
            if band and all(s[v] == lin[v] for v in variant):
                rows.append((band, ratio, s, lin))
    if not rows:
        raise SystemExit(nothing)
    print("| SRAM band | A = streaming net (B) | B = linear (B) | SRAM B / A | final, A − B | AUC, A − B |")
    print("|---|---|---|---|---|---|")
    count = {}
    for band, ratio, s, lin in sorted(rows, key=lambda r: (r[0] != "about equal (within 1.5x)", r[1])):
        cells = []
        for metric in ("final", "auc"):
            r = analysis.paired_diff(exp, s["run_id"], lin["run_id"], metric)
            m, h = r["mean_diff"], r["ci95"]
            v = verdict(m, h)
            tally = count.setdefault((band, metric), {"A higher": 0, "inconclusive": 0, "A lower": 0})
            tally[v] += 1
            cells.append(f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}] {v}")
        print(f"| {band} | {s['run_id']} ({int(s['peak_bytes'])}) | {lin['run_id']} ({int(lin['peak_bytes'])}) | "
              f"{ratio:.2f} | {cells[0]} | {cells[1]} |")
    print()
    print("| SRAM band | metric | pairs | streaming net higher | inconclusive | streaming net lower |")
    print("|---|---|---|---|---|---|")
    for (band, metric), n in count.items():
        print(f"| {band} | {metric} | {sum(n.values())} | {n['A higher']} | {n['inconclusive']} | {n['A lower']} |")
    if variant:
        linear = d[d["label"] == "linear"]
        best = linear.loc[linear.groupby("sweep_features")["final_mean"].idxmax()]
        print()
        print("| about equal SRAM, against linear's better encoding: A = streaming net (B) | B = linear (B) | "
              "B's final | final, A − B |")
        print("|---|---|---|---|")
        for _, lin in best.iterrows():
            for _, s in d[d["label"] == "net-stream"].iterrows():
                if (memory_band(lin["peak_bytes"] / s["peak_bytes"]) or "").startswith("about equal"):
                    r = analysis.paired_diff(exp, s["run_id"], lin["run_id"], "final")
                    m, h = r["mean_diff"], r["ci95"]
                    print(f"| {s['run_id']} ({int(s['peak_bytes'])}) | {lin['run_id']} ({int(lin['peak_bytes'])}) | "
                          f"{lin['final_mean']:.0f} | {m:+.1f} [{m - h:+.1f}, {m + h:+.1f}] {verdict(m, h)} |")


MATCHED = {"pendulum_main_v2_tuned": (4, 16, 64), "cartpole_main_v2_tuned": (6, 16, 64)}   # KB (1 KB = 1024 B)


def family(label: str) -> str | None:
    return "network" if label.startswith("net") else "linear" if label.startswith("linear") else None


def pick(budget: int, sram, auc) -> dict:
    """{family: run_id with the highest selection AUC among the runs whose M4 peak SRAM fits the budget}."""
    best = {}
    for run, b in sram.items():
        f = family(run)
        if f and b <= budget and run in auc and (f not in best or auc[run] > auc[best[f]]):
            best[f] = run
    return best


def matched(exp: str) -> None:
    """D70 comparison 2: per SRAM budget, best network − best linear, chosen on the selection runs only."""
    rep, sel = analysis.summarize(exp), analysis.summarize(exp + "_select")
    if rep.empty or sel.empty:
        raise SystemExit(f"compare: need results/raw/{exp}/ and results/raw/{exp}_select/")
    rep = rep[rep["label"].map(family).notna()]
    if (rep["peak_source"] != "m4-qemu").any():
        raise SystemExit("compare: every network and linear run needs its M4 measurement (measure_m4.py) first")
    sram = dict(zip(rep["run_id"], rep["peak_bytes"]))
    auc = dict(zip(sel["run_id"], sel["auc_mean"]))
    print("| SRAM budget | A = best network (B) | B = best linear (B) | metric | pairs | mean | 95% CI | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for kb in MATCHED[exp]:
        best = pick(kb * 1024, sram, auc)
        if len(best) < 2:
            print(f"| {kb} KB | {best.get('network', 'none fits')} | {best.get('linear', 'none fits')} | | | | | |")
            continue
        a, b = best["network"], best["linear"]
        for metric in ("final", "auc"):
            r = analysis.paired_diff(exp, a, b, metric)
            m, h = r["mean_diff"], r["ci95"]
            print(f"| {kb} KB | {a} ({int(sram[a])}) | {b} ({int(sram[b])}) | {metric} | {r['n']} | {m:+.1f} | "
                  f"[{m - h:+.1f}, {m + h:+.1f}] | {verdict(m, h)} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=sorted(COMPARISONS))
    ap.add_argument("--by-sram", action="store_true", help="exploratory equal-memory table (D63)")
    ap.add_argument("--matched", action="store_true", help="D70: best network vs best linear per SRAM budget")
    a = ap.parse_args()
    if a.by_sram:
        by_sram(a.experiment)
        return
    if a.matched:
        matched(a.experiment)
        return
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
