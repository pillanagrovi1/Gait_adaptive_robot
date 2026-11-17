"""Figures for one experiment. Called by scripts/make_plots.py.

  <exp>_curves.png     learning curves, one panel per learner (small multiples);
                       within a panel, darker blue = larger sweep value
  <exp>_headline.png   final performance vs peak SRAM (the project's main plot)
  <exp>_compute.png    instructions per learning step vs peak SRAM (if measured)
  <exp>_summary.csv    the numbers behind the plots (the table view)

Colour by learner kind (fixed order, validated for colour-vision deficiency),
plus a different marker shape per kind so identity never relies on colour.
Hollow markers = memory NOT measured on the M4 (host estimate only).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import analysis, sweep  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
KIND_COLOR = {"tabular": "#2a78d6", "linear": "#eb6834", "mlp": "#1baf7a"}
KIND_MARKER = {"tabular": "o", "linear": "s", "mlp": "^"}
KIND_NAME = {"tabular": "Table", "linear": "Linear (tile coding)", "mlp": "Tiny network"}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
F446_SRAM = 128 * 1024


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _ramp(n):
    if n == 1:
        return [RAMP[3]]
    idx = np.linspace(0, len(RAMP) - 1, n).round().astype(int)
    return [RAMP[i] for i in idx]


def learning_curves(exp: dict, specs, path: Path):
    is_base = lambda s: s["learner"]["kind"].startswith("baseline")  # noqa: E731
    labels = list(dict.fromkeys(s["label"] for s in specs if not is_base(s)))
    base_curves = []
    for s in specs:
        if is_base(s):
            df, _ = analysis.load_run(exp["name"], s["run_id"])
            if df is not None:
                eps, M = analysis.eval_matrix(df)
                base_curves.append((s["run_id"], eps, np.nanmean(M, 0)))
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 3.4),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    shift_at = exp.get("run", {}).get("shift_at")
    for ax, lab in zip(axes[0], labels):
        _style(ax)
        for (bname, beps, bmu), ls in zip(base_curves, (":", "-.", "--")):
            ax.plot(beps, bmu, color=INK2, linestyle=ls, linewidth=1, label=f"baseline: {bname}")
        group = [s for s in specs if s["label"] == lab]
        for s, col in zip(group, _ramp(len(group))):
            df, _ = analysis.load_run(exp["name"], s["run_id"])
            if df is None:
                continue
            eps, M = analysis.eval_matrix(df)
            mu = np.nanmean(M, 0)
            ci = np.array([analysis.ci95(M[:, j]) for j in range(M.shape[1])])
            name = ", ".join(f"{k}={v}" for k, v in s["sweep_values"].items()) or s["run_id"]
            ax.plot(eps, mu, color=col, linewidth=1.5, label=f"{name} (n={M.shape[0]})")
            if M.shape[0] > 1:
                ax.fill_between(eps, mu - ci, mu + ci, color=col, alpha=0.15, linewidth=0)
        if shift_at is not None:
            ax.axvline(shift_at, color=INK2, linestyle="--", linewidth=0.8)
            ax.text(shift_at, ax.get_ylim()[1], " shift", color=INK2, fontsize=7, va="top")
        ax.set_title(lab, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("training episode", color=INK2, fontsize=8)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    axes[0][0].set_ylabel("greedy eval return (mean, 95% CI)", color=INK2, fontsize=8)
    fig.suptitle(f"{exp['name']}: learning curves", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def headline(exp: dict, summ, path: Path, y="final", ylabel="final performance (greedy eval return)"):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), facecolor=SURFACE)
    _style(ax)
    base = summ[summ["kind"].str.startswith("baseline")]
    summ = summ[~summ["kind"].str.startswith("baseline")]
    for _, r in base.iterrows():
        if np.isnan(r.get(f"{y}_mean", np.nan)):
            continue
        ax.axhline(r[f"{y}_mean"], color=INK2, linestyle=":", linewidth=1)
        ax.annotate(f"baseline: {r['label']}", (0.01, r[f"{y}_mean"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK2)
    for kind in ("tabular", "linear", "mlp"):
        d = summ[summ["kind"] == kind].sort_values("peak_bytes")
        if d.empty:
            continue
        c, mk = KIND_COLOR[kind], KIND_MARKER[kind]
        ax.plot(d["peak_bytes"], d[f"{y}_mean"], color=c, linewidth=1.2, alpha=0.6)
        for _, r in d.iterrows():
            measured = r["peak_source"] == "m4-qemu"
            ax.errorbar(r["peak_bytes"], r[f"{y}_mean"], yerr=r[f"{y}_ci95"] if r["n_seeds"] > 1 else None,
                        fmt=mk, markersize=7, color=c, ecolor=c, elinewidth=1, capsize=2,
                        markerfacecolor=c if measured else SURFACE, markeredgewidth=1.5, zorder=3)
        last = d.iloc[-1]
        ax.annotate(KIND_NAME[kind], (last["peak_bytes"], last[f"{y}_mean"]), xytext=(7, 9),
                    textcoords="offset points", fontsize=8, color=INK2, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    ax.axvline(F446_SRAM, color=INK2, linestyle="--", linewidth=0.8)
    ax.text(F446_SRAM, ax.get_ylim()[0], " STM32F446RE SRAM (128 KB)", color=INK2,
            fontsize=7, rotation=90, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("peak learner SRAM, bytes (log scale)", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    # legend: kinds + measured/estimate
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=KIND_MARKER[k], color=KIND_COLOR[k], linestyle="", markersize=7,
                      label=KIND_NAME[k]) for k in ("tabular", "linear", "mlp") if (summ["kind"] == k).any()]
    handles += [Line2D([], [], marker="o", color=INK2, linestyle="", markersize=7, label="memory measured (QEMU M4)"),
                Line2D([], [], marker="o", color=INK2, markerfacecolor=SURFACE, linestyle="", markersize=7,
                       label="memory estimated (host)")]
    ax.legend(handles=handles, fontsize=7, frameon=False, labelcolor=INK2, loc="best")
    n = int(summ["n_seeds"].min()) if len(summ) else 0
    ax.set_title(f"{exp['name']}: performance vs peak working memory  (error bars: 95% CI, n>={n} seeds)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def compute_plot(exp: dict, summ, path: Path):
    d0 = summ.dropna(subset=["insns_learn_step"]) if "insns_learn_step" in summ else summ.iloc[0:0]
    if d0.empty:
        return False
    fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor=SURFACE)
    _style(ax)
    for kind in ("tabular", "linear", "mlp"):
        d = d0[d0["kind"] == kind].sort_values("peak_bytes")
        if d.empty:
            continue
        ax.plot(d["peak_bytes"], d["insns_learn_step"], marker=KIND_MARKER[kind], color=KIND_COLOR[kind],
                markersize=7, linewidth=1.2, label=KIND_NAME[kind])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("peak learner SRAM, bytes (log)", color=INK2, fontsize=9)
    ax.set_ylabel("instructions per act+learn step (QEMU, log)", color=INK2, fontsize=9)
    ax.set_title(f"{exp['name']}: compute vs memory (instruction counts, not cycles)",
                 color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def make_all(exp_name: str):
    exp = sweep.load(exp_name)
    specs = sweep.expand(exp)
    OUT.mkdir(parents=True, exist_ok=True)
    summ = analysis.summarize(exp_name)
    if summ.empty:
        print("no runs found for", exp["name"])
        return []
    name = exp["name"]
    files = []
    summ.to_csv(OUT / f"{name}_summary.csv", index=False)
    files.append(OUT / f"{name}_summary.csv")
    learning_curves(exp, specs, OUT / f"{name}_curves.png")
    files.append(OUT / f"{name}_curves.png")
    headline(exp, summ, OUT / f"{name}_headline.png")
    files.append(OUT / f"{name}_headline.png")
    if exp.get("run", {}).get("shift_at") is not None and "recovery_episodes_mean" in summ:
        headline(exp, summ, OUT / f"{name}_adaptation.png", y="recovery_episodes",
                 ylabel="episodes to recover after shift (lower = faster)")
        files.append(OUT / f"{name}_adaptation.png")
    if compute_plot(exp, summ, OUT / f"{name}_compute.png"):
        files.append(OUT / f"{name}_compute.png")
    return files
