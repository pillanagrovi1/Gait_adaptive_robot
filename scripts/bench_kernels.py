#!/usr/bin/env python3
"""Measure candidate learning-update kernels on a Cortex-M4F in QEMU and draw a
memory x compute trade-off map (docs/research_review_2.md, "Map + measure").

Each kernel (m4/kernels/kernels_main.c) is built for several sizes, run for
N1 and N2 steps under the instruction-counting plugin, and reported as
instructions per learning step and static learner memory.

    python3 scripts/bench_kernels.py
Output: results/m4/kernels.csv and results/figures/tradeoff_map.png

Caveats (also on the figure): QEMU counts instructions, not cycles. A float
multiply-accumulate (VMLA, 3 cycles on the M4) counts as 1, like a 1-cycle
integer SMLAL, so float looks cheaper here than it is. Data efficiency is not
measured by this script.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "tools" / "qemu_plugin" / "libinsn_count.so"
BUILD = ROOT / "build" / "kernels"
OUT_CSV = ROOT / "results" / "m4" / "kernels.csv"
OUT_FIG = ROOT / "results" / "figures" / "tradeoff_map.png"
N1, N2 = 50, 150
CFLAGS = ["-mcpu=cortex-m4", "-mthumb", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16", "-O2", "-std=c11",
          "-ffunction-sections", "-fdata-sections", "-Wall", "-Wno-unused-function"]
QEMU = ["qemu-system-arm", "-M", "netduinoplus2", "-nographic", "-monitor", "none",
        "-semihosting-config", "enable=on,target=native"]

# (family, label, kernel id, defines)
VARIANTS = []
for cells in (121, 1331, 10000):
    for kid, fmt in ((1, "Q16.16"), (2, "float32"), (3, "q15")):
        VARIANTS.append(("table", f"table {fmt} {cells} cells", kid, {"CELLS": cells}))
for m in (256, 1024, 4096):
    for kid, fmt in ((4, "Q16.16"), (5, "float32"), (6, "q15")):
        VARIANTS.append(("tile coding", f"tiles {fmt} M={m}", kid, {"M": m, "T": 8}))
for h in (16, 32, 64):
    for kid, fmt in ((7, "Q16.16"), (8, "float32")):
        VARIANTS.append(("network", f"net {fmt} H={h}", kid, {"D": 3, "H": h}))
for k in (32, 64, 128):
    VARIANTS.append(("LSTD-Q", f"LSTD-Q float32 k={k}", 9, {"K": k}))
for n in (0, 5, 20, 50):
    VARIANTS.append(("Dyna-Q", f"Dyna-Q n={n} (1331 cells)", 10, {"CELLS": 1331, "NPLAN": n}))


def build_run(kid, defs, steps, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    elf = out / "k.elf"
    d = [f"-DKERNEL={kid}", f"-DSTEPS={steps}"] + [f"-D{k}={v}" for k, v in defs.items()]
    src = [str(ROOT / "m4" / "kernels" / "kernels_main.c"), str(ROOT / "m4" / "startup.c")]
    cmd = ["arm-none-eabi-gcc", *CFLAGS, *d, f"-I{ROOT / 'm4'}", *src, "-T", str(ROOT / "m4" / "link.ld"),
           "-nostartfiles", "--specs=nano.specs", "-Wl,--gc-sections", "-lgcc", "-o", str(elf)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, ("does not fit" if "static RAM too large" in r.stderr or "overflowed" in r.stderr
                      else r.stderr.strip()[-200:])
    q = subprocess.run(QEMU + ["-plugin", str(PLUGIN), "-d", "plugin", "-kernel", str(elf)],
                       capture_output=True, text=True, timeout=900)
    kv = {k: int(v) for k, v in re.findall(r"^(\w+)=(\d+)$", q.stdout + q.stderr, re.M)}
    return kv, ""


def main():
    if not PLUGIN.exists():
        sys.exit("build the QEMU plugin first: make -C tools/qemu_plugin")
    rows = []
    for fam, label, kid, defs in VARIANTS:
        tag = f"k{kid}_" + "_".join(f"{k}{v}" for k, v in defs.items())
        a, err = build_run(kid, defs, N1, BUILD / tag / "n1")
        b, _ = build_run(kid, defs, N2, BUILD / tag / "n2") if a else (None, err)
        row = {"family": family_name(fam), "label": label, "kernel": kid, **defs}
        if a and b:
            row.update(bytes=b["bytes"], stack=b["stack"], peak_bytes=b["bytes"] + b["stack"],
                       insns_per_step=round((b["insns"] - a["insns"]) / (N2 - N1), 1))
        else:
            row["error"] = err
        rows.append(row)
        print(f"{label:34s} {row.get('peak_bytes', 'n/a')!s:>8} B  {row.get('insns_per_step', 'n/a')!s:>10} insn/step  {row.get('error', '')}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    keys = ["family", "label", "kernel", "CELLS", "M", "T", "D", "H", "K", "NPLAN",
            "bytes", "stack", "peak_bytes", "insns_per_step", "error"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    plot(rows)
    print("wrote", OUT_CSV.relative_to(ROOT), "and", OUT_FIG.relative_to(ROOT))


def family_name(f):
    return f


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fams = ["table", "tile coding", "network", "LSTD-Q", "Dyna-Q"]
    colors = dict(zip(fams, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]))
    fmt_marker = {"Q16.16": "o", "float32": "s", "q15": "^"}
    fig, ax = plt.subplots(figsize=(8.0, 5.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    for fam in fams:
        rr = [r for r in rows if r["family"] == fam and r.get("peak_bytes")]
        for fmt, mk in fmt_marker.items():
            pts = [r for r in rr if fmt in r["label"]] if fam not in ("LSTD-Q", "Dyna-Q") else (rr if fmt == "Q16.16" else [])
            if fam == "LSTD-Q":
                pts = rr if fmt == "float32" else []
            if not pts:
                continue
            pts = sorted(pts, key=lambda r: r["peak_bytes"])
            ax.plot([r["peak_bytes"] for r in pts], [r["insns_per_step"] for r in pts], marker=mk,
                    color=colors[fam], linewidth=1.0, markersize=7, alpha=0.9,
                    linestyle="-" if fmt == "Q16.16" else ("--" if fmt == "float32" else ":"))
        if rr:
            last = max(rr, key=lambda r: r["insns_per_step"])
            ax.annotate(fam, (last["peak_bytes"], last["insns_per_step"]), xytext=(6, 4),
                        textcoords="offset points", fontsize=8, color="#52514e")
    ax.axvline(128 * 1024, color="#52514e", linestyle="--", linewidth=0.8)
    ax.text(128 * 1024, ax.get_ylim()[0] * 1.5 if ax.get_yscale() == "log" else 0, "  128 KB SRAM",
            color="#52514e", fontsize=7, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learner memory: static arrays + peak stack (bytes, log)", color="#52514e", fontsize=9)
    ax.set_ylabel("instructions per learning step (QEMU, log)", color="#52514e", fontsize=9)
    ax.set_title("On-board learning updates: memory vs compute on a Cortex-M4F (kernels, not full learners)",
                 color="#0b0b0b", fontsize=10, loc="left")
    handles = [Line2D([], [], color=colors[f], marker="o", linestyle="-", label=f) for f in fams]
    handles += [Line2D([], [], color="#52514e", marker=m, linestyle=ls, label=fm)
                for fm, m, ls in (("Q16.16", "o", "-"), ("float32", "s", "--"), ("q15 storage", "^", ":"))]
    ax.legend(handles=handles, fontsize=7, frameon=False, ncol=2, loc="upper left")
    fig.text(0.01, 0.01, "Kernels only: excludes feature hashing, normalisation and action selection (see results/m4/*.csv for full learners).\n"
            "QEMU counts instructions, not cycles: float MACs (3 cycles) count like integer ones (1).\n"
            "At 4 Hz decisions the M4 has ~45 M cycles per step; at 50 Hz ~3.6 M.",
            ha="left", va="bottom", fontsize=7, color="#52514e")
    ax.set_ylim(20, None)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
