#!/usr/bin/env python3
"""Processor-in-the-loop check: learning ON the (emulated) Cortex-M4 must reproduce
learning on the laptop bit for bit.

Starts the PIL firmware (m4/pil) in QEMU with its UART on a TCP socket, then runs
the same short experiment twice per learner: once with the host library, once with
the learner on the chip (the laptop only steps MuJoCo and relays observations).
Every training return and every evaluation return must be identical.

    python3 scripts/pil_check.py                  # QEMU
    python3 scripts/pil_check.py --link serial:/dev/ttyACM0 --no-qemu   # a flashed Nucleo board

On a real board this also records cycles per act() and observe() call, the
first timing numbers the project is allowed to claim (on that board, that clock).
Output: results/m4/pil_check.json
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tinyrl import harness  # noqa: E402

LEARNERS = {
    "table": {"kind": "tabular", "alpha": 0.3, "gamma": 0.95, "bins": 9},
    "linear": {"kind": "linear", "alpha": 0.2, "gamma": 0.95, "tilings": 8, "tiles": 8, "features": 1024},
    "net-replay": {"kind": "mlp", "alpha": 0.02, "gamma": 0.95, "hidden": 16, "td_clip": 1.0,
                   "target_period": 100, "replay_cap": 300, "batch": 4, "replay_compact": 1},
    "net-stream": {"kind": "mlp", "alpha": 1.0, "gamma": 0.95, "hidden": 16, "lam": 0.8, "obgd_kappa": 2.0,
                   "norm": 1, "round_mode": 1},
}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", help="PIL link; default: start QEMU on a free TCP port")
    ap.add_argument("--no-qemu", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    qemu = None
    link = a.link
    if not a.no_qemu:
        subprocess.run(["make", "-s", "-C", str(ROOT / "m4" / "pil"), "qemu"], check=True, capture_output=True)
        port = free_port()
        qemu = subprocess.Popen(["qemu-system-arm", "-M", "netduinoplus2", "-nographic", "-monitor", "none",
                                 "-semihosting-config", "enable=on,target=native",
                                 "-serial", f"tcp:127.0.0.1:{port},server=on,wait=off,nodelay=on",
                                 "-kernel", str(ROOT / "build" / "pil" / "qemu" / "pil.elf")],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        link = link or f"tcp:127.0.0.1:{port}"
        time.sleep(0.5)
    task = {"name": "pendulum", "obs_mode": "trig"}
    run = {"steps": a.steps, "decay_steps": a.steps // 2, "eval_every_steps": a.steps // 3, "eval_episodes": 1}
    report, ok = {}, True
    try:
        for name, L in LEARNERS.items():
            t0 = time.time()
            host = harness.run(task, L, seed=3, run_cfg=run)
            t1 = time.time()
            chip = harness.run(task, {**L, "pil": link}, seed=3, run_cfg=run)
            t2 = time.time()
            same_train = [r["train_return"] for r in host["rows"]] == [r["train_return"] for r in chip["rows"]]
            same_eval = [e["eval_return"] for e in host["evals"]] == [e["eval_return"] for e in chip["evals"]]
            match = same_train and same_eval
            ok &= match
            report[name] = {"match": match, "episodes": len(host["rows"]), "host_s": round(t1 - t0, 2),
                            "chip_s": round(t2 - t1, 2), "board": chip["meta"]["pil"],
                            "board_arena_bytes": chip["meta"]["arena_bytes"]}
            print(f"{name:11s} {'MATCH' if match else 'DIFFERENT':9s} episodes {len(host['rows']):3d}  "
                  f"host {t1 - t0:5.1f}s  chip {t2 - t1:5.1f}s  arena {chip['meta']['arena_bytes']} B  "
                  f"stack {chip['meta']['pil']['stack_peak_bytes']} B  "
                  f"cycles act/observe (mean) {chip['meta']['pil']['mean_act_cycles']}/{chip['meta']['pil']['mean_observe_cycles']}")
    finally:
        if qemu:
            qemu.kill()
    out = ROOT / "results" / "m4" / "pil_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"link": "qemu" if qemu else link, "steps": a.steps, "learners": report}, indent=2))
    print("PIL check", "passed: on-chip learning reproduces the host bit for bit" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
