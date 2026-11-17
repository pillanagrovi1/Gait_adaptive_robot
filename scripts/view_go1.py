#!/usr/bin/env python3
"""Watch the Go1 trot on rough ground in the MuJoCo viewer (needs a display).

    python3 scripts/view_go1.py                     # nominal trot (action 0)
    python3 scripts/view_go1.py --action 1          # always "high step"
    python3 scripts/view_go1.py --random            # random gait adjustments
    python3 scripts/view_go1.py --roughness 0.09    # rougher ground

Actions: 0 nominal, 1 high step, 2 extend front, 3 extend rear, 4 extend left, 5 extend right.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import mujoco.viewer  # noqa: E402

from tinyrl.envs.go1 import Go1Gait  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--action", type=int, default=0)
ap.add_argument("--random", action="store_true")
ap.add_argument("--roughness", type=float, default=0.06)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

env = Go1Gait(roughness=a.roughness)
rng = np.random.default_rng(a.seed)
env.reset(seed=a.seed)
with mujoco.viewer.launch_passive(env.model, env.data) as v:
    v.cam.trackbodyid = env.model.body("trunk").id
    v.cam.distance = 1.6
    while v.is_running():
        act = int(rng.integers(6)) if a.random else a.action
        t0 = time.time()
        o, r, term, trunc = env.step(act)
        v.sync()
        print(f"t={env.t * env.decision_dt:5.2f}s  action {act}  roll {o[0]:+.2f}  pitch {o[1]:+.2f}  reward {r:+.2f}")
        time.sleep(max(0.0, env.decision_dt - (time.time() - t0)))
        if term or trunc:
            print("fell" if term else "episode over", "- resetting")
            env.reset(seed=int(rng.integers(1 << 30)))
