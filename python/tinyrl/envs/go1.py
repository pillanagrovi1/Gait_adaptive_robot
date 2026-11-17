"""Unitree Go1 trotting over rough ground (Stage 7: the gait task).

Structure: a fixed central pattern generator (CPG) produces an open-loop trot.
The learner does NOT output joint angles. Every `decision_dt` seconds it picks
one of a few discrete gait adjustments, based on what an IMU would measure.
That keeps the action and state spaces small enough for a table, a hashed
linear model or a tiny network on a Cortex-M4, while the CPG does the fast
rhythmic work.

    learner (slow, ~4 Hz)  --action-->  CPG + leg mapping (50 Hz)  -->  PD joints (MuJoCo, 500 Hz)
          ^                                                                   |
          +------------------- IMU: roll, pitch, roll rate, pitch rate -------+

CPG: phase phi advances at `freq` Hz. Diagonal pairs (FR+RL, FL+RR) are half a
cycle apart (trot). For leg phase phi_l:
    thigh = thigh0 + A_thigh * cos(phi_l)        fore-aft swing
    calf  = calf0  - A_calf * max(0, sin(phi_l)) foot lift during swing
    plus per-leg extension offsets chosen by the learner (pitch/roll compensation)

Actions (6):
    0 nominal   1 high step   2 extend front   3 extend rear   4 extend left   5 extend right

Observation modes:
    "imu4": roll, pitch, roll rate, pitch rate (instantaneous, at the decision)
    "imu6": imu4 + forward speed + trunk height above start ground
    "imu4p": phase-aware (17 Nov 2026 revision, see docs/research_review.md):
             roll, pitch, roll rate, pitch rate AVERAGED over the last decision period,
             plus one bit: which diagonal pair swings next. Use with phase_lock=True.

phase_lock=True: the CPG starts at phase 0 and the learner decides exactly every half
cycle (decision_dt = 1 / (2 freq)), i.e. at the moment one diagonal pair lifts off.
Without it, the same IMU reading can mean "pair A about to swing" or "pair B about to
swing" (phase aliasing): every published CPG-modulation method observes the phase.
Optional extras (default off): friction randomisation per episode, roll/pitch-rate
penalty, action-switch penalty.

Reward per decision: 10 * forward progress (m) - 2 * (roll^2 + pitch^2), and -5 on a fall.
Shift (adaptation): terrain roughness scale *= (1 + level).

STATUS: pipeline-complete, NOT tuned. The CPG constants were set by a short
manual check that the open-loop trot moves forward on flat ground. Treat every
number here as a starting point for your own decisions.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .base import Env

ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "assets" / "unitree_go1" / "scene_rough.xml"

LEGS = ("FR", "FL", "RR", "RL")
LEG_PHASE = {"FR": 0.0, "RL": 0.0, "FL": np.pi, "RR": np.pi}
FRONT = {"FR": 1, "FL": 1, "RR": -1, "RL": -1}
LEFT = {"FR": -1, "FL": 1, "RR": -1, "RL": 1}

HOME = np.array([0.0, 0.9, -1.8] * 4)


def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class Go1Gait(Env):
    name = "go1"
    reward_scale = 0.1
    n_actions = 6

    def __init__(self, obs_mode: str = "imu4", roughness: float = 0.04,
                 freq: float = 2.0, decision_dt: float = 0.25, episode_s: float = 10.0,
                 a_thigh: float = 0.25, a_calf: float = 0.35, ext: float = 0.15,
                 thigh_sign: float = 1.0, phase_lock: bool = False,
                 friction_range: tuple | None = None, rate_penalty: float = 0.0,
                 switch_penalty: float = 0.0, fall_penalty: float = 5.0):
        if phase_lock:
            decision_dt = 1.0 / (2.0 * freq)
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.obs_mode = obs_mode
        self.roughness0 = roughness
        self.roughness = roughness
        self.freq, self.decision_dt = freq, decision_dt
        self.max_steps = int(round(episode_s / decision_dt))
        self.a_thigh, self.a_calf, self.ext = a_thigh, a_calf, ext
        self.thigh_sign = thigh_sign
        self.ctrl_dt = 0.02                                   # CPG update rate 50 Hz
        if phase_lock:
            # the half cycle must be a whole number of CPG ticks, or the decision phase
            # drifts: 0.25 s = 25 ticks of 0.01 s (each 5 physics steps of 0.002 s)
            self.ctrl_dt = 0.01
        self.n_sub = int(round(self.ctrl_dt / self.model.opt.timestep))
        self.n_ctrl = int(round(decision_dt / self.ctrl_dt))
        if phase_lock and abs(self.n_ctrl * self.ctrl_dt - decision_dt) > 1e-9:
            raise ValueError("phase_lock needs 1/(2 freq) to be a multiple of 0.01 s")
        self.hf = self.model.hfield("terrain").id
        self.hf_nrow = int(self.model.hfield_nrow[self.hf])
        self.hf_ncol = int(self.model.hfield_ncol[self.hf])
        if obs_mode == "imu4":
            self.obs_lo = np.array([-0.5, -0.5, -3.0, -3.0])
            self.obs_hi = np.array([0.5, 0.5, 3.0, 3.0])
        elif obs_mode == "imu6":
            self.obs_lo = np.array([-0.5, -0.5, -3.0, -3.0, -0.5, 0.15])
            self.obs_hi = np.array([0.5, 0.5, 3.0, 3.0, 1.5, 0.40])
        elif obs_mode == "imu4p":
            self.obs_lo = np.array([-0.4, -0.4, -1.5, -1.5, 0.0])
            self.obs_hi = np.array([0.4, 0.4, 1.5, 1.5, 1.0])
        else:
            raise ValueError(obs_mode)
        self.phase_lock = phase_lock
        self.friction_range = friction_range
        self.rate_penalty, self.switch_penalty, self.fall_penalty = rate_penalty, switch_penalty, fall_penalty
        self._terrain_geom = self.model.geom("terrain").id
        self._friction0 = self.model.geom_friction[self._terrain_geom].copy()
        self._imu_sum = np.zeros(4)
        self._imu_n = 0
        self._last_action = 0
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.phase = 0.0

    # ---------------- terrain ----------------
    def set_shift(self, level: float) -> None:
        self.roughness = self.roughness0 * (1.0 + level)

    def _make_terrain(self):
        nr, nc = self.hf_nrow, self.hf_ncol
        noise = self.rng.uniform(0.0, 1.0, (nr, nc))
        # smooth with a separable box filter so bumps are ~0.25 m wide
        k = 5
        ker = np.ones(k) / k
        noise = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, noise)
        noise = np.apply_along_axis(lambda c: np.convolve(c, ker, mode="same"), 0, noise)
        noise = (noise - noise.min()) / max(noise.max() - noise.min(), 1e-9)
        # flat start pad: the field spans x in [-2, 14] m; keep x < 0.5 m flat
        xs = np.linspace(-2.0, 14.0, nc)
        noise[:, xs < 0.5] = 0.0
        self.model.hfield_data[:] = noise.ravel()
        self.model.hfield_size[self.hf, 2] = max(self.roughness, 1e-4)

    # ---------------- gait ----------------
    def _targets(self, action: int):
        tgt = HOME.copy()
        a_calf = self.a_calf * (1.6 if action == 1 else 1.0)
        for i, leg in enumerate(LEGS):
            ph = self.phase + LEG_PHASE[leg]
            ext = 0.0
            if action == 2:
                ext = self.ext * (FRONT[leg] > 0)
            elif action == 3:
                ext = self.ext * (FRONT[leg] < 0)
            elif action == 4:
                ext = self.ext * (LEFT[leg] > 0)
            elif action == 5:
                ext = self.ext * (LEFT[leg] < 0)
            tgt[3 * i + 1] += self.thigh_sign * self.a_thigh * np.cos(ph) - 0.5 * ext
            tgt[3 * i + 2] += -a_calf * max(0.0, np.sin(ph)) + ext
        return tgt

    def _imu(self):
        d = self.data
        roll, pitch, _ = quat_to_rpy(d.qpos[3:7])
        return np.array([roll, pitch, d.qvel[3], d.qvel[4]])   # angular velocity in the local frame

    def _obs(self):
        d = self.data
        imu = self._imu()
        if self.obs_mode == "imu4p":
            avg = self._imu_sum / max(1, self._imu_n) if self._imu_n else imu
            if self.phase_lock:          # decisions alternate exactly between the two pairs
                pair_next = float(self.t % 2)
            else:                        # FR+RL swing while sin(phase) > 0
                pair_next = 0.0 if np.sin(self.phase + 1e-6) >= 0 else 1.0
            return np.append(avg, pair_next)
        o = list(imu)
        if self.obs_mode == "imu6":
            o += [d.qvel[0], d.qpos[2]]
        return np.array(o)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._make_terrain()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, 0.30]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[7:] = HOME
        self.data.ctrl[:] = HOME
        mujoco.mj_forward(self.model, self.data)
        # settle on the ground before the gait starts
        mujoco.mj_step(self.model, self.data, nstep=250)
        if self.friction_range is not None:
            k = self.rng.uniform(*self.friction_range)
            self.model.geom_friction[self._terrain_geom] = self._friction0 * np.array([k, 1.0, 1.0])
        self.phase = 0.0 if self.phase_lock else self.rng.uniform(0, 2 * np.pi)
        self._imu_sum[:] = 0
        self._imu_n = 0
        self._last_action = 0
        self.t = 0
        return self._obs()

    def step(self, a: int):
        x0 = float(self.data.qpos[0])
        self._imu_sum[:] = 0
        self._imu_n = 0
        for _ in range(self.n_ctrl):
            self.data.ctrl[:] = self._targets(a)
            mujoco.mj_step(self.model, self.data, nstep=self.n_sub)
            self.phase = (self.phase + 2 * np.pi * self.freq * self.ctrl_dt) % (2 * np.pi)
            self._imu_sum += self._imu()
            self._imu_n += 1
        self.t += 1
        o = self._obs()
        roll, pitch, wx, wy = self._imu()
        fell = abs(roll) > 0.7 or abs(pitch) > 0.7 or self.data.qpos[2] < 0.12
        r = 10.0 * (float(self.data.qpos[0]) - x0) - 2.0 * (roll ** 2 + pitch ** 2)
        if self.rate_penalty:
            mean_rates = self._imu_sum[2:] / max(1, self._imu_n)
            r -= self.rate_penalty * float(mean_rates @ mean_rates)
        if self.switch_penalty and a != self._last_action:
            r -= self.switch_penalty
        self._last_action = a
        if fell:
            r -= self.fall_penalty
        return o, r, bool(fell), self.t >= self.max_steps
