"""Unitree Go1 trotting over rough ground (Stage 7: the gait task).

Structure: a central pattern generator (CPG) produces a trot. The learner does
NOT output joint angles. Every `decision_dt` seconds it picks one of a few
discrete gait adjustments, based on what an IMU (and optionally foot contacts)
would measure. That keeps the action and state spaces small enough for a table,
a hashed linear model or a tiny network on a Cortex-M4, while the CPG does the
fast rhythmic work.

    learner (slow, ~4 Hz)  --action-->  CPG + reflexes + leg mapping (50-100 Hz)  -->  PD joints (MuJoCo, 500 Hz)
          ^                                                                                  |
          +---------- IMU: roll, pitch, roll rate, pitch rate (+ foot contacts) -------------+

CPG: each leg has a phase phi_l advancing at `freq` Hz. Diagonal pairs (FR+RL, FL+RR)
are half a cycle apart (trot). For leg phase phi_l:
    thigh = thigh0 + A_thigh * cos(phi_l) - 0.5 * ext_l        fore-aft swing
    calf  = calf0  - lift_l * max(0, sin(phi_l)) + ext_l      foot lift during swing (sin > 0)
where ext_l (leg extension) and lift_l come from the action, the persistent gait
parameters and the reflexes below.

---------------------------------------------------------------------------------
SELF-ADAPTING MECHANISMS (no learning; default off; docs/research_review_2.md)
  tegotae_sigma > 0  load-based phase feedback (Owaki et al.; Tegotae):
                       dphi_l/dt = 2 pi freq - sigma * N_l * cos(phi_l)
                     N_l = foot normal force / (body weight / 2). A loaded leg near the end of
                     stance or early in swing is held back; a leg loaded late in swing is
                     pushed into stance. Legs then keep their own phases.
  posture_gain > 0   IMU posture reflex (vestibular reflex, Kimura et al. 2007):
                       ext_l += posture_gain * (pitch * FRONT_l - roll * LEFT_l)   (clipped to +-0.2 rad)
                     from a low-passed IMU, at the CPG rate: legs on the low side extend.
  stumble = True     stumble reflex: a swing foot that touches the ground before mid-swing is
                     lifted 60% higher for the rest of that swing.
These give non-learning BASELINES the learner has to beat, and (action_mode "incremental")
parameters the learner can tune online.
---------------------------------------------------------------------------------

Action modes:
  "oneshot" (6, original): 0 nominal, 1 high step, 2 extend front, 3 extend rear,
                           4 extend left, 5 extend right; lasts one decision.
  "factored" (18): step height {nominal, high} x pitch {0, front, rear} x roll {0, left, right},
                   all applied together for one decision.
  "incremental" (7): persistent gait parameters, nudged one step at a time:
                   0 keep, 1/2 clearance +/-, 3/4 posture-reflex gain +/-, 5/6 body height +/-.
                   The three parameters (scaled to [0, 1]) are appended to the observation,
                   so the state stays Markov.

Observation modes:
  "imu4":   roll, pitch, roll rate, pitch rate (instantaneous, at the decision)
  "imu6":   imu4 + forward speed + trunk height
  "imu4p":  imu4 AVERAGED over the last decision period + "which diagonal pair swings next" bit
  "imu4pc": imu4p + foot-contact mismatch over the last decision period:
            [contact fraction of the pair that should have been in stance,
             contact fraction of the pair that should have been in swing]
            (CPG-RL found contacts necessary for learning to walk: Bellegarda & Ijspeert 2022)

phase_lock=True: the CPG starts at phase 0 and the learner decides exactly every half
cycle (decision_dt = 1 / (2 freq)), at the moment one diagonal pair lifts off. With
tegotae_sigma > 0 the leg phases drift from the clock; the pair bit then follows leg FR.

Reward per decision: 10 * forward progress (m) - 2 * (roll^2 + pitch^2), -fall_penalty on a fall.
Optional: rate_penalty * |mean roll/pitch rate|^2, switch_penalty when the action changes.
Shift (adaptation): terrain roughness scale *= (1 + level).

STATUS: pipeline-complete, NOT tuned. Reflex gains and signs were checked only by a
short probe (see docs/build_notes.md). Treat every number here as a starting point.
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
PAIR_A = (0, 3)          # FR, RL: swing while sin(phi_FR) > 0
PAIR_B = (1, 2)          # FL, RR

HOME = np.array([0.0, 0.9, -1.8] * 4)
ACTION_MODES = {"oneshot": 6, "factored": 18, "incremental": 7}
EXT_MAX = 0.2

# incremental parameters: (name, low, high, step, initial)
INC_PARAMS = (("clearance", 0.8, 1.8, 0.2, 1.0),     # x a_calf
              ("posture_gain", 0.0, 1.0, 0.2, 0.0),  # reflex gain, rad/rad
              ("height", -0.1, 0.1, 0.05, 0.0))      # all-leg extension, rad


def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class Go1Gait(Env):
    name = "go1"
    reward_scale = 0.1

    def __init__(self, obs_mode: str = "imu4", roughness: float = 0.04,
                 freq: float = 2.0, decision_dt: float = 0.25, episode_s: float = 10.0,
                 a_thigh: float = 0.25, a_calf: float = 0.35, ext: float = 0.15,
                 thigh_sign: float = 1.0, phase_lock: bool = False,
                 friction_range: tuple | None = None, rate_penalty: float = 0.0,
                 switch_penalty: float = 0.0, fall_penalty: float = 5.0,
                 action_mode: str = "oneshot", tegotae_sigma: float = 0.0,
                 posture_gain: float = 0.0, stumble: bool = False):
        if phase_lock:
            decision_dt = 1.0 / (2.0 * freq)
        if action_mode not in ACTION_MODES:
            raise ValueError(action_mode)
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.n_actions = ACTION_MODES[action_mode]
        self.roughness0 = roughness
        self.roughness = roughness
        self.freq, self.decision_dt = freq, decision_dt
        self.max_steps = int(round(episode_s / decision_dt))
        self.a_thigh, self.a_calf, self.ext = a_thigh, a_calf, ext
        self.thigh_sign = thigh_sign
        self.tegotae_sigma, self.posture_gain0, self.stumble = tegotae_sigma, posture_gain, stumble
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
        lo, hi = {
            "imu4": ([-0.5, -0.5, -3.0, -3.0], [0.5, 0.5, 3.0, 3.0]),
            "imu6": ([-0.5, -0.5, -3.0, -3.0, -0.5, 0.15], [0.5, 0.5, 3.0, 3.0, 1.5, 0.40]),
            "imu4p": ([-0.4, -0.4, -1.5, -1.5, 0.0], [0.4, 0.4, 1.5, 1.5, 1.0]),
            "imu4pc": ([-0.4, -0.4, -1.5, -1.5, 0.0, 0.0, 0.0], [0.4, 0.4, 1.5, 1.5, 1.0, 1.0, 1.0]),
        }.get(obs_mode, (None, None))
        if lo is None:
            raise ValueError(obs_mode)
        if action_mode == "incremental":
            lo, hi = lo + [0.0] * len(INC_PARAMS), hi + [1.0] * len(INC_PARAMS)
        self.obs_lo, self.obs_hi = np.array(lo), np.array(hi)
        self.phase_lock = phase_lock
        self.friction_range = friction_range
        self.rate_penalty, self.switch_penalty, self.fall_penalty = rate_penalty, switch_penalty, fall_penalty
        self._terrain_geom = self.model.geom("terrain").id
        self._friction0 = self.model.geom_friction[self._terrain_geom].copy()
        self._foot_geoms = np.array([self.model.geom(n).id for n in LEGS])
        self._half_weight = 0.5 * float(self.model.body_subtreemass[1]) * 9.81
        self._f6 = np.zeros(6)
        self._imu_sum = np.zeros(4)
        self._imu_n = 0
        self._contact_sum = np.zeros(4)
        self._stance_sum = np.zeros(4)
        self._last_action = 0
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.phase = 0.0
        self.leg_phase = np.zeros(4)
        self._reset_params()

    def _reset_params(self):
        self.params = {name: init for name, _, _, _, init in INC_PARAMS}
        if self.action_mode != "incremental":
            self.params["posture_gain"] = self.posture_gain0
        else:
            self.params["posture_gain"] = max(self.params["posture_gain"], self.posture_gain0)
        self._imu_lp = np.zeros(2)
        self._stumbled = np.zeros(4, bool)

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

    # ---------------- sensing ----------------
    def _imu(self):
        d = self.data
        roll, pitch, _ = quat_to_rpy(d.qpos[3:7])
        return np.array([roll, pitch, d.qvel[3], d.qvel[4]])   # angular velocity in the local frame

    def _foot_forces(self):
        """Normal force on each foot (N), from all its contacts."""
        f = np.zeros(4)
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            for k, g in enumerate(self._foot_geoms):
                if c.geom1 == g or c.geom2 == g:
                    mujoco.mj_contactForce(self.model, d, i, self._f6)
                    f[k] += abs(self._f6[0])
        return f

    def _phases(self):
        if self.tegotae_sigma > 0:
            return self.leg_phase
        return np.array([self.phase + LEG_PHASE[leg] for leg in LEGS])

    # ---------------- gait ----------------
    def _action_offsets(self, action: int):
        """Per-leg extension and lift multiplier for this decision's action."""
        ext = np.zeros(4)
        lift = 1.0
        if self.action_mode == "oneshot":
            if action == 1:
                lift = 1.6
            for i, leg in enumerate(LEGS):
                if action == 2:
                    ext[i] = self.ext * (FRONT[leg] > 0)
                elif action == 3:
                    ext[i] = self.ext * (FRONT[leg] < 0)
                elif action == 4:
                    ext[i] = self.ext * (LEFT[leg] > 0)
                elif action == 5:
                    ext[i] = self.ext * (LEFT[leg] < 0)
        elif self.action_mode == "factored":
            h, rest = divmod(action, 9)
            p, r = divmod(rest, 3)
            lift = 1.6 if h else 1.0
            for i, leg in enumerate(LEGS):
                if p == 1:
                    ext[i] += self.ext * (FRONT[leg] > 0)
                elif p == 2:
                    ext[i] += self.ext * (FRONT[leg] < 0)
                if r == 1:
                    ext[i] += self.ext * (LEFT[leg] > 0)
                elif r == 2:
                    ext[i] += self.ext * (LEFT[leg] < 0)
        else:   # incremental: parameters already applied in step()
            lift = self.params["clearance"]
            ext[:] = self.params["height"]
        return ext, lift

    def _targets(self, ext_act, lift_mult, forces):
        tgt = HOME.copy()
        ph = self._phases()
        ext = ext_act.copy()
        g = self.params["posture_gain"]
        if g > 0:
            roll, pitch = self._imu_lp
            for i, leg in enumerate(LEGS):
                ext[i] += np.clip(g * (pitch * FRONT[leg] - roll * LEFT[leg]), -EXT_MAX, EXT_MAX)
        for i in range(4):
            s = np.sin(ph[i])
            lift = self.a_calf * lift_mult
            if self.stumble:
                if s <= 0:
                    self._stumbled[i] = False                  # stance: reset
                elif forces[i] > 0.2 * self._half_weight and (ph[i] % (2 * np.pi)) < 0.5 * np.pi:
                    self._stumbled[i] = True                   # touched down before mid-swing
                if self._stumbled[i]:
                    lift *= 1.6
            tgt[3 * i + 1] += self.thigh_sign * self.a_thigh * np.cos(ph[i]) - 0.5 * ext[i]
            tgt[3 * i + 2] += -lift * max(0.0, s) + ext[i]
        return tgt

    def _advance_phase(self, forces):
        w = 2 * np.pi * self.freq * self.ctrl_dt
        self.phase = (self.phase + w) % (2 * np.pi)
        if self.tegotae_sigma > 0:
            N = forces / self._half_weight
            self.leg_phase = (self.leg_phase + w - self.tegotae_sigma * N * np.cos(self.leg_phase)
                              * self.ctrl_dt) % (2 * np.pi)

    # ---------------- interface ----------------
    def _pair_next(self):
        if self.tegotae_sigma > 0:
            return 0.0 if np.sin(self.leg_phase[0] + 1e-6) >= 0 else 1.0
        if self.phase_lock:          # decisions alternate exactly between the two pairs
            return float(self.t % 2)
        return 0.0 if np.sin(self.phase + 1e-6) >= 0 else 1.0     # FR+RL swing while sin > 0

    def _obs(self):
        d = self.data
        imu = self._imu()
        if self.obs_mode in ("imu4p", "imu4pc"):
            avg = self._imu_sum / self._imu_n if self._imu_n else imu
            o = list(avg) + [self._pair_next()]
            if self.obs_mode == "imu4pc":
                n = max(1, self._imu_n)
                stance_frac = self._contact_sum[self._stance_sum > 0.5 * n].mean() / n if (self._stance_sum > 0.5 * n).any() else 1.0
                swing_frac = self._contact_sum[self._stance_sum <= 0.5 * n].mean() / n if (self._stance_sum <= 0.5 * n).any() else 0.0
                o += [stance_frac, swing_frac]
        else:
            o = list(imu)
            if self.obs_mode == "imu6":
                o += [d.qvel[0], d.qpos[2]]
        if self.action_mode == "incremental":
            for name, lo, hi, _, _ in INC_PARAMS:
                o.append((self.params[name] - lo) / (hi - lo))
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
        self.leg_phase = np.array([self.phase + LEG_PHASE[leg] for leg in LEGS]) % (2 * np.pi)
        self._reset_params()
        self._imu_sum[:] = 0
        self._contact_sum[:] = 0
        self._stance_sum[:] = 0
        self._imu_n = 0
        self._last_action = 0
        self.t = 0
        return self._obs()

    def step(self, a: int):
        if self.action_mode == "incremental" and a > 0:
            name, lo, hi, step, _ = INC_PARAMS[(a - 1) // 2]
            self.params[name] = float(np.clip(self.params[name] + (step if a % 2 else -step), lo, hi))
        ext_act, lift_mult = self._action_offsets(a)
        x0 = float(self.data.qpos[0])
        self._imu_sum[:] = 0
        self._contact_sum[:] = 0
        self._stance_sum[:] = 0
        self._imu_n = 0
        need_forces = self.tegotae_sigma > 0 or self.stumble or self.obs_mode == "imu4pc"
        forces = np.zeros(4)
        for _ in range(self.n_ctrl):
            if self.params["posture_gain"] > 0:
                imu = self._imu()
                self._imu_lp += 0.3 * (imu[:2] - self._imu_lp)     # ~3 Hz low-pass at 100 Hz
            if need_forces:
                forces = self._foot_forces()
            self._stance_sum += np.sin(self._phases()) <= 0
            self.data.ctrl[:] = self._targets(ext_act, lift_mult, forces)
            mujoco.mj_step(self.model, self.data, nstep=self.n_sub)
            self._advance_phase(forces)
            self._imu_sum += self._imu()
            self._contact_sum += forces > 0.05 * self._half_weight
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
