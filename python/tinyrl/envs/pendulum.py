"""Pendulum swing-up in MuJoCo (Stage 2).

Physics follow gymnasium's Pendulum-v1 so numbers are comparable to the
literature: a uniform rod (m = 1 kg, l = 1 m) on a frictionless hinge,
torque limited to +-2 N m, dt = 0.05 s, 200 steps per episode.
The torque limit is too small to lift the rod directly, so the agent must
learn to swing.

theta = 0 is upright. Reward per step: -(theta^2 + 0.1 thetadot^2 + 0.001 u^2),
in [-16.3, 0]. reward_scale = 1/16 maps it to about [-1, 0].

Observation modes (the "state dimension" sweep):
    "angle" : (theta, thetadot)               2-D
    "trig"  : (cos theta, sin theta, thetadot) 3-D
Actions: 3 torques {-2, 0, +2}  (or 5 with n_actions=5).

Shift (adaptation): rod mass *= (1 + level).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .base import Env, wrap_angle

XML = """
<mujoco model="pendulum">
  <option timestep="0.01" gravity="0 0 -10" integrator="RK4"/>
  <worldbody>
    <body name="pole" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 1" size="0.02" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""


class Pendulum(Env):
    name = "pendulum"
    frame_skip = 5
    max_steps = 200
    reward_scale = 1.0 / 16.0
    max_speed = 8.0

    def __init__(self, obs_mode: str = "angle", n_actions: int = 3):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.obs_mode = obs_mode
        self.n_actions = n_actions
        self.torques = np.linspace(-2.0, 2.0, n_actions)
        self._mass0 = self.model.body_mass.copy()
        self._inertia0 = self.model.body_inertia.copy()
        if obs_mode == "angle":
            self.obs_lo = np.array([-np.pi, -self.max_speed])
            self.obs_hi = np.array([np.pi, self.max_speed])
        elif obs_mode == "trig":
            self.obs_lo = np.array([-1.0, -1.0, -self.max_speed])
            self.obs_hi = np.array([1.0, 1.0, self.max_speed])
        else:
            raise ValueError(obs_mode)
        self.rng = np.random.default_rng(0)
        self.t = 0

    def set_shift(self, level: float) -> None:
        k = 1.0 + level
        self.model.body_mass[:] = self._mass0 * k
        self.model.body_inertia[:] = self._inertia0 * k

    def _obs(self):
        th = wrap_angle(float(self.data.qpos[0]))
        thd = float(self.data.qvel[0])
        if self.obs_mode == "angle":
            return np.array([th, thd])
        return np.array([np.cos(th), np.sin(th), thd])

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = self.rng.uniform(-np.pi, np.pi)
        self.data.qvel[0] = self.rng.uniform(-1.0, 1.0)
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self._obs()

    def step(self, a: int):
        u = float(self.torques[a])
        th = wrap_angle(float(self.data.qpos[0]))
        thd = float(self.data.qvel[0])
        cost = th ** 2 + 0.1 * thd ** 2 + 0.001 * u ** 2
        self.data.ctrl[0] = u
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        # speed limit as in Pendulum-v1
        self.data.qvel[0] = np.clip(self.data.qvel[0], -self.max_speed, self.max_speed)
        self.t += 1
        return self._obs(), -cost, False, self.t >= self.max_steps
