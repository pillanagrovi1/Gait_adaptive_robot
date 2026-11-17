"""Cart-pole balance in MuJoCo (Stage 5).

Parameters follow the classic Barto, Sutton & Anderson (1983) setup used by
gymnasium CartPole-v1: cart 1.0 kg, pole 0.1 kg and 1.0 m long (hinged at one
end), push force +-10 N, dt = 0.02 s. Episode ends when |x| > 2.4 m or
|theta| > 12 deg, or after 500 steps. Reward +1 per step.

Observation (4-D): x, xdot, theta, thetadot.  Actions: 2 (push left/right).
Shift (adaptation): pole mass *= (1 + level).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .base import Env

XML = """
<mujoco model="cartpole">
  <option timestep="0.01" gravity="0 0 -9.8" integrator="RK4"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-5 5" damping="0"/>
      <geom name="cart" type="box" size="0.2 0.1 0.05" mass="1.0"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0"/>
        <geom name="pole" type="capsule" fromto="0 0 0 0 0 1.0" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="push" joint="slider" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""

THETA_LIMIT = 12 * np.pi / 180
X_LIMIT = 2.4


class CartPole(Env):
    name = "cartpole"
    frame_skip = 2
    max_steps = 500
    reward_scale = 0.1

    def __init__(self, n_actions: int = 2):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.n_actions = n_actions
        self.forces = np.linspace(-10.0, 10.0, n_actions)
        self.obs_lo = np.array([-X_LIMIT, -3.0, -THETA_LIMIT, -3.5])
        self.obs_hi = np.array([X_LIMIT, 3.0, THETA_LIMIT, 3.5])
        self._pole = self.model.body("pole").id
        self._mass0 = self.model.body_mass.copy()
        self._inertia0 = self.model.body_inertia.copy()
        self.rng = np.random.default_rng(0)
        self.t = 0

    def set_shift(self, level: float) -> None:
        k = 1.0 + level
        self.model.body_mass[self._pole] = self._mass0[self._pole] * k
        self.model.body_inertia[self._pole] = self._inertia0[self._pole] * k

    def _obs(self):
        d = self.data
        return np.array([d.qpos[0], d.qvel[0], d.qpos[1], d.qvel[1]])

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.rng.uniform(-0.05, 0.05, 2)
        self.data.qvel[:] = self.rng.uniform(-0.05, 0.05, 2)
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self._obs()

    def step(self, a: int):
        self.data.ctrl[0] = self.forces[a]
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        o = self._obs()
        failed = abs(o[0]) > X_LIMIT or abs(o[2]) > THETA_LIMIT
        return o, 1.0, bool(failed), self.t >= self.max_steps
