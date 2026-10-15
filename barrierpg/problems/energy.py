"""A legged robot on a battery: running out of charge before the end of the mission as the failure.

The robot is Gymnasium's HalfCheetah (MuJoCo), whose task is to run forward; its utility per step is the task's own
reward (forward velocity minus a small control charge). It now carries a battery. Each step draws an idle load plus
the resistive losses of its motors, proportional to the squared torques, scaled by an efficiency factor that is drawn
once per mission and not observed: battery health and temperature vary from mission to mission, so the same gait
drains the battery faster on some days than on others. A robot whose battery runs flat before the end of its mission
of 1000 steps is stranded, and the mission ends there.

Running flat out earns the most per step and drains the battery fastest. A gait that spends the charge evenly
finishes the mission; a fast gait finishes it only on good days. The robot observes its remaining charge, so it can
slow down when the battery drains faster than expected.

Mapping to the budget problem. The budget is the battery's charge; the step cost is the energy drawn; the horizon is
the mission, H = 1000, with utility discount beta = 0.999. The task's reward has no useful bound below a price, so the
stage should use clip_utility=False (only the failure is priced). Use Gymnasium >= 1.0 with MuJoCo installed
(pip install "gymnasium[mujoco]").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..spec import BudgetSpec, Stage
from ..wrapper import BudgetWrapper

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None


@dataclass(frozen=True)
class Battery:
    """Battery and mission parameters (energy in units of one squared unit torque per step)."""
    charge: float = 2500.0               # the budget: a gait trained without a battery uses about 3100 per mission
    idle: float = 0.5                    # idle draw per step
    efficiency_sd: float = 0.15          # log-standard deviation of the per-mission efficiency factor (mean one)
    mission: int = 1000                  # steps
    beta: float = 0.999
    ubar: float = 20.0                   # assumed bound on the task's reward per step (forward speed); see the tutorial
    u_minus: float = 21.0                # its negative counterpart (running backwards at full control)
    env_id: str = "HalfCheetah-v5"

    def spec(self) -> BudgetSpec:
        return BudgetSpec(budgets=(self.charge,), beta=self.beta, ubar=self.ubar, gammas=(1.0,), u_minus=self.u_minus,
                          horizon=self.mission)


def half_cheetah(**make_kwargs):
    """Gymnasium's HalfCheetah-v5, built so that MuJoCo 3 does not warn about the model file's deprecated settotalmass
    attribute. The attribute is removed from a copy of the model file, and the total mass it set is applied with
    mujoco.mj_setTotalmass instead, as the warning advises; the masses, inertias and dynamics are identical to the stock
    environment's. If a later Gymnasium drops the attribute, the stock environment is returned unchanged."""
    import hashlib
    import os
    import re
    import tempfile
    import mujoco
    import gymnasium.envs.mujoco as gym_mujoco
    xml = open(os.path.join(os.path.dirname(gym_mujoco.__file__), "assets", "half_cheetah.xml")).read()
    found = re.search(r'settotalmass="([0-9.eE+-]+)"', xml)
    if found is None:
        return gym.make("HalfCheetah-v5", **make_kwargs)
    patched = re.sub(r'\s*settotalmass="[^"]*"', "", xml)
    path = os.path.join(tempfile.gettempdir(), f"barrierpg-half_cheetah-{hashlib.sha1(patched.encode()).hexdigest()[:12]}.xml")
    if not os.path.exists(path):                       # written once, atomically (parallel runs may start together)
        fd, tmp = tempfile.mkstemp(suffix=".xml", dir=tempfile.gettempdir())
        with os.fdopen(fd, "w") as f:
            f.write(patched)
        os.replace(tmp, path)
    env = gym.make("HalfCheetah-v5", xml_file=path, **make_kwargs)
    core = env.unwrapped
    mujoco.mj_setTotalmass(core.model, float(found.group(1)))
    mujoco.mj_setConst(core.model, core.data)
    core.init_qpos = core.data.qpos.ravel().copy()
    core.init_qvel = core.data.qvel.ravel().copy()
    return env


if gym is not None:
    class BatteryRobot(gym.Wrapper):
        """Adds the battery draw to info['energy'] and the mission's efficiency factor to info['efficiency']."""

        def __init__(self, battery: Battery = Battery(), **make_kwargs):
            env = (half_cheetah(max_episode_steps=battery.mission, **make_kwargs) if battery.env_id == "HalfCheetah-v5"
                   else gym.make(battery.env_id, max_episode_steps=battery.mission, **make_kwargs))
            super().__init__(env)
            self.battery = battery
            self._rng = np.random.default_rng()
            self.efficiency = 1.0

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._rng = np.random.default_rng([int(seed), 0x62617474])
            s = self.battery.efficiency_sd
            self.efficiency = float(np.exp(s * self._rng.standard_normal() - 0.5 * s * s))
            return self.env.reset(seed=seed, options=options)

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
            info = dict(info)
            info["energy"] = self.efficiency * (self.battery.idle + float(a @ a))
            info["efficiency"] = self.efficiency
            return obs, reward, terminated, truncated, info


def energy_use(obs, action, next_obs, reward, info):
    return [info["energy"]]


def make_robot_env(stage: Stage, battery: Battery = Battery(), *, true_budget_scale: float = 1.0,
                   observe_time: bool = True) -> BudgetWrapper:
    """The robot under the given stage. The wrapper appends the remaining charge as a fraction and the elapsed fraction
    of the mission: the task's own observation has no clock, and pacing the battery needs one (observe_time=False
    removes it, to see the difference)."""
    if gym is None:
        raise ImportError("the robot needs gymnasium with MuJoCo: pip install 'gymnasium[mujoco]'")
    return BudgetWrapper(BatteryRobot(battery), energy_use, battery.spec(), stage, observe="fraction",
                         true_budget_scale=true_budget_scale, observe_time=observe_time)


__all__ = ["Battery", "energy_use", "make_robot_env", "half_cheetah"] + (["BatteryRobot"] if gym is not None else [])
