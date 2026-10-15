"""The fuel-tank environment of the paper's study, as a test fixture: lines 36-96 of fuel-tank/lunar.py in the replication
repository (SHA256 of that file: 896b1fbe1b81b9971b158950f6209e79313d3294e2289e7017a6b4c6b06a5852), copied verbatim.
tests/test_sb3.py checks that BudgetWrapper reproduces it step by step."""
import gymnasium as gym
import numpy as np

H = 1000


def fuel_use(action):
    a0, a1 = float(action[0]), float(action[1])
    m_power = (min(max(a0, 0.0), 1.0) + 1.0) * 0.5 if a0 > 0.0 else 0.0
    s_power = min(max(abs(a1), 0.5), 1.0) if abs(a1) > 0.5 else 0.0
    return 0.30 * m_power + 0.03 * s_power


class FuelLander(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, budget, objective="failure", m=0.0, beta=0.999, multiplier=0.0):
        super().__init__()
        self.inner = gym.make("LunarLanderContinuous-v3")
        self.budget, self.objective, self.m, self.beta = float(budget), objective, float(m), beta
        self.multiplier = float(multiplier)
        self.action_space = self.inner.action_space
        low = np.append(self.inner.observation_space.low, 0.0).astype(np.float32)
        high = np.append(self.inner.observation_space.high, 1.0).astype(np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def set_multiplier(self, value):
        self.multiplier = float(value)

    def _obs(self, o):
        return np.append(o, max(self.reserve, 0.0) / self.budget).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        o, info = self.inner.reset(seed=seed)
        self.reserve, self.t, self.used = self.budget, 0, 0.0
        return self._obs(o), info

    def step(self, action):
        o, r, terminated, truncated, info = self.inner.step(action)
        g = fuel_use(action)
        utility = float(r)                          # native reward, fuel charge included
        self.reserve -= g
        self.used += g
        failed = self.reserve < 0.0
        if failed:
            if self.objective == "chance":
                reward = -self.multiplier
            else:
                reward = -self.m * (1 - self.beta ** (H - self.t)) / (1 - self.beta)
            utility = 0.0
            terminated = True
        else:
            reward = utility
        self.t += 1
        core = self.inner.unwrapped
        landed = bool(terminated and not failed and not core.game_over and not core.lander.awake)
        crashed = bool(terminated and not failed and core.game_over)
        info = dict(fuel=g, failure=bool(failed), utility=utility, landed=landed, crashed=crashed, used=self.used)
        if terminated or truncated:
            info["episode_failed"] = bool(failed)
        return self._obs(o), float(reward), bool(terminated), bool(truncated), info


