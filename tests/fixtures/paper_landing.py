"""The landing problem of the paper's lander study, as a test fixture: lines 62-156 of fuel-tank/lunar.py in the
replication repository (SHA256 of that file: 669c1633a63a33fab9dae19e15af0f6ca1c662f4245e426aab5d8f8c17aef86c), copied verbatim; the
imports and H are those of that file. tests/test_sb3.py checks barrierpg.problems.landing against it step by step."""
import gymnasium as gym
import numpy as np

H = 1000


def fuel_use(action):
    a0, a1 = float(action[0]), float(action[1])
    m_power = (min(max(a0, 0.0), 1.0) + 1.0) * 0.5 if a0 > 0.0 else 0.0
    s_power = min(max(abs(a1), 0.5), 1.0) if abs(a1) > 0.5 else 0.0
    return 0.30 * m_power + 0.03 * s_power


def potential(o):
    """The potential of the benchmark's reward, from the first eight coordinates of the observation."""
    return float(-100.0 * np.hypot(o[0], o[1]) - 100.0 * np.hypot(o[2], o[3]) - 100.0 * abs(o[4]) + 10.0 * o[6] + 10.0 * o[7])


class FuelLander(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, budget, objective="failure", m=0.0, beta=0.999, multiplier=0.0, crash_fails=False, tank_until_touchdown=False, fuel_price=1.0,
                 revenue=None, progress="benchmark", aid_scale=1.0 / 12.0):
        super().__init__()
        self.crash_fails, self.tank_until_touchdown, self.fuel_price = bool(crash_fails), bool(tank_until_touchdown), float(fuel_price)
        # Formulations added on 2026-11-17 (stages formulations-*). With `revenue` set, the utility is a ledger: the revenue is
        # paid once, when the lander comes to rest on the pad, and fuel costs fuel_price per unit; nothing else enters it.
        # With progress == "aid" the benchmark's potential enters the training reward as beta * Phi(s') - Phi(s), with
        # Phi = 0 at the end of an episode, so that its discounted sum is -Phi(s_0) on every path and the objective is unchanged.
        self.revenue = None if revenue is None else float(revenue)
        self.progress, self.aid_scale = progress, float(aid_scale)
        assert progress in ("benchmark", "aid") and (self.revenue is not None or progress == "benchmark")
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
        self.episode_utility = 0.0                   # the discounted utility of the episode so far
        self.touched = False                         # whether a leg has touched the ground in an earlier frame
        self.prev_phi = potential(o)
        return self._obs(o), info

    def step(self, action):
        after_touchdown = self.tank_until_touchdown and self.touched
        if after_touchdown and self.reserve <= 0.0:
            action = np.array([-1.0, 0.0], dtype=np.float32)      # the tank is empty: both engines are off
        o, r, terminated, truncated, info = self.inner.step(action)
        g = fuel_use(action)
        if after_touchdown:
            g = min(g, max(self.reserve, 0.0))      # the last frame of thrust uses what is left
        if self.revenue is None:
            utility = float(r) - (self.fuel_price - 1.0) * g  # native reward, with fuel charged at fuel_price per unit (native: 1)
        else:
            utility = -self.fuel_price * g                    # the ledger: fuel now, the revenue at completion (below)
        self.reserve -= g
        self.used += g
        exhausted = self.reserve < 0.0 and not after_touchdown
        self.touched = self.touched or bool(o[6] > 0.5 or o[7] > 0.5)
        core = self.inner.unwrapped
        crashed = bool(terminated and not exhausted and core.game_over)
        landed = bool(terminated and not exhausted and not core.game_over and not core.lander.awake)
        lost = bool(terminated and not exhausted and not landed)          # crashed, or left the frame
        failed = exhausted or (self.crash_fails and lost)
        on_pad = bool(abs(float(o[0])) <= 0.2)                 # the pad spans the horizontal coordinates -0.2 to 0.2
        completed = bool(landed and on_pad)
        if self.revenue is not None and completed:
            utility += self.revenue
        if failed:
            if self.objective == "chance":
                reward = -self.multiplier
            else:
                reward = -self.m * (1 - self.beta ** (H - self.t)) / (1 - self.beta)
            utility = 0.0
            terminated = True
        else:
            reward = utility
        if self.progress == "aid":
            phi = 0.0 if (terminated or truncated) else potential(o)
            reward += self.aid_scale * (self.beta * phi - self.prev_phi)
            self.prev_phi = phi
        self.episode_utility += self.beta ** self.t * utility
        self.t += 1
        info = dict(fuel=g, failure=bool(failed), utility=utility, landed=landed, crashed=crashed, used=self.used, episode_utility=self.episode_utility,
                    exhausted=bool(exhausted), lost=lost, touched=self.touched, empty=bool(self.reserve <= 0.0), completed=completed)
        if terminated or truncated:
            info["episode_failed"] = bool(failed)
        return self._obs(o), float(reward), bool(terminated), bool(truncated), info

