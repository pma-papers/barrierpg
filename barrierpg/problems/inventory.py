"""A retailer that finances its stock from cash: insolvency as the failure (numpy only).

The retailer sells three lines over a year of 52 weeks: a staple with a thin margin and steady demand, a seasonal
line with a high margin whose demand is concentrated in the weeks before the year end, and a fashion line with a high
margin and volatile demand. It pays suppliers when it orders. The staple arrives the next week, the fashion line after
four weeks and the seasonal line after eight, so the seasonal stock has to be bought before the season shows how
strong it is. Demand is random and moves with a persistent market factor that sets how good the year is; the
seasonal and fashion lines are more exposed to it than the staple. Revenue arrives as goods are sold. Holding costs
and a fixed overhead are paid every week. Seasonal stock left after the season is marked down, and whatever is left
at the end of the year is sold off below cost. The firm starts with a cash balance and has no credit line: the first
week that ends with negative cash is insolvency, and the year ends there.

A large pre-season buy captures a strong season and earns the most on average, but in a weak year it leaves the cash
in unsold goods while the overhead keeps falling due. The more seasonal stock the retailer buys, the higher both its
expected profit and its risk of insolvency; a price of failure risk decides that trade-off.

Mapping to the budget problem. The budget is the starting cash. The resource use of a week is its net cash outflow
(orders, holding and overhead minus revenue), which is negative in good weeks, so the cost passed to the wrapper is
the increment of the running maximum of the cumulative outflow (`barrierpg.RunningMaxCost`): the first week in which the
cumulative outflow exceeds the starting cash is the first week with negative cash. The utility is the week's
accounting profit (margin on goods sold, minus holding, overhead and markdown losses), which lies in [-u_minus, ubar].
The horizon is the physical year, H = 52.

InventoryCore is the raw simulator (reward = utility; info carries the signed cash outflow); make_inventory_env wraps
it. BaseStockRule is the incumbent: order up to a cover of expected demand over the lead time, never spending cash
below a reserve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from ..spec import BudgetSpec, Stage
from ..wrapper import BudgetWrapper, RunningMaxCost

try:
    import gymnasium as gym
    _Box = gym.spaces.Box
except ImportError:  # pragma: no cover
    gym = None
    _Box = None


@dataclass(frozen=True)
class Retailer:
    """Parameters of the retailer (money in hundreds of dollars, goods in units). One entry per line: staple,
    seasonal, fashion."""
    price: Tuple[float, ...] = (10.0, 30.0, 60.0)          # selling price per unit
    cost: Tuple[float, ...] = (7.0, 14.0, 30.0)            # purchase cost per unit, paid on ordering
    holding: Tuple[float, ...] = (0.05, 0.2, 0.4)          # holding cost per unit and week, on the stock at the end of the week
    demand: Tuple[float, ...] = (40.0, 12.0, 4.0)          # mean weekly demand at the seasonal average
    season: Tuple[float, ...] = (0.1, 0.95, 0.4)           # seasonal amplitude; the season peaks in week `peak`
    lead: Tuple[int, ...] = (1, 8, 4)                      # weeks between ordering and arrival
    exposure: Tuple[float, ...] = (0.2, 1.0, 1.5)          # exposure of each line's demand to the log market factor
    shape: Tuple[float, ...] = (20.0, 6.0, 3.0)            # gamma mixing of Poisson demand (lower: more dispersion)
    salvage: Tuple[float, ...] = (0.8, 0.3, 0.4)           # markdown value as a fraction of cost
    overhead: float = 200.0                                # fixed cost per week
    cash: float = 2500.0                                   # starting cash: the budget
    weeks: int = 52
    peak: int = 46
    season_end: int = 50                                   # seasonal stock is marked down after this week
    market_rho: float = 0.97                               # persistence of the log market factor (how good the year is)
    market_sd: float = 0.06                                # its weekly innovation
    max_order_weeks: float = 6.0                           # an order is at most this many weeks of peak mean demand
    beta: float = 0.999                                    # weekly utility discount

    @property
    def K(self) -> int:
        return len(self.price)

    @property
    def L(self) -> int:
        return int(max(self.lead))

    @property
    def market_stationary_sd(self) -> float:
        return self.market_sd / np.sqrt(1.0 - self.market_rho ** 2)

    def seasonal_mean(self, week) -> np.ndarray:
        phase = 2.0 * np.pi * (np.asarray(week, dtype=float) - self.peak) / self.weeks
        return np.asarray(self.demand) * (1.0 + np.asarray(self.season) * np.cos(phase))

    @property
    def max_order(self) -> np.ndarray:
        return self.max_order_weeks * np.asarray(self.demand) * (1.0 + np.asarray(self.season))

    @property
    def demand_cap(self) -> np.ndarray:
        """Demand is capped at four times its peak mean in a year three standard deviations above average."""
        return 4.0 * np.asarray(self.demand) * (1.0 + np.asarray(self.season)) * np.exp(3.0 * self.market_stationary_sd * np.asarray(self.exposure))

    @property
    def ubar(self) -> float:
        """Largest weekly profit: every capped demand met at full margin, no holding cost, the overhead paid."""
        margin = np.asarray(self.price) - np.asarray(self.cost)
        return float(margin @ self.demand_cap - self.overhead)

    @property
    def u_minus(self) -> float:
        """Largest weekly loss: the overhead, holding on the largest stock, and the markdown loss on it."""
        c, h = np.asarray(self.cost), np.asarray(self.holding)
        stock = self.max_order * (np.asarray(self.lead) + 1)
        return float(self.overhead + h @ stock + ((1.0 - np.asarray(self.salvage)) * c) @ stock)

    def spec(self) -> BudgetSpec:
        return BudgetSpec(budgets=(self.cash,), beta=self.beta, ubar=self.ubar, gammas=(1.0,), u_minus=self.u_minus,
                          horizon=self.weeks)


class InventoryCore:
    """Raw simulator. Observation (scaled to order one): stock on hand per line, goods in transit by weeks to arrival
    (L x K, zero where a line's lead time is shorter), last week's demand per line (each relative to mean demand), the
    season (cos, sin of its phase), the week as a fraction of the year, and cash relative to the starting cash. Action:
    one number per line in [-1, 1]; the order is (a + 1)/2 times the largest order. Reward: the week's profit. info:
    'cash_out' (signed net cash outflow), 'cash', 'sales', 'lost_sales', 'stock', 'order', 'revenue'."""
    metadata = {"render_modes": []}

    def __init__(self, retailer: Retailer = Retailer()):
        self.r = retailer
        self.np_random = np.random.default_rng()
        K, L = retailer.K, retailer.L
        self.n_obs = K + L * K + K + 4
        if _Box is not None:
            self.observation_space = _Box(-np.inf, np.inf, (self.n_obs,), dtype=np.float32)
            self.action_space = _Box(-1.0, 1.0, (K,), dtype=np.float32)

    def _obs(self):
        r = self.r
        scale = np.asarray(r.demand)
        phase = 2.0 * np.pi * (self.week - r.peak) / r.weeks
        return np.concatenate([self.stock / scale, (self.transit / scale).reshape(-1), self.last_demand / scale,
                               [np.cos(phase), np.sin(phase), self.week / r.weeks, self.cash_now / r.cash]]).astype(np.float32)

    def order_quantities(self, action) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        return (a + 1.0) / 2.0 * self.r.max_order

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        r = self.r
        self.week = 0
        self.log_market = self.np_random.normal(0.0, r.market_stationary_sd)
        mean0 = r.seasonal_mean(0)
        self.stock = 2.0 * mean0                                 # two weeks of cover to start with, already paid for
        self.transit = np.zeros((r.L, r.K))                      # transit[j] arrives in j + 1 weeks
        self.last_demand = mean0.copy()
        self.cash_now = float(r.cash)
        return self._obs(), {}

    def step(self, action):
        r, rng = self.r, self.np_random
        price, cost, hold = np.asarray(r.price), np.asarray(r.cost), np.asarray(r.holding)
        sv = np.asarray(r.salvage)
        q = self.order_quantities(action)
        if self.week > r.season_end - 1:                         # no seasonal orders once the season is over
            q[1] = 0.0
        # arrivals, then this week's orders enter the pipeline
        self.stock = self.stock + self.transit[0]
        self.transit = np.vstack([self.transit[1:], np.zeros((1, r.K))])
        for k in range(r.K):
            self.transit[r.lead[k] - 1, k] += q[k]
        # demand and sales
        mean = r.seasonal_mean(self.week) * np.exp(np.asarray(r.exposure) * self.log_market)
        shape = np.asarray(r.shape)
        demand = np.minimum(rng.poisson(mean * rng.gamma(shape, 1.0 / shape)), r.demand_cap)
        sales = np.minimum(demand, self.stock)
        self.stock = self.stock - sales
        revenue = float(price @ sales)
        holding = float(hold @ self.stock)
        profit = float((price - cost) @ sales) - holding - r.overhead
        cash_out = float(cost @ q) + holding + r.overhead - revenue
        # markdowns: seasonal stock after the season, everything at the end of the year
        markdown = np.zeros(r.K)
        if self.week == r.season_end:
            markdown[1] = self.stock[1] + self.transit[:, 1].sum()
        if self.week == r.weeks - 1:
            markdown = self.stock + self.transit.sum(axis=0)
        if markdown.any():
            profit -= float(((1.0 - sv) * cost) @ markdown)
            cash_out -= float((sv * cost) @ markdown)
            self.stock = np.where(markdown > 0, 0.0, self.stock)
            self.transit[:, markdown > 0] = 0.0
        self.cash_now -= cash_out
        self.last_demand = demand.astype(float)
        self.log_market = r.market_rho * self.log_market + r.market_sd * rng.standard_normal()
        self.week += 1
        last = self.week >= r.weeks
        info = dict(cash_out=cash_out, cash=self.cash_now, sales=sales, lost_sales=demand - sales, stock=self.stock.copy(),
                    order=q, revenue=revenue)
        return self._obs(), profit, False, last, info


def cash_outflow(obs, action, next_obs, reward, info):
    """Signed resource use of a week: its net cash outflow (negative when the week brings in cash)."""
    return [info["cash_out"]]


def make_inventory_env(stage: Stage, retailer: Retailer = Retailer(), *, true_budget_scale: float = 1.0) -> BudgetWrapper:
    """The retailer under the given stage. The wrapper appends the running minimum of cash relative to the starting cash,
    ends the year at insolvency, and charges the stage's price or lump sum there."""
    return BudgetWrapper(InventoryCore(retailer), RunningMaxCost(cash_outflow), retailer.spec(), stage, observe="fraction",
                         true_budget_scale=true_budget_scale)


@dataclass
class BaseStockRule:
    """The incumbent: order up to the expected demand over the lead time plus `cover` weeks on the stock position
    (stock plus goods in transit), scaling expected demand by last week's demand relative to its seasonal mean, and
    never spend cash below `reserve` (orders are scaled down to fit). `cover` is one number or one per line. Optional
    Gaussian dispersion `std` in the action variable."""
    retailer: Retailer = field(default_factory=Retailer)
    cover: object = 2.0
    reserve: float = 300.0
    std: float = 0.0
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def mean_action(self, obs) -> np.ndarray:
        r = self.retailer
        K, L = r.K, r.L
        scale = np.asarray(r.demand)
        stock = obs[:K] * scale
        transit = obs[K:K + L * K].reshape(L, K) * scale
        last = obs[K + L * K:2 * K + L * K] * scale
        week = int(round(float(obs[2 * K + L * K + 2]) * r.weeks))
        cash = float(obs[2 * K + L * K + 3]) * r.cash
        level = np.clip(last / np.maximum(r.seasonal_mean(max(week - 1, 0)), 1e-9), 0.5, 2.0)
        cover = np.broadcast_to(np.asarray(self.cover, dtype=float), (K,))
        span = np.asarray(r.lead) + cover
        weeks_ahead = np.arange(week, week + int(np.ceil(span.max())) + 1)
        ahead = np.array([r.seasonal_mean(w) if w < r.weeks else np.zeros(K) for w in weeks_ahead])
        if week + 1 <= r.season_end:                              # seasonal demand after the season is sold at markdown
            ahead[weeks_ahead > r.season_end, 1] = 0.0
        target = np.array([level[k] * (ahead[:int(np.floor(span[k])), k].sum() + (span[k] % 1.0) * ahead[int(np.floor(span[k])), k])
                           for k in range(K)])
        q = np.clip(target - stock - transit.sum(axis=0), 0.0, r.max_order)
        spend = float(np.asarray(r.cost) @ q)
        room = max(0.0, cash - self.reserve)
        if spend > room:
            q = q * (room / spend)
        return np.clip(2.0 * q / r.max_order - 1.0, -1.0, 1.0)

    def __call__(self, obs):
        a = self.mean_action(obs)
        if self.std > 0:
            a = np.clip(a + self.std * self.rng.standard_normal(a.shape), -1.0, 1.0)
        return a.astype(np.float32)

    def deterministic(self, obs):
        return self.mean_action(obs).astype(np.float32)


__all__ = ["Retailer", "InventoryCore", "cash_outflow", "make_inventory_env", "BaseStockRule"]
