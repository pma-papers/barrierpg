"""The four tutorial problems as campaign definitions: environment, requirement, PPO settings and configurations.

A configuration fixes everything a run needs: the stage (price, lump sum or adaptive multiplier), the number of
training steps, the regularizers, and where the policy starts (from scratch, from a fitted incumbent, or from the
policy of another configuration). run.py trains one (problem, configuration, seed) and writes its record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from barrierpg import Stage


@dataclass
class PPOSettings:
    n_envs: int = 16
    n_steps: int = 512
    batch_size: int = 1024
    n_epochs: int = 10
    lr: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    net_arch: tuple = (64, 64)
    log_std_init: float = -1.0
    norm_obs: bool = False
    norm_reward: bool = True


@dataclass
class Config:
    stage: Stage
    steps: int
    note: str = ""
    dual: Optional[dict] = None          # dict(alpha=..., step=..., multiplier0=...): adaptive multiplier
    anchor: Optional[float] = None       # kappa_p of a ParameterAnchor to the starting parameters
    anchor_decoupled: bool = False       # the shipped records were trained with the coupled pull (the paper's form); True: the toolbox default
    noise: Optional[float] = None        # kappa1 of ParameterNoise
    action_anchor: Optional[float] = None  # eta of an ActionAnchor to the starting policy (requires init)
    prior: Optional[str] = None          # 'incumbent': fit the actor to the problem's incumbent before training
    init: Optional[str] = None           # name of a configuration whose trained policy (same seed) is the start
    env_kw: dict = field(default_factory=dict)
    eval_episodes: int = 2000
    eval_kw: List[dict] = field(default_factory=list)   # extra evaluations, e.g. [dict(true_budget_scale=0.9)]


@dataclass
class Problem:
    name: str
    make_env: Callable                   # make_env(stage, **env_kw) -> BudgetWrapper
    spec: object
    T: int                               # requirement horizon: p_T over T + 1 actions
    ppo: PPOSettings
    configs: Dict[str, Config]
    summarize: Optional[Callable] = None # summarize(list of info dicts of one episode) -> dict of extras
    incumbent: Optional[Callable] = None # () -> (mean_action(obs) -> action, prior log_std, policy whose states the fit covers)


# --- the retailer ---------------------------------------------------------------------------------------------------
def _retailer() -> Problem:
    from barrierpg.problems.inventory import Retailer, make_inventory_env, BaseStockRule
    r = Retailer()

    def lump(F):
        return Stage(m=0.0, penalty="multiplier", multiplier=float(F), clip_utility=False)

    def summarize(infos):
        sales = np.sum([i["sales"] for i in infos], axis=0); lost = np.sum([i["lost_sales"] for i in infos], axis=0)
        return dict(fill_rate=(sales / np.maximum(sales + lost, 1e-9)).tolist(), final_cash=float(infos[-1]["cash"]),
                    min_cash=float(min(i["cash"] for i in infos)))

    def incumbent():
        # the fit covers the states a noisy version of the rule visits, low cash included, where the rule's cash check acts
        rule = BaseStockRule(r, cover=2.0, reserve=300.0)
        return rule.mean_action, float(np.log(0.03)), BaseStockRule(r, cover=2.0, reserve=300.0, std=0.2, seed=3)

    steps = 10_000_000               # the learning curves were still rising after 3 and 6 million steps
    configs = {
        "unpriced": Config(lump(0.0), steps, "no charge for insolvency"),
        "lump-2000": Config(lump(2000.0), steps, "insolvency costs 2,000"),
        "lump-5000": Config(lump(5000.0), steps, "insolvency costs 5,000"),
        "lump-20000": Config(lump(20000.0), steps, "insolvency costs 20,000"),
        "adaptive-1pct": Config(lump(0.0), steps, "tolerance 1%: adaptive multiplier, step from the level 20,000 that attains about 1%",
                                dual=dict(alpha=0.01, step=10000.0, multiplier0=0.0)),
        "lump-20000-long": Config(lump(20000.0), 3_000_000, "insolvency costs 20,000, continued for 3 million more steps",
                                  init="lump-20000"),
        "adaptive-1pct-warm": Config(lump(20000.0), 3_000_000, "tolerance 1%: adaptive multiplier started from the policy and level of "
                                     "the 20,000 run, with a twentieth of that level as its step", init="lump-20000",
                                     dual=dict(alpha=0.01, step=1000.0, multiplier0=20000.0)),
        "prior-only": Config(lump(5000.0), steps, "starts from the incumbent, no anchor", prior="incumbent"),
        "anchor-1": Config(lump(5000.0), steps, "starts from the incumbent, anchored (kappa_p = 1)", prior="incumbent", anchor=1.0),
        "anchor-10": Config(lump(5000.0), steps, "starts from the incumbent, anchored (kappa_p = 10)", prior="incumbent", anchor=10.0),
    }
    return Problem("retailer", lambda stage, **kw: make_inventory_env(stage, r, **kw), r.spec(), r.weeks - 1,
                   PPOSettings(n_envs=16, n_steps=520, batch_size=1040, gamma=r.beta), configs, summarize, incumbent)


# --- the robot on a battery -----------------------------------------------------------------------------------------
def _robot() -> Problem:
    from barrierpg.problems.energy import Battery, make_robot_env
    b = Battery()

    def price(m):
        return Stage(m=float(m), clip_utility=False)

    def summarize(infos):
        return dict(distance=float(infos[-1].get("x_position", 0.0)), energy=float(sum(i["energy"] for i in infos)),
                    efficiency=float(infos[-1]["efficiency"]))

    steps = 2_000_000
    weak = [dict(true_budget_scale=0.9)]           # the battery holds 10% less than the robot believes
    configs = {
        "no-battery": Config(price(0.0), steps, "a gait trained without a battery", env_kw=dict(battery=Battery(charge=1e9)),
                             eval_kw=[dict(battery=b)]),
        "unpriced": Config(price(0.0), steps, "battery, no price", eval_kw=weak),
        "price-1": Config(price(1.0), steps, "battery, price 1", eval_kw=weak),
        "price-3": Config(price(3.0), steps, "battery, price 3", eval_kw=weak),
        "price-10": Config(price(10.0), steps, "battery, price 10", eval_kw=weak),
        "barrier": Config(Stage(m=3.0, theta=0.5, clip_utility=False), steps, "battery, price 3 with an interior barrier (theta = 0.5)",
                          eval_kw=weak),
        "unpriced-no-clock": Config(price(0.0), steps, "battery, no price, without the elapsed time in the observation",
                                    env_kw=dict(observe_time=False)),
        "price-3-no-clock": Config(price(3.0), steps, "battery, price 3, without the elapsed time in the observation",
                                   env_kw=dict(observe_time=False)),
    }
    for c in configs.values():
        c.eval_episodes = 500
    return Problem("robot", lambda stage, battery=b, **kw: make_robot_env(stage, battery, **kw), b.spec(), b.mission - 1,
                   PPOSettings(n_envs=8, n_steps=1024, batch_size=256, n_epochs=10, gamma=b.beta, norm_obs=True,
                               log_std_init=-0.5), configs, summarize)


# --- the lander with a fuel tank ------------------------------------------------------------------------------------
def _lander() -> Problem:
    from barrierpg.problems.lander import Tank, make_lander_env
    tank = Tank()

    def price(m):
        return Stage(m=float(m), clip_utility=False)

    def summarize(infos):
        last = infos[-1]
        return dict(landed=bool(last.get("landed", False)), crashed=bool(last.get("crashed", False)),
                    fuel=float(sum(i["cost"][0] for i in infos)))

    fine = 3_000_000
    configs = {
        # the controller in use observes a fuel gauge against the tank but never runs dry while it is trained (the failure is
        # judged against a practically unlimited tank), so the gauge is a familiar input when the real tank arrives
        "controller": Config(price(0.0), 5_000_000, "the controller in use: trained with a fuel gauge but no limit",
                             env_kw=dict(true_budget_scale=1e6), eval_kw=[dict(true_budget_scale=1.0)]),
        "tank-unpriced": Config(price(0.0), fine, "the tank imposed, no price", init="controller"),
        "price-0.25": Config(price(0.25), fine, "tank, price 0.25", init="controller"),
        "price-0.5": Config(price(0.5), fine, "tank, price 0.5", init="controller"),
        "price-1": Config(price(1.0), fine, "tank, price 1", init="controller"),
        "price-2": Config(price(2.0), fine, "tank, price 2", init="controller"),
        "anchor-0.01": Config(price(0.5), fine, "tank, price 0.5, anchored to the controller (kappa_p = 0.01)", init="controller", anchor=0.01),
        "anchor-0.1": Config(price(0.5), fine, "tank, price 0.5, anchored to the controller (kappa_p = 0.1)", init="controller", anchor=0.1),
        "anchor-1": Config(price(0.5), fine, "tank, price 0.5, anchored to the controller (kappa_p = 1)", init="controller", anchor=1.0),
        "anchor-10": Config(price(0.5), fine, "tank, price 0.5, anchored to the controller (kappa_p = 10)", init="controller", anchor=10.0),
        "action-anchor": Config(price(0.5), fine, "tank, price 0.5, action relative entropy to the controller (eta = 0.05)",
                                init="controller", action_anchor=0.05),
    }
    for c in configs.values():
        c.eval_episodes = 1000
    return Problem("lander", lambda stage, tank=tank, **kw: make_lander_env(stage, tank, **kw), tank.spec(), tank.horizon - 1,
                   PPOSettings(n_envs=16, n_steps=1024, batch_size=64, n_epochs=4, gamma=tank.beta, log_std_init=0.0,
                               norm_reward=False), configs, summarize)


# --- the navigation robot with a damage allowance (Python 3.10 environment with Safety-Gymnasium) --------------------
def _navigation() -> Problem:
    from barrierpg.problems.navigation import Allowance, make_navigation_env
    allow = Allowance()

    def price(m):
        return Stage(m=float(m), clip_utility=False)

    def summarize(infos):
        return dict(cost=float(sum(i["cost"][0] for i in infos)), ret=float(sum(i["utility"] for i in infos)))

    steps = 2_000_000
    configs = {
        "unpriced": Config(price(0.0), steps, "no price"),
        "price-0.003": Config(price(0.003), steps, "price 0.003"),
        "price-0.01": Config(price(0.01), steps, "price 0.01"),
        "price-0.03": Config(price(0.03), steps, "price 0.03"),
        "price-0.1": Config(price(0.1), steps, "price 0.1"),
    }
    for c in configs.values():
        c.eval_episodes = 300
    return Problem("navigation", lambda stage, allowance=allow, **kw: make_navigation_env(stage, allowance, **kw), allow.spec(),
                   allow.horizon - 1, PPOSettings(n_envs=8, n_steps=1024, batch_size=64, n_epochs=4, gamma=allow.beta,
                                                  log_std_init=0.0, norm_obs=True), configs, summarize)


PROBLEMS: Dict[str, Callable[[], Problem]] = {"retailer": _retailer, "robot": _robot, "lander": _lander, "navigation": _navigation}


def get(name: str) -> Problem:
    return PROBLEMS[name]()
