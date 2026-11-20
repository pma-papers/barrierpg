"""Paired wear-allowance pilot with a common actor and trust-region solver.

Run from bpg-toolbox with Torch installed:
    python examples/benchmark_wear.py --output examples/wear_comparison.json
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from barrierpg import upper_cvar
from barrierpg.examples.wear import WearModel
from barrierpg.on_policy import CostVariance, OnPolicyBatch, cpo_policy_update, cvar_policy_update
from barrierpg.validation import binomial_upper


MODEL = WearModel()
RATES = np.array([0.08, 0.35, 0.8])
SETTINGS = (
    ("fixed price m=0.10", "price", 0.10),
    ("fixed price m=0.30", "price", 0.30),
    ("fixed price m=1.00", "price", 1.00),
    ("CPO q <= 0.10", "chance", 0.10),
    ("log barrier, m=0.30", "barrier_price", 0.30),
    ("CPO + log, q <= 0.10", "barrier_chance", 0.10),
    ("Gaussian CVaR <= 0.16", "tail", 0.16),
    # Sensitivity check added after the nominal-cap pilot.
    ("CPO q <= 0.08", "chance", 0.08),
    ("CPO + log, q <= 0.08", "barrier_chance", 0.08),
    ("log barrier, m=0.45", "barrier_price", 0.45),
)
INTERIOR_STRENGTH = 0.10
CLIP_PRICE = 0.30
TAIL_MASS = 0.20
KL_LIMIT = 0.015
UPDATES = 18
TRAIN_EPISODES = 768
TEST_EPISODES = 10000
SEEDS = tuple(range(6))
EVALUATION_STREAM = 8021


class WearActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor([[0., 0., -1.0], [0., 0., -3.0]], dtype=torch.float64))

    def forward(self, observations):
        logits = observations @ self.weights.T
        return Categorical(logits=torch.cat((torch.zeros_like(logits[:, :1]), logits), dim=1))


def collect(actor, rng, episodes, *, training):
    """Simulate the wear process with independent maintenance access.

    Draw exogenous values for every episode at each active time. Resetting
    the seed gives paired shocks and action uniforms even when policies
    terminate at different times.
    """
    price = np.zeros(episodes)
    reserve = np.full(episodes, MODEL.budget)
    total_utility = np.zeros(episodes)
    failed = np.zeros(episodes, dtype=bool)
    overshoot = np.zeros(episodes)
    interior_loss = np.zeros(episodes)
    min_reserve = reserve.copy()
    alive = np.ones(episodes, dtype=bool)
    rows = []
    decisions = 0
    while alive.any():
        active = np.flatnonzero(alive)
        decisions += active.size
        obs = torch.from_numpy(np.column_stack((price[active], reserve[active] / MODEL.budget,
                                                 np.ones(active.size))))
        with torch.no_grad():
            probabilities = actor(obs).probs.numpy()
        uniforms = rng.random(episodes)
        action = (uniforms[active, None] > probabilities.cumsum(axis=1)[:, :-1]).sum(axis=1)
        rate = RATES[action]
        wear = rng.gamma(MODEL.wear_shape, 1 / MODEL.wear_shape, size=episodes)
        price_draw = rng.uniform(-1, 1, episodes)
        access = rng.random(episodes)
        reserve[active] -= MODEL.wear_rate * rate * wear[active]
        crossing = reserve[active] < 0
        utility = np.where(crossing, 0.0, MODEL.utility(price[active], rate))
        with np.errstate(divide="ignore"):
            safe_reserve = np.where(crossing, np.inf, reserve[active])
            barrier = INTERIOR_STRENGTH * np.log1p(MODEL.budget / safe_reserve)
        loss = np.minimum(barrier, utility + CLIP_PRICE)
        total_utility[active] += utility
        interior_loss[active] += loss
        min_reserve[active] = reserve[active]
        failed[active] = crossing
        overshoot[active] = np.maximum(-reserve[active], 0.0)
        price[active] = MODEL.rho * price[active] + (1 - MODEL.rho) * price_draw[active]
        alive[active] = ~crossing & (access[active] < MODEL.beta)
        if training:
            rows.append((active, obs, action, utility, loss))
    return total_utility, failed, overshoot, interior_loss, min_reserve, decisions, rows


def batch_from_rollouts(actor, rng, method):
    _, failure, overshoot, _, _, decisions, rows = collect(actor, rng, TRAIN_EPISODES, training=True)
    cost = overshoot if method == "tail" else failure.astype(float)
    reward_to_go = np.zeros(TRAIN_EPISODES)
    reward_samples = []
    for active, _, _, reward, loss in reversed(rows):
        reward_to_go[active] += reward - loss if method in {"barrier_price", "barrier_chance"} else reward
        reward_samples.append(reward_to_go[active].copy())
    targets = np.concatenate(reward_samples[::-1])
    obs = torch.cat([o for _, o, _, _, _ in rows])
    actions = torch.from_numpy(np.concatenate([a for _, _, a, _, _ in rows]))
    costs = np.concatenate([cost[active] for active, *_ in rows])
    cost_mean = float(cost.mean())
    scale = decisions / TRAIN_EPISODES
    batch = OnPolicyBatch(obs, actions, torch.from_numpy(targets - targets.mean()),
                          torch.from_numpy(costs - costs.mean()), cost_mean, scale)
    if method == "tail":
        centered_square = (costs - cost_mean) ** 2
        variance = CostVariance(float(np.mean((cost - cost_mean) ** 2)),
                                torch.from_numpy(centered_square - centered_square.mean()), scale)
    else:
        variance = None
    return batch, variance, decisions


def train_and_evaluate(seed, label, method, setting):
    actor = WearActor()
    training_seeds = np.random.SeedSequence(seed).spawn(UPDATES)
    training_steps = 0
    accepted = 0
    for epoch_seed in training_seeds:
        batch, variance, decisions = batch_from_rollouts(actor, np.random.default_rng(epoch_seed), method)
        training_steps += decisions
        if method == "tail":
            result = cvar_policy_update(actor, batch, variance, cost_limit=setting,
                                        kl_limit=KL_LIMIT, tail_mass=TAIL_MASS)
        else:
            if method in {"price", "barrier_price"}:
                price = setting / (1 - MODEL.beta)
                batch = replace(batch, reward_advantage=batch.reward_advantage - price * batch.cost_advantage,
                                cost_advantage=torch.zeros_like(batch.cost_advantage), cost_mean=0.0)
            result = cpo_policy_update(actor, batch, cost_limit=setting, kl_limit=KL_LIMIT)
        accepted += result.update.status != "rejected"

    evaluation_seed = np.random.SeedSequence([seed, EVALUATION_STREAM])
    utility, failure, overshoot, loss, reserve, evaluation_steps, _ = collect(
        actor, np.random.default_rng(evaluation_seed), TEST_EPISODES, training=False,
    )
    mean_overshoot = float(np.mean(overshoot))
    return {
        "setting": label,
        "method": method,
        "training_price_or_limit": setting,
        "seed": seed,
        "stopped_utility": float(utility.mean()),
        "net_utility_with_log": float((utility - loss).mean()),
        "low_reserve_fraction": float(np.mean(reserve <= 0.05 * MODEL.budget)),
        "failures": int(failure.sum()),
        "failure_fraction": float(failure.mean()),
        "failure_upper_95": binomial_upper(int(failure.sum()), TEST_EPISODES, 0.05),
        "mean_overshoot": mean_overshoot,
        "overshoot_CVaR": upper_cvar(overshoot, TAIL_MASS),
        "accepted_updates": accepted,
        "training_decisions": training_steps,
        "evaluation_decisions": evaluation_steps,
        "actor_weights": actor.weights.detach().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write run settings and per-seed endpoints as JSON")
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = [train_and_evaluate(seed, *setting) for setting in SETTINGS for seed in SEEDS]
    print("Setting                     Utility     Failure    Overshoot CVaR    Decisions")
    for label, *_ in SETTINGS:
        rows = [row for row in records if row["setting"] == label]
        utility, risk, cvar, steps = (np.mean([row[key] for row in rows]) for key in
                                     ("stopped_utility", "failure_fraction", "overshoot_CVaR", "training_decisions"))
        print(f"{label:25s}  {utility:7.3f}     {risk:7.3f}        {cvar:7.3f}       {steps:9.0f}")

    if args.output:
        initial = [
            float(collect(WearActor(), np.random.default_rng(np.random.SeedSequence([seed, EVALUATION_STREAM])),
                          TEST_EPISODES, training=False)[1].mean())
            for seed in SEEDS
        ]
        results = {
            "model": vars(MODEL),
            "rates": RATES.tolist(),
            "updates": UPDATES,
            "episodes_per_update": TRAIN_EPISODES,
            "test_episodes_per_seed": TEST_EPISODES,
            "evaluation_stream": EVALUATION_STREAM,
            "kl_limit": KL_LIMIT,
            "tail_mass": TAIL_MASS,
            "interior_strength": INTERIOR_STRENGTH,
            "interior_clip_price": CLIP_PRICE,
            "initial_failure_fractions_by_seed": initial,
            "runs": records,
        }
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
