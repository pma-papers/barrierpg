"""Train a dispatch policy under a mean-cost or upper-tail cost limit.

The actor chooses between two actions. Utility increases from 0.6 to 1.0
under action one, and that action incurs an extra unit of cost with
probability one half. CPO constrains mean cost. The CVaR update constrains
the Gaussian surrogate for the worst quarter of costs. The two baseline
runs use the original utility and cost directly. A third run adds action
and parameter penalties to the CVaR objective.

Let x be the probability of action one. Expected utility is 0.6 + 0.4*x.
Cost equals 1.25 with probability x/2 and 0.25 otherwise. Both updates
improve the utility surrogate within a KL trust region. For CPO the
local constraint uses the mean of this loss. The CVaR version uses the
same local solver with the Gaussian tail statistic and its gradient.
See ../docs/learning-algorithms.md for the optimization and line search.

The printed exact CVaR is computed from the final policy's two-point cost
distribution. Use it to compare the deployment measure with the Gaussian
quantity used during the updates. With the RL extra installed, run
    python examples/train_risk_constrained.py
"""
import numpy as np
import torch
from torch.distributions import Categorical

from barrierpg import categorical_action_divergence, parameter_potential, upper_cvar
from barrierpg.on_policy import (
    CostVariance, OnPolicyBatch, cpo_policy_update, cvar_policy_update, gaussian_tail_weight,
)


class DispatchPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(-4.0, dtype=torch.float64))

    def forward(self, observations):
        logit = self.logit.expand(len(observations))
        return Categorical(logits=torch.stack((torch.zeros_like(logit), logit), dim=-1))


def rollout(policy, rng, episodes=4096):
    """One decision yields utility, with a random extra cost for action one."""
    obs = torch.zeros(episodes, 1, dtype=torch.float64)
    with torch.no_grad():
        p = policy(obs).probs[:, 1]
        actions = (torch.rand(episodes, generator=rng) < p).long()
        costs = 0.25 + (torch.rand(episodes, generator=rng) < 0.5).to(obs.dtype) * actions
        reward = 0.6 + 0.4 * actions.to(obs.dtype)
        mean = costs.mean()
        centered_square = (costs - mean).square()
        var = centered_square.mean()
    batch = OnPolicyBatch(obs, actions, reward - reward.mean(), costs - mean, float(mean))
    return batch, CostVariance(float(var), centered_square - var)


def train(mode, anchored):
    actor = DispatchPolicy()
    rng = torch.Generator().manual_seed(19)
    prior = actor.logit.detach().clone()

    def action_penalty(obs, dist):
        return 0.03 * categorical_action_divergence(dist.probs, [0.9, 0.1], form="hellinger").mean()

    def parameter_penalty(vector):
        return 0.045 * parameter_potential(vector - prior, form="pseudo_huber")

    penalties = dict(action_penalty=action_penalty, parameter_penalty=parameter_penalty) if anchored else {}
    accepted = 0
    for _ in range(24):
        batch, variance = rollout(actor, rng)
        settings = dict(cost_limit=0.62, kl_limit=0.01, **penalties)
        if mode == "CPO":
            outcome = cpo_policy_update(actor, batch, **settings)
        else:
            outcome = cvar_policy_update(actor, batch, variance, tail_mass=0.25, **settings)
        accepted += outcome.update.status != "rejected"

    p = float(actor.logit.detach().sigmoid())
    mean = 0.25 + p/2
    std = np.sqrt((p/2) * (1-p/2))
    gaussian_cvar = mean + gaussian_tail_weight(0.25) * std
    exact_cvar = upper_cvar([0.25, 1.25], 0.25, probabilities=[1 - p/2, p/2])
    return p, 0.6 + 0.4*p, mean, gaussian_cvar, exact_cvar, accepted


if __name__ == "__main__":
    torch.set_num_threads(1)
    print("Mode            P(action 1)  Utility  Mean cost  Gaussian CVaR  Exact CVaR  Updates")
    for mode, anchored in [("CPO", False), ("CVaR", False), ("CVaR", True)]:
        p, u, mean, gaussian, exact, accepted = train(mode, anchored)
        label = mode + (" + anchors" if anchored else "")
        print(f"{label:16s} {p:8.3f} {u:9.3f} {mean:10.3f} {gaussian:14.3f} {exact:11.3f} {accepted:8d}")
