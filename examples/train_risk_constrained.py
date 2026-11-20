"""Train a small actor with the native CPO and Gaussian CVaR updates."""
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
