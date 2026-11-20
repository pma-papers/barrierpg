from math import exp, log, pi, sqrt

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from barrierpg import categorical_action_divergence, parameter_potential
from barrierpg.on_policy import CostVariance, OnPolicyBatch, cpo_policy_update, cvar_policy_update, gaussian_tail_weight


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, observations):
        logits = self.logit.expand(len(observations))
        return torch.distributions.Categorical(logits=torch.stack((torch.zeros_like(logits), logits), dim=1))


def batch():
    observations = torch.zeros(4, 1, dtype=torch.float64)
    actions = torch.tensor([0, 0, 1, 1])
    costs = torch.tensor([0., 0., 0., 2.], dtype=torch.float64)
    rewards = actions.double()
    variance = (costs - costs.mean()).square()
    return (
        OnPolicyBatch(observations, actions, rewards - rewards.mean(), costs - costs.mean(), float(costs.mean())),
        CostVariance(float(variance.mean()), variance - variance.mean()),
    )


def test_cpo_native_actor_uses_one_accepted_surrogate_step():
    actor = Actor()
    b, _ = batch()
    out = cpo_policy_update(actor, b, cost_limit=0.55, kl_limit=0.03)
    assert out.update.status == "accepted"
    p = float(actor.logit.detach().sigmoid())
    # On this balanced batch the mean-cost surrogate is exactly p and the
    # KL constraint is KL(Bernoulli(1/2), Bernoulli(p)).
    assert 0.5 < p <= 0.55
    assert out.risk_after == pytest.approx(p)
    assert out.utility_gain == pytest.approx(p - 0.5)
    assert out.mean_kl == pytest.approx(-0.5*log(4*p*(1-p)))


def test_gaussian_cvar_native_actor_uses_both_surrogates():
    actor = Actor()
    b, variance = batch()
    # At tail mass 1/2 the exact Gaussian coefficient is sqrt(2/pi).
    coef = sqrt(2/pi)
    out = cvar_policy_update(actor, b, variance, cost_limit=1.3, tail_mass=0.5, kl_limit=0.03)
    assert out.update.status == "accepted"
    p = float(actor.logit.detach().sigmoid())
    assert out.risk_before == pytest.approx(0.5 + coef * sqrt(0.75))
    assert out.risk_after == pytest.approx(p + coef * sqrt(0.25 + p))
    assert out.risk_after <= 1.3


def test_regularizers_change_native_update_and_rejection_keeps_actor():
    b, variance = batch()
    mean = Actor()
    out = cpo_policy_update(
        mean, b, cost_limit=3, kl_limit=0.03,
        action_penalty=lambda obs, dist: 4 * categorical_action_divergence(dist.probs, [0.95, 0.05]).mean(),
    )
    assert out.update.status == "accepted" and float(mean.logit.detach()) < 0

    tail = Actor()
    out = cvar_policy_update(
        tail, b, variance, cost_limit=3, tail_mass=0.25, kl_limit=0.03,
        parameter_penalty=lambda w: 4 * parameter_potential(w + 1, form="pseudo_huber"),
    )
    assert out.update.status == "accepted" and float(tail.logit.detach()) < 0

    actor = Actor()
    out = cpo_policy_update(actor, b, cost_limit=3, kl_limit=0.03, damping=0, backtracks=0,
                            parameter_penalty=lambda w: 1e4*parameter_potential(w))
    assert out.update.status == "rejected"
    np.testing.assert_array_equal(out.update.step, [0])
    assert float(actor.logit.detach()) == 0.0


def test_gaussian_tail_weight_matches_normal_tail_integral():
    from scipy.integrate import quad
    from scipy.stats import norm

    for mass in (0.01, 0.2, 0.5, 0.75, 1.0):
        boundary = norm.isf(mass)
        integral = quad(lambda z: z * exp(-z*z/2)/sqrt(2*pi), boundary, np.inf)[0] / mass
        assert gaussian_tail_weight(mass) == pytest.approx(integral)
