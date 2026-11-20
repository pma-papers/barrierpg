"""Compare several resource-budget formulations on explicit finite examples.

The first comparison uses the same candidate utilities and consumption
scenarios for a linear price, a mean-cost cap, a chance constraint, CVaR
and pathwise feasibility. The later examples compare reserve penalties
and regularized objective gradients under a common cost constraint.
The finite-scenario comparisons use the supplied outcomes and weights.

The decision is a candidate index i. Choose argmax U[i] - price * E[Z_i]
for a mean-consumption price, or maximize U[i] subject to the risk
statistic meeting its cap. The CPO calculation then solves a local linear
objective with a linear cost constraint and a quadratic KL bound. The
worked calculations are in ../docs/formulation-comparison.md.

Run from the repository root after installing the core package:
    python examples/compare_formulations.py
"""
import numpy as np
from scipy.optimize import approx_fprime

from barrierpg import (
    InverseBarrier, LogBarrier, ReserveHinge, categorical_action_divergence,
    cpo_step, cpo_update, parameter_potential, regularized_utility, select_scenario_policy,
)


def compare_costs():
    utility = np.array([1.0, 1.1])
    deterministic = [[1, 1], [2, 2]]
    stochastic = [[1, 1], [0, 2]]
    labels = ("steady", "variable")
    price = 0.2
    constraints = (
        ("mean", {"cost_limit": 1}),
        ("chance", {"cost_limit": 0, "budget": 1}),
        ("cvar", {"cost_limit": 1, "tail_mass": 0.5}),
        ("pathwise", {"cost_limit": 1}),
    )
    for name, costs in (("deterministic", deterministic), ("stochastic", stochastic)):
        priced = np.argmax(utility - price * np.mean(costs, axis=1))
        print(f"{name}: mean consumption price selects {labels[priced]}")
        for formulation, kwargs in constraints:
            result = select_scenario_policy(utility, costs, formulation=formulation, **kwargs)
            print(f"  {formulation:8s} risks={result.risk}   selects {labels[result.index]}")

    # The 2-unit CVaR cap admits both policies. The action reference can still
    # change which one is preferred; it does not change their measured risks.
    action_cost = categorical_action_divergence([[0.9, 0.1], [0.2, 0.8]], [0.9, 0.1], form="hellinger")
    weight_cost = np.array([parameter_potential([d], form="pseudo_huber") for d in (0.0, 0.25)])
    for strength in (0.0, 0.5):
        target = regularized_utility(utility, action_penalty=action_cost, eta=strength,
                                     parameter_penalty=weight_cost, kappa=strength)
        result = select_scenario_policy(target, stochastic, formulation="cvar", cost_limit=2, tail_mass=0.5)
        print(f"  CVaR cap 2, regularizer strengths {strength:.1f}: {labels[result.index]}")

    no_anchor = cpo_step([2, 1], [1, 0], np.eye(2), -0.6, 0.5)
    # At displacement (0, 1), the second gradient component of the quadratic
    # is 1 and that of pseudo-Huber (scale 1) is 1/sqrt(2).
    quadratic = cpo_step([2, 1 - 1.2], [1, 0], np.eye(2), -0.6, 0.5)
    huber = cpo_step([2, 1 - 1.2 / np.sqrt(2)], [1, 0], np.eye(2), -0.6, 0.5)
    print(f"CPO proposed steps at the same slack and KL limit: {no_anchor.step} "
          f"(none), {quadratic.step} (quadratic), {huber.step} (pseudo-Huber)")


def compare_cpo_training():
    # A one-state Gaussian policy with fixed identity covariance. The mean is
    # trained from exact expectations, so only objective and update effects enter.
    target, reference, prior = np.array([1.0, 1.5]), np.array([0.2, 0.5]), np.zeros(2)
    cap = np.exp(0.6)

    def utility(mean):
        # Expected reward -||action - target||**2/2, up to an additive constant.
        return -0.5 * np.sum((mean - target) ** 2)

    print("CPO objective       final means      utility   expected cost")
    for form, eta, kappa in (("quadratic", 0, 0), ("quadratic", 0.3, 0.8), ("pseudo_huber", 0.3, 0.8)):
        def objective(mean):
            # Gaussian KL to a fixed reference with the same covariance.
            return regularized_utility(
                utility(mean), action_penalty=0.5 * np.sum((mean - reference) ** 2), eta=eta,
                parameter_penalty=parameter_potential(mean - prior, form=form), kappa=kappa,
            )

        mean = prior.copy()
        for _ in range(40):
            expected_cost = np.exp(mean[0])  # E exp(action[0] - 1/2)
            initial_value = objective(mean)

            def evaluate_step(step):
                candidate = mean + step
                return (objective(candidate) - initial_value, np.exp(candidate[0]) - cap, step @ step / 2)

            update = cpo_update(
                approx_fprime(mean, objective), [expected_cost, 0], np.eye(2),
                expected_cost - cap, 0.02, evaluate_step,
            )
            if update.status == "rejected":
                break
            mean = mean + update.step
        label = form if kappa else "unregularized"
        print(f"{label:17s} ({mean[0]:.3f}, {mean[1]:.3f})    {utility(mean):+.3f}       {np.exp(mean[0]):.3f}")


def compare_barriers():
    actions = np.linspace(0.01, 0.99, 99)
    probabilities = np.array([0.9, 0.1])
    consumption = actions[:, None] + [0.0, 0.18]
    nominal_reserve = 1 - consumption
    failure = consumption > 1
    utility = np.where(failure, 0, 2 * actions[:, None])
    failure_price = 1.0
    level = 0.12
    # Each penalty is level at 20% reserve, with hinges activated below 40%.
    barriers = {
        "failure only": None,
        "log": LogBarrier(level / np.log(6), [1]),
        "inverse": InverseBarrier(level * 0.2, [1]),
        "linear hinge": ReserveHinge(level / 0.2, [1], fraction=0.4),
        "quadratic hinge": ReserveHinge(level / 0.2**2, [1], fraction=0.4, power=2),
    }
    print("shape              action  utility  nominal risk  smaller budget risk  min reserve")
    for label, barrier in barriers.items():
        if barrier is None:
            interior_rewards = utility
        else:
            charges = np.array([barrier([reserve]) if reserve >= 0 else 0
                                for reserve in nominal_reserve.flat]).reshape(consumption.shape)
            interior_rewards = np.maximum(-failure_price, utility - charges)
        objective = np.where(failure, -failure_price, interior_rewards) @ probabilities
        best = np.argmax(objective)
        smaller_risk = (consumption[best] > 0.8) @ probabilities
        print(f"{label:18s} {actions[best]:.2f}    {utility[best] @ probabilities:.3f}"
              f"     {failure[best] @ probabilities:.2f}          {smaller_risk:.2f}"
              f"                 {min(nominal_reserve[best]):+.2f}")


if __name__ == "__main__":
    compare_costs()
    print()
    compare_cpo_training()
    print()
    compare_barriers()
