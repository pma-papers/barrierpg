from itertools import product

import numpy as np

from barrierpg import Stage, Transition, solve_budget_dp


def test_price_matches_enumeration_with_stochastic_consumption_and_terminal_rewards():
    transitions = [
        [[Transition(1.0, 1, 1.0)], [Transition(0.7, 1, 6.0, 1), Transition(0.3, 0, 8.0, 2)]],
        [[Transition(1.0, None, 0.5)], [Transition(0.75, None, 5.0, 1), Transition(0.25, None, 7.0, 2)]],
    ]
    beta, price, horizon, budget = 0.75, 1.0, 2, 1
    solution = solve_budget_dp(transitions, horizon=horizon, budget=budget,
                               beta=beta, stage=Stage(m=price))

    def enumerate_returns(policy, t, state, reserve):
        value = utility = risk = geometric = 0.0
        for outcome in transitions[state][policy[t, state, reserve]]:
            remaining = reserve - outcome.consumption
            p = outcome.probability
            if remaining < 0:
                value -= p * sum(beta ** k * price for k in range(t, horizon))
                risk += p
                geometric += p * beta ** t
            else:
                u = p * beta ** t * outcome.utility
                value += u
                utility += u
                if outcome.next_state is not None and t + 1 < horizon:
                    v, u, r, g = enumerate_returns(policy, t + 1, outcome.next_state, remaining)
                    value += p * v
                    utility += p * u
                    risk += p * r
                    geometric += p * g
        return value, utility, risk, geometric

    start = (0, 0, budget)
    policies = (np.array(bits).reshape(2, 2, 2) for bits in product((0, 1), repeat=8))
    exact = max(enumerate_returns(policy, *start)[0] for policy in policies)
    assert np.isclose(solution.value[start], exact)
    assert np.allclose(enumerate_returns(solution.policy, *start),
                       [solution.value[start], solution.stopped_utility[start],
                        solution.failure_probability[start], solution.geometric_risk[start]])


def test_pathwise_feasibility_is_backed_up_from_later_actions():
    transitions = [
        [[Transition(1.0, None, 1.0, 1), Transition(0.0, None, 100.0, 2)],
         [Transition(1.0, 1, 10.0, 0)]],
        [[Transition(1.0, None, 0.0, 2)]],
    ]
    solution = solve_budget_dp(transitions, horizon=2, budget=1, pathwise=True)
    assert solution.policy[0, 0, 1] == 0
    assert solution.value[0, 0, 1] == 1.0
    assert solution.failure_probability[0, 0, 1] == 0.0
    assert solution.policy[1, 1, 1] == -1
    assert np.isneginf(solution.value[1, 1, 1])
    assert np.isnan(solution.failure_probability[1, 1, 1])


def test_fixed_lump_charge_and_barrier_at_exact_depletion():
    transitions = [[[Transition(0.75, None, 2.0, 1), Transition(0.25, None, 9.0, 2)]]]
    plain = Stage(m=1.0, penalty="multiplier", multiplier=4.0)
    barrier = plain.with_(theta=0.2, clip_utility=False)
    initial = (0, 0, 1)
    ordinary = solve_budget_dp(transitions, horizon=2, budget=1, stage=plain)
    clipped = solve_budget_dp(transitions, horizon=2, budget=1, stage=barrier)
    assert ordinary.value[initial] == 0.75 * 2.0 - 0.25 * 4.0
    assert clipped.value[initial] == -0.75 * 1.0 - 0.25 * 4.0
    assert clipped.stopped_utility[initial] == 1.5
    assert clipped.failure_probability[initial] == 0.25


def test_invalid_transition_models_are_rejected():
    cases = [
        [[[Transition(0.5, None, 1.0)]]],
        [[[Transition(1.0, 1, 1.0)]]],
        [[]],
    ]
    for transitions in cases:
        try:
            solve_budget_dp(transitions, horizon=1, budget=1)
        except ValueError:
            continue
        raise AssertionError("invalid transition model was accepted")
