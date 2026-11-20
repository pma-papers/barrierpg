"""Solve a two-period production problem with one unit of wear allowance.

Each period, standing still consumes no wear. Operating earns two units and
consumes one or two units of wear. The reserve and remaining time are known.
Compare the optimal priced policy with a policy that must retain a
nonnegative reserve on every realization of the finite model.

Run from the repository root with the core package installed:
    python examples/solve_finite_budget.py
"""
from barrierpg import Stage, Transition, solve_budget_dp


def main():
    # One state. Actions are idle (0) and operate (1).
    model = [[
        [Transition(1.0, 0, 0.0, 0)],
        [Transition(0.75, 0, 2.0, 1), Transition(0.25, 0, 2.0, 2)],
    ]]
    initial = (0, 0, 1)
    priced = solve_budget_dp(model, horizon=2, budget=1, beta=0.9,
                             stage=Stage(m=0.0, penalty="multiplier", multiplier=1.0))
    feasible = solve_budget_dp(model, horizon=2, budget=1, beta=0.9, pathwise=True)
    for label, solution in (("Fixed failure charge", priced), ("Pathwise feasibility", feasible)):
        action = ("idle", "operate")[solution.policy[initial]]
        print(f"{label}: first action {action}, value {solution.value[initial]:.2f}, "
              f"failure probability {solution.failure_probability[initial]:.0%}")


if __name__ == "__main__":
    main()
