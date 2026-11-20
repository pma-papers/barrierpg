# Exact dynamic programming for finite budget models

Use backward induction when transition probabilities, utility and consumption are known and the state, action, reserve and time sets are small enough to enumerate. `solve_budget_dp` takes an explicitly specified finite Markov decision process with one integer-valued budget. It optimizes the supplied model without training or sampled transitions. Values are computed using floating-point arithmetic.

## Specify the model

List the outcomes of each state-action pair as `Transition(probability, next_state, utility, consumption)`. Probabilities for a state-action pair sum to one. Use `next_state=None` for an outcome that terminates the task. Consumption is a nonnegative integer. A transition that reduces the reserve to zero still earns its utility. A strict crossing terminates the task at that action and excludes its utility.

The reachable state is the triple of action time, model state and reserve. The transition law is time homogeneous. The optimal decision may depend on time and reserve. The `beta` argument discounts utility. Resource consumption is undiscounted.

```python
from barrierpg import Stage, Transition, solve_budget_dp

transitions = [[
    [Transition(1.0, 0, 0.0, 0)],
    [Transition(0.75, 0, 2.0, 1), Transition(0.25, 0, 2.0, 2)],
]]
initial = (0, 0, 1)

priced = solve_budget_dp(
    transitions,
    horizon=2,
    budget=1,
    beta=0.9,
    stage=Stage(m=0.0, penalty="multiplier", multiplier=1.0),
)
feasible = solve_budget_dp(transitions, horizon=2, budget=1, beta=0.9, pathwise=True)

print(priced.policy[initial], priced.value[initial], priced.failure_probability[initial])
print(feasible.policy[initial], feasible.value[initial], feasible.failure_probability[initial])
```

There is one state and two actions. Idle uses no wear and earns zero. Operating earns two units on the successful outcome and crosses the budget with probability 0.25. Under a lump charge of one unit, the optimal policy operates at the first action and idles after success, giving value 1.25. The pathwise feasible policy idles at both actions and has zero failure risk. [Runnable example](../examples/solve_finite_budget.py)

## Choose the Bellman objective

In priced mode, each action is evaluated by its expected immediate reward plus its discounted continuation value on surviving outcomes. On a crossing the reward is the negative failure charge and continuation is zero. A price `m` is charged for each remaining period including the failing action. For `beta=1` and a failure at action `t`, the charge is `m * (horizon - t)`. In multiplier mode, the failure charge is constant. A positive `theta` deducts the log barrier described in the [practitioner guide](practitioner-guide.md#the-method-in-brief).

In pathwise mode, backward induction considers only actions whose positive-probability outcomes preserve a nonnegative reserve and have a feasible continuation through the horizon. This includes checking whether a current action leaves enough reserve to finish. When there is no feasible action, the result uses policy index `-1` and value `-inf`. With no stage specified, the solver maximizes expected discounted utility among pathwise feasible policies.

The result contains `policy`, `value`, `stopped_utility`, `failure_probability` and `geometric_risk`. The two risk arrays give the probability of crossing by the end of the model and the expected discount weight on the first crossing, respectively. The arrays are indexed by `[time, state, reserve]`. Value and risk arrays include a zero continuation at the terminal time. If several actions attain the same objective at an entry, the first one is returned.

## Interpret exactness and choose a method

The solutions are exact for the finite model up to numerical arithmetic. If continuous dynamics or consumption are discretized, compare the resulting policy in the original simulator to assess approximation error. The solver's cost grows with the horizon, number of reserve levels, and total number of state-action outcomes. The wear study uses a separate discretized continuous model and simulator evaluations.

A fixed price can identify a candidate with a suitable model-based failure probability. A positive chance tolerance and a CVaR cap impose different constraints. Use finite-scenario policy selection when comparing a supplied list of policies against those criteria. For sampled policy updates, see the [CPO and CVaR workflow](formulation-comparison.md#cpo-and-cvar-training-without-an-interior-barrier).
