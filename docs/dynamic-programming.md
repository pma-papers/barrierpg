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

Let $(t,s,z)$ denote the current time, state and integer reserve, and let $A(s)$ be the list of actions. For each $a\in A(s)$, the model lists outcomes $j$ with probability $p_j$, next state $s_j$, utility $u_j$ and consumption $c_j$. The next reserve is $z_j=z-c_j$. Let $F_t$ be the charge at failure. For a physical horizon, $F_t=m\sum_{\ell=0}^{H-1-t}\beta^\ell$ under per-period pricing. In lump-penalty mode, $F_t=F$. Define $r(u_j,z_j)$ on surviving outcomes to be $u_j$ under failure-only pricing or the specified barrier-adjusted reward.

Let $\Pi$ be the set of feedback policies on the tabular state $(t,s,z)$. Write $J(\pi)$ for the expected discounted reward up to task termination under the stated reward and failure charge. In priced mode the optimization is $\operatorname{maximize}_{\pi\in\Pi}J(\pi)$. With a pathwise requirement it is

$$
\begin{aligned}
\underset{\pi\in\Pi}{\operatorname{maximize}}\quad & J(\pi)\\
\text{subject to}\quad & \Pr_\pi(\tau\le H-1)=0.
\end{aligned}
$$

Here $\tau$ is the first negative reserve update. With no reserve penalty, $J$ in the pathwise problem is expected discounted utility along feasible trajectories.

Starting with $V_H(s,z)=0$, compute the Bellman recursion backward:

$$
\begin{aligned}
Q_t(s,z,a)
&=\sum_{j:z_j<0}p_j(-F_t)
  +\sum_{j:z_j\ge0}p_j\big[r(u_j,z_j)+\beta V_{t+1}(s_j,z_j)\big],\\
V_t(s,z)&=\max_{a\in A(s)}Q_t(s,z,a).
\end{aligned}
$$

A successful terminal outcome has zero continuation value and still earns its current utility. For each entry, save an action attaining the maximum. The recursion works because, after the outcome, the best remaining value depends only on the next state, reserve and time. The current action only needs to compare those optimal continuations, so there is no need to enumerate complete policies.

In pathwise mode, the maximization is restricted to actions whose positive-probability outcomes have $z_j\ge0$ and either terminate safely or reach a feasible next entry. A state with no admissible action receives value $-\infty$ and action index `-1`. For the model above at the last action, operating with reserve one has value $0.75(2)-0.25(1)=1.25$. With reserve zero, the best last action is idle with value zero. At time zero, idling is worth $0.9(1.25)=1.125$. Operating is worth $1.25$ because either it succeeds and leaves zero reserve or it terminates at failure. For the pathwise problem, the probability-$0.25$ crossing rules out operating at reserve one.

The result contains `policy`, `value`, `stopped_utility`, `failure_probability` and `geometric_risk`. For the selected action, the two risk arrays are propagated using the same transition probabilities. An immediate crossing contributes one to its undiscounted and discounted event weights. A surviving transition contributes its successor risk, multiplied by $\beta$ only for the discounted risk. This gives an exact model-based evaluation of the selected policy. The arrays are indexed by `[time, state, reserve]`. Value and risk arrays include a zero continuation at the terminal time. If several actions attain the same objective at an entry, the first one is returned.

## Interpret exactness and choose a method

The solutions are exact for the finite model up to numerical arithmetic. If continuous dynamics or consumption are discretized, compare the resulting policy in the original simulator to assess approximation error. The solver's cost grows with the horizon, number of reserve levels, and total number of state-action outcomes. The wear study uses a separate discretized continuous model and simulator evaluations.

A fixed price can identify a candidate with a suitable model-based failure probability. A positive chance tolerance and a CVaR cap impose different constraints. Use finite-scenario policy selection when comparing a supplied list of policies against those criteria. For sampled policy updates, see the [CPO and CVaR workflow](formulation-comparison.md#cpo-and-cvar-training-without-an-interior-barrier).
