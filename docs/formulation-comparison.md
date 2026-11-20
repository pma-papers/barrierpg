# Formulations, learning algorithms and regularization

A resource-budget formulation specifies what is optimized, what is constrained and when that quantity is measured. The same consumption process can enter an expected-cost constraint, a probability of exceeding the budget, or a tail constraint on the size of the excess. In a fixed-price formulation, the stated cost enters the training objective. These criteria can select different policies from the same class.

Proximal policy optimization (PPO) can optimize a reward supplied by the wrapper. Constrained policy optimization (CPO) estimates an update subject to a local expected-cost limit. A trust-region update under a Gaussian conditional value-at-risk (CVaR) constraint uses a different cost surrogate. Action and parameter regularizers change the optimized utility term. A comparison should specify the objective, constraint and learning algorithm separately.

Run the comparison from the toolbox repository after installing the core package.

```bash
python examples/compare_formulations.py
```

## Specify the optimization problem

Let $t=0,\ldots,H-1$ index the $H$ possible decisions of a physical episode. Let $s_t$ be the task state, $a_t$ an action and $u_t$ its utility. The policy $\pi$ draws actions from a distribution conditional on the information available to it. The set $\Pi$ specifies the policies under consideration, for example all feasible feedback tables or the actors represented by a neural network. An expectation $\mathbb E_\pi$ averages over the initial state, simulator transitions and policy draws. Task termination sets subsequent utility and consumption to zero.

For resource $k$, let $b_k$ be its initial budget, $c_{t,k}$ the consumption and $\gamma_k\in(0,1]$ its cost discount. The reserve satisfies

$$
z_{0,k}=b_k,\qquad z_{t+1,k}=\frac{z_{t,k}-c_{t,k}}{\gamma_k}.
$$

Failure is the first action index $\tau$ for which some $z_{t+1,k}<0$. Set $\tau=\infty$ if no crossing occurs. Let $T\le H-1$ be the last action covered by the requirement and $p_T(\pi)=\Pr_\pi(\tau\le T)$. With utility discount $\beta$, the original-task utility through $H$ actions and the stopped utility are

$$
\begin{aligned}
V^{(H)}(\pi)&=\mathbb E_\pi\left[\sum_{t=0}^{H-1}\beta^t u_t\right],\\
U_H(\pi)&=\mathbb E_\pi\left[\sum_{t=0}^{H-1}\beta^t u_t\mathbf 1_{\{t<\tau\}}\right].
\end{aligned}
$$

In the original task, crossing the budget can leave subsequent utility. Use $V^{(H)}$ to evaluate that task, as in the mean-damage formulation for navigation. Use $U_H$ when failure ends operations before the failing action earns utility, as in the retailer example. The symbols $V$ and $U$ denote the corresponding continuing-horizon quantities.

Let $Z$ be the loss on a trajectory drawn under $\pi$, for example total consumption or overshoot. The mean-cost problem on the original task is

$$
\begin{aligned}
\underset{\pi\in\Pi}{\mathrm{maximize}}\quad & V^{(H)}(\pi)\\
\text{subject to}\quad & \mathbb E_\pi[Z]\le d.
\end{aligned}
$$

Here $d$ is the cap in units of $Z$. The chance-constrained problem on a task that stops at failure is

$$
\begin{aligned}
\underset{\pi\in\Pi}{\mathrm{maximize}}\quad & U_H(\pi)\\
\text{subject to}\quad & p_T(\pi)\le\alpha .
\end{aligned}
$$

Here $\alpha$ is a probability tolerance. The same choice of utility applies when the following constraints are used.

| Formulation | Constraint | Meaning |
|---|---|---|
| Expected cost | $\mathbb E_\pi[Z]\le d$ | The average loss must be at most $d$. |
| Chance constraint | $p_T(\pi)\le\alpha$ | Failure by action $T$ may occur with probability at most $\alpha$. |
| Conditional value-at-risk (CVaR) | $\mathrm{CVaR}_\varepsilon(Z)\le d$ | The mean of the worst $\varepsilon$ fraction of losses is bounded by $d$. |
| Pathwise feasibility | $p_{H-1}(\pi)=0$ | Every positive-probability trajectory of a finite model must respect the budget throughout the episode. |

An expected-cost method can impose the chance constraint by using $Z=\mathbf 1_{\{\tau\le T\}}$ and setting the cost limit to $\alpha$. For nonnegative consumption, iteration of the reserve recursion gives $\gamma_k^{T+1}z_{T+1,k}=b_k-\sum_{t=0}^T\gamma_k^t c_{t,k}$. Each discounted cumulative sum is nondecreasing, so crossing by action $T$ occurs if any of these sums exceeds $b_k$. For undiscounted net use that can decrease after replenishment, use its maximum through $T$ or record the first crossing directly.

For CVaR, let $(x)_+=\max(x,0)$. The formula below holds for integrable $Z$ with $0<\varepsilon<1$.

$$
\mathrm{CVaR}_\varepsilon(Z)=
\min_{\xi\in\mathbb R}\left[\xi+\frac1\varepsilon\mathbb E_\pi[(Z-\xi)_+]\right].
$$

The variable $\xi$ is a threshold in the units of the loss. At the minimum it is an upper-tail quantile, with the appropriate fraction of any mass at that threshold included in the tail. A CVaR constraint on overshoot differs from a CVaR constraint on total consumption. Set the loss definition before choosing its cap. `upper_cvar` calculates the exact tail average for a finite discrete loss distribution.

## Write the priced training objective

For a continuing task, let $q_\beta(\pi)=\mathbb E_\pi[\beta^\tau\mathbf 1_{\{\tau<\infty\}}]$. This is the probability of failure before an independent geometric stop with survival probability $\beta$ after each surviving action. Let $U(\pi)$ be discounted stopped utility and let $\mathcal I_m(\pi)$ be the expected discounted deduction from the pre-failure reward. On a nonfailing action the floor is applied after subtracting the barrier at the updated reserve. The change in utility is $u_t-\max\{-m,u_t-B(z_{t+1})\}$. It is nonnegative when $u_t\ge-m$. If the raw utility is below the floor, clipping can raise the reward. With failure-only pricing, $\mathcal I_m=0$. The continuing training problem is

$$
\underset{\pi\in\Pi}{\mathrm{maximize}}\quad U(\pi)-\mathcal I_m(\pi)-\frac{m}{1-\beta}q_\beta(\pi).
$$

The charge $m/(1-\beta)$ at the failing step represents a discounted sequence of rewards $-m$ from failure onward. Under a physical horizon $H$, the charge at a failure at $t$ is $F_t=m\sum_{j=0}^{H-1-t}\beta^j$. Training then maximizes $U_H(\pi)-\mathcal I_{m,H}(\pi)-\mathbb E_\pi[\beta^\tau F_\tau\mathbf 1_{\{\tau<H\}}]$, where $\mathcal I_{m,H}$ is the interior deduction through the physical horizon. A lump charge sets $F_t=F$ at all failure times. Validation uses the stated deployment event $p_T$. Its confidence level is a separate statistical input.

## Choose a numerical method

For a known finite model, backward induction computes the priced optimum or an optimum over pathwise feasible policies. For a given finite list of policies and scenarios, compute the utility and constraint statistic of each candidate and maximize utility over those that meet the limit. For a simulator with a large or continuous state space, improve a parametrized policy using rollouts. PPO uses a scalar reward, CPO uses utility and a mean-cost surrogate, and Gaussian CVaR training uses a surrogate for an upper-tail constraint. [Bellman recurrence](dynamic-programming.md#choose-the-bellman-objective) · [How the actor updates are computed](learning-algorithms.md)

## When can the solutions agree?

| Known model or requirement | Formulation or solution method | Example algorithm or implementation | What to measure |
|---|---|---|---|
| Small finite model with known outcomes | Exact dynamic programming | `solve_budget_dp` for priced or pathwise feasible decisions | Model-based value, utility and failure probability |
| Running out ends the task and has a known cost | Failure-only price | PPO on a reward wrapped with `BudgetWrapper` | Stopped utility and failure frequency at the deployment horizon |
| Operating with low reserve has a cost | Deduct a reserve penalty from utility | PPO with the modified reward or CPO with an expected-cost constraint and utility net of the reserve penalty | Minimum reserve and performance under a budget shift |
| Mean cumulative cost must meet a cap | Expected-cost constraint | CPO or a policy-gradient algorithm using an adaptive multiplier | Mean cost and update-time violations |
| Failure by a specified action must have probability at most a tolerance | Chance constraint encoded as the expectation of the first-crossing indicator | CPO with the specified event cost | First-crossing probability by the specified horizon |
| Large losses in the worst outcomes must be limited | CVaR constraint on total consumption, overshoot or another specified loss | TRC or exact selection from finite-scenario candidates | Held-out tail loss and its uncertainty |
| Every trajectory must meet the limit | Pathwise feasibility | A model-based feasible policy class or a safety layer | The model assumptions and the resulting pathwise guarantee |

For example, compare two formulations on the same stopped task with loss $Z$ and cap $d$. Let $J_c(\pi)=\mathbb E_\pi[Z]$. Suppose $\pi^\star$ maximizes $U_H(\pi)-\nu[J_c(\pi)-d]$ for a fixed $\nu\ge0$. It also maximizes $U_H$ subject to $J_c\le d$ if

$$
J_c(\pi^\star)\le d,\qquad \nu[J_c(\pi^\star)-d]=0 .
$$

Indeed, for any feasible $\pi$, $U_H(\pi)\le U_H(\pi)-\nu[J_c(\pi)-d]\le U_H(\pi^\star)$, where the last inequality uses priced optimality and complementary slackness. This argument compares the same utility, cost, stopping rule and policy class. Some constrained optima in a discrete or restricted class cannot be obtained by any fixed multiplier.

A one-stage deterministic example uses utility 1.0 at consumption 1 and utility 1.1 at consumption 2. A consumption cap of 1 and a linear price of 0.2 both select the first policy. In the stochastic example, the second policy consumes either 0 or 2 with equal probability. Both policies now have mean consumption 1, so a mean-cost cap or linear price selects the higher-utility policy. The variable policy exceeds budget 1 with probability 0.5 and has upper-half CVaR 2. The steady policy has crossing probability 0 and upper-half CVaR 1. A deterministic simulator can still produce this distinction if the initial condition or policy is randomized.

For a single fixed terminal time and undiscounted consumption, a linear penalty proportional to used reserve is the fixed-price counterpart of a mean consumption cap. A running penalty on remaining reserve charges early consumption over more periods, so it must be matched to the corresponding time-weighted cost. For failure-only pricing, use the same failure indicator and discount as the price. The continuing convention can be encoded by a cost of one in every failed absorbing period. A physical-horizon crossing probability can be encoded by a single undiscounted indicator of first crossing before the horizon.

### CPO for an expected-cost constraint

`cpo_step` solves the local single-cost quadratic subproblem with a dense positive-definite Fisher matrix. `cpo_update` applies the local proposal to a caller-supplied line search that measures utility change, cost and KL. An accepted recovery step can still exceed the cap. A fixed-price step in the same trust region can coincide with an active CPO step if its price matches the local cost multiplier and the same proposal passes both line searches. That multiplier depends on the current slack, policy gradients and trust region. [Achiam et al., Section 6.1](https://proceedings.mlr.press/v70/achiam17a.html)

For full CPO training, use the optional adapter in a separate OmniSafe-compatible environment. The environment supplies a reward and a cost through OmniSafe's registered CMDP interface.
The integration was exercised with Python 3.10, OmniSafe 0.5.0, Safety-Gymnasium 1.0.0, NumPy 1.26 and MoviePy 1.x. That environment uses Gymnasium 0.28.1 and should be kept separate from the toolbox's SB3 extras.

```python
from barrierpg import omnisafe_cpo

agent = omnisafe_cpo("SafetyPointGoal1-v0", cost_limit=25)
agent.learn()
```

In OmniSafe 0.5.0, the slack uses the logged sum of costs per episode, while the cost advantage has a configurable `cost_gamma`. Match both the cost signal and its horizon and discount to your intended comparison. For an undiscounted first-crossing cap, deliver the indicator once and use `cost_gamma=1.0`. Verify the measured crossing rate independently. The adapter takes upstream configuration through `custom_cfgs`. [OmniSafe CPO](https://omnisafe.readthedocs.io/en/latest/saferl/cpo.html) · [OmniSafe CMDP interface](https://omnisafe.readthedocs.io/en/latest/start/env.html)

### CVaR constraints and algorithms that use them

`upper_cvar(costs, tail_mass, probabilities)` averages the worst `tail_mass` of the cost distribution, splitting a probability atom at the boundary when needed. For example, `tail_mass=0.05` means the worst 5% and corresponds to a CVaR level of 0.95. `select_scenario_policy` picks the highest-utility feasible candidate from a fixed list and reports each candidate's measured risk. Use actual end-of-trajectory costs for the comparison. For a sequence of candidate checkpoints, estimate tail risk during selection and use a separate scenario set or fresh rollouts for final evaluation. The finite-scenario routine is checked against the CVaR linear program in SciPy HiGHS. For a learned continuous controller, the authors' TRC implementation provides a policy-search baseline with its own cost-distribution approximation. [CVaR formulation](https://web.stanford.edu/~boyd/papers/pdf/cvar_qp.pdf) · [SciPy HiGHS](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html) · [TRC implementation](https://github.com/rllab-snu/Trust-Region-CVaR)

### Select from a finite policy list

`select_scenario_policy` takes one expected utility per candidate and a matrix with one row per candidate and one column per common scenario. Supply scenario probabilities where they are unequal. For `formulation="chance"`, `budget` sets the strict crossing threshold for the supplied path statistic and `cost_limit` is a probability. Use a first-crossing statistic such as the largest cumulative use through the horizon if consumption can decrease after a crossing. For `formulation="pathwise"`, the cost limit applies to each positive-probability scenario.

The returned risk array contains the selected statistic for every candidate. The index is `None` when the given policy list contains no feasible candidate. For a finite model with known probabilities this comparison is exact on the listed scenarios. For sampled scenarios it is a sample-based comparison and calls for independent evaluation after selection.

## Compare reserve penalties

Let `r = z/b` be the fraction of a single budget remaining after a decision. The package sums the corresponding terms over resources. Pass the shape as `BudgetWrapper(..., barrier=penalty)`. Leaving it out gives the original `Stage(theta=...)` convention, with a log barrier at positive strength and failure-only pricing at zero.

| Shape | Penalty per resource | Effect on the reward signal |
|---|---|---|
| Failure-only | Zero before crossing | Retains the original utility on successful steps. When nearly all rollouts succeed or fail, the event alone may give little information about which actions preserve reserve. |
| `LogBarrier` | `theta * log(1 + 1/r)` | Charges reserve use across the interior and grows near zero. Its derivative increases in magnitude as the reserve approaches zero. |
| `InverseBarrier` | `theta / r` | Has a steeper rise near zero than log. Early clipping can flatten this distinction. Check the fraction of rewards at the floor and the variability across seeds. |
| `ReserveHinge(..., fraction=h)` | `theta * max(0, h-r)` | Applies no deduction above the specified reserve fraction and a linear deduction below it. With `h=1` and reserve never above its initial value, it prices used budget linearly. |
| `ReserveHinge(..., fraction=h, power=2)` | `theta * max(0, h-r)**2` | Gives a continuous first derivative at the specified reserve fraction and an increasing marginal deduction below it. The penalty stays finite at zero. |

Log and inverse diverge at zero, where the wrapper uses the clipped reward floor. Finite hinges also permit an explicit charge at zero without treating zero as a failure. With clipping, changes in a penalty beyond `u + m` have no effect on that step's reward. With no clipping, near-zero reserve may produce very large charges under singular shapes. All shapes retain the strict crossing event. A finite hinge is a soft reserve penalty with a different boundary behavior from the paper's singular barrier.

In the one-stage example, $a$ is chosen on a grid. Consumption is $c=a+X$, where $X$ equals $0.18$ with probability $0.1$ and zero otherwise. With budget one and the penalty shape $B$, the script maximizes

$$
\underset{a\in\{0.01,0.02,\ldots,0.99\}}{\mathrm{maximize}}\quad\mathbb E[r(a,c)],\qquad
r(a,c)=
\begin{cases}
\max\{-1,2a-B(1-c)\}, & c\le1,\\
-1, & c>1.
\end{cases}
$$

In this equation the reward takes the value $-1$ at a crossing. The barrier is evaluated only when $c\le1$. Each action is scored using the probabilities of the two outcomes. In the failure-only case the nonfailing reward is $2a$. The example matches all positive penalties at 20% reserve and activates the hinges below 40%. After selecting the action, evaluate it at the original budget and with a 20% reduction.

| Training reward | Action | Nominal stopped utility | Nominal failure | Failure with smaller budget | Smallest nominal reserve |
|---|---:|---:|---:|---:|---:|
| Failure-only | 0.99 | 1.782 | 10% | 100% | −0.17 |
| Log | 0.81 | 1.620 | 0% | 100% | 0.01 |
| Inverse | 0.78 | 1.560 | 0% | 10% | 0.04 |
| Linear hinge | 0.82 | 1.640 | 0% | 100% | 0.00 |
| Quadratic hinge | 0.82 | 1.640 | 0% | 100% | 0.00 |

These are grid optima for the stated example. The inverse penalty selects a policy with a larger minimum reserve and lower stopped utility in this comparison. Both hinges choose the same action here despite different slopes. In a learning experiment, use the same policy class, horizon, environment calls and training seeds. Calibrate weights at a meaningful reserve level or match the slope there. Record raw utility, the reserve distribution, the share of clipped steps, failure timing and the deployment risk. Compare reserve shapes at matched deployment risk when possible, and use the budget uncertainty actually relevant to the task.

## Vary action and parameter regularization independently

Use the unregularized objective as a baseline. Add one form at a time and save the initial or reference policy as a candidate. Across shapes, matching the numeric strength alone can induce very different costs and gradients.

| Choice | Use | What may change |
|---|---|---|
| Action KL to a fixed reference | Give the action distributions a common support and measure their discounted stopped KL | Penalizes departure from the reference distribution. The risk bound uses the achieved stopped KL and the reference's risk. |
| Squared Hellinger or squared probability distance | Compare categorical distributions in a differentiable custom loss | Both are convex in policy probabilities. Their gradients and treatment of small probabilities differ from KL. Evaluate trajectory risk separately. |
| Fixed action cost | Penalize movement, energy or distance from a reference action using the ordinary reward or the SB3 `ActionCost` callback | The penalty is linear in the action law. Its scale and whether it applies to the failing decision affect behavior. |
| Quadratic parameter potential | Anchor weights to a trained reference, including dispersion when it is trainable | The pull grows linearly with displacement and supplies uniform curvature in parameter space. |
| Pseudo-Huber parameter potential | Allow selected weights to move farther while retaining a quadratic region near the reference | The coordinatewise gradient is bounded. Curvature decays with distance, so a global positive curvature bound does not follow. |
| Elastic-net parameter potential | Add a preference for leaving individual weights at the reference | The absolute-value term is nonsmooth. A standard gradient optimizer uses a subgradient at the kink and may behave differently from a proximal method. |
| KL plus power divergence of a parameter law | Control a distribution of parameters relative to a prior with overlapping support | `parameter_law_divergence(..., power=2)` adds Pearson chi-squared on a common grid. Concentrating mass where the prior is small costs more. A finite network's empirical measure against a continuous prior is a different case and has infinite KL. |

For SB3, fixed action costs can be combined with a different parameter potential using existing callback and optimizer interfaces.

```python
import numpy as np
from barrierpg import parameter_potential
from barrierpg.callbacks import ActionCost, ParameterAnchor

movement = ActionCost(0.01, lambda obs, actions: np.sum(actions**2, axis=-1))
anchor = ParameterAnchor(
    model, kappa_p=0.05, decoupled=False,
    potential=lambda d: parameter_potential(d, form="pseudo_huber", scale=0.2),
).install()
model.learn(10_000, callback=movement)
```

With CPO or an algorithm for a CVaR constraint, use `regularized_utility` to form the same objective from expected utility and the two penalties. In CPO, differentiate it for the utility-gradient input and evaluate its change during backtracking. Keep the original cost gradient, cost limit and old-to-new KL trust region. The anchor to an incumbent and the KL trust region have different reference policies and different purposes. When a differentiable action divergence is evaluated on a fixed batch of states, it gives a local surrogate. To target a policy's full stopped action penalty, account for how the policy changes visited states. The upstream OmniSafe adapter exposes native CPO. It can take fixed action costs through the registered environment's reward. The reference update accepts a custom objective gradient, including a parameter penalty, without changing the cost gradient.

For finite-scenario policy selection under a CVaR constraint, pass regularized utilities as the candidate scores and keep the same CVaR constraint. In the runnable example, both policies satisfy a CVaR cap of 2. Adding action and parameter penalties switches the selected policy while leaving the two candidates' CVaRs unchanged. The example also shows a CPO step changing direction under a quadratic anchor of the same strength as a pseudo-Huber anchor. These comparisons isolate the effect of the objective. The paper's generalized convex regularization theorem concerns its specified population gradient flow. Using the same penalties in another optimizer calls for a separate analysis of its updates and validation of the returned policies.

The one-state Gaussian example uses $a\sim N(w,I)$. Up to a constant, expected utility is $J_u(w)=-\|w-(1,1.5)\|^2/2$ and expected cost is $\mathbb E[\exp(a_1-1/2)]=\exp(w_1)$. Thus the barrier-free problem maximizes $J_u(w)$ subject to $\exp(w_1)\le \exp(0.6)$. The regularized variants subtract action KL to a fixed Gaussian reference and a parameter potential from $J_u$. Each update uses the same cap and the same KL radius. With the cost cap of 1.822, the unregularized run reaches cost 1.822. Adding the action KL and a quadratic potential yields cost 1.657. Keeping the same action KL and using a pseudo-Huber potential yields 1.696 and higher raw utility than the quadratic run. These are endpoints of the specified updates. The strength, reference, and optimizer affect what happens in a learned controller.

## CPO and CVaR training without an interior barrier

Use the original task utility to compute reward advantages. Supply separate trajectory costs for the constraint. For instance, a mean fuel cap uses cumulative fuel as the cost, a finite-horizon chance constraint uses the indicator of first crossing, and an overshoot CVaR constraint uses the amount by which the allowance is exceeded. No `BudgetWrapper`, failure price or reserve penalty is required in these actor updates.

The runnable dispatch example collects fresh episodes for each update. It compares barrier-free CPO with barrier-free Gaussian CVaR training and reports both the Gaussian surrogate and exact upper-tail CVaR in the specified dispatch model. Install `-e ".[rl]"` for PyTorch. [Dispatch example](../examples/train_risk_constrained.py)

```bash
python examples/train_risk_constrained.py
```

The two calls below use the same actor interface. In `batch`, provide observations, sampled actions, utility advantages, cost advantages and the current mean cumulative cost. For CVaR, also provide `variance`, containing the cost variance and its advantages. The limits in the two calls have different meanings even when their numerical values are equal.

```python
from barrierpg.on_policy import cpo_policy_update, cvar_policy_update

# Use separate actors and collect on-policy data before each update.
cpo = cpo_policy_update(mean_actor, mean_batch, cost_limit=0.62, kl_limit=0.01)
cvar = cvar_policy_update(
    tail_actor, tail_batch, variance, cost_limit=0.62, tail_mass=0.25, kl_limit=0.01
)
```

For CPO the cap concerns mean cumulative cost. For the CVaR update it concerns the Gaussian surrogate for the average of the worst quarter of costs. In the dispatch example, compare the resulting mean cost, Gaussian CVaR and exact CVaR in the output table. The modeled cost distribution is discrete, so its exact CVaR can differ from the Gaussian surrogate. Estimate the corresponding deployment quantity on independent trajectories. In a finite model, `upper_cvar` computes CVaR exactly from the loss distribution, including atoms at the tail boundary.

The [actor update source](../barrierpg/on_policy.py) takes a module that returns a PyTorch distribution. Its inputs include the observed reward and cost advantages and the current mean cumulative cost. The Gaussian CVaR update also takes the current cost variance and its advantages. Both compute the old-to-new KL Hessian, solve a dense single-cost trust-region subproblem and check the trial step on the batch before changing the actor. Use the dense implementation for small actors. The caller supplies rollouts and any critic estimates, making the reward, cost and advantage estimators visible. Keep the actor forward pass deterministic when constructing the distribution, for example by disabling dropout during sampling and line search.

For vector observations, the [residual network guide](residual-networks.md) describes a PyTorch backbone
with independently configurable skip connections and LayerNorm.

`cpo_policy_update` accepts a limit on mean cumulative cost. `cvar_policy_update` uses the Gaussian surrogate from TRC, equal to the mean cost plus a multiple of its standard deviation. Use variance advantages and set the mean and variance scales to match their units. The upstream TRC conventions use `1 / (1 - gamma)` and `1 / (1 - gamma**2)` with centered advantages. The dispatch example has a single decision and uses unit scales. The Gaussian surrogate and empirical CVaR can differ substantially for skewed or discrete costs. The script prints both for the final policy. A sampled batch can also underestimate the model's risk, so check the deployment criterion separately. [TRC paper](https://arxiv.org/abs/2312.00344) · [Authors' PyTorch surrogate](https://github.com/rllab-snu/Trust-Region-CVaR/blob/main/torch/agent.py)

Both updates accept `action_penalty(observations, distribution)` and `parameter_penalty(flat_parameters)`. The functions return scalar penalties, including their strengths. They enter the objective gradient and the line search. The dispatch example uses squared Hellinger distance to an action reference and a pseudo-Huber parameter potential. A fixed-batch action penalty regularizes the distributions at the sampled states. A penalty under the full policy-dependent state distribution requires the corresponding training objective and advantage estimator.

On matched gradients and a matched Fisher matrix, the dense solver agreed with the OpenAI Safety Starter Agents numerical CPO block in unconstrained, active and recovery cases. OmniSafe 0.5.0 returned a different proposal in the active example with utility gradient `[2, 1]`, cost gradient `[1, 0]`, gap `-0.6`, identity Fisher and KL limit `0.5`. The original numerical block and dense solver give `[0.6, 0.8]`. On a matched batch, the local Gaussian CVaR surrogate agreed with the authors' PyTorch code before and after an accepted step. These checks cover the solver and surrogate on controlled inputs. The surrounding rollout, critic and line-search implementations differ across projects. [Original CPO numerical block](https://github.com/openai/safety-starter-agents/blob/4151a283967520ee000f03b3a79bf35262ff3509/safe_rl/pg/agents.py#L259) · [OmniSafe CPO source](https://omnisafe.readthedocs.io/en/latest/_modules/omnisafe/algorithms/on_policy/second_order/cpo.html)

## Use CPO to solve an objective with an interior barrier

When reserve depletion has a cost and failure risk has a prescribed limit, use stopped utility net of the clipped interior loss as the reward objective, then constrain the desired failure cost in CPO. Define the same interior loss for all candidates before comparing constraints or solvers. If the full priced barrier return is supplied as reward under the same cap, the fixed failure charge is still part of the objective. This can favor a safer policy with lower utility even when both candidates satisfy the limit.

The [paired wear-allowance pilot](wear-benchmark.md) runs both CPO variants, fixed prices with and without a log barrier, and a separate overshoot-CVaR constraint. The full design and seed-level results are kept with the code.

For a small, explicitly specified model, backward induction can compute an optimal policy for a fixed failure charge or a pathwise budget. [Dynamic programming guide](dynamic-programming.md)
