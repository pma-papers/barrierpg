# From a resource-budget problem to a validated policy

A resource-budget problem has a reserve that can become negative. Failure is the first action after which any reserve is negative. The training objective can assign a price to that event and deduct an additional penalty when reserves are small. The deployment requirement specifies how often that event may occur by a stated horizon.

The toolbox also provides expected-cost actor updates, chance and tail-risk comparisons, and policy selection on finite scenarios. The first three steps identify the formulation. The later steps describe the priced-reward and first-crossing validation workflow used in the applied notebooks.

The steps below apply to the [custom task template](../examples/your_problem.py) and to the lander, retailer, battery and navigation notebooks listed in the [toolbox overview](../README.md#tutorials-and-examples). The [theory guide](under-the-hood/decisions-and-theory.md) gives the conditions behind the reward identities, value bounds and regularization results.

## The method in brief

The reserve for resource $k$ starts at $b_k$ and is updated as $z_{t+1,k}=(z_{t,k}-c_{t,k})/\gamma_k$, where $c_{t,k}$ is the consumption on action $t$ and $\gamma_k$ is the cost discount. With $\gamma_k=1$, the reserve is the initial budget less cumulative consumption. The observation supplied to the policy includes the reserve. If the task has a fixed end and optimal decisions depend on time remaining, the observation should include time as well.

A failure-only stage pays the task's full utility up to the action before failure. With an interior barrier $B$, the reward on a nonfailing step is $\max\{-m,u-B(z_{t+1})\}$ under clipping. In the continuing formulation, a failure ends the simulated episode with a discounted charge of $m/(1-\beta)$ at that action. This represents a charge of $m$ in every remaining period of the absorbing failed state. Under a physical horizon of $H$ actions, a failure at $t$ costs $m(1-\beta^{H-t})/(1-\beta)$ at that action.

Choose a stage for training, then assess the candidate that will be deployed. The price and the statistical test address different decisions. The price sets the objective used to train the candidate, while the test estimates its failure probability under the deployment conditions.

## Step 1. Check that the problem has a budget

Specify the consumed quantity, the initial allowance, and what happens at the first crossing. In the fuel-tank example, the fuel consumed on a step depends on the engine commands. In the retailer example, cash decreases with purchases and overhead and increases with receipts. Both have a first crossing. The current landing problem also accounts for loss of the lander with a second failure condition.

For undiscounted signed net use, `RunningMaxCost` converts it to the increments of its running maximum. This preserves the first crossing of the original budget. The transformed reserve measures the distance from the largest net cumulative use so far. If the policy needs to react to replenishment, retain the current cash balance or corresponding state variable in the underlying observation. See the [retailer notebook](../notebooks/02_retailer_cash.ipynb).

## Step 2. State the requirement before choosing a method

```python
from barrierpg import BudgetSpec, Requirement

spec = BudgetSpec(budgets=[24.0], beta=0.999, ubar=100.0, u_minus=100.0, horizon=1000)
req = Requirement(T=999, alpha=0.05, zeta=0.05, family=1)
```

Here the physical horizon consists of 1,000 actions and the deployment test covers all of them, indexed 0 to 999. Set `alpha` to the maximum tolerated failure probability and `zeta` to the statistical error budget. `zeta=0.05` gives 95% one-sided confidence for the single statement above. If the validation data are used to assess multiple candidates, horizons or bounds together, include those statements in the error allocation before inspection.

The `beta` field discounts utility and, in the continuing objective, weights failure times. The cost discounts in `gammas` default to one. For a value certificate, establish an upper bound `ubar` on each step's utility and the required lower bound `-u_minus`. The scales above illustrate the interface. When those bounds are unavailable, use direct failure validation and pass `include_physical_value=False` to `interpret` to omit the empirical physical-horizon value bound.

## Step 3. Check whether another method is the better choice

| Requirement | Formulation and evidence |
|---|---|
| A failure has a known operating cost | Charge the stated loss at failure or use the discounted total of continuing downtime costs. Evaluate stopped utility and failure probability separately. |
| Probability of failure by a specified action | Use the first-crossing event for validation. A fixed price or an adaptive multiplier can produce candidate policies. |
| Mean cumulative consumption must stay below a limit | Use an expected-cost formulation. Constrained policy optimization (CPO) uses a local cost constraint in its update. Estimate the final policy's mean cost on fresh trajectories. |
| Large losses in the worst outcomes must be limited | Use a tail measure such as conditional value-at-risk (CVaR) of a specified loss. Evaluate that same loss on deployment trajectories or scenarios. |
| Every trajectory must respect the budget | Use a formulation with pathwise feasibility, supported by a model, a verified safe set or an admissible policy class. |
| A small model is available | Use [dynamic programming](dynamic-programming.md) when the finite transition model is known. Report the resulting model-based utility and risk. |

The [formulation tutorial](formulation-comparison.md) includes barrier-free CPO and CVaR updates. It also shows how the same two policies can rank differently under mean, chance and tail constraints. `plan(spec, req, ...)` summarizes relevant conditions and sample-size calculations for a specified problem.

## Step 4. Wrap the environment

```python
import gymnasium as gym

from barrierpg import BudgetWrapper, Stage
from barrierpg.problems.lander import fuel_cost

env = BudgetWrapper(
    gym.make("LunarLanderContinuous-v3"),
    cost=fuel_cost,
    spec=spec,
    stage=Stage(m=0.5),
    observe_time=True,
)
```

The cost function takes `(obs, action, next_obs, reward, info)` and returns one finite nonnegative number per budget. This example uses the earlier fuel-tank task. The [landing example](../README.md#train-on-a-budget-constrained-task) uses the current paper's simulator and its two budgets.

By default, the wrapper appends each reserve as a fraction of its initial budget. With `observe_time=True`, it also appends the elapsed fraction of the physical horizon. Before training, check an ordinary step, the case of exactly zero reserve, the first crossing, and termination at the physical horizon. Verify the failure charge against its discounted sum. Premature simulator truncation cannot be treated as a completed failure-free evaluation.

`Stage(theta=0)` pays full utility on each nonfailing step, including negative task rewards. With a positive barrier, `clip_utility=False` permits the reward $u-B$ at positive reserve even when it is below $-m$. Confirm which reward bounds hold before applying a value certificate.

## Step 5. Choose the price

When failure has a stated one-off cost $F$, `Stage(m=0, penalty="multiplier", multiplier=F)` with no dual-ascent callback charges that amount once. The [retailer notebook](../notebooks/02_retailer_cash.ipynb) uses this convention for insolvency. With a continuing per-period cost, the equivalent fixed price is $m=(1-\beta)F$. For a physical horizon the price that matches $F$ depends on the failure time.

When the objective is to meet a risk tolerance, compare a few fixed prices on pilot runs. Track stopped utility, failure by the specified horizon, the training objective, and other task outcomes such as crashes. Inspect the failure charge in the same units as episode utility. The earlier fuel-tank task illustrates why a change in one failure cost can alter the mix of termination outcomes.

A conservative price from `plan` requires a valid comparison policy and the assumptions of the price–risk bound, including control of the optimization error. Under exact failure-only optimization in a fixed class, geometric risk is nonincreasing in $m$. With approximate training, geometric risk or the finite-horizon failure rate can rise across the tested prices. A shortfall from the best tested own-price benchmark estimates a lower bound on the true optimization gap. The pilot results guide the choice of candidates for independent validation. [Theory and price bounds](under-the-hood/decisions-and-theory.md#3-fix-the-price-then-certify)

An adaptive multiplier updates the flat penalty in the direction of the observed rate minus its prescribed level. Use `Stage(penalty="multiplier")` and `DualAscent` with compatible on-policy training. The callback counts the failure flags of episodes that ended in each rollout. Use a task termination or stopping clock that makes this event the risk quantity intended for training. Fixed-price pilots can help set the initial penalty and step size. Inspect both the multiplier and the measured risk, because a small multiplier change can occur even while the measured rate is inconsistent with the level. Select a candidate with enough slack for the planned validation sample.

## Step 6. Add a reserve when the budget is uncertain

The failure-only objective charges a budget crossing. An interior barrier also makes small positive reserves costly before crossing. Consider both stopped utility and the distribution of minimum reserve when choosing the barrier strength. A larger barrier penalty can reduce utility or make more rewards reach the clipping floor.

If a possible budget reduction is specified, test it explicitly. With `true_budget_scale=0.9`, the failure event is measured against 90% of the nominal budget, and the policy continues to observe the nominal reserve. This is a sensitivity experiment. Training with a reduced budget implements a different policy specification. The [battery notebook](../notebooks/03_robot_battery.ipynb) illustrates the sensitivity experiment, and the [formulation tutorial](formulation-comparison.md#compare-reserve-penalties) compares reserve penalties at a common reference level.

## Step 7. Choose the regularizers for what they do

**Action relative entropy.** `ActionAnchor` deducts a penalty for departure from a reference action distribution. Use its observed or bounded stopped relative entropy, together with the reference policy's risk, to assess the risk comparison under the paper's assumptions. The penalty is a training term. The risk comparison depends on the relative entropy achieved by the candidate.

**Parameter regularization.** `ParameterAnchor` penalizes departures from specified actor parameters. Evaluate the policy at the anchor before training, including its action dispersion. If a network is fitted to an incumbent rule, measure the fitted network's risk as a separate candidate. Initialization and an anchor are separate choices. Compare from-scratch training, initialization from the incumbent, and continued anchoring using fixed evaluation conditions.

```python
from barrierpg.callbacks import ActionAnchor, ParameterAnchor

anchor = ParameterAnchor(model, kappa_p=1.0).install()
model.learn(steps, callback=[ActionAnchor(0.05, log_reference), monitor])
```

In this fragment, `model` is a compatible Stable-Baselines3 actor-critic learner. `log_reference` gives the reference log probability or density of each sampled action. `monitor` is a `BudgetMonitor`. To fit an initial actor to a rule, use `fit_actor_mean` on representative states and set the policy dispersion with `set_log_std` before installing the anchor.

The default decoupled anchor applies a separate update proportional to `lr * kappa_p` after the optimizer step. With `decoupled=False`, its gradient is added before the optimizer step, where Adam rescales the combined gradient. Strengths are comparable within a specified optimizer and implementation. Record the form, effective actor distance and measured action dispersion. `ParameterNoise` adds Gaussian increments after the update. Its effect under PPO is an empirical question. The population-flow convergence theorem uses additional curvature and representation hypotheses. [Regularizer choices](formulation-comparison.md#vary-action-and-parameter-regularization-independently) · [Theory and optimizer implementation](under-the-hood/decisions-and-theory.md#5-choose-regularizers-for-their-role)

## Step 8. Validate the policy you deploy

A training trace describes the learner's behavior during training. `BudgetMonitor.convergence()` compares recent windows for changes in measured quantities. The resulting diagnostic is descriptive. In particular, a decayed learning rate can leave a trace nearly constant. To assess the final policy, freeze its weights, action sampling rule and observation preprocessing before collecting validation episodes.

Record the evaluation policy and inputs with each candidate. For a Stable-Baselines3 actor, `sb3_policy(model)`
samples its actions and `sb3_policy(model, deterministic=True)` uses its deterministic prediction. If training
used observation normalization, freeze the corresponding statistics and apply the same transform at evaluation.
The evaluator rejects an episode censored by an administrative time limit. Configure the physical horizon
explicitly or use simulator termination for a genuine early endpoint.

For a finite-horizon requirement, run independent episodes that observe the first crossing through action $T$ or end earlier at a genuine terminal state. A stochastic policy and the policy formed from its mean action are distinct candidates. Changing the dispersion, task horizon or budget also changes the statement being tested. Use a separate seed range for pilot comparisons and final validation.

`validation.validate_direct(failures, n, alpha, zeta, family)` computes the exact one-sided binomial upper endpoint. The policy meets the stated tolerance when that endpoint is at most $\alpha$. With zero failures in $N$ independent episodes the endpoint is $1-(\zeta/\mathrm{family})^{1/N}$. Use `validation.episodes_for_tolerance` to plan the sample at the per-statement error. An endpoint above $\alpha$ leaves the requirement unestablished at that confidence level.

For a single 5% tolerance at statistical error 0.05, a sample of 59 zero-failure episodes is just sufficient.
For two simultaneous candidates at the same tolerance and total error, each candidate's zero-failure sample must
have at least 72 episodes because its endpoint uses error 0.025. These are minimum sizes for a zero-failure outcome.
If failures are expected, plan for a larger sample using a plausible failure rate below the tolerance.

For a process with an independent geometric stopping time, the Bernoulli event is failure before that stop. The bound $p_T\le\beta^{-T}q_\beta$ can transfer that result to a fixed horizon. Its informativeness depends on $\beta$ and $T$. [Validation tutorial](../notebooks/under-the-hood/T4_validation_in_detail.ipynb)

## Step 9. Write down what the evidence supports

`interpret(spec, req, stage, evaluation)` returns a report for the evaluated candidate. Use `summary()` for the sample size, observed frequency, confidence bound and decision. Use `text()` to inspect the assumptions and inputs of each additional bound. Report the horizon, tolerance, validation error, policy variant and simulator conditions with the result.

| Step | Functions |
|---|---|
| Specification | `BudgetSpec`, `Requirement`, `plan` |
| Environment | `BudgetWrapper`, `RunningMaxCost`, `Stage`, `barrierpg.problems` |
| Price and reserve | `frontier.check_runs`, `frontier.price_sequence_check`, `DualAscent`, `BudgetWrapper(true_budget_scale=...)` |
| Regularization | `ActionAnchor`, `ParameterAnchor`, `ParameterNoise`, `fit_actor_mean`, `set_log_std` |
| Evaluation | `BudgetMonitor`, `evaluate`, `evaluation_from_episodes`, `Evaluation.validate`, `validation.episodes_for_tolerance` |
| Reporting | `interpret(...).summary()`, `interpret(...).text()`, `plan(...).explain()` |
