# Practitioner guide: from an RL problem with a budget to a validated policy

This guide is for someone who trains reinforcement-learning policies and has a resource that can run out: fuel, a
battery, cash, an allowance of damage, an emissions quota. It assumes no knowledge of the paper behind the toolbox.
Where the paper's results tell you what to do, the guide states the conclusion in plain words and shows it on four
problems, each with its own notebook:

| Problem | The resource | What it shows |
|---|---|---|
| [A lunar lander with a fuel tank](../notebooks/01_lander_fuel_tank.ipynb) | fuel | imposing a budget on a controller already in use; what to anchor to |
| [A retailer that finances its stock from cash](../notebooks/02_retailer_cash.ipynb) | cash | a failure with a known cost; an incumbent rule as the starting point and as a benchmark |
| [A legged robot on a battery](../notebooks/03_robot_battery.ipynb) | charge | what the policy must observe (a clock); prices; a reserve margin against an uncertain battery |
| [A navigation robot with a damage allowance](../notebooks/04_navigation_allowance.ipynb) | allowance of damage | which requirement the task means; a price that is too high |

Each notebook builds its problem and trains it in view, with notes on what the theory implies at each step;
[`examples/your_problem.py`](../examples/your_problem.py) is a complete template, from the task to the report, to copy
for your own problem. The [under-the-hood track](under-the-hood/decisions-and-theory.md) ties each step to the paper's
results, using test problems with exact solutions.

## The method in brief

Barrier policy gradients turn a budget into something a standard learner can handle. A wrapper around your
environment adds the remaining budget to the observation, ends the episode when the budget is exhausted, and charges
a penalty at that moment. Any policy-gradient learner (PPO, SAC, ...) then trains on the wrapped environment
unchanged. Two quantities set the penalty.

* The **price of failure risk** `m` is a charge per period from the failure to the end of the horizon. When failure
  costs a known lump sum `F` in your application, the toolbox charges `F` once instead (a flat penalty).
* The **interior barrier** of scale `theta` charges a small amount every period that grows as the reserve shrinks,
  which keeps a reserve. `theta = 0`, the default, prices only the failure.

What the price does, in the paper's results and in plain words:

* A policy that is optimal for the priced objective is the best policy among all those that fail no more often
  than it does. The price does not choose a risk level for you; it chooses a point on the trade-off between
  utility and risk, and a higher price chooses a safer point.
* Raising the price lowers the attained risk, up to the learner's own error. A learner that falls further short of
  the optimum as the price rises does not reach a smaller risk by a higher price.
* The risk of the policy you deploy is established by testing that policy, not by trusting the training run.
  The toolbox's validation test is exact for any policy, however it was trained.

## Step 1. Check that the problem has a budget

The method applies when

1. a resource is consumed over time and the consumption is nonnegative, or can be made so (a resource that is also
   replenished, such as cash, is handled by `RunningMaxCost`);
2. running out is a failure that ends the episode or makes the rest of it worthless;
3. the requirement is a tolerated probability of running out within a horizon, or a known cost of running out.

It does not apply as it stands when the requirement holds at every step (a forbidden region, a speed limit), when
it limits the expected total cost rather than the probability of exhausting a limit, or when the loss grows with the size
of the excess rather than with the event. The [navigation notebook](../notebooks/04_navigation_allowance.ipynb) shows how one
benchmark task can be read in both ways, and why the two requirements differ.

## Step 2. State the requirement before choosing a method

```python
from barrierpg import BudgetSpec, Requirement
spec = BudgetSpec(budgets=[24.0], beta=0.999, ubar=100.0, u_minus=100.0, horizon=1000)
req = Requirement(T=999, alpha=0.05, zeta=0.05, family=10)
```

* `budgets` is the size of the resource; `horizon` is the length of an episode in steps if it has one.
* `beta` is the discount of the utility. With a horizon, a discount close to one keeps late failures from looking
  cheap.
* `ubar` and `u_minus` bound the reward per step from above and below. Only the certificates use them; a loose bound
  makes the certificates weaker, not wrong.
* `alpha` is the tolerated probability of running out within `T + 1` steps. `zeta` is the probability that the
  validation test certifies a policy that in fact violates the tolerance. They are different quantities.
* `family` is the number of policies (or horizons, or tests) you will look at before deploying one. The validation
  error is shared among them, so that picking the best-looking candidate does not inflate it.

If failure has a cost in the application (the cost of a bankruptcy, of a stranded robot, a contractual sanction),
write it down too. It is then the price, not a tuning parameter.

## Step 3. Check whether another method is the better choice

`plan(spec, req, ...)` states, for your problem, the cases in which another method is the better choice. In short:

* no failure is permissible and a set of states from which failure can always be avoided is known: use a method that
  guarantees safety with probability one (a safety layer or an infinite penalty);
* a prescribed tolerance must be met closely and you can tune a multiplier's step: an adaptive multiplier (a
  Lagrangian method) targets the tolerance directly;
* violations during training matter: use constrained policy optimization;
* the loss grows with the amount of the excess: use a constraint on the tail of the cumulative consumption (CVaR);
* the model is known and small: solve it exactly; the toolbox's validation applies to that policy too.

Two further cases concern the barrier itself: a price that charges far more for a failure than an episode earns
(normalize the reward or use a prescribed-level method), and a budget error of known size (tighten the budget instead
of adding a reserve).

## Step 4. Wrap the environment

```python
from barrierpg import BudgetWrapper, Stage
from barrierpg.problems.lander import fuel_cost    # fuel_cost(obs, action, next_obs, reward, info) -> [fuel burned]
env = BudgetWrapper(gym.make("LunarLanderContinuous-v3"), cost=fuel_cost, spec=spec, stage=Stage(m=0.5, clip_utility=False))
```

* `cost(obs, action, next_obs, reward, info)` returns the resource consumed by one step (one number per budget).
  For a resource that is also replenished, wrap the signed net use in `RunningMaxCost`: the wrapper then fails at the
  first time the cumulative use exceeds the budget.
* The wrapper appends the remaining budget, as a fraction, to the observation. The policy needs it.
* With a fixed horizon, the policy also needs to know how much of the episode is left: whether the remaining budget
  suffices depends on it. `observe_time=True` appends the elapsed fraction of the horizon. The robot of the
  [battery notebook](../notebooks/03_robot_battery.ipynb), whose task has no clock, was stranded far more often
  without it.
* If your task has large negative rewards (a crash penalty, a write-off), set `clip_utility=False`: only the failure
  is priced and the task's own rewards are left as they are.
* `true_budget_scale=0.9` judges failure against a budget 10% smaller than the one the policy observes. Use it to
  test how a policy copes with a misjudged budget.

`barrierpg.problems` contains the four problems of this guide; each module documents its mapping from the task to the
budget problem and can serve as a template.

## Step 5. Choose the price

**When failure has a cost, use it.** The retailer goes insolvent at a cost; charging that cost at insolvency makes the
objective the economic one, and no tolerance needs to be chosen. `Stage(m=0, penalty="multiplier", multiplier=F,
clip_utility=False)` charges a lump sum `F` once.

**When you have a tolerance, train at a few prices.** Short pilot runs at a few prices show the range in which the
attained risk crosses the tolerance; validation then decides. `plan` gives a conservative starting price that is
guaranteed to suffice, but it is usually far too high to be useful in practice. Four cautions follow from the
paper's results and from the notebooks.

* Compare the price with the other ways an episode can end badly. The price applies to running out only; if it makes
  running out costlier than, say, a crash, the learner may avoid running out by crashing. The lander did exactly
  that: higher prices barely changed how often it ran dry and made it crash more often.
* A price that charges far more for a failure than an episode earns degrades the learner: it may give up earning in
  order not to fail. The navigation robot almost stopped reaching goals at a price that charged about four times what
  an episode earns, while the legged robot and the retailer did well at two to three times. Watch the return in pilot
  runs; normalizing the reward helps; `plan(..., episode_earnings=...)` warns above three times.
* A higher price lowers the risk only as far as the learner can follow. `frontier.price_sequence_check` reads from
  pilot runs whether the learner's shortfall grows with the price; if it does, a higher price is not the remedy.
* A tolerance below what any policy of the network can reach is not reached by any price. Exploration noise alone
  keeps a stochastic policy's risk above zero.

**An adaptive multiplier** (`Stage(penalty="multiplier")` with the `DualAscent` callback) moves the penalty after
every rollout toward the level at which the failure rate equals the tolerance. Size its step from the level it has
to reach, read from your fixed-price pilot runs: `dual_step_suggestion(expected_multiplier, rollouts_to_reach)`.
A step sized from a level that is too low leaves the multiplier rising when training ends; a step far too large
makes it rise past the level at which the tolerance is met before the policy can respond. A learner trained from
scratch fails often at first, so a multiplier started with it rises for many batches, and with a small tolerance it
comes down only slowly: the retailer's multiplier for a 1% tolerance ended fifty times above the level it needed, and
the policies gave up much of their profit. Train at a few fixed prices first; a multiplier, if you use one, should
start from the policy and the level of the fixed-price run that comes closest to the tolerance.
`DualAscent.diagnose` reports whether the multiplier has converged (stable), is still moving (trending), or rose
while every episode failed (increasing). Set the target
below the tolerance by the width of the validation's confidence bound: a policy whose failure rate equals the
tolerance cannot pass a test at that tolerance.

## Step 6. Add a reserve margin when the budget is uncertain

A price on failure alone produces policies that use the budget up whenever that pays, and such policies fail often
when the budget turns out to be smaller than assumed. The interior barrier (`Stage(theta=...)`) keeps a graded
reserve at a cost in utility. The [robot notebook](../notebooks/03_robot_battery.ipynb) tests both against a battery
that holds 10% less than the robot believes. There, a price high enough to meet the tolerance with room to spare also
kept a reserve under the weaker battery, while the barrier at a lower price helped in two of three seeds and varied
widely across them; a reserve is worth testing for, whichever way it is obtained. If you know how large the error can
be, tightening the budget by that amount is cheaper than either.

## Step 7. Choose the regularizers for what they do

Two kinds of regularization act on different things, and each has a use.

* **Action regularization** (`ActionAnchor`) charges the policy, at every step, for how much its action distribution
  departs from a reference policy you trust. It keeps the policy close to that reference and keeps its risk of
  failure close to the reference's. It helps when the reference is trusted and the price or the simulator is not; with
  a correct price it costs utility.
* **Parameter regularization** (`ParameterAnchor`) pulls the network's weights toward a starting point, the prior.
  It makes the learning problem better conditioned, and it biases the learned policy toward the **prior policy**,
  the policy the network computes at the prior's weights. The learner inherits that policy's risk and utility.

The notebooks show the code of each: [notebook 1](../notebooks/01_lander_fuel_tank.ipynb) trains from a controller in
use with a parameter anchor and with an action anchor, and [notebook 2](../notebooks/02_retailer_cash.ipynb) fits a
network to a rule and trains from it, with and without an anchor. In short:

```python
anchor = ParameterAnchor(model, kappa_p=1.0).install()        # weight decay toward the current weights (the prior)
model.learn(steps, callback=[ActionAnchor(0.05, log_reference), monitor])   # charge log pi(a|s) - log reference(a|s)
fit_actor_mean(model, rule_mean_action, states); set_log_std(model, np.log(0.03))   # a prior fitted to a rule
```

How to choose, by what you have:

* **A controller in use** (the lander). Start training from it. If its behavior violates the new requirement, do not
  anchor its weights: the learner inherits the prior policy's risk. The lander's controller ran dry under the new
  tank, and a parameter anchor to it kept it running dry at every strength tried, down to 0.01 (runs with the coupled
  form of the anchor under Adam, the form of the paper's runs, where the pull's step is about the learning rate
  whatever its strength; see the end of this step). An action anchor to the controller, which charges departures of
  the action distribution instead, let it learn to save fuel while keeping its landings; its effect on crashes was
  mixed across seeds. When a parameter anchor is appropriate (a prior whose behavior you accept), anchor every
  parameter that shapes the policy, the action dispersion included; otherwise the learner can change the policy
  through the dispersion alone.
* **An incumbent rule** (the retailer's base-stock rule). Fit the network to the rule, on states that include the bad
  ones (fit on the states a noisy version of the rule visits), give it a small dispersion, and evaluate the fitted
  network as a policy before training: it is the prior policy, and it can be riskier than the rule. A learner started
  there stays near it, with or without an anchor; the retailer's learners started from scratch found more profitable
  policies. Starting from a rule begins near behavior you accept; it does not improve on the rule. Compare the rule,
  the learner from scratch and the learner from the rule. The rule is a candidate too, and it may be the better one.
* **Nothing trusted** (the navigation robot, the legged robot). Do not anchor. Keep any parameter regularization weak
  and use it for conditioning only.

Set the strength of a parameter anchor for how well the optimizer behaves, not to reach a convergence guarantee: the
strength the theory requires is far larger than useful and would leave the policy at the prior. Judge an anchor by
what the trained policy does and by its distance from the prior (`ParameterAnchor.distance()`), not by the number.
Treat a policy that fails validation as the signal to lower the anchor or raise the price, not to anchor harder.

How the anchor enters the learner must match the optimizer (the paper's guidance, its four implementation rules).
`ParameterAnchor` has two forms:

* **Decoupled** (`decoupled=True`, the default): the pull is applied to the weights after the optimizer's step,
  `theta <- theta - lr kappa_p (theta - theta_0)`, the weight decay of AdamW. Every weight then moves toward the
  anchor by the factor `1 - lr kappa_p` per step whatever Adam does with the policy gradient, so the strength keeps
  its meaning: calibrate it by the half-life, about `ln 2 / (lr kappa_p)` steps (`ParameterAnchor.half_life()`), and
  by the measured distance. With `ParameterNoise`, compare the measured spread with `sqrt(kappa1/kappa_p)`
  (`roles.noise_and_anchor_spread`): the injected noise adds no pull of its own. Under Adam this is a reference, not
  a prediction, since the policy-gradient step stays adaptive. A learning rate decayed to zero switches the pull off
  with it.
* **Coupled** (`decoupled=False`): the pull is added to the policy gradient before the step, the form of the paper's
  runs and of the shipped campaign records. Adam divides every gradient coordinate by its running root-mean-square, so
  the pull's step is about the learning rate whatever the distance to the anchor and whatever the strength, it is weak
  where the policy gradient is large and of fixed size where it is small, and it inflates Adam's normalizer, which
  shrinks the policy gradient's own step. The strength then loses its meaning as a curvature, and the spread under
  noise is proportional to the noise strength whatever the pull. Use this form to reproduce the paper's runs.

Strengths are comparable only within one form. Pull the actor's weights including the action dispersion (unless the
dispersion is meant to be free) and never the critic, which has no prior; `ParameterAnchor` does both. Record the
optimizer, its stability constant, the order of gradient clipping and pull, and the form.

## Step 8. Validate the policy you deploy

* **Compare policies only after training has converged.** `BudgetMonitor` records the failure rate, the utility, the
  priced objective and the action dispersion of every batch of training episodes. `monitor.convergence()` checks
  whether the run has stopped changing: it compares the last tenth of the batches with the tenth before, and a
  quantity counts as still changing when its change exceeds both its batch-to-batch noise and 5% of its size.
  `monitor.plot()` draws the curves, and `BudgetMonitor(check_every=k, stop_when_converged=True)` checks during
  training and can stop it. A short run is reported as too short to judge. Converged means that the run stopped
  changing, not that its policy is good: a learning rate that decays to zero makes any run stop changing at its end,
  and a run can stop in a bad place (the lander's anchored runs). Pass `training=monitor.convergence()` to
  `interpret`, and the report states it. The retailer's learners were still improving after a few million steps;
  compared then, they looked worse than the rule in use, and trained longer they earned more than it.
* **Validate the candidate that is deployed.** The stochastic policy and its mean action (the network without its
  noise) are different candidates with different risks; so are the policy before and after any change to its
  dispersion. Noise in the actions can carry much of the risk: the retailer's rule goes insolvent several times more
  often with noise in its orders. The mean action is often the better candidate (the robot, the retailer); validate
  it if you deploy it.
* **Test at the specified horizon.** A test at one horizon does not certify a longer one.
* **Share the error.** `validation.validate_direct(failures, n, alpha, zeta, family)` uses an exact binomial bound
  with the error divided among all the candidates you inspected.
* **Mind the resolution.** No failures in `N` episodes certify the floor `1 - zeta^(1/N)`, never zero, and a failed
  test does not prove a violation. `validation.episodes_for_tolerance` sizes the test.
* **Certificates from the value** bound the risk without a validation run, but over long horizons they are valid and
  uninformative; the direct test then carries the decision (the navigation notebook shows this).

## Step 9. Write down what the evidence supports

`interpret(spec, req, stage, evaluation, ...)` writes the report. Its `summary()` states the verdict in plain words:
how often the candidate ran out on fresh episodes, the validated bound on its risk, whether the requirement is met,
and what the test does not show. Its `text()` adds every bound with its assumptions, and what the evidence does not
establish: the risk after the horizon, other horizons, other candidates, convergence of the training run, and
behavior under a different simulator.

## Function index by step

| Step | Functions |
|---|---|
| Requirement | `BudgetSpec`, `Requirement` |
| Another method? | `plan`, `planning.when_to_use_another_method` |
| Wrap | `BudgetWrapper`, `RunningMaxCost`, `Stage`, `barrierpg.problems` |
| Price | `plan`, `frontier.check_runs`, `frontier.price_sequence_check`, `DualAscent`, `callbacks.dual_step_suggestion`, `DualAscent.diagnose` |
| Reserve | `Stage(theta=...)`, `BudgetWrapper(true_budget_scale=...)` |
| Regularizers | `ActionAnchor`, `ParameterAnchor`, `ParameterNoise`, `fit_actor_mean`, `set_log_std`, `advise_regularizers`, `regularizer_effects` |
| Validation | `BudgetMonitor.convergence`, `BudgetMonitor.plot`, `evaluate`, `evaluation_from_episodes`, `Evaluation.validate`, `validation.validate_direct`, `validation.resolution_floor`, `validation.episodes_for_tolerance` |
| Report | `interpret(...).summary()`, `interpret(...).text()`, `plan(...).explain()` |
