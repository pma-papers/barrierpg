# Decisions and the theory behind them

This document belongs to the toolbox's "under the hood" track, for readers who want each decision of the
[practitioner guide](../practitioner-guide.md) tied to the result of the paper *The Price of Failure Risk: Clipped
Barrier Policy Gradients with Convex Regularization* (Malo, Viitasaari, Suominen, Vilkkumaa, Tahvonen) that supports
it. It uses the paper's notation and its test problems, which have exact solutions. The
[under-the-hood notebooks](../../notebooks/under-the-hood/) run these problems with outputs, and the replication
repository walks through the paper's studies.

The paper does not rank barrier policy gradients against other constrained methods, and neither
does this guide. Its purpose is to say what the bounded barrier delivers, what can be
certified about the policy it returns, what each regularizer does, and when another method is the
better choice.

Results are cited by the titles the paper gives them, which do not change with its numbering;
[`results-map.md`](results-map.md) lists them with the functions that implement them.

## Vocabulary

The guide uses the paper's names, one name per concept.

| Symbol | Name | In the toolbox |
|---|---|---|
| `b_k`, `gamma_k`, `beta` | budgets, cost discounts, utility discount | `BudgetSpec` |
| `tau` | failure index: the first action after which some reserve is negative | `BudgetWrapper` |
| `p_T = P(tau <= T)` | finite-horizon risk over `T + 1` actions | `Requirement(T, alpha)` |
| `q_beta = E[beta^tau 1{tau < inf}]` | geometric risk: failure before an independent geometric time | `Evaluation.q_beta` |
| `m` | price of failure risk; a failure costs `m/(1-beta)` in discounted total | `Stage.m` |
| `max{-m, u - B(z')}` | the bounded barrier, the one-step reward clipped at `-m` | `BudgetWrapper` |
| `V_m`, `U`, `I_m` | clipped value, stopped utility, interior loss: `V_m = U - I_m - m q_beta/(1-beta)` | `Evaluation` |
| `theta` | scale of the interior barrier `B(z) = theta sum_k log(1 + b_k/z_k)`; `theta = 0` is the failure-only price | `Stage.theta` |
| `nu` | flat failure penalty (frozen multiplier, or adaptive multiplier under dual ascent) | `Stage(penalty="multiplier")`, `DualAscent` |
| `rho`, `eta`, `A` | action reference, weight and value of the action relative entropy | `ActionAnchor` |
| `Gamma`, `pi_Gamma`, `kappa` | prior law of the network's units, the prior policy it induces, strength of the parameter regularizer | `ParameterAnchor` |
| `kappa1`, `kappa2` | noise strength (the entropy part of a divergence) and pull strength (a potential such as an anchor) in the particle form | `ParameterNoise`, `ParameterAnchor`, `RegularizerMix` |
| `alpha`, `zeta` | tolerated failure probability; statistical error of validation (not interchangeable) | `Requirement` |

A *configuration* is a rule for the per-period reward (fixed price, fixed price with the interior
barrier, adaptive multiplier, anchored). A *candidate* is one fixed policy that may be deployed; the
stochastic policy and its mean-action policy are different candidates.

## The workflow at a glance

1. **Specify the requirement**: `BudgetSpec`, `Requirement`.
2. **Check whether another method is the better choice** (the cases below; `plan` states them for your
   problem).
3. **Fix the price, then certify**: `plan` derives the conservative price and the validation budget;
   train any learner on `BudgetWrapper`; `frontier.check_runs` checks runs at several prices.
4. **Buy margin deliberately**: `Stage(theta=...)` for the interior barrier, or a tightened budget when
   the size of the possible error is known; test with `BudgetWrapper(true_budget_scale=...)`.
5. **Choose regularizers for their role, and the prior from the problem**: `advise_regularizers`
   (also through `plan(..., regularizers=...)`), `ActionAnchor`, `ParameterAnchor`, `ParameterNoise`; afterwards
   `regularizer_effects`.
6. **Validate the deployed candidate**: `evaluate`, `Evaluation.validate`, `validate_direct`, the value
   certificates, and `interpret(...).text()` for the report.

```python
from bpg import BudgetSpec, Requirement, Stage, BudgetWrapper, evaluate, interpret, plan

spec = BudgetSpec(budgets=[1.0], beta=0.95, ubar=1.0)            # one budget, utility in [0, 1]
req = Requirement(T=19, alpha=0.05, zeta=0.05, family=2)        # p_19 <= 5%, two candidates share the error
p = plan(spec, req, sup_U=5.4, expected_failure_rate=0.01)      # conservative price, validation budget, notes
print(p.explain())

stage = Stage(m=1.0)                                             # a fixed failure-only price
env = BudgetWrapper(my_simulator, cost=my_step_cost, spec=spec, stage=stage)
# ... train any policy-gradient learner on env ...
ev = evaluate(lambda: BudgetWrapper(my_simulator, cost=my_step_cost, spec=spec, stage=stage),
              policy, 20000, spec=spec, stage=stage, T=19)
print(interpret(spec, req, stage, ev, sup_U=5.4).text())
```

## 1. Specify the requirement

State the budgets, the horizon and the tolerated failure probability before choosing a learning
mechanism. The choice of mechanism follows from the requirement, not the other way round.

* **The problem.** `BudgetSpec(budgets, beta, gammas, ubar, u_minus, horizon)`: the budgets `b_k`, the
  cost discounts (`1` for a plain cumulative allowance), the utility discount, a valid upper bound
  `ubar` on the one-step utility (every certificate needs it, and a loose `ubar` weakens them), the
  magnitude `u_minus` of a lower bound on the utility, and the physical horizon `H` if the task ends
  after `H` actions (`None` for the absorbing, infinite-horizon form).
* **The requirement.** `Requirement(T, alpha, zeta, kind, family)`. `kind="finite"` is the
  operational requirement `p_T <= alpha`. `kind="geometric"` is `q_beta <= alpha`, the natural
  criterion when the horizon is random, as for an asset whose maintenance access arrives at random;
  substituting it for a fixed operational horizon needs a justification, and the two are linked by
  `p_T <= beta^(-T) q_beta` (`certificates.finite_from_geometric`). `kind="eventual"` bounds
  `P(tau < inf)`; neither a small `q_beta` nor a test at one horizon certifies it, and the toolbox
  certifies it only with known bounded discounted costs (`certificates.eventual_risk_structural`).
  `family` is the number of simultaneous statements (candidates, horizons, tests) that share the
  error `zeta`; allocate it before taking minima or selecting a policy.
* **When failure has a price in the application itself** (downtime until the next maintenance access,
  a contractual penalty, a lost season), the failure-only clipped value with that per-period price is
  the problem to be solved, and the price is not a tuning parameter. A lump sanction at an
  observation date is a flat penalty at the failing step, the frozen-multiplier form:
  `Stage(m=0, penalty="multiplier", multiplier=F, clip_utility=False)` without a dual-ascent callback. With a
  physical horizon its value certificate is `validation.lump_sanction_risk`, and
  `certificates.conservative_multiplier` gives a conservative starting penalty.
* **A resource whose step use can be negative** (net emissions in a week, a battery that recharges) needs
  nonnegative costs for the wrapper: `RunningMaxCost(signed_use)` charges the increment of the running maximum of
  the cumulative use, whose first crossing of the budget is the first crossing of the cumulative use.
* **For a chance requirement** the bounded barrier is a surrogate, not the constraint. The
  requirement is met by a certificate when it is informative and by validation of the deployed
  candidate otherwise (section 6 of this guide).
* `plan(..., violations_during_training_matter=True)` and a zero tolerance trigger warnings: both point
  to another method.

## 2. When to use another method

The paper's section *Which formulation to prefer* states when the clipped barrier is the choice and when another
method is the better one. The cases are repeated here so that this guide can be read on its own; each rests on a
result or a measurement of the paper. `planning.when_to_use_another_method` returns them, and `plan` states them for
your problem.

The clipped barrier is the choice when failures can be permitted in simulation and deployment tolerates a positive
failure probability or carries a finite failure cost; when the tolerance is uncertain or may be unattainable, since
a fixed price does not change during training and the attained risk is monotone in it, so the tolerance is
approached by training at several prices and validating each; when the failure cost is known (a lump sum or an
overage tariff), which sets the price directly, a lump sum `F` charged once at failure giving `m = (1-beta) F`; and
when a reserve margin before exhaustion is worth its cost. Another method is the better choice in five cases.

1. **No failure is permissible and a controlled-invariant set is known.** A finite price prices failure
   risk; it does not prohibit failure. If the action reference can exhaust a budget with positive
   probability, so can every represented policy: bounded scores impose a risk floor,
   `p_T(pi_mu) >= exp(-2 L0 (T+1)) p_T(rho)` (*The represented class*; `certificates.risk_floor`), and
   no price removes it. An almost-sure method (an infinite failure penalty, a control-barrier safety
   layer, or a method that learns which actions keep probability-one budget constraints) addresses this
   specification when a controlled-invariant set is known. `plan` flags `alpha = 0`: a positive price
   never prohibits failure, no multiplier converges when every policy of the class has positive risk,
   and a failure-free validation certifies only the resolution floor.
2. **A prescribed level is attainable with slack and the multiplier's step can be tuned.** A fixed price
   leaves the attained risk to the optimizer, and unused risk costs utility (*Matched-risk performance
   comparison*). An adaptive multiplier, as in a Lagrangian chance-constrained policy gradient, targets
   the level directly when it lies above the risk floor of the class. Its step must be tuned: with too
   large a step, and an initial policy that fails often, the multiplier increases without bound (the
   fuel-tank study). The toolbox's own prescribed-level option is `Stage(penalty="multiplier")` with
   `DualAscent`; see section 3 of this guide for its step and diagnosis.
3. **Violations during training matter.** Constrained policy optimization keeps the iterates near the
   constraint; the bounded barrier is trained where failures are permitted, in a simulator.
4. **The loss grows with the amount by which the consumption exceeds the budget.** A failure price
   charges the event of exhaustion, not its size; the CVaR constraint on the cumulative consumption
   charges the size.
5. **The model is known and small.** In the emissions application the exact dynamic program, and a
   re-planning rule tuned for each sanction, were better than either learner (PPO and SAC). The
   certificates and the validation apply to such policies as well, since they hold for whatever policy
   is deployed: evaluate the policy through `BudgetWrapper`, validate it with `Evaluation.validate` or
   `validation.validate_direct`, and read it with `interpret`.

Two further cases concern the barrier itself.

* **The required price charges far more for a failure than an episode earns.** A large price degraded
  the controllers of the fuel-tank study; reward normalization or a prescribed-level formulation is the
  alternative. `plan(..., episode_earnings=...)` warns when the discounted total of the price exceeds
  ten times what an episode earns.
* **The size of a possible allowance error is known.** Tightening the allowance is then cheaper than the
  interior margin (section 4 of this guide).

## 3. Fix the price, then certify

The parts of *Fixed price of failure risk* (optimality at the attained risk, the price-risk law,
monotonicity in the price, the learning form) and *Matched-risk performance comparison* hold at a fixed
price `m`, so training at many prices is not required.

* **Start from the conservative price.** When the class contains a policy that never fails at finite
  barrier cost (the idle policy, for example), the price-risk law
  `q_beta <= q_0 + (1-beta)[sup U - U_0 + I_0 + delta]/m` gives a price at which the attained risk is
  certified whatever the optimizer returns, up to its optimization error `delta`. `plan(spec, req,
  sup_U=...)` computes it with the idle comparison policy; for another comparison policy use
  `certificates.conservative_price(alpha, beta, sup_U, U0, I0, q0, delta, u_minus)` directly. The law
  is conservative (by a factor of twenty to thirty in the wear study), so the price it gives is a safe
  start, not the price to aim for.
* **Let the certificates and validation bound the attained risk.** After training, the value
  certificates (`certificates.risk_from_value`, `validation.risk_from_validation_value`,
  `validation.physical_horizon_risk`) and direct validation (section 6 of this guide) bound the risk of
  the candidate. A bound on `q_beta` also bounds when failures occur: the earliest `eps`-fraction of
  failures occurs on average no earlier than `log(eps/q_beta)/log(1/beta)` periods
  (`certificates.earliest_failures_time`; *The geometric risk as a risk measure of the failure time*).
* **Retrain at a higher price only if the tolerance is missed.** `certificates.price_monotonicity`
  bounds how much the attained risk can rise between two prices, given the optimization errors.
  `frontier.check_runs(runs, beta, sup_U, U0, I0, q0, delta)` checks runs at several prices against the
  price-risk law, monotonicity and dominance.
* **Check that raising the price can work.** A sequence of prices reaches a tolerance when some policy
  has risk to spare and the charged errors (the comparison policy's interior loss, the regularization
  cost and the optimization error) grow slower than the price (*Price sequences reaching a tolerance*). In
  the emissions study PPO failed this condition: its shortfall grew in proportion to the sanction. On short
  pilot runs at a few prices,
  `frontier.price_sequence_check(prices, shortfalls)` reads the condition from measured shortfalls; on a
  new problem, the shortfall from the best structured policy at the same price is a lower bound on the
  optimization error. A rising shortfall or crash rate is the signal to stop raising the price.
* **Where on the frontier the optimum lies is not controlled.** A fixed-price optimum is optimal at the
  risk it attains, but risk left unused against a tolerance costs utility at the price;
  `certificates.matched_risk_utility_gap` bounds the shortfall.
* **An adaptive multiplier targets a tolerance directly** when the tolerance lies above the risk floor
  of the class; no multiplier converges below it. Use `Stage(penalty="multiplier")` and `DualAscent(alpha,
  step)`. Take the step from `callbacks.dual_step_suggestion(expected_multiplier, rollouts_to_reach)`,
  with the expected multiplier the penalty at which fixed-price runs of the same learner attain about the
  tolerance; a learned policy attains more risk than the exact optimum at the same price, so the exact
  frontier understates it. The step moves the multiplier by `expected_multiplier / rollouts_to_reach` in a
  rollout in which every episode fails, and by the fraction `rate - alpha` of that otherwise, so near the
  tolerance it moves slowly relative to the policy's updates. The default, a twentieth, suits long runs;
  with a few hundred rollouts a larger fraction is needed (notebook 2 uses one half). Read the trace with
  `DualAscent.diagnose(beta)`, which applies the paper's two criteria for convergence: the multiplier
  stops moving, and the attained risk lies within the confidence bound of the tolerance. The status is
  *stable* (both hold), *increasing* (every episode fails and the flat penalty carries no gradient:
  lower the step, raise a fixed price, or start from a controller that sometimes survives), *inactive*
  (the tolerance does not bind), *trending* (still moving; one that keeps rising while the failure
  rate stays above the tolerance signals a tolerance below the risk floor), *off-target* (no drift, but
  the rollouts cannot resolve the tolerance), or *short* (too few rollouts). The drift is tested on the
  multiplier's updates, since a slope fitted to the levels of a running sum reports a trend in almost
  any window. A multiplier that converges at the tolerance itself
  does not pass validation at that tolerance: set the prescribed level below the tolerance by the width
  of the confidence bound, just as a fixed price has to be chosen with one.
* **Price and conditioning.** A higher price raises the curvature that a convergence argument requires
  and can worsen the conditioning of the learner (the reward scale). Reward normalization helps; a
  price that charges far more for a failure than an episode earns is one of the cases of section 2.
* **Clipping and utilities below `-m`.** Clipping at `-m` truncates utility values below `-m`, such as a
  crash penalty. Choose `m >= u_minus`, or charge the price at failure alone: the frozen-multiplier
  form `U - nu q_beta`, `Stage(clip_utility=False)`, to which the fixed-price results apply verbatim.
  `plan` warns when `m < u_minus` with clipping on.
* **Physical horizon.** For a task that ends after `H` actions, `BudgetSpec(horizon=H)` charges the tail
  `m (1 - beta^(H-t))/(1 - beta)` at a failure at step `t`; `bpg.spec.failure_tail` gives it, and
  `bpg.spec.price_from_multiplier_physical` converts a flat penalty into a price at a typical failure time.
* **Check the resolution before retraining.** `validation.value_resolution_check` says whether the
  value-based test can certify the tolerance with the planned sample and horizon at all; increasing
  `m` cannot repair a failure of this check. `validation.episodes_for_tolerance` sizes the direct test.

## 4. Buy margin deliberately

* A failure-only price leaves no reserve: its optimal policies exhaust the budget whenever that is
  profitable, and a modest error in the budget then causes frequent failures. In the wear study an
  allowance ten percent smaller than assumed raised the breakdown probability of the failure-only
  price `m = 1` from about two percent to a third of the episodes.
* The interior barrier, `Stage(theta > 0)`, buys graded protection against such errors at a utility
  cost bounded by its interior loss; the fixed-price results still apply, net of that loss. In the wear
  study `theta = 0.1` lowered near-exhaustion and the breakdown probability under the reduced
  allowance by an order of magnitude.
* If the size of the possible error is known, a correspondingly tightened budget is the more
  economical margin; the barrier is the graded margin for an error of unknown size.
  `plan(..., allowance_error_known=True/False)` states the corresponding advice.
* Test the margin before deployment: `BudgetWrapper(..., true_budget_scale=0.9)` judges failure
  against a reduced true budget while the policy observes the nominal reserve, and `BudgetMonitor`
  and `Evaluation.near` report near-exhaustion frequencies. With a positive barrier,
  `certificates.near_exhaustion_bound` with `LogBarrier.lower_level` certifies the probability of
  coming within a chosen distance of exhaustion.

## 5. Choose regularizers for their role

Two regularizers act on different objects, and they do different things. Each creates no safety
that its target lacks.

### 5.1 Action regularization and parameter regularization

* **The action relative entropy** (weight `eta`, to the action reference `rho`) is a reshaped reward;
  `ActionAnchor` charges it at every decision up to and including the failing one, as the paper's `A`
  does (`include_failing_action=False` charges the steps before failure only).
  It holds the policy near the action reference and bounds the departure of the trajectory risk from
  the reference's: `kl(q || q_rho) <= A` (*Action regularization and trajectory risk*; `certificates.action_risk_bound`,
  `kl_upper_inverse`). The bound is informative only for small `A`, that is, when the learned policy
  stays close to the reference and gains little. Use the action relative entropy when the action
  reference is trusted and the price or the model is not; otherwise a correctly chosen price dominates
  it, and at a correct price it costs utility. It convexifies the action-law problem, not the
  network's, so the convergence theorem credits it with no curvature and charges it the deficit
  `eta C_A`. An entropy bonus is the special case of a uniform action reference; it is not a parameter
  regularizer. Implementation: `ActionAnchor(eta, log_reference)`; estimate `A` with
  `evaluate.stopped_action_relative_entropy`.
* **The parameter regularizer** (strength `kappa`, toward a prior law `Gamma` of the network's units)
  supplies the curvature that makes the population learning problem converge, equally for every
  admissible regularizer (*Fixed-stage population convergence*). The policy-wise certificates do not need it, and its bias is never
  silent, since the certificates and validation are evaluated on the learned policy.

### 5.2 The roles: noise, pull and spread

In the particle form of the population flow (*The prior and the particle form of the flow*) each unit's parameters move by the policy
gradient, a pull toward the prior and any reference, and noise:

```
dx_i = [policy gradient] dl - [kappa1 grad W(x_i) + kappa1 omega grad G'(rho_n)(x_i) + kappa2 grad P(x_i)] dl + sqrt(2 kappa1) dB_i
```

* **Noise.** The entropy part of relative entropy is Langevin noise of variance `2 kappa1` per unit time on
  every parameter. It is what lets units leave a region where the value gradient vanishes. No standard
  learner injects it; `ParameterNoise(model, kappa1).install()` adds Gaussian noise of variance
  `2 kappa1 x lr` after each optimizer step, the learning rate being the step of flow time.
* **Pull.** The prior's potential pulls toward its center (for `N(x0, sigma^2 I)`, weight decay of weight
  `kappa1/sigma^2`), and a potential `P` with strength `kappa2` pulls toward a reference of one's choice
  with no noise attached (`ParameterAnchor`). A potential alone collapses the population to a single
  point in the certified regime (*Collapse under a potential alone*).
* **Spread.** An integrand of faster growth limits how far the law crowds relative to the prior. It needs
  a density, so it has no finite-network form without a density estimate.

Relative entropy alone ties noise and pull to one strength; the general class unties them, and every
combination converges by the same argument. Under a plain gradient pull the stationary spread is
`sqrt(kappa1/(kappa1/sigma^2 + kappa2 lambda_P))` (`roles.stationary_spread`, `kappa1_for_spread`). In
the paper's PPO study of the roles Adam normalized the pull, and the measured spread was proportional
to `kappa1` instead, about `sqrt(2) kappa1`: the roles survived the finite network, the formula did not.
Risk and utility stayed at the anchored values while the spread was small against the policy's
dispersion and then moved, with seeds splitting; noise without a pull acted as exploration and abandoned
the prior. Treat `kappa1` as a knob to vary and measure. `RegularizerMix(kappa1, kappa2).explain()`
describes a combination.

### 5.3 The prior is a policy

* The regularized solution inherits the prior policy's risk and utility (*Improvement over the prior policy*). Specify
  the prior as the policy the application accepts when learning gains nothing:
  * a **risk-free or hard-feasible policy** when one exists (the idle policy of the wear study); the
    bias then lowers risk outright, and learning improves on that policy within the risk slack;
  * the **incumbent** when one is in operation: fit the actor to it (`callbacks.fit_actor_mean`), set the
    dispersion to the incumbent's (`callbacks.set_log_std`), and anchor all actor parameters to the fitted
    `theta_0` with `ParameterAnchor(model, kappa_p).install()`;
  * the **centering construction**, which makes the prior policy equal to the action reference, when
    nothing is trusted.
* **Evaluate the prior policy as a candidate before training** (its `q_beta`, `U` and near-exhaustion),
  with `evaluate` on the actor at `theta_0`. In the paper's parameter-anchor study the actor fitted to
  the incumbent rule but carrying the learner's default dispersion was a policy with about six times
  the rule's breakdowns, and the anchored learner inherited that policy, not the rule.
* **The prior must cover every parameter that shapes the policy**, the action dispersion included. A
  prior on the mean network alone leaves the dispersion free, and the learner can move the whole
  distance between the prior policy and the optimum through that one parameter: in the wear study the
  mean-only anchor kept the mean network at `theta_0` while the dispersion grew from 0.37 to 7.5 and
  the policy broke down in a third of the episodes. `ParameterAnchor(include_log_std=True)`, the
  default, anchors the dispersion; `regularizer_effects` flags a dispersion escape.
* **Gaussian prior width.** For a Gaussian prior `N(x_0, sigma^2 I)` the relative entropy is negative
  entropy of weight `kappa` plus weight decay toward `x_0` of weight `kappa/sigma^2`. The rate depends
  on `(kappa, sigma)` only through `kappa/sigma^2`, the minimizer on both, and the width also changes
  the induced prior policy unless the features are centered at the prior. Do not concentrate the units
  where the features are flat: a prior pinned in a flat region turns regularization into stagnation
  (the plateau of the solvable flow). `convergence.gaussian_prior_curvature(sigma)` gives the curvature.
* **Expect the learned policy at the prior's risk, not at the risk slack the bounds grant.** The slack
  terms (`convergence.risk_slack_from_prior`, `action_slack_from_prior`) bound every policy with the
  given divergence; a learner that also optimizes utility lies orders of magnitude inside them (two to
  five orders in the solvable flow). What the bounds get right is the direction of the pull and the
  inheritance of the prior policy's risk.

### 5.4 The integrand

Every regularizer convex along generalized geodesics gives the same rate above the same threshold, so
the integrand is chosen for what it controls:

* relative entropy alone for the smallest bias;
* a faster-growing integrand, such as the power integrand `G(v) = v^r/(r-1)` added to relative entropy,
  to bound the mass of the unit law where it is much denser than the prior (a diversity requirement
  stated in advance) or to match a divergence the application prescribes; it keeps the minimizer closer
  to the prior;
* a convex potential when the initial law is a finite network, since a potential is finite on
  finitely supported laws while the relative-entropy families are not.

`advise_regularizers(..., want_unit_spread=..., specified_divergence=..., starting_from_finite_network=...)`
states the corresponding recommendation with its reason.

### 5.5 Weight decay and the strength

* Quadratic weight decay toward `theta_0`, `kappa_p (theta - theta_0)` added to the actor's gradient,
  is an admissible potential regularizer with curvature equal to its weight, and it is the pull part of a
  Gaussian prior that a finite network can carry; `kappa_p` is `kappa/sigma^2` in the paper's notation.
  `ParameterAnchor` implements it. Under a potential, noiseless continuous-time training of a network of
  any width is the population flow from the network's own empirical law, with the same contraction and
  rate (*Finite-network training under a potential*; `convergence.finite_training_transfer`). Discrete steps and PPO's or SAC's
  estimators are outside that result.
* **Set `kappa` for conditioning, not at the certified threshold.** The threshold of the convergence
  theorem is sufficient, not necessary, and grows with the price; in the solvable flow every flow
  converged orders of magnitude below it, and at the certified strength the minimizer was the prior
  itself. Choose the smallest strength at which the optimizer is well behaved.
  `convergence.curvature_budget(constants, beta, m, ubar, kappa, lambda_H, eta)` and
  `convergence.certified_kappa` report what the theorem says for a stage, as a diagnostic. The deficits
  come from bounds on how fast the policy kernel and its relative entropy change when the units move
  (*Value and action-law curvature*): `convergence.KernelConstants(B1, B2, K0, K1, K2)` states them for any actor class,
  and `FeatureConstants(L0, L1, L2)` is the softmax instance.
* Treat a failed certificate as the signal to lower `kappa` or to raise the price, not to raise
  `kappa`.

### 5.6 The joint effect

When both regularizers are used, their biases add in every certificate, `R = eta A + kappa H` of the
comparison law; the action term pulls the action law toward `rho`, the parameter term pulls the unit
law toward `pi_Gamma`, and the two targets coincide only under the centering construction. The action
term also adds `eta C_A` to the curvature that the parameter term must supply.

```python
from bpg import advise_regularizers, regularizer_effects

advice = advise_regularizers(spec, stage, action_reference_trusted=True, price_trusted=False,
                             incumbent_available=True, prior_sigma=1.0)
print(advice.explain())
# ... after training, with anchor = ParameterAnchor(model, kappa_p).install() ...
eff = regularizer_effects(spec, stage, dict(q_beta=ev.q_beta, U=ev.U, q_se=ev.q_se, U_se=ev.U_se),
                          prior_policy=dict(q_beta=..., U=...), **anchor.effects_inputs())
print(interpret(spec, req, stage, ev, effects=eff).text())
```

## 6. Validate the deployed candidate

* **Validate the candidate that is deployed.** Reward units, action dispersion, critic gradients and
  update counts can decide outcomes at a fixed interaction budget. Taking the deterministic mean,
  reducing the dispersion after training or changing an entropy coefficient creates a different
  candidate; evaluate and validate it directly with `evaluate(..., T=T)` and
  `Evaluation.validate(alpha, zeta, family)`.
* **At its specified horizon.** A passed test at one horizon does not certify a longer one.
* **With the error allocated.** `validation.validate_direct(failures, n, alpha, zeta, family)` uses the
  one-sided exact binomial upper endpoint at the error `zeta/family`; allocate the error over every
  simultaneous statement before selecting among candidates.
* **With the resolution in mind.** Zero failures in `N` episodes certify the resolution floor
  `1 - zeta^(1/N)` (`validation.resolution_floor`), never zero; failure to certify does not prove
  violation. `validation.episodes_for_tolerance(alpha, err, expected_failure_rate)` sizes the sample.
* **Under geometric stopping** (`BudgetWrapper(geometric_stop=True)`, the convention under which a
  learner with algorithmic discount one optimizes the discounted objective exactly), the episode
  event is `q_beta` itself and `p_T` is censored; validate `q_beta` and use `p_T <= beta^(-T) q_beta`,
  or evaluate without geometric stopping to test `p_T` directly. `interpret` handles both cases.
* **What a small objective gap does not give.** It removes neither representation bias nor a poor
  action reference. An uninformative sufficient bound does not establish failure; direct validation or
  additional cost-tail structure may still certify the requirement.
* **Finite networks.** *Population-to-network transfer* gives widths
  (`certificates.width_for_target`, `width_for_direct_target`, `finite_network_transfer`) that are
  typically far above the networks used, which is why deployed candidates are validated directly.
* **The report.** `interpret(spec, req, stage, ev, ...)` states in what sense the candidate solves the
  problem (optimal at the risk it attains, bounded risk, validated at a horizon), what each regularizer
  did to it, and what the evidence does not establish (eventual risk, other horizons, other
  candidates, discrete stochastic training, model error). `BudgetMonitor` supplies the training traces
  it reads.
* **Your own evaluator.** For many candidates on common random numbers a vectorized simulator is faster
  than the wrapper's one-episode-at-a-time loop; `evaluation_from_episodes(failed, tau, U, I, ...)` builds
  the same `Evaluation` from its per-episode arrays.

## Function index by decision

| Decision | Functions |
|---|---|
| Specify the requirement | `BudgetSpec`, `Requirement`, `certificates.finite_from_geometric`, `certificates.earliest_failures_time`, `certificates.eventual_risk_structural` |
| Detect conditions for another method | `planning.when_to_use_another_method`, `plan`, `certificates.risk_floor`, `certificates.prescribed_level_attainable` |
| Fix the price | `plan`, `certificates.conservative_price`, `certificates.conservative_multiplier`, `certificates.price_risk_law`, `certificates.price_monotonicity`, `frontier.check_runs`, `frontier.price_sequence_check`, `certificates.matched_risk_utility_gap`, `certificates.matched_risk_signed` |
| Run any learner | `BudgetWrapper`, `RunningMaxCost`, `Stage`, `bpg.spec.failure_tail`, `BudgetMonitor` |
| Adaptive multiplier | `Stage(penalty="multiplier")`, `DualAscent`, `callbacks.dual_step_suggestion`, `DualAscent.diagnose` |
| Buy margin | `Stage(theta=...)`, `BudgetWrapper(true_budget_scale=...)`, `certificates.near_exhaustion_bound`, `LogBarrier.lower_level` |
| Choose regularizers | `advise_regularizers`, `ActionAnchor`, `ParameterAnchor`, `ParameterNoise`, `fit_actor_mean`, `set_log_std`, `RegularizerMix`, `convergence.gaussian_prior_curvature`, `convergence.curvature_budget`, `convergence.certified_kappa`, `convergence.finite_training_transfer` |
| Read their effects | `regularizer_effects`, `ParameterAnchor.effects_inputs`, `ParameterNoise.spread`, `evaluate.stopped_action_relative_entropy`, `certificates.action_risk_bound` |
| Certify and validate | `evaluate`, `evaluation_from_episodes`, `Evaluation.validate`, `validation.validate_direct`, `validation.resolution_floor`, `validation.episodes_for_tolerance`, `certificates.risk_from_value`, `validation.risk_from_validation_value`, `validation.physical_horizon_risk`, `validation.lump_sanction_risk`, `validation.value_resolution_check` |
| Report | `interpret(...).text()`, `plan(...).explain()` |
