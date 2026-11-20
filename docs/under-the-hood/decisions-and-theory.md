# Theory and implementation for resource-budget policies

The central distinction is between the objective used during training and the condition that must hold at deployment. In the paper's continuing formulation, the clipped value prices geometric failure risk against stopped utility net of interior loss. A deployment requirement can instead limit failure by a fixed horizon. A bound on the value or independent evaluation of the policy can establish that requirement under the corresponding assumptions.

This guide follows *The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization* (Malo, Viitasaari, Suominen, Vilkkumaa and Tahvonen). The technical notebooks use a wear-allowance problem with a grid-based benchmark and an analytically tractable population model. The results map cites each result by title so references remain usable across manuscript revisions. [Technical notebooks](../../notebooks/under-the-hood/) · [Results map](results-map.md) · [Replication repository](https://github.com/pma-papers/bpg-replication)

## Vocabulary

| Symbol | Name | In the toolbox |
|---|---|---|
| `b_k`, `gamma_k`, `beta` | budgets, cost discounts, utility discount | `BudgetSpec` |
| `tau` | first action index after which any reserve is negative | `BudgetWrapper` |
| `p_T = P(tau <= T)` | finite-horizon risk over `T + 1` actions | `Requirement(T, alpha)` |
| `q_beta = E[beta^tau 1{tau < inf}]` | geometric risk: failure before an independent geometric time | `Evaluation.q_beta` |
| `m` | per-period price of failure risk, with a continuing charge at failure of `m/(1-beta)` in discounted total | `Stage.m` |
| `max{-m, u - B(z')}` | the clipped barrier, the one-step reward clipped at `-m` | `BudgetWrapper` |
| $V$, $U$, $V_m$, $\mathcal I_m$ | original-task utility, stopped utility, clipped value and interior loss | `Evaluation.U`, `Evaluation.V_m`, `Evaluation.I_m` |
| $\theta_B$ | scale of the log barrier $B(z)=\theta_B\sum_k\log(1+b_k/z_k)$ on positive reserves | `Stage.theta` |
| `nu` | flat failure penalty (fixed multiplier, or adaptive multiplier under dual ascent) | `Stage(penalty="multiplier")`, `DualAscent` |
| `rho`, `eta`, `A` | action reference, weight and value of the action relative entropy | `ActionAnchor` |
| `Gamma`, `pi_Gamma`, `kappa` | prior law of the network's units, the prior policy it induces, strength of the parameter regularizer | `ParameterAnchor` |
| `kappa1`, `kappa2` | noise strength (the entropy part of a divergence) and drift strength (a potential such as an anchor) in the particle system | `ParameterNoise`, `ParameterAnchor`, `RegularizerMix` |
| `alpha`, `zeta` | tolerated failure probability and statistical error budget for validation | `Requirement` |

A training configuration fixes the stage, learner settings and any regularization. A candidate is a fixed policy considered for deployment. A stochastic policy and the policy using its mean action are distinct candidates.

## The workflow at a glance

1. Specify the budget, first-crossing event, horizon, tolerance and validation error in `BudgetSpec` and `Requirement`.
2. Choose the objective and any constraint. `plan` describes the cases relevant to the inputs. The [formulation tutorial](../formulation-comparison.md) compares the available objectives.
3. For a priced objective, use `BudgetWrapper` and compare fitted policies at selected prices. Use `frontier.check_runs` to compare results under the stated benchmark assumptions.
4. When small reserves carry a cost, specify an interior barrier or other reserve penalty. Evaluate the candidate under nominal and reduced budgets if the latter represent relevant uncertainty.
5. Select references and strengths separately for action and parameter regularization. Evaluate the policy induced by each reference.
6. Freeze the candidate and evaluate it on independent episodes. Use `interpret` to inspect the direct test and any applicable value bounds.

```python
from barrierpg import Requirement, Stage, evaluate, interpret, plan
from barrierpg.examples.wear import IncumbentRule, WearModel, make_wear_env

wear = WearModel()
spec = wear.spec()
req = Requirement(T=19, alpha=0.05, zeta=0.05)
p = plan(spec, req, sup_U=spec.ubar / (1 - spec.beta), expected_failure_rate=0.01)
print(p.explain())

stage = Stage(m=1.0)
ev = evaluate(lambda: make_wear_env(stage, wear, geometric_stop=True),
              IncumbentRule(seed=0), 20_000, spec=spec, stage=stage, T=req.T, seed=100_000)
print(interpret(spec, req, stage, ev).text())
```

The geometric clock produces an estimate of breakdown risk before maintenance. The report tests the finite-horizon
requirement by comparing its upper confidence bound with $\alpha\beta^T$.

## 1. Specify the requirement

`BudgetSpec` describes the resource problem independently of the learning algorithm. Its `budgets` and `gammas` give the initial reserves and their cost discounts. Its `beta` discounts utility. The fields `ubar` and `u_minus` give bounds $-u_-\le u\le\bar u$ where the bound being used requires them. Set `horizon=H` for a task ending after $H$ actions.

`Requirement(kind="finite")` tests $p_T$ through action $T$, so it covers $T+1$ actions. Under independent geometric stopping with per-action survival probability $\beta$, the observed breakdown event has probability $q_\beta$. Use `kind="geometric"` when that is the deployment requirement. A geometric risk bound can also bound $p_T$ through $p_T\le\beta^{-T}q_\beta$. An eventual-failure requirement calls for additional structural information, such as the uniform cost bounds in the [supplementary certificate](supplementary-results.md#2-structural-certificate-for-the-probability-of-ever-failing).

For `family` simultaneous statements, allocate the error budget before examining the validation data. At a fixed validation horizon, the one-sided confidence bound applies to the tested event and the specified candidate. Where an economic loss is already given, it can set the failure charge independently of the testing criterion.

## 2. When to use another method

The appropriate formulation is determined by the quantity the decision maker intends to control. A known cost of failure can enter a priced objective. A time-dependent failure tolerance can be represented as the expectation of an undiscounted first-crossing indicator by the required horizon. A limit on mean cumulative consumption instead constrains the expectation of the consumption itself. A CVaR constraint limits the mean of a specified loss in its upper tail. See the [formulation tutorial](../formulation-comparison.md).

Pathwise feasibility calls for a model or a policy class that can establish feasibility on every relevant realization. Bounded-score policies with a reference that fails with positive probability retain a positive lower bound on risk. Under the paper's representation assumptions, `certificates.risk_lower_bound` computes $p_T(\pi_\mu)\ge e^{-2L_0(T+1)}p_T(\rho)$. An exact one-sided binomial bound from a finite sample is positive even when every validation episode succeeds.

CPO uses a local constraint on estimated cost during an update. The final policy's cost must still be measured at the intended horizon. If the model is small and known, backward induction can compute an optimal priced or pathwise feasible policy. Finite-scenario selection compares a supplied policy list. [Dynamic programming guide](../dynamic-programming.md) The toolbox's evaluation functions apply to these policies as well.

## 3. Fix the price, then certify

At a continuing failure price $m$, the clipped value is $V_m=U-\mathcal I_m-mq_\beta/(1-\beta)$. Here $U$ is stopped utility and $\mathcal I_m$ is the expected discounted loss from the interior penalty before failure. An exact maximizer of this objective maximizes $U-\mathcal I_m$ among policies in the same class with no greater geometric risk. For an approximate maximizer, the corresponding comparison incurs its optimization error.

A sufficient starting price can be computed from a comparison policy, an upper bound on stopped utility and a justified optimization gap. With comparison quantities $U_0$, $\mathcal I_{m,0}$ and $q_0$ and an upper bound $\delta$ on the gap, the price–risk bound is
$q_\beta\le q_0+(1-\beta)(\sup U-U_0+\mathcal I_{m,0}+\delta)/m$. Use `plan(..., sup_U=..., comparator=...)` or `certificates.conservative_price` with the hypotheses of the bound. Use `certificates.learned_law_risk` for the corresponding comparison with a regularization cost.

Under exact failure-only optimization, geometric risk is nonincreasing in $m$. With an interior barrier or approximate training, the monotonicity bound accounts for changes in interior loss and the optimization errors. A sequence of increasing prices is sufficient only under the relevant conditions, which include a feasible comparison with slack and charged terms that grow sublinearly in the price. At any one price, subtracting the candidate's value from the best known feasible benchmark value gives a lower bound on the exact gap when both values are known. Simulator estimates of those values give a descriptive shortfall subject to sampling error. `frontier.price_sequence_check` summarizes observed shortfalls across prices.

For a task with $H$ physical actions, the discounted charge at failure $t$ is $m(1-\beta^{H-t})/(1-\beta)$. Use `spec.failure_charge` to compute it. To charge a known lump sum at every failure time, use `Stage(penalty="multiplier", multiplier=F)` without dual ascent. A failure-only stage pays full utility before failure. With an interior barrier, check that any clipping and reward bounds match the claimed value certificate.

For adaptive pricing, `DualAscent(alpha, step, multiplier0=...)` updates the lump penalty using the failure frequency of completed episodes in a rollout. Use a stopping rule that yields the event specified by the training requirement. `dual_step_suggestion` initializes a scale from a fixed-price pilot for that event. `DualAscent.diagnose(beta)` reports multiplier and rate diagnostics on recent rollouts. A rising multiplier can indicate an unattainable level or insufficient policy response. A stable multiplier remains a training diagnostic. Validation tests the final candidate independently.

The multiplier diagnostic compares recent changes in the multiplier with the measured failure frequency. A stable
status requires both a small trend and a frequency consistent with the target. A persistently positive trend
warrants inspecting whether any tested policy attains the level, how the initial policy performs, and whether the
dual step is large relative to policy updates. A rollout in which all episodes fail gives little information about
how a change in the action distribution would affect the event rate.

A bound on geometric risk also bounds the mean failure time of the earliest $\varepsilon$-fraction of all trajectories. Assign infinite failure time to trajectories that never fail. The lower bound is $\log(\varepsilon/q_\beta)/\log(1/\beta)$ when positive. [Price and value functions](results-map.md#risk-certificates-and-the-fixed-price-of-failure-risk)

## 4. Keep a reserve deliberately

A fixed failure price assigns a cost to the crossing event. A positive interior barrier also deducts utility on nonfailing steps with small reserves. The objective therefore gives a separate incentive to preserve reserve. The attained reserve distribution depends on the policy and should be measured.

In the wear study, a reduction of 10% in the true allowance increased the failure frequency of the price-only policy at $m=1$. The interior-barrier runs had lower frequency under that reduction. To reproduce this type of comparison, set `true_budget_scale=0.9` and retain the nominal observations. Use the same evaluation conditions across candidates and report the frequency of small reserves, failure and stopped utility. If the reduced allowance is known before training, it can instead be specified as the initial budget.

`Evaluation.near` gives empirical frequencies for reserve thresholds. A value-based bound is also available when the interior barrier is bounded below by a positive amount in the region of interest. [Small-reserve certificate](supplementary-results.md#1-certificate-for-a-small-reserve)

## 5. Choose regularizers for their role

### 5.1 Action relative entropy

Let $\rho(\cdot\mid o)$ be the reference action distribution at observation $o$. Define $\mathcal A$ as the expected discounted sum of $\log[\pi(a_t\mid o_t)/\rho(a_t\mid o_t)]$ through the failing decision, assuming absolute continuity of the policy with respect to the reference. Equivalently, each conditional mean is a KL divergence between the two action distributions. Under a common environment law and the geometric clock, the chain rule for KL accumulates these terms over the stopped trajectory. Mapping a trajectory to its failure flag can only reduce KL. Hence $\mathrm{kl}(q_\beta(\pi)\Vert q_\beta(\rho))\le\mathcal A$, where $\mathrm{kl}$ is the divergence between Bernoulli laws. Use the attained action divergence and reference risk for this comparison. `ActionAnchor(eta, log_reference)` applies the sampled charge. `evaluate.stopped_action_relative_entropy` estimates it.

Relative entropy to a reference and the trust region used by CPO have distinct roles. The reference appears in the objective. The trust region measures the change from the old policy to the proposed policy at each update. [Regularization in constrained actor updates](../formulation-comparison.md#vary-action-and-parameter-regularization-independently)

### 5.2 Parameter regularization in the population model

The population model represents the actor by a law $\mu$ of unit parameters. Over the represented laws we minimize $-V_m(\mu)+\eta\mathcal A(\mu)+\kappa\mathcal H(\mu)$, where $V_m$ is the clipped value and $\mathcal H$ is an admissible parameter regularizer. The gradient flow moves the unit law in the direction of decreasing objective. At fixed strengths the rate theorem requires the net curvature

$$
\Lambda_m=\kappa\lambda_{\mathcal H}-C_m-\eta C_{\mathcal A}>0.
$$

Here $\lambda_{\mathcal H}$ is a lower bound on regularizer curvature along the specified interpolations. The quantities $C_m$ and $C_{\mathcal A}$ bound the curvature deficits of the negative value and action terms. These are conditions on the population flow. The numerical updates of a finite PPO actor have their own measured learning traces.

`convergence.curvature_budget` evaluates the sufficient criterion once the representation bounds are supplied. `KernelConstants` gives bounds in terms of the policy kernel and its derivatives. `FeatureConstants` gives the softmax instance. With a decreasing strength, the separate near-minimizer result requires that both the strength and the gap to the corresponding objective minimum tend to zero. A training schedule is evaluated by its observed outcomes, including comparisons with training at the final strength from the start.

The common-potential result concerns continuous-time, width-scaled training with the same smooth potential applied to each unit. The finite-network callback that assigns each parameter its own incumbent value uses a different penalty. Its effect under discrete PPO is measured on the trained policies. [Curvature calculation](../../examples/under-the-hood/curvature_budget.py)

### 5.3 Noise, drift and spread

In the population particle system, relative entropy with respect to a Gaussian prior $N(x_0,\sigma^2 I)$ gives both noise and drift. At strength $\kappa_1$, each coordinate has noise variance $2\kappa_1$ per unit of flow time, and the Gaussian potential contributes a pull of strength $\kappa_1/\sigma^2$ toward $x_0$. A separate quadratic potential contributes drift without noise. A faster-growing divergence integrand penalizes concentration relative to the prior and requires a density with respect to it.

In the finite-network implementation, `ParameterNoise` adds Gaussian increments with variance `2 * kappa1 * lr` after each update. `ParameterAnchor` separately applies the pull. Under plain gradient descent with constant step $h$, quadratic pull $K$, zero objective gradient and $0<hK<2$, the stationary standard deviation per coordinate is $\sqrt{\kappa_1/[K(1-hK/2)]}$. The continuous-time approximation is $\sqrt{\kappa_1/K}$. Use `roles.noise_and_anchor_spread` for this calculation.

With a decoupled anchor, the anchor substep multiplies the displacement from its reference by `1 - lr * kappa_p`. A coupled anchor adds its derivative to the loss before Adam's adaptive update. The stationary formula above belongs to the specified constant-step model. For PPO with Adam, compare realized parameter distance, action dispersion, stopped utility and failure risk at the recorded learning rates. [Regularization notebook](../../notebooks/under-the-hood/T3_roles_of_the_regularizers.ipynb)

### 5.4 Evaluate the prior policy

The prior distribution of network units induces a policy in the population model. Under the stated alignment and divergence assumptions, *Improvement over the prior policy* bounds the risk of the population flow initialized at that prior by the prior's risk plus a slack term. It also compares clipped values. The bound on the population trajectory is distinct from the measured effect of initializing or anchoring a finite actor.

If an actor is initialized by fitting an incumbent policy, evaluate it before training. Its observation preprocessing and action dispersion are part of the policy. Fit the mean with `fit_actor_mean` where appropriate and set the dispersion with `set_log_std`. Include all intended actor parameters in the anchor. `ParameterAnchor` includes the trainable log standard deviation by default. A weakly constrained dispersion can alter the deployed policy even when its mean parameters remain close to the reference.

For comparative bounds that use a comparison law, enter its regularization cost $\eta\mathcal A+\kappa\mathcal H$ along with its risk, utility and interior loss. For implementation, use `advise_regularizers` to inspect the options and record the actual anchor, optimizer and update form. `regularizer_effects` can summarize comparisons with observations from the prior and trained candidate.

## 6. Validate the deployed candidate

For a fixed policy and independently simulated episodes, count the failures of the specified event. `validation.validate_direct` gives the one-sided exact binomial upper endpoint at error `zeta/family`. If $N$ episodes contain zero failures, the endpoint is $1-(\zeta/\mathrm{family})^{1/N}$. A test passes when the endpoint is at most the tolerance. An upper endpoint above the tolerance leaves the requirement unresolved at that confidence level.

Under independent geometric stopping, directly validate $q_\beta$ and apply $p_T\le\beta^{-T}q_\beta$ where it gives a useful bound. To count the finite-horizon event directly, allow the simulator to observe first crossing through $T$ or a genuine earlier terminal state. The finite-horizon test then covers that horizon and policy only. Refit weights, change the action sampling rule, or change the budget, and the test concerns a different candidate or task.

Value certificates use a valid lower bound on value and the appropriate reward bounds. Check the horizon convention and the validation resolution before relying on an empirical value bound. The population-to-network results concern a specified sampling construction. For a deployed finite network, direct simulation is available under the same independent-episode assumptions as for any other fixed policy.

`evaluate` produces per-episode outcomes through the wrapper. For a vectorized simulator, use `evaluation_from_episodes` with failure flags, first failure times, stopped utilities and interior losses. Supply the actual clock and horizon convention. `interpret` reports the direct test, any applicable value bounds and their inputs. [Validation notebook](../../notebooks/under-the-hood/T4_validation_in_detail.ipynb)

## Function index by decision

| Decision | Functions |
|---|---|
| Specify the requirement | `BudgetSpec`, `Requirement`, `certificates.finite_from_geometric`, `certificates.earliest_failures_time`, `certificates.ever_failing_bound` |
| Detect conditions for another method | `planning.when_to_use_another_method`, `plan`, `certificates.risk_lower_bound`, `certificates.prescribed_level_attainable` |
| Fix the price | `plan`, `certificates.conservative_price`, `certificates.conservative_multiplier`, `certificates.price_risk_bound`, `certificates.price_monotonicity`, `frontier.check_runs`, `frontier.price_sequence_check`, `certificates.matched_risk_utility_gap`, `certificates.matched_risk_signed` |
| Run any learner | `BudgetWrapper`, `RunningMaxCost`, `Stage`, `barrierpg.spec.failure_charge`, `BudgetMonitor` |
| Adaptive multiplier | `Stage(penalty="multiplier")`, `DualAscent`, `callbacks.dual_step_suggestion`, `DualAscent.diagnose` |
| Keep a reserve | `Stage(theta=...)`, `BudgetWrapper(true_budget_scale=...)`, `certificates.small_reserve_bound`, `LogBarrier.lower_level` |
| Choose regularizers | `advise_regularizers`, `ActionAnchor`, `ParameterAnchor`, `ParameterNoise`, `fit_actor_mean`, `set_log_std`, `RegularizerMix`, `convergence.gaussian_prior_curvature`, `convergence.curvature_budget`, `convergence.certified_kappa`, `convergence.finite_training_transfer` |
| Read their effects | `regularizer_effects`, `ParameterAnchor.effects_inputs`, `ParameterNoise.spread`, `evaluate.stopped_action_relative_entropy`, `certificates.action_risk_bound` |
| Certify and validate | `evaluate`, `evaluation_from_episodes`, `Evaluation.validate`, `validation.validate_direct`, `validation.zero_failure_bound`, `validation.episodes_for_tolerance`, `certificates.risk_from_value`, `validation.risk_from_validation_value`, `validation.physical_horizon_risk`, `validation.lump_sanction_risk`, `validation.value_resolution_check` |
| Report | `interpret(...).text()`, `plan(...).explain()` |
