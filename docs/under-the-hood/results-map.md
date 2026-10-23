# Results map: from the paper to the code

Results of *The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization* are listed by
the titles the paper gives them, which can be searched in any revision. The toolbox never cites statement, section
or equation numbers, since they change between revisions. Inside the code, every citation goes through the registry
in [`barrierpg/paper.py`](../../barrierpg/paper.py); `barrierpg.paper.results_table()` prints it, and
`barrierpg.PAPER["checked_against"]` records the last revision against which the toolbox's statements were checked.

## Risk certificates and the fixed price of failure risk

| Result in the paper | What it says | Function |
|---|---|---|
| The inequality $\beta^Tp_T\le q_\beta$ (since $\beta^\tau\ge\beta^T$ on $\tau\le T$) | The geometric risk bounds the finite-horizon risk | `certificates.finite_from_geometric`, `geometric_from_finite` |
| *The represented class* | Bounded scores impose a risk floor, $p_T(\pi_\mu)\ge e^{-2L_0(T+1)}p_T(\rho)$ | `certificates.risk_floor` |
| The clipping identity (*Utility at the attained risk level*) | $V_m=U-\mathcal I_m-\frac{m}{1-\beta}q_\beta$ | `certificates.clipped_value_identity`; checked per episode by `evaluate` |
| *Value and objective certificates* | $q_\beta\le(\bar u-(1-\beta)V_m)/(m+\bar u)$, and $p_T$ with the factor $\beta^{-T}$ | `certificates.risk_from_value`, `risk_from_objective` |
| *The geometric risk as a risk measure of the failure time* | The earliest $\varepsilon$-fraction of failures occurs on average no earlier than $\log(\varepsilon/q_\beta)/\log(1/\beta)$ periods | `certificates.earliest_failures_time` |
| *Matched-risk performance comparison* | Utility shortfall charged with interior loss, regularization cost, optimization error and unused risk; its signed form | `certificates.matched_risk_utility_gap`, `matched_risk_signed`, `frontier.complementarity_gap` |
| *Fixed price of failure risk*, optimality at the attained risk | A fixed-price optimum is optimal for $U-\mathcal I_m$ at the risk it attains | `frontier.check_runs` (dominance), `report.interpret` (the fixed-price reading) |
| *Fixed price of failure risk*, price–risk law | The attained risk is at most a comparison policy's plus order $1/m$; a second bound with the $(m+\bar u)$ denominator | `certificates.price_risk_law`, `conservative_price`, `conservative_multiplier` |
| *Fixed price of failure risk*, monotonicity in the price | A higher price raises the attained risk only through the errors and the interior loss | `certificates.price_monotonicity`, `frontier.check_runs` |
| *Fixed price of failure risk*, learning form | The price–risk law for a minimizer of the regularized objective, with the regularization cost charged | `certificates.learning_form_risk`, `comparison_law_risk` |
| *The fixed price of failure risk* (the section) | A prescribed level below the risk floor is unattainable, and no multiplier converges | `certificates.prescribed_level_attainable`, `convergence.multiplier_diagnosis` |
| *Action regularization and failure risk* | $\mathrm{kl}(q\,\|\,q_\rho)\le\mathcal A$: departure from the action reference's risk | `certificates.action_risk_bound`, `kl_upper_inverse`, `kl_lower_inverse` |

## Convergence, the prior and the roles of the regularizers

| Result in the paper | What it says | Function |
|---|---|---|
| *Admissible parameter regularizer*, *Admissible families* | Relative entropy with any displacement-convex integrand, convex potentials, and their sums are admissible; the integrand adds no curvature | `convergence.supplied_curvature`, `gaussian_prior_curvature`, `roles.RegularizerMix.curvature` |
| *Fixed-stage population convergence* | Exponential convergence when $\Lambda_m=\kappa\lambda_{\mathcal H}-C_m-\eta C_{\mathcal A}>0$; two flows contract for every sign | `convergence.curvature_budget`, `rate`, `certified_kappa`, `time_to_fraction` |
| *Value and action-law curvature* | The deficits from the kernel bounds $B_1,B_2,K_0,K_1,K_2$; the softmax instance | `convergence.KernelConstants`, `FeatureConstants`, `curvature_constants` |
| *Learning error and budget risk* | The risk bound along the population flow at learning time $\ell$ | `convergence.learning_risk_bound`, `learning_time_for_slack`, `initial_gap_upper`; `certificates.learning_slack` |
| *Improvement over the prior policy* (with its proof) | Clipped value at least the prior policy's; risk within a slack of order $\kappa^{-1/2}$; a second slack with $\eta>0$; the $\kappa$ that meets a tolerance | `convergence.risk_slack_from_prior`, `action_slack_from_prior`, `kappa_for_tolerance` |
| *The regularization bias is priced and detected* | The bias enters only through the comparison law's regularization cost and shows as a failed certificate | `regularizers.regularizer_effects` (bias terms), `report.interpret` |
| *The prior and the particle form of the flow* | Noise $\sqrt{2\kappa_1}\,dB$, pull $-\kappa_1\nabla W-\kappa_2\nabla P$, spread from the integrand; stationary spread under a gradient pull; with injected noise the anchor is the only pull | `roles.RegularizerMix`, `stationary_spread`, `kappa1_for_spread`, `noise_and_anchor_spread`; `callbacks.ParameterNoise` |
| *Finite-network training under a potential* | Noiseless finite-network training is the flow from its empirical law, with the same contraction; value and risk transfer | `convergence.finite_training_transfer` |
| *The Gaussian prior as entropy, weight decay and a policy* | Entropy plus weight decay of weight $\kappa/\sigma^2$; the rate depends on $\kappa/\sigma^2$ only | `convergence.gaussian_prior_curvature`; `callbacks.ParameterAnchor` |
| *Potential regularizers and weight decay* | Potentials are admissible and finite on a finite network | `callbacks.ParameterAnchor.regularizer_value` |
| *Collapse under a potential alone* | A potential alone collapses the units to one point in the certified regime | `roles.RegularizerMix.collapses` |
| *First variations* | The per-unit policy gradient | `examples.plateau.PlateauModel.value_gradient` (the solvable flow) |

## Deployment and validation

| Result in the paper | What it says | Function |
|---|---|---|
| *Population-to-network transfer* | From a learned law to one sampled network; widths for a target | `certificates.finite_network_transfer`, `width_for_target`, `width_for_direct_target` |
| *Validate the deployed candidate at its specified horizon*, the direct test | Exact binomial upper bound; resolution floor $1-\zeta^{1/N}$; errors allocated over a family | `validation.validate_direct`, `binomial_upper`, `resolution_floor`, `episodes_for_tolerance`, `allocate_error`; `Evaluation.validate` |
| The same section, the value test | Hoeffding lower bound on $V_m$ from truncated returns; a necessary condition for the test to certify a tolerance | `validation.hoeffding_value_lower`, `risk_from_validation_value`, `value_resolution_check` |
| The same section, a physical horizon | The certificate when the task ends after $H$ actions | `validation.physical_horizon_risk` |
| *The risk priced by a physical horizon*, with *Value and objective certificates* | The value certificate with a lump sum at failure, $p_T\le(\bar uS_H-V^{(H)})/(F\beta^T)$ | `validation.lump_sanction_risk` |
| *Fixed-stage learning and independent deployment validation* (the algorithm) | Train at a fixed stage, then validate the deployed candidate | `evaluate`, `evaluation_from_episodes`, `Evaluation.validate`, `interpret` |
| *Delay without eventual feasibility* | A small geometric risk is not an eventual-risk guarantee | stated in every report |
| *Price sequences reaching a tolerance* | A sequence of prices reaches a tolerance when the charged errors grow slower than the price | `frontier.price_sequence_check` |

## Related formulations

The paper relates the price to other formulations (*The price supports the frontier*, *The one-shot equivalent*,
*Chance, time and overshoot penalties*, *The price as a CVaR multiplier*, *CVaR of the failure time*). The toolbox
uses two consequences: a lump sum $F$ charged once at failure corresponds to the price $m=(1-\beta)F$ in the absorbing
form (`spec.price_from_multiplier`), and a multiplier method that converges returns the fixed-price optimum at its
price-equivalent.

## Practice

| Part of the paper | Function |
|---|---|
| *Which formulation to prefer*: the cases in which another method is the better choice | `planning.when_to_use_another_method`, `plan` |
| *Using the clipped barrier* | `plan`, `advise_regularizers` |
| *Scope of the guarantees* | `report.interpret` (the "not established" list) |
| The wear-allowance study | `examples.wear` (model, rules, exact benchmark numbers) |
| The solvable flow and the roles of the regularizers | `examples.plateau` (model, particle simulation, the paper's roles table) |
| The parameter-anchor study | `callbacks.ParameterAnchor`, `fit_actor_mean`, `set_log_std` |
| *The companion toolbox* (online appendix) | this repository |

## Supplementary results of the toolbox

Not stated in the paper; proved in [`supplementary-results.md`](supplementary-results.md).

| Result | Function |
|---|---|
| Near-exhaustion certificate | `certificates.near_exhaustion_bound` with `LogBarrier.lower_level` |
| Structural certificate for eventual risk | `certificates.eventual_risk_structural`, `unresolved_threshold` |
