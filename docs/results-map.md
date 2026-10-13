# Results map: from the paper to the code

Numbers refer to revision 62 of *The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex
Regularization*. The toolbox cites results through the registry in [`bpg/paper.py`](../bpg/paper.py), so the strings
it prints follow this table; `bpg.paper.results_table()` prints the registry itself.

## Risk certificates and the fixed-price reading (Section 3)

| Result | What it says | Function |
|---|---|---|
| Section 1.1 | $\beta^Tp_T\le q_\beta$: the geometric risk bounds the finite-horizon risk | `certificates.finite_from_geometric`, `geometric_from_finite` |
| Section 2.3 | Bounded scores impose a risk floor, $p_T(\pi_\mu)\ge e^{-2L_0(T+1)}p_T(\rho)$ | `certificates.risk_floor` |
| Section 3.3, the clipping identity | $V_m=U-\mathcal I_m-\frac{m}{1-\beta}q_\beta$ | `certificates.clipped_value_identity`; checked per episode by `evaluate` |
| Proposition 3.1 | Value and objective certificates: $q_\beta\le(\bar u-(1-\beta)V_m)/(m+\bar u)$, and $p_T$ with $\beta^{-T}$ | `certificates.risk_from_value`, `risk_from_objective` |
| Theorem 3.2 | Matched-risk comparison: utility shortfall charged with interior loss, regularization cost, optimization error and unused risk; the signed form | `certificates.matched_risk_utility_gap`, `matched_risk_signed`, `frontier.complementarity_gap` |
| Proposition 3.3(i) | Optimality at the attained risk | `frontier.check_runs` (dominance), `report.interpret` (the reading) |
| Proposition 3.3(ii) | The price–risk law and its second bound with the $(m+\bar u)$ denominator | `certificates.price_risk_law`, `conservative_price`, `conservative_multiplier` |
| Proposition 3.3(iii) | Monotonicity in the price | `certificates.price_monotonicity`, `frontier.check_runs` |
| Proposition 3.3(iv) | Learning form, with $\delta'=\delta+\mathcal R(\bar\mu)+2\varepsilon_r/(1-\beta)$ | `certificates.learning_form_risk`, `comparison_law_risk` |
| Section 3.4 | A prescribed level below the risk floor is unattainable; no multiplier settles | `certificates.prescribed_level_attainable`, `convergence.multiplier_diagnosis` |
| Proposition 3.4 | Action relative entropy and trajectory risk: $\mathrm{kl}(q\,\|\,q_\rho)\le\mathcal A$ | `certificates.action_risk_bound`, `kl_upper_inverse`, `kl_lower_inverse` |

## Convergence, the prior and the roles (Section 4, Appendix B)

| Result | What it says | Function |
|---|---|---|
| Definition 4.1, Proposition 4.2 | Admissible regularizers: relative entropy with any displacement-convex integrand, convex potentials, and sums; the integrand adds no curvature | `convergence.supplied_curvature`, `gaussian_prior_curvature`, `roles.RegularizerMix.curvature` |
| Theorem 4.3 | Population convergence when $\Lambda_m=\kappa\lambda_{\mathcal H}-C_m-\eta C_{\mathcal A}>0$; two-flow contraction for every sign | `convergence.curvature_budget`, `rate`, `certified_kappa`, `time_to_fraction` |
| Proposition B.1 | The deficits from kernel bounds $B_1,B_2,K_0,K_1,K_2$; the softmax instance | `convergence.KernelConstants`, `FeatureConstants`, `curvature_constants` |
| Corollary 4.4 | Learning error and budget risk at learning time $\ell$ | `convergence.learning_risk_bound`, `learning_time_for_slack`, `initial_gap_upper`; `certificates.learning_slack` |
| Remark 4.5, Online Proposition F.3 | Improvement over the prior policy within a risk slack of order $\kappa^{-1/2}$; the second slack with $\eta>0$; the $\kappa$ that meets a tolerance | `convergence.risk_slack_from_prior`, `action_slack_from_prior`, `kappa_for_tolerance` |
| Remark 4.6 | The regularization bias is priced and detected | `regularizers.regularizer_effects` (bias terms), `report.interpret` |
| Section 4.2, the particle form | Noise $\sqrt{2\kappa_1}\,dB$, pull $-\kappa_1\nabla W-\kappa_2\nabla P$, spread from the integrand; stationary spread under a gradient pull | `roles.RegularizerMix`, `stationary_spread`, `kappa1_for_spread`; `callbacks.ParameterNoise` |
| Proposition 4.7 | Under a potential, noiseless finite-network training is the flow from its empirical law, with the same contraction; value and risk transfer | `convergence.finite_training_transfer` |
| Remark 4.8 | A Gaussian prior is entropy plus weight decay of weight $\kappa/\sigma^2$; the rate depends on $\kappa/\sigma^2$ only | `convergence.gaussian_prior_curvature`; `callbacks.ParameterAnchor` |
| Online Remark F.1 | Potential regularizers and weight decay are admissible; finite on a finite network | `callbacks.ParameterAnchor.regularizer_value` |
| Online Remark F.2 | A potential alone collapses the units to one point in the certified regime | `roles.RegularizerMix.collapses` |
| Online Proposition E.2 | First variations: the per-unit policy gradient | `examples.plateau.PlateauModel.value_gradient` (solvable flow) |

## Deployment and validation (Section 5)

| Result | What it says | Function |
|---|---|---|
| Theorem 5.1 | From a learned law to one sampled network; widths for a target | `certificates.finite_network_transfer`, `width_for_target`, `width_for_direct_target` |
| Section 5.2, direct test | Exact binomial upper bound; resolution floor $1-\zeta^{1/N}$; errors allocated over a family | `validation.validate_direct`, `binomial_upper`, `resolution_floor`, `episodes_for_tolerance`, `allocate_error`; `Evaluation.validate` |
| Section 5.2, value test | Hoeffding lower bound on $V_m$ from truncated returns; the resolution check | `validation.hoeffding_value_lower`, `risk_from_validation_value`, `value_resolution_check` |
| Section 5.2, physical horizon | The certificate when the task ends after $H$ actions | `validation.physical_horizon_risk` |
| Section 6.6 | The value certificate with a lump sanction, $p_T\le(\bar uS_H-V^{(H)})/(F\beta^T)$ | `validation.lump_sanction_risk` |
| Algorithm 1 | Fixed-stage learning and independent validation | `evaluate`, `evaluation_from_episodes`, `Evaluation.validate`, `interpret` |
| Online Example E.3 | A small geometric risk is not an eventual-risk guarantee | stated in every report |
| Online Proposition E.4 | A sequence of prices reaches a tolerance when the charged errors grow slower than the price | `frontier.price_sequence_check` |

## Practice (Sections 1.2, 6 and 7)

| Result | Function |
|---|---|
| Section 1.2, when another method is better | `planning.when_to_use_another_method`, `plan` |
| Section 7.1, using the bounded barrier | `plan`, `advise_regularizers` |
| Section 7.2, scope of the guarantees | `report.interpret` (the "not established" list) |
| Section 6.1, the wear-allowance study | `examples.wear` (model, rules, exact benchmark numbers) |
| Section 6.3 and 6.5, the solvable flow and the roles | `examples.plateau` (model, particle simulation, the paper's roles table) |
| Section 6.4, the parameter anchor | `callbacks.ParameterAnchor`, `fit_actor_mean`, `set_log_std` |
| Online Appendix M, the toolbox | this repository |

## Supplementary results of the toolbox

Not stated in revision 62; proved in [`supplementary-results.md`](supplementary-results.md).

| Result | Function |
|---|---|
| Near-exhaustion certificate | `certificates.near_exhaustion_bound` with `LogBarrier.lower_level` |
| Structural certificate for eventual risk | `certificates.eventual_risk_structural`, `unresolved_threshold` |
