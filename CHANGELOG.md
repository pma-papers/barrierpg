# Changelog

## 0.6.1 (2026-10-20): the parameter anchor decoupled from the optimizer; checked against revision 68

- `ParameterAnchor` has two forms. The new default, `decoupled=True`, applies the pull to the actor's parameters after
  the optimizer's step, `theta <- theta - lr kappa_p (theta - theta_0)` with the current learning rate of the
  optimizer's first parameter group (the weight decay of AdamW). Every coordinate then contracts toward the anchor by
  `1 - lr kappa_p` per step whatever the optimizer does with the objective's gradient, so the strength keeps its meaning
  as a curvature (`contraction()`, `half_life()`), and with `ParameterNoise` the pair is the Euler-Maruyama step of the
  particle form, and the measured spread can be checked against its stationary spread (`roles.stationary_spread`).
  `install` refuses `lr kappa_p > 1`. `decoupled=False` is the previous form, the pull added to the gradient after
  clipping and before the step; under Adam it is rescaled coordinate by coordinate to a step of about `lr` whatever the
  distance, the form of the paper's runs and of the shipped campaign records. The paper's guidance (the first subsection
  of *Practical choices, scope and conclusion*) states the four implementation rules behind this.
- `ParameterAnchor` and `ParameterNoise` compose in either installation order through one hook on the optimizer's
  step: coupled pull, step, decoupled pull, noise. The noise always comes after the pull.
- Tests: the decoupled pull contracts a parameter with zero objective gradient by exactly `1 - lr kappa_p` per step
  under Adam; the coupled pull's step under Adam is about `lr` at a displacement of 1 as at 0.01; anchor and noise
  compose to `theta_0 + (1 - lr kappa_2) delta + sqrt(2 kappa_1 lr) xi` in either order.
- The claim that under Adam even a weak pull on the weights outweighs the policy gradient is withdrawn from the
  practitioner guide, the decisions document and notebook 1: it was not established. In its place the documents
  explain the coupled and the decoupled forms. `roles.adam_stationary_spread` and the printed explanations say that
  the measured `sqrt(2) kappa1` describes the coupled form.
- Reproducibility of what was trained before this change: `campaigns.problems.Config.anchor_decoupled` (default
  `False`) pins the coupled form for the shipped records, new records carry `regularizers.anchor_form`, the live
  anchored runs of notebooks 1 and 2 and of T3 pass `decoupled=False` so that their stored outputs and text remain
  those of the form they ran, and `examples/under-the-hood/wear_ppo.py` has `--anchor-form` (default coupled, the
  paper's). Whether the anchored lander still runs dry under the decoupled form, and what the roles study gives under
  it, is untested: the notebook sentences say so. The template `examples/your_problem.py` uses the default.
- Notebooks T3 and T4 are re-executed from their sources, whose wording had changed in 0.4.1 without a rebuild; only
  labels and sentences differ, every number is unchanged.
- `PAPER["checked_against"]` is revision 68 (2026-10-20): the result titles are unchanged, and revision 68's statement
  of the coupling in its roles study and its guidance are reflected in the callbacks' docstrings.

## 0.6.0 (2026-10-15): the toolbox is called `barrierpg`; the fuel-tank test runs anywhere

- The distribution and the package are renamed from `bpg` to `barrierpg`, because the name `bpg` is taken on PyPI by
  an unrelated package. Code changes `import bpg` to `import barrierpg` and `from bpg...` to `from barrierpg...`;
  nothing else changes. An environment with the old editable install needs `pip uninstall bpg` and
  `pip install -e .` again. Code that must keep the old name can pin the tag `v0.5.1`. The run records in
  `campaigns/results/`, written before the rename, keep the key `bpg` for the toolbox's version. The entries below
  use the old name.
- The test that `BudgetWrapper` reproduces the paper's fuel-tank environment step by step no longer needs a checkout
  of the replication repository: the environment's class is shipped as a test fixture
  (`tests/fixtures/paper_fuel_lander.py`, copied verbatim, with the hash of its source file). The CI job with the
  learners installs Box2D and MuJoCo, so this test and the robot's run there.
- Releases from 0.3.0 on carry git tags (`v0.3.0`, ...), so a version can be pinned.

## 0.5.1 (2026-10-15): HalfCheetah without MuJoCo's deprecation warning

- `bpg.problems.energy.half_cheetah()` builds Gymnasium's HalfCheetah-v5 from a copy of its model file without the
  `settotalmass` attribute, which MuJoCo 3 deprecates and warned about at every environment (and logged to
  MUJOCO_LOG.TXT); the total mass is set with `mujoco.mj_setTotalmass` instead. Masses, inertias and trajectories are
  identical to the stock environment's (a test checks it). The robot problem and notebook 3 use it.

## 0.5.0 (2026-10-15): convergence monitoring; regularizers as examples

- Convergence monitoring of a training run: `bpg.convergence.training_convergence` compares the last tenth of the
  training batches with the tenth before, for the failure rate, the utility, the priced objective, the action
  dispersion and an adaptive multiplier, and reports converged, not converged (with what is still changing), or too
  short to judge (fewer than two windows of 10 batches). `BudgetMonitor.convergence()` and `BudgetMonitor.plot()` apply
  it to a run; `BudgetMonitor(check_every=k, stop_when_converged=True, patience=, min_steps=)` checks during training
  and can stop it; `interpret(..., training=...)` states it in the report. The explanation says what convergence does
  not mean: a learning rate that decays to zero makes any run stop changing, and a run can stop in a bad place.
- The notebooks run the regularizers live: the lander from its controller with a parameter anchor and with an action
  anchor, the retailer from a prior fitted to its rule with and without an anchor; every short run is followed by the
  convergence check, and every campaign table by the convergence of its runs (`records.Campaign.convergence_table`).
- The template `examples/your_problem.py` has optional regularizers (start from a policy in use, parameter anchor,
  action anchor) and checks convergence during training.
- Shipped models load anywhere: `records.Campaign.model()` and the runner replace the pickled schedules on loading.

## 0.4.1 (2026-10-15): the notebooks build and train their problems in view

- Every problem notebook now defines its problem and trains it in visible cells: the resource (a cost function, or a
  wrapper that adds the resource to a task), the budget specification with each argument explained, the requirement,
  the price or the cost of failure, the wrapper and a check that it fails when it should, `plan`, and the training code
  (PPO settings, normalization, `BudgetMonitor`, regularizers, an adaptive multiplier), run live on a short budget
  before the full campaign's records are loaded. Notes marked Theory say what the results behind the method imply at
  each step. The retailer's simulator is written in the notebook, as a template for writing one's own, and checked
  against the copy the campaign used.
- `examples/your_problem.py`: a complete, runnable template from the task to the report.
- Experiment wording: no "predicted" or "pre-registered" (fixed before the campaign).

## 0.4.0 (2026-10-15): a toolbox for practitioners; the paper's examples as a separate track

Structure
- The toolbox is its own git repository, separate from the paper's working tree; the paper's studies are reproduced in
  a separate replication repository, to which the walk-through of the paper's studies moved.
- Two tracks. The primary track assumes no knowledge of the paper: four notebooks on RL-level problems and a rewritten
  practitioner guide. The under-the-hood track keeps the paper's test problems, which have exact solutions
  (notebooks T1 to T4, `docs/under-the-hood/`).

New
- `bpg.problems`: four problems as templates for mapping a task to a budget. `lander` (LunarLander with a fuel tank),
  `inventory` (a retailer that finances its stock from cash; insolvency is the failure), `energy` (HalfCheetah on a
  battery whose efficiency varies by mission), `navigation` (Safety-Gymnasium's PointGoal1 with a damage allowance).
- `campaigns/`: definitions, a runner, launch scripts and the records of the training campaigns of the four problems,
  read by the notebooks through `campaigns/records.py`.
- `BudgetWrapper(observe_time=True)` appends the elapsed fraction of the horizon; with a fixed horizon the policy needs
  it to pace a budget.
- `Report.summary(resource, period)`: the verdict in plain words (failures on fresh episodes, the validated bound,
  whether the requirement is met, utility, what the test does not show); `text()` keeps every bound.
- The `problems` extra installs Box2D and MuJoCo.

Fixed
- With whole-unit costs the reserve can reach exactly zero without being exceeded, where the barrier is infinite;
  under the unclipped form (`clip_utility=False`) the step's reward was then minus infinity. With no interior barrier
  the unclipped form now charges only the failure, and with a barrier a zero reserve is charged as the clipped reward
  charges it. No result with continuous costs is affected.

Changed
- The practitioner guide is rewritten for readers who have not read the paper, around the four problems; its advice
  on prices, multipliers, anchors and validation rests on the campaigns (a fixed horizon needs a clock in the
  observation; a price can trade one failure for another; a parameter anchor to a controller that violates the new
  limit keeps the violation; a multiplier started with an untrained learner rises far past its level; compare
  policies only after training has converged).
- `plan(..., episode_earnings=...)` compares the failure charge at the start of an episode with what an episode
  earns: above three times it warns and the case for another method applies, above once it adds a note (it warned at
  ten times before). In the campaigns, the navigation robot's return collapsed at about four times, while the legged
  robot and the retailer did well at two to three times.
- The theory-oriented guide is now `docs/under-the-hood/decisions-and-theory.md`, with the results map and the
  supplementary results; the former notebooks 1 to 4 are T1 to T4, and notebook 5 is replaced by the four problems.
- The landing page and its figure use the problems' campaigns; the former figure belongs to the under-the-hood track.

## 0.3.0 (2026-10-15): checked against the paper's revision 65; results cited by name

Citations
- The toolbox no longer cites statement, section, table or equation numbers of the paper, in code, printed output,
  documentation or notebooks, since they change between revisions. Results are cited by the titles the paper gives
  them: `cite("fixed_price", "ii")` returns `paper, 'Fixed price of failure risk' (price-risk law)`. The parts of that
  statement are printed by name (optimality at the attained risk, price-risk law, monotonicity in the price, learning
  form). `PAPER["checked_against"]` replaces `PAPER["revision"]`, and `paper.number()` is removed.
- A test fails if a printed citation, plan or report contains a statement number. `docs/results-map.md` lists the
  results by title.

New
- `certificates.earliest_failures_time`: the earliest eps-fraction of failures occurs on average no earlier than
  log(eps/q_beta)/log(1/beta) periods (the paper's remark on the geometric risk as a risk measure of the failure time).
- The benchmark-suite study in `docs/worked-examples.md`, with the use of a Safety-Gymnasium task through the wrapper.

Changed
- `planning.when_to_use_another_method` and `plan` state the cases of the paper's *Which formulation to prefer*: five
  in which another method is the better choice (no failure permissible and a controlled-invariant set known; a
  prescribed level attainable with slack and a tunable multiplier step; violations during training that matter; a
  loss that grows with the excess consumption; a known small model) and two that concern the barrier (a price that
  charges far more for a failure than an episode earns; an allowance error of known size). New keyword arguments
  replace `violations_during_learning_acceptable`, `prescribed_tolerance_least_conservatism`,
  `expected_cost_requirement`, `structured_solution_available` and `task_stakes` (now `episode_earnings`).
- `convergence.multiplier_diagnosis` and `DualAscent.diagnose` apply the paper's two criteria for convergence; the
  statuses are `stable`, `increasing`, `inactive`, `trending`, `off-target` and `short`. Whether the multiplier still
  moves is tested on its rollout-to-rollout updates (their mean is its drift), not by a slope fitted to its levels:
  the multiplier is a running sum, and the slope test reported a trend in almost every window, also after
  convergence.
- Notebook 2 sizes the dual step from the price at which the learned runs attain the prescribed level, half that
  level per rollout in which every episode fails; the multiplier now converges within the run.
- `callbacks.dual_step_suggestion(expected_multiplier, rollouts_to_reach=...)` replaces `rollouts_to_settle`.
- The value test is described as a necessary condition for certifying a tolerance, as in the paper.
- Wording follows the paper: a multiplier converges or increases without bound; prices are compared by training at
  several prices.

## 0.2.0 (2026-10-12): the paper's revision 62

Citations
- All citations go through one registry, `bpg/paper.py`, keyed by stable names; numbers follow revision 62
  (`cite("fixed_price", "ii")`). `docs/results-map.md` maps every result to its function.

New
- The particle form and the roles of the parameter regularizers: `bpg.roles` (`RegularizerMix`, `stationary_spread`,
  `kappa1_for_spread`, `adam_stationary_spread`) and the `ParameterNoise` callback (the noise role on a finite actor).
- General actor classes in the convergence budget: `convergence.KernelConstants` (Proposition B.1), with softmax as one
  instance; `supplied_curvature` for a divergence plus a potential.
- Corollary 4.4 in its revision-62 form (`convergence.learning_risk_bound`, `certificates.learning_slack`) and
  Proposition 4.7 (`convergence.finite_training_transfer`).
- `certificates.matched_risk_signed`, `risk_floor`, `prescribed_level_attainable`, `conservative_multiplier`.
- `frontier.price_sequence_check`: can raising the price reach a tolerance (Online Proposition E.4)?
- `validation.lump_sanction_risk` and `hoeffding_lower`; the report adds the physical-horizon and lump-sanction
  certificates when they apply.
- `planning.when_to_use_another_method`: the five conditions of Section 1.2, stated by `plan`.
- `RunningMaxCost` for resources whose step use can be negative; `evaluation_from_episodes` for vectorized evaluators;
  `fit_actor_mean` and `set_log_std` to build a prior from an incumbent; `BudgetMonitor(multiplier=...)`.
- `bpg.examples.plateau`: the paper's solvable flow and its particle simulation (numpy only).
- Five executed tutorial notebooks, generated from sources in `tools/notebook_sources/`.

Changed
- `ActionAnchor` charges the failing action by default, as the paper's action relative entropy does
  (`include_failing_action=False` restores the previous behavior).
- `plan` handles flat-penalty stages (a lump sanction) and warns when the price is far above the stakes.
- The regularizer effects compare clipped values, not utilities, with the prior policy, and read a utility drop through
  the signed comparison of Theorem 3.2.

Fixed
- Under geometric stopping the wrapper seeded its stopping clock with the environment's own seed, so maintenance access
  and the environment's shocks came from correlated random streams. The clock now has an independent stream. Results
  under `geometric_stop=True` change: the wear example's incumbent rule now gives 1.05% breakdowns and utility 2.73,
  matching the paper (1.08%, 2.73); before the fix it gave about 0.8%. Results without geometric stopping (physical
  horizons, as in the fuel-tank and emissions studies) are unaffected.

## 0.1.0 (2026-10-08)

First version: wrapper, callbacks, certificates, validation, planning and report for revisions 40 to 42 of the paper.
