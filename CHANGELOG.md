# Changelog

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
