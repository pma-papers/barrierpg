"""Evaluate fixed operating policies under the wear allowance.

This example defines a finite-horizon failure tolerance, evaluates two
fixed rules with independent geometric maintenance access and translates
the geometric confidence bound to the specified horizon. It compares
the outcomes with the tabular grid benchmark and prints the evidence for
the incumbent rule.

Run with the core package installed:
    python examples/under-the-hood/wear_quickstart.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from barrierpg import Requirement, Stage, evaluate, interpret, plan
from barrierpg.convergence import FeatureConstants
from barrierpg.frontier import Run, check_runs
from barrierpg.examples.wear import EXACT_FRONTIER, INCUMBENT, IncumbentRule, WearModel, standing_still_policy, make_wear_env

model = WearModel()
spec = model.spec(ubar=1.0)                              # b = 1, beta = 0.95, utility in [0, 1]
req = Requirement(T=19, alpha=0.05, zeta=0.05, family=2) # p_19 <= 5%, two candidates share the 5% error

print("=" * 100); print("1. PLAN"); print("=" * 100)
# Use a valid utility upper bound. Standing still is the comparison policy.
sup_U = spec.ubar / (1 - spec.beta)
p = plan(spec, req, sup_U=sup_U, expected_failure_rate=0.01, validation_H=300, validation_N=20000,
         features=FeatureConstants.tanh_unit(scale=1.0, input_bound=2.0), lambda_H=1.0, allowance_error_known=False,
         regularizers=dict(action_reference_trusted=True, price_trusted=False, incumbent_available=True, prior_sigma=1.0))
print(p.explain())

print(); print("=" * 100); print("2. EVALUATE TWO CANDIDATES AT ILLUSTRATIVE PRICE m = 1"); print("=" * 100)
stage = Stage(m=1.0)
env = lambda: make_wear_env(stage, model)
idle = evaluate(env, standing_still_policy, 2000, spec=spec, stage=stage, seed=0, T=19)
rule = IncumbentRule(span=3.0, std=0.1, seed=1)
inc = evaluate(env, rule, 20000, spec=spec, stage=stage, seed=0, T=19)
print("standing still   ", idle.summary().replace("\n", "\n                  "))
print("incumbent rule   ", inc.summary().replace("\n", "\n                  "))
print(f"paper's incumbent: q_beta = {INCUMBENT['q_beta']}, U = {INCUMBENT['U']}")

print(); print("=" * 100); print("3. VALIDATE AT THE SPECIFIED HORIZON"); print("=" * 100)
for name, ev in (("standing still", idle), ("incumbent", inc)):
    v = ev.validate(req.alpha * spec.beta ** req.T, req.zeta, req.family)
    print(f"{name:10s} {v.failures} failures / {v.n}: q_beta <= {v.upper:.4g}, "
          f"implies p_{req.T} <= {min(1.0, v.upper / spec.beta ** req.T):.4g} -> "
          f"{'certifies the requirement' if v.passed else 'does not certify the requirement'}")

print(); print("=" * 100); print("4. THE PRICE-RISK EXPRESSION ON THE GRID BENCHMARK"); print("=" * 100)
runs = [Run(m=m, q_beta=q, U=U, label="grid") for m, (q, U) in EXACT_FRONTIER.items() if m > 0]
print("Plug-in calculation using the no-price grid estimate and setting the illustrative gap to zero.")
print("Certification in the continuous model requires a valid utility bound and optimization gap.")
for row in check_runs(runs, beta=spec.beta, sup_U=EXACT_FRONTIER[0.0][1], delta=0.0).table:
    print(f"m={row['m']:<5g} grid risk {row['q_beta']:.5f}, plug-in price-risk expression {row['price_risk_bound']:.5f}")

print(); print("=" * 100); print("5. INTERPRETATION OF THE INCUMBENT AS A CANDIDATE"); print("=" * 100)
rep = interpret(spec, req, stage, inc, candidate_name="the incumbent rule")
print(rep.text())
