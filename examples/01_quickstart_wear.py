"""Quickstart without any learning library: the wear-allowance problem of the manuscript.

Runs with numpy and scipy only. It (1) states the requirement and derives a plan, (2) evaluates two
hand-written candidates (the idle policy and the incumbent operating rule) through the budget
wrapper, (3) validates them at the specified horizon, (4) checks the exact benchmark's frontier
against the price-risk law, and (5) prints the interpretation report of one candidate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpg import Requirement, Stage, evaluate, interpret, plan
from bpg.convergence import FeatureConstants
from bpg.frontier import Run, check_runs
from bpg.examples.wear import EXACT_FRONTIER, INCUMBENT, IncumbentRule, WearModel, idle_policy, make_wear_env

model = WearModel()
spec = model.spec(ubar=1.0)                              # b = 1, beta = 0.95, utility in [0, 1]
req = Requirement(T=19, alpha=0.05, zeta=0.05, family=2) # p_19 <= 5%, two candidates share the 5% error

print("=" * 100); print("1. PLAN"); print("=" * 100)
# sup_U: the stopped utility of the unpriced exact optimum (grid value); the idle policy is the comparator
p = plan(spec, req, sup_U=EXACT_FRONTIER[0.0][1], expected_failure_rate=0.01, validation_H=300, validation_N=20000,
         features=FeatureConstants.tanh_unit(scale=1.0, input_bound=2.0), lambda_H=1.0, allowance_error_known=False,
         regularizers=dict(action_reference_trusted=True, price_trusted=False, incumbent_available=True, prior_sigma=1.0))
print(p.explain())

print(); print("=" * 100); print("2. EVALUATE TWO CANDIDATES AT THE PLANNED PRICE"); print("=" * 100)
stage = Stage(m=1.0)                                     # the conservative price is 5.4; the paper's campaign also used m = 1
env = lambda: make_wear_env(stage, model)
idle = evaluate(env, idle_policy, 2000, spec=spec, stage=stage, seed=0, T=19)
rule = IncumbentRule(span=3.0, std=0.1, seed=1)
inc = evaluate(env, rule, 20000, spec=spec, stage=stage, seed=0, T=19)
print("idle policy      ", idle.summary().replace("\n", "\n                  "))
print("incumbent rule   ", inc.summary().replace("\n", "\n                  "))
print(f"paper's incumbent: q_beta = {INCUMBENT['q_beta']}, U = {INCUMBENT['U']}")

print(); print("=" * 100); print("3. VALIDATE AT THE SPECIFIED HORIZON"); print("=" * 100)
for name, ev in (("idle", idle), ("incumbent", inc)):
    v = ev.validate(req.alpha, req.zeta, req.family)
    print(f"{name:10s} {v.failures} failures / {v.n}: upper endpoint {v.upper:.4g}, floor {v.floor:.4g} -> {'pass' if v.passed else 'fail'}")

print(); print("=" * 100); print("4. THE EXACT FRONTIER AGAINST THE PRICE-RISK LAW"); print("=" * 100)
runs = [Run(m=m, q_beta=q, U=U, label="exact") for m, (q, U) in EXACT_FRONTIER.items() if m > 0]
print(check_runs(runs, beta=spec.beta, sup_U=EXACT_FRONTIER[0.0][1], delta=0.0).explain())

print(); print("=" * 100); print("5. INTERPRETATION OF THE INCUMBENT AS A CANDIDATE"); print("=" * 100)
rep = interpret(spec, req, stage, inc, sup_U=EXACT_FRONTIER[0.0][1], delta=None, candidate_name="the incumbent rule")
print(rep.text())
