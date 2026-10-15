"""The curvature budget of the paper's convergence theorem ('Fixed-stage population convergence') for a stage and
an actor class, and what it does and does not say.

Reproduces the constants of the paper's solvable flow: with the plateau feature's suprema,
C_m = 1940 and the certified strength kappa = 1.05 C_m / lambda_W = 2037 at beta = 0.9, m = 1, ubar = 1.
Run: python examples/under-the-hood/curvature_budget.py   (numpy only)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bpg.convergence import (FeatureConstants, KernelConstants, curvature_budget, gaussian_prior_curvature,
                             kappa_for_tolerance, risk_slack_from_prior)
from bpg.examples.plateau import feature_bounds
from bpg.roles import RegularizerMix

# 1. A tanh unit psi = tanh(x . phi) with |phi| <= 2 and the prior N(0, 1): the deficit grows like (1-beta)^-3.
for beta in (0.9, 0.95, 0.99):
    fc = FeatureConstants.tanh_unit(scale=1.0, input_bound=2.0)
    cb = curvature_budget(fc, beta=beta, m=1.0, ubar=1.0, kappa=1.0, lambda_H=gaussian_prior_curvature(1.0))
    print(f"beta = {beta}: C_m = {cb.constants['C_m']:.4g}, C_A = {cb.constants['C_A']:.4g}, certified kappa = {cb.kappa_certified:.4g}")
print()

# 2. The paper's solvable flow: the plateau feature's bounds, computed on a fine grid.
fc = FeatureConstants(**feature_bounds())
cb = curvature_budget(fc, beta=0.9, m=1.0, ubar=1.0, kappa=2037.0, lambda_H=1.0)
print("solvable flow:"); print(cb.explain()); print()

# 3. Any actor class enters through its kernel bounds ('Value and action-law curvature'); softmax is one instance.
print("softmax kernel bounds of the plateau feature:", fc.kernel()); print()

# 4. The roles of a combined regularizer (the particle form of the flow): noise from relative entropy, pull from a potential.
print(RegularizerMix(kappa1=0.1, kappa2=10.0).explain()); print()

# 5. 'Improvement over the prior policy': the risk slack of the flow started at the prior, and the kappa that keeps p_T <= alpha.
R_Gamma = 12.2   # ubar/(1-beta) - V_m(Gamma) in the solvable flow
for kappa in (0.1, 1.0, 10.0, 100.0):
    print(f"kappa = {kappa:6g}: risk slack {risk_slack_from_prior(L0=1.0, T=0, c_H=2.0, R_Gamma=R_Gamma, kappa=kappa):.4g} "
          "(the measured deviation lies orders of magnitude inside it)")
print(f"kappa keeping p_0 <= 0.12 when the prior policy has p_0 = 0.0895: {kappa_for_tolerance(1.0, 0, 2.0, R_Gamma, 0.12, 0.0895):.4g}")
print("\nReading: the threshold is sufficient, not necessary; near it the minimizer lies close to the prior; set kappa for conditioning")
print("and let the certificates, evaluated on the learned policy, detect a bias that is not affordable.")
