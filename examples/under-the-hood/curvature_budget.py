"""The curvature budget of the paper's convergence theorem ('Convergence of the population gradient flow') for a stage and
an actor class, and what it does and does not say.

Reproduces the constants of the paper's analytical example: with the plateau feature's suprema,
C_m = 1940 and the strength five percent above the threshold, kappa = 1.05 C_m / lambda_W = 2037 at beta = 0.9, m = 1, ubar = 1.
Run: python examples/under-the-hood/curvature_budget.py   (numpy only)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from barrierpg.convergence import (FeatureConstants, curvature_budget, gaussian_prior_curvature,
                             kappa_for_tolerance, risk_slack_from_prior)
from barrierpg.examples.plateau import feature_bounds
from barrierpg.roles import RegularizerMix

# 1. A tanh unit psi = tanh(x . phi) with |phi| <= 2 and the prior N(0, 1): the deficit grows like (1-beta)^-3.
for beta in (0.9, 0.95, 0.99):
    fc = FeatureConstants.tanh_unit(scale=1.0, input_bound=2.0)
    cb = curvature_budget(fc, beta=beta, m=1.0, ubar=1.0, kappa=1.0, lambda_H=gaussian_prior_curvature(1.0))
    print(f"beta = {beta}: C_m = {cb.constants['C_m']:.4g}, C_A = {cb.constants['C_A']:.4g}, certified kappa = {cb.kappa_certified:.4g}")
print()

# 2. The paper's analytical example: the plateau feature's global derivative bounds.
fc = FeatureConstants(**feature_bounds())
cb = curvature_budget(fc, beta=0.9, m=1.0, ubar=1.0, kappa=2037.0, lambda_H=1.0)
print("analytical example:"); print(cb.explain()); print()

# 3. Any actor class enters through its kernel bounds ('Value and action-law curvature'); softmax is one instance.
print("softmax kernel bounds of the plateau feature:", fc.kernel()); print()

# 4. The roles of a combined regularizer (the particle system of the gradient flow): noise from relative entropy, drift from a potential.
print(RegularizerMix(kappa1=0.1, kappa2=10.0).explain()); print()

# 5. 'Improvement over the prior policy': the risk bound of the flow started at the prior, and the kappa that keeps p_T <= alpha.
# Since every reward lies in [-1, 1], 20 bounds R_Gamma = ubar/(1-beta) - V_m(Gamma).
R_Gamma_upper = (1.0 + 1.0) / (1.0 - 0.9)
# The prior induces a nonnegative exponential tilt, shifting actions upward relative to uniform.
# Under uniform actions P(a < 0.2 x) = 0.1, which upper-bounds the prior risk.
p_prior_upper = 0.1
for kappa in (0.1, 1.0, 10.0, 100.0):
    slack = risk_slack_from_prior(L0=1.0, T=0, c_H=2.0, R_Gamma=R_Gamma_upper, kappa=kappa)
    print(f"kappa = {kappa:6g}: upper bound on the increase in risk {slack:.4g}")
strength = kappa_for_tolerance(1.0, 0, 2.0, R_Gamma_upper, 0.12, p_prior_upper)
print(f"Sufficient kappa for p_0 <= 0.12 using p_0(prior) <= {p_prior_upper:g} and R_Gamma <= {R_Gamma_upper:g}: {strength:.4g}")
