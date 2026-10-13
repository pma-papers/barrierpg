"""The solvable flow of the paper (Online Appendix E.1 and Sections 6.3, 6.5), numpy only.

One physical state coordinate x uniform on [0, 1], action a in [0, 1] with a uniform action reference,
utility u = a at the first action and zero after, a first cost g = 1 - a + eps x against the budget b = 1
with cost discount gamma, and zero cost after. Failure happens at the first action exactly when a < eps x.
The barrier is B(z) = log(1 + 1/z). The feature is psi(a, theta) = a chi(theta), where chi is flat (zero)
on [-1, 1] and smooth outside it, so a law mu of unit parameters acts only through its tilt
c = int chi dmu, and the policy has density proportional to exp(c a).

From the plateau initial law (units in [-1/2, 1/2]) the value gradient is zero: unregularized transport
cannot move. This makes the example the place where the roles of the regularizers are visible: only noise
lets units reach |theta| > 1, where the feature acts.

PlateauModel computes the clipped value V_m, the stopped utility, the interior loss and the geometric risk as
functions of the tilt by quadrature (no simulation); simulate_particles runs the particle form of the flow.
The paper's parameters are eps = 0.2, m = 1, beta = 0.9, gamma = 1/2 (plateau value V_m = -2.2533).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


def chi(theta):
    """Flat on [-1, 1], exp(-1/(|theta|-1)^2) outside: smooth, bounded by one, with bounded derivatives."""
    theta = np.asarray(theta, dtype=float)
    s = np.abs(theta) - 1.0
    out = np.zeros_like(theta)
    pos = s > 0
    out[pos] = np.exp(-1.0 / s[pos] ** 2)
    return out


def chi_prime(theta):
    theta = np.asarray(theta, dtype=float)
    s = np.abs(theta) - 1.0
    out = np.zeros_like(theta)
    pos = s > 0
    out[pos] = np.sign(theta[pos]) * 2.0 * s[pos] ** -3 * np.exp(-1.0 / s[pos] ** 2)
    return out


def chi_second(theta):
    theta = np.asarray(theta, dtype=float)
    s = np.abs(theta) - 1.0
    out = np.zeros_like(theta)
    pos = s > 0
    sp = s[pos]
    out[pos] = (4.0 * sp ** -6 - 6.0 * sp ** -4) * np.exp(-1.0 / sp ** 2)
    return out


def feature_bounds() -> Dict[str, float]:
    """(L0, L1, L2) of the feature a chi(theta), a in [0, 1], computed on a fine grid."""
    th = np.linspace(-12, 12, 480001)
    return dict(L0=1.0, L1=float(np.max(np.abs(chi_prime(th)))), L2=float(np.max(np.abs(chi_second(th)))))


class PlateauModel:
    """Complete clipped returns as functions of the first action, by Gauss-Legendre quadrature with the failure
    boundary a = eps x resolved exactly and a square-root substitution for the barrier singularity."""

    def __init__(self, eps: float = 0.2, m: float = 1.0, beta: float = 0.9, gamma: float = 0.5, n_x: int = 120, n_s: int = 120,
                 t_max: int = 400):
        self.eps, self.m, self.beta, self.gamma = eps, m, beta, gamma
        xg, xw = np.polynomial.legendre.leggauss(n_x)
        x = 0.5 * (xg + 1.0); wx = 0.5 * xw
        sg, sw = np.polynomial.legendre.leggauss(n_s)
        s = 0.5 * (sg + 1.0); ws = 0.5 * sw
        A, X, W, F = [], [], [], []
        for xi, wxi in zip(x, wx):
            a0 = eps * xi                                   # failure iff a < a0
            a_s = a0 + (1.0 - a0) * s ** 2                  # survivors
            w_s = wxi * ws * 2.0 * (1.0 - a0) * s
            a_f = a0 * 0.5 * (sg + 1.0)                     # failures
            w_f = wxi * 0.5 * sw * a0
            A += [a_s, a_f]; X += [np.full_like(a_s, xi), np.full_like(a_f, xi)]
            W += [w_s, w_f]; F += [np.zeros_like(a_s, dtype=bool), np.ones_like(a_f, dtype=bool)]
        self.a = np.concatenate(A); self.x = np.concatenate(X); self.w = np.concatenate(W); self.fail = np.concatenate(F)
        z1 = (1.0 - (1.0 - self.a + eps * self.x)) / gamma          # updated reserve after the first action
        surv = ~self.fail
        with np.errstate(divide="ignore", invalid="ignore"):
            B1 = np.where(z1 > 0, np.log1p(1.0 / np.where(z1 > 0, z1, 1.0)), np.inf)
        R = np.where(surv, np.maximum(-m, self.a - B1), -m)
        I = np.where(surv, np.minimum(B1, self.a + m), 0.0)
        t = np.arange(1, t_max + 1); disc = beta ** t
        zs = z1[surv][:, None] / gamma ** t[None, :]
        Bt = np.log1p(1.0 / zs)
        R[surv] += np.sum(disc[None, :] * np.maximum(-m, -Bt), axis=1)
        I[surv] += np.sum(disc[None, :] * np.minimum(Bt, m), axis=1)
        R[self.fail] = -m / (1.0 - beta)                            # absorbing failure tail, exact
        self.R, self.I = R, I
        self.U = np.where(surv, self.a, 0.0)

    def _weights(self, c: float) -> np.ndarray:
        wt = self.w * np.exp(c * (self.a - 0.5))
        return wt / wt.sum()

    def stats(self, c: float) -> Dict[str, float]:
        """V_m, U, I_m and q_beta of the policy with tilt c; the identity V_m = U - I_m - m q/(1-beta) holds to quadrature accuracy."""
        p = self._weights(c)
        q = float(p @ self.fail.astype(float))
        out = dict(c=float(c), V_m=float(p @ self.R), U=float(p @ self.U), I_m=float(p @ self.I), q_beta=q)
        out["identity_residual"] = out["V_m"] - (out["U"] - out["I_m"] - self.m * q / (1.0 - self.beta))
        return out

    def dV(self, c: float) -> float:
        """d V_m / d c: the covariance of the action and the return under the tilted policy."""
        p = self._weights(c)
        return float(p @ (self.a * self.R) - (p @ self.a) * (p @ self.R))

    def value_gradient(self, theta, c: float):
        """Gradient of the first variation of V_m with respect to one unit's parameter: V_m'(c) chi'(theta)."""
        return self.dV(c) * chi_prime(theta)


def plateau_initial_law(n: int, rng: np.random.Generator) -> np.ndarray:
    """N(0, 1) conditioned on [-1/2, 1/2]: units inside the flat region of the feature."""
    out = np.empty(0)
    while out.size < n:
        x = rng.standard_normal(4 * n)
        out = np.concatenate([out, x[np.abs(x) <= 0.5]])
    return out[:n]


@dataclass
class ParticleRun:
    times: np.ndarray
    V_m: np.ndarray
    q_beta: np.ndarray
    spread: np.ndarray
    final_units: np.ndarray


def simulate_particles(model: PlateauModel, kappa1: float, kappa2: float, *, n: int = 2000, seed: int = 0,
                       horizon: Optional[float] = None, dt: Optional[float] = None, prior_sigma: float = 1.0,
                       n_record: int = 80, initial: Optional[np.ndarray] = None) -> ParticleRun:
    """Euler-Maruyama for the particle form with relative entropy to N(0, prior_sigma^2) at strength kappa1 and the
    potential x^2/2 at strength kappa2:
        dx_i = [V_m'(c_n) chi'(x_i) - (kappa1/sigma^2 + kappa2) x_i] dl + sqrt(2 kappa1) dB_i,  c_n = mean chi(x_j).
    Defaults follow the paper: horizon 40/(pull), step 0.02/(pull)."""
    pull = kappa1 / prior_sigma ** 2 + kappa2
    if pull <= 0:
        raise ValueError("need kappa1 > 0 or kappa2 > 0")
    horizon = 40.0 / pull if horizon is None else horizon
    dt = 0.02 / pull if dt is None else dt
    rng = np.random.default_rng(seed)
    x = plateau_initial_law(n, rng) if initial is None else np.array(initial, dtype=float)
    rec = np.unique(np.concatenate([[0.0], np.geomspace(horizon * 1e-4, horizon, n_record)]))
    T, V, Q, S = [], [], [], []
    l, i = 0.0, 0
    while True:
        while i < len(rec) and l >= rec[i] - 1e-12:
            st = model.stats(float(np.mean(chi(x))))
            T.append(l); V.append(st["V_m"]); Q.append(st["q_beta"]); S.append(float(np.std(x))); i += 1
        if i >= len(rec):
            break
        h = min(dt, rec[i] - l)
        c = float(np.mean(chi(x)))
        x = x + (model.dV(c) * chi_prime(x) - pull * x) * h + math.sqrt(2.0 * kappa1 * h) * rng.standard_normal(x.size)
        l += h
    return ParticleRun(np.array(T), np.array(V), np.array(Q), np.array(S), x)


PAPER_ROLES_TABLE = [  # (regularizer, kappa1, kappa2, final V_m, final spread, predicted spread): Online Table K.1
    ("relative entropy", 0.1, 0.0, -1.9644, 2.288, 1.000),
    ("relative entropy", 1.0, 0.0, -2.2176, 1.071, 1.000),
    ("potential", 0.0, 1.0, -2.2533, 0.000, 0.000),
    ("potential", 0.0, 10.0, -2.2533, 0.000, 0.000),
    ("sum", 0.1, 1.0, -2.2533, 0.304, 0.302),
    ("sum", 0.1, 10.0, -2.2533, 0.100, 0.100),
]

__all__ = ["chi", "chi_prime", "chi_second", "feature_bounds", "PlateauModel", "plateau_initial_law", "ParticleRun",
           "simulate_particles", "PAPER_ROLES_TABLE"]
