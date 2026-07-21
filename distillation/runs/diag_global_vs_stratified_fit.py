"""Diagnostic: is a single GLOBAL lognormal an adequate fit to the banded cost
data, or does cost genuinely need per-size stratification?

Fits a zero-inflated lognormal by censored MLE (band = interval [L,U)) two ways:
  (A) GLOBAL   — one (p0, mu, sigma) for all attacked business firms
  (B) PER-SIZE — one (p0, mu, sigma) per size band (the estimate's choice)
and reports a chi-square goodness-of-fit of predicted vs observed band counts
for each, plus how mu/sigma move across size. GOF here compares the fitted
band probabilities (conditional on cost > 0) against observed band-2..10 counts.

Prints only; no results file (pure diagnostic informing the fitting-layer design).
"""

import os
import sys
import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import load_business_data, BAND_BOUNDS, SIZE_LABELS


def fit_lognormal(bands):
    """bands: array of integer band codes (1..10). Returns (p0, mu, sigma, n, ncond)."""
    bands = bands[~np.isnan(bands)].astype(int)
    n = len(bands)
    counts = np.array([np.sum(bands == b) for b in range(1, 11)], dtype=float)
    p0 = counts[0] / n
    cond = counts[1:]  # bands 2..10
    if cond.sum() < 5:
        return None

    def nll(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        pred = np.zeros(9)
        for i, b in enumerate(range(2, 11)):
            L, U = BAND_BOUNDS[b]
            lo = norm.cdf(np.log(max(L, 1)), mu, sigma)
            hi = norm.cdf(np.log(U), mu, sigma)
            pred[i] = hi - lo
        s = pred.sum()
        if s < 1e-9:
            return 1e10
        pred = np.clip(pred / s, 1e-12, 1)
        return -(cond * np.log(pred)).sum()

    best = None
    for mu0 in (5, 7, 9, 11):
        for ls0 in (0.5, 1.0, 1.5):
            r = minimize(nll, [mu0, ls0], method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
            if best is None or r.fun < best.fun:
                best = r
    return p0, best.x[0], float(np.exp(best.x[1])), n, int(cond.sum())


def gof(bands, mu, sigma):
    """Chi-square GOF of predicted vs observed band-2..10 counts (cost>0 firms).
    Pools adjacent bands so every expected cell >= 5. Returns (chi2, df, p, table)."""
    bands = bands[~np.isnan(bands)].astype(int)
    obs = np.array([np.sum(bands == b) for b in range(2, 11)], dtype=float)
    N = obs.sum()
    pred = np.zeros(9)
    for i, b in enumerate(range(2, 11)):
        L, U = BAND_BOUNDS[b]
        pred[i] = norm.cdf(np.log(U), mu, sigma) - norm.cdf(np.log(max(L, 1)), mu, sigma)
    pred = pred / pred.sum()
    exp = pred * N

    # pool adjacent bins left-to-right so each expected >= 5
    o_pool, e_pool = [], []
    co = ce = 0.0
    for o, e in zip(obs, exp):
        co += o; ce += e
        if ce >= 5:
            o_pool.append(co); e_pool.append(ce); co = ce = 0.0
    if ce > 0:  # merge leftover into last bin
        if e_pool:
            o_pool[-1] += co; e_pool[-1] += ce
        else:
            o_pool.append(co); e_pool.append(ce)
    o_pool, e_pool = np.array(o_pool), np.array(e_pool)
    stat = float(((o_pool - e_pool) ** 2 / e_pool).sum())
    df = max(len(o_pool) - 1 - 2, 1)  # bins - 1 - (mu,sigma)
    p = 1 - chi2.cdf(stat, df)
    return stat, df, p, len(o_pool)


df = load_business_data()
att = df[df["attacked"] & df["band"].notna()]

print("=" * 70)
print("(A) GLOBAL lognormal — one shape for all attacked firms")
print("=" * 70)
g = fit_lognormal(att["band"].values)
p0, mu, sigma, n, ncond = g
stat, dof, pval, nbin = gof(att["band"].values, mu, sigma)
print(f"  n={n}  cost>0={ncond}  p0={p0:.3f}  mu={mu:.2f}  sigma={sigma:.2f}")
print(f"  E[cost|>0]=£{np.exp(mu + sigma**2/2):,.0f}")
print(f"  GOF: chi2={stat:.1f}  df={dof}  p={pval:.3f}  ({nbin} pooled bins)  "
      f"-> {'ADEQUATE' if pval > 0.05 else 'REJECTED at 5%'}")

print("\n" + "=" * 70)
print("(B) PER-SIZE lognormal — the estimate's choice")
print("=" * 70)
print(f"  {'size':<7} {'n':>5} {'cost>0':>7} {'p0':>6} {'mu':>6} {'sigma':>6} "
      f"{'E[cost|>0]':>12} {'GOF chi2':>9} {'df':>3} {'p':>6}")
for sz, lab in SIZE_LABELS.items():
    b = att.loc[att["sizeb"] == sz, "band"].values
    f = fit_lognormal(b)
    if f is None:
        print(f"  {lab:<7} (too few)")
        continue
    p0, mu, sigma, n, ncond = f
    stat, dof, pval, nbin = gof(b, mu, sigma)
    print(f"  {lab:<7} {n:>5} {ncond:>7} {p0:>6.3f} {mu:>6.2f} {sigma:>6.2f} "
          f"£{np.exp(mu + sigma**2/2):>10,.0f} {stat:>9.1f} {dof:>3} {pval:>6.3f}")

print("\nRead: if the GLOBAL fit is adequate AND per-size mu/sigma barely move,")
print("stratification is unnecessary. If mu shifts strongly with size (bigger")
print("firms -> bigger costs), the global fit misprices the bands and per-size")
print("is justified.")
