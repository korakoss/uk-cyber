"""EXPLORATORY (approach B) — size x type cell-mean estimate, NO per-firm debanding.

Motivation: the production estimate (national_simulation.py, type-based) fits the
cost SHAPE by SIZE only and lets attack TYPE enter solely through the bridge
multiplier; it then debands each firm's own band and averages. Approach B instead
fits a zero-inflated lognormal per (size, type) CELL, collapses each cell to a
single analytic mean single-incident cost, and throws the per-firm band away.
The firm's individual band is replaced by its cell's mean. Type is thus allowed
to move the COST distribution, not just the bridge.

    National total = Sum_size N(size) * E[annual cost per firm | size]
    E[... | size]  = weighted mean over firms in size of
                        attacked * cellmean(size, type) * bridge(type, freq)

Consistency with the production baseline (kept identical so the number is
comparable):
  - band 10 BOUNDED at £100k-£500k (BAND_BOUNDS[10]=(100000,500000)); the cell
    mean is computed by summing bounded band conditional-means weighted by the
    FITTED band probabilities -> no unbounded lognormal tail.
  - bridge = 1.0 when freq==1 (clean anchor), else the type multiplier.
  - Impersonation (disrupta==5) is a RANGE 3.11-6.25; reported low & high.
  - thin cells (few non-zero obs or implausible sigma) fall back to the
    well-populated per-SIZE shape (same spirit as national_simulation's rule).

Point estimate only (weighted means = the bootstrap mean); no resampling here —
the goal is to read off the number approach B gives and compare to ~£2.2bn.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}
SIZE_LABELS = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}
BAND_BOUNDS = {1: (0, 0), 2: (1, 100), 3: (100, 500), 4: (500, 1000),
               5: (1000, 5000), 6: (5000, 10000), 7: (10000, 20000),
               8: (20000, 50000), 9: (50000, 100000), 10: (100000, 500000)}

# type-based bridge (disrupta -> multiplier), from bridge_specification.py
BRIDGE = {1: 2.21, 2: 1.93, 3: 0.36, 4: 1.01, 7: 1.01, 9: 1.01, 6: 1.02, 11: 5.50}
IMP_LOW, IMP_HIGH = 3.11, 6.25  # disrupta==5
VALID_FREQ = {1, 2, 3, 4, 5, 6}

# thin-cell fallback gates (looser than the per-(size,freq) rule since size x type
# cells are inherently thinner; sigma cap identical)
MIN_NONZERO = 10
SIGMA_CAP = 3.5


def fit_lognormal(bands):
    """Censored-MLE zero-inflated lognormal on band codes. Returns (p0,mu,sigma,n_nonzero)."""
    bands = bands[~np.isnan(bands)].astype(int)
    n = len(bands)
    if n == 0:
        return None
    counts = np.array([np.sum(bands == b) for b in range(1, 11)], dtype=float)
    p0 = counts[0] / n
    cond = counts[1:]
    if cond.sum() < 5:
        return None

    def nll(params):
        mu, ls = params
        sigma = np.exp(ls)
        pred = np.array([norm.cdf(np.log(BAND_BOUNDS[b][1]), mu, sigma)
                         - norm.cdf(np.log(max(BAND_BOUNDS[b][0], 1)), mu, sigma)
                         for b in range(2, 11)])
        s = pred.sum()
        if s < 1e-9:
            return 1e10
        pred = np.clip(pred / s, 1e-12, 1)
        return -(cond * np.log(pred)).sum()

    best = None
    for mu0 in (5, 7, 9, 11):
        for ls0 in (0.5, 1.0, 1.5):
            r = minimize(nll, [mu0, ls0], method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
            if best is None or r.fun < best.fun:
                best = r
    return p0, best.x[0], float(np.exp(best.x[1])), int(cond.sum())


def band_condmean(mu, sigma, b):
    """Analytic E[X | L<=X<U], band 10 BOUNDED (U finite). Band 1 -> 0."""
    if b == 1:
        return 0.0
    L, U = BAND_BOUNDS[b]
    a = (np.log(max(L, 1)) - mu) / sigma
    bU = (np.log(U) - mu) / sigma
    denom = norm.cdf(bU) - norm.cdf(a)
    if denom < 1e-12:
        return np.exp(mu + sigma**2 / 2)
    num = norm.cdf(bU - sigma) - norm.cdf(a - sigma)
    return np.exp(mu + sigma**2 / 2) * num / denom


def bounded_cell_mean(fit):
    """Single analytic mean single-incident cost for a cell, using FITTED band
    probabilities and BOUNDED band conditional-means. Includes the zero-cost
    mass via (1-p0). Returns £."""
    p0, mu, sigma = fit[0], fit[1], fit[2]
    # fitted probability in each band 2..10 (normalised over the non-zero part)
    praw = np.array([norm.cdf(np.log(BAND_BOUNDS[b][1]), mu, sigma)
                     - norm.cdf(np.log(max(BAND_BOUNDS[b][0], 1)), mu, sigma)
                     for b in range(2, 11)])
    praw = praw / praw.sum()
    nonzero_mean = sum(w * band_condmean(mu, sigma, b) for w, b in zip(praw, range(2, 11)))
    return (1 - p0) * nonzero_mean


# ---------------------------------------------------------------------------
df = get_business_data()
df = df[df['sizeb'].notna()].copy()
df['sizeb'] = df['sizeb'].astype(int)
df['attacked'] = df['freq'].isin(VALID_FREQ)
db = df['damage_bands']
df['band'] = np.where(df['attacked'] & db.notna() & (db < SPECIAL_CODE_THRESHOLD) & (db >= 1),
                      db, np.nan)

# per-size fallback shapes + their bounded cell mean
size_fit = {sz: fit_lognormal(df.loc[df['sizeb'] == sz, 'band'].values) for sz in (1, 2, 3, 4)}
size_mean = {sz: bounded_cell_mean(size_fit[sz]) for sz in (1, 2, 3, 4)}

# per (size, type) bounded cell means, with thin-cell fallback to per-size
types_present = sorted(int(t) for t in df.loc[df['attacked'], 'disrupta'].dropna().unique())
cellmean = {}
cellsrc = {}
print("Per (size, type) bounded cell mean single-incident cost (£):")
print(f"  {'size':<7} {'type':>5}  {'n_nz':>5}  {'mu':>6} {'sig':>5}  {'cellmean£':>10}  src")
for sz in (1, 2, 3, 4):
    for t in types_present:
        bands = df.loc[(df['sizeb'] == sz) & (df['attacked']) & (df['disrupta'] == t), 'band'].values
        fit = fit_lognormal(bands)
        if fit is not None and fit[3] >= MIN_NONZERO and fit[2] <= SIGMA_CAP:
            cellmean[(sz, t)] = bounded_cell_mean(fit)
            cellsrc[(sz, t)] = 'cell'
            print(f"  {SIZE_LABELS[sz]:<7} {t:>5}  {fit[3]:>5}  {fit[1]:>6.2f} {fit[2]:>5.2f}  "
                  f"{cellmean[(sz,t)]:>10,.0f}  cell")
        else:
            cellmean[(sz, t)] = size_mean[sz]
            cellsrc[(sz, t)] = 'size-pooled'
n_cell = sum(v == 'cell' for v in cellsrc.values())
print(f"  -> {n_cell}/{len(cellsrc)} (size,type) cells use own shape; rest fall back to per-size")


def bridge(t, freq, imp):
    if freq == 1:
        return 1.0
    if t == 5:
        return imp
    return BRIDGE.get(int(t), 1.0)


def national_total(imp):
    total = 0.0
    per_size = {}
    for sz in (1, 2, 3, 4):
        sub = df[df['sizeb'] == sz]
        w = sub['weight'].values
        contrib = np.zeros(len(sub))
        att = sub['attacked'].values
        freq = sub['freq'].values
        dis = sub['disrupta'].values
        for i in range(len(sub)):
            if not att[i]:
                continue
            t = dis[i]
            cm = cellmean.get((sz, int(t)), size_mean[sz]) if not pd.isna(t) else size_mean[sz]
            br = bridge(t if not pd.isna(t) else -1, freq[i], imp)
            contrib[i] = cm * br
        epb = np.average(contrib, weights=w)   # E[annual cost per firm | size]
        c = N_BY_SIZE[sz] * epb
        per_size[sz] = (epb, c)
        total += c
    return total, per_size


print("\n" + "=" * 70)
print("APPROACH B — national annual total (point estimate)")
print("=" * 70)
for label, imp in [("Impersonation LOW (3.11x)", IMP_LOW), ("Impersonation HIGH (6.25x)", IMP_HIGH)]:
    total, per_size = national_total(imp)
    print(f"\n{label}:  TOTAL = £{total/1e9:,.2f}bn")
    for sz in (1, 2, 3, 4):
        epb, c = per_size[sz]
        print(f"    {SIZE_LABELS[sz]:<7} £{c/1e9:>6.2f}bn   (£{epb:,.0f}/business x {N_BY_SIZE[sz]:,})")

print("\nCompare: production type-based (per-size cost shape, per-firm deband) ~£2.2bn.")

# ---------------------------------------------------------------------------
# DIAGNOSIS: why is B ~half of production? Hypothesis — replacing each firm's
# OWN band with a fitted analytic cell mean discards the empirical tail: the
# lognormal predicts fewer top-band (band 10) firms than are actually observed,
# so the fitted mean sits below the empirical debanded mean. Compare, per size,
# the EMPIRICAL debanded mean (production: average each firm's own bounded band
# condmean) vs the FITTED analytic cell mean (B), plus observed vs fitted count
# of band-10 firms.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DIAGNOSIS: empirical band mix vs fitted analytic mean, per size")
print("=" * 70)
print(f"  {'size':<7} {'emp.mean£':>10} {'fit.mean£':>10} {'emp/fit':>8}  "
      f"{'obs band10':>10} {'fit band10':>10}")
for sz in (1, 2, 3, 4):
    fit = size_fit[sz]
    mu, sigma = fit[1], fit[2]
    bands = df.loc[(df['sizeb'] == sz) & df['band'].notna(), 'band'].values.astype(int)
    w = df.loc[(df['sizeb'] == sz) & df['band'].notna(), 'weight'].values
    # empirical: each firm's own bounded band condmean, weighted
    emp = np.average([band_condmean(mu, sigma, b) for b in bands], weights=w)
    fitm = bounded_cell_mean(fit)
    # observed weighted count-share in band 10 vs lognormal-predicted P(band10|>0)
    obs10 = np.average((bands == 10).astype(float), weights=w)
    p2_10 = np.array([norm.cdf(np.log(BAND_BOUNDS[b][1]), mu, sigma)
                      - norm.cdf(np.log(max(BAND_BOUNDS[b][0], 1)), mu, sigma) for b in range(2, 11)])
    fit10 = (1 - fit[0]) * (p2_10[-1] / p2_10.sum())
    print(f"  {SIZE_LABELS[sz]:<7} {emp:>10,.0f} {fitm:>10,.0f} {emp/fitm:>7.2f}x  "
          f"{obs10:>9.2%} {fit10:>9.2%}")
print("\nIf emp.mean > fit.mean and obs band10 > fit band10, the lognormal")
print("under-predicts the observed tail; B's analytic mean throws it away.")
print("Done.")
