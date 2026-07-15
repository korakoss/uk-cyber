"""
Body vs. tail decomposition of the national total (2026-07-15).

Question: is the estimated UK business cybercrime cost driven by BREADTH (a
huge number of small-value incidents) or DEPTH (a few catastrophic ones)?
The national_simulation.py output shows Micro firms dominate BY SIZE BAND, but
that does not by itself say whether the £ within each band come from the body
of the cost distribution or its heavy upper tail.

This script decomposes the expected national total (type-based bridge, point
estimate — no bootstrap needed for shares) two ways:
  1. By observed worst-incident cost band: contribution from firms whose worst
     incident is in the OPEN TOP BAND (£100k+) vs. all lower bands (<£100k).
     This is the direct "catastrophic vs. ordinary incident" split.
  2. Same split within each size band (esp. Micro), to check whether even
     Micro's contribution is tail-inflated.

Each attacked firm contributes  band_conditional_mean(band; per-size lognormal)
x type_bridge(disrupta, freq), weighted by survey weight and scaled to N(size)
— exactly the quantity national_simulation.py averages, just partitioned by
band instead of aggregated. Impersonation's bridge is set to its range midpoint
(4.68) here for a deterministic point estimate.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}
SIZE_LABELS = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}
BAND_BOUNDS = {
    1: (0, 0), 2: (1, 100), 3: (100, 500), 4: (500, 1000), 5: (1000, 5000),
    6: (5000, 10000), 7: (10000, 20000), 8: (20000, 50000),
    9: (50000, 100000), 10: (100000, 500000),  # bounded per codebook; no firm above band 10
}
BEST_ESTIMATE_BRIDGE = {1: 2.21, 2: 1.93, 3: 0.36, 4: 1.01, 7: 1.01, 9: 1.01, 6: 1.02, 11: 5.50}
IMP_BRIDGE_MID = (3.11 + 6.25) / 2
VALID_FREQ = {1, 2, 3, 4, 5, 6}
TOP_BAND = 10  # £100k+, the open catastrophic band


def fit_lognormal(bands):
    bands = bands[~np.isnan(bands)].astype(int)
    if len(bands) == 0:
        return None
    counts = np.array([np.sum(bands == b) for b in range(1, 11)], float)
    if counts[1:].sum() < 5:
        return None
    cond = counts[1:]

    def nll(p):
        mu, ls = p
        sigma = np.exp(ls)
        pred = np.array([
            (1.0 if np.isinf(BAND_BOUNDS[b][1]) else norm.cdf(np.log(BAND_BOUNDS[b][1]), mu, sigma))
            - norm.cdf(np.log(max(BAND_BOUNDS[b][0], 1)), mu, sigma) for b in range(2, 11)])
        s = pred.sum()
        if s < 1e-9:
            return 1e10
        return -(cond * np.log(np.clip(pred / s, 1e-12, 1))).sum()

    best = None
    for mu0 in (5, 7, 9, 11):
        for ls0 in (0.5, 1.0, 1.5):
            r = minimize(nll, [mu0, ls0], method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
            if best is None or r.fun < best.fun:
                best = r
    return best.x[0], float(np.exp(best.x[1]))


def band_conditional_mean(mu, sigma, b):
    if b == 1:
        return 0.0
    L, U = BAND_BOUNDS[b]
    a = (np.log(max(L, 1)) - mu) / sigma
    denom = (1.0 if np.isinf(U) else norm.cdf((np.log(U) - mu) / sigma)) - norm.cdf(a)
    if denom < 1e-12:
        return np.exp(mu + sigma ** 2 / 2)
    num = (1.0 if np.isinf(U) else norm.cdf((np.log(U) - mu) / sigma - sigma)) - norm.cdf(a - sigma)
    return np.exp(mu + sigma ** 2 / 2) * num / denom


def type_bridge(disrupta, freq):
    if freq == 1:
        return 1.0
    if disrupta == 5:
        return IMP_BRIDGE_MID
    return BEST_ESTIMATE_BRIDGE.get(int(disrupta), 1.0) if not pd.isna(disrupta) else 1.0


df = get_business_data()
df = df[df['sizeb'].notna()].copy()
df['sizeb'] = df['sizeb'].astype(int)
df['attacked'] = df['freq'].isin(VALID_FREQ)
db = df['damage_bands']
df['band'] = np.where(df['attacked'] & db.notna() & (db < SPECIAL_CODE_THRESHOLD) & (db >= 1), db, np.nan)

print("=" * 78)
print("BODY (<£100k worst incident) vs TAIL (£100k+ worst incident) — type-based")
print("=" * 78)

grand_body = grand_tail = 0.0
rows = []
for sz in (1, 2, 3, 4):
    sub = df[df['sizeb'] == sz]
    fit = fit_lognormal(sub['band'].values)
    mu, sigma = fit
    w = sub['weight'].values
    wsum = w.sum()
    body_pb = tail_pb = 0.0  # expected £ per business, split by band group
    for _, r in sub.iterrows():
        if not r['attacked'] or np.isnan(r['band']):
            continue
        b = int(r['band'])
        contrib = band_conditional_mean(mu, sigma, b) * type_bridge(r['disrupta'], r['freq']) * r['weight'] / wsum
        if b == TOP_BAND:
            tail_pb += contrib
        else:
            body_pb += contrib
    body_nat = body_pb * N_BY_SIZE[sz]
    tail_nat = tail_pb * N_BY_SIZE[sz]
    grand_body += body_nat
    grand_tail += tail_nat
    n_top = int(((sub['band'] == TOP_BAND)).sum())
    tot = body_nat + tail_nat
    rows.append((sz, body_nat, tail_nat, tot, n_top))
    print(f"\n{SIZE_LABELS[sz]} (sigma={sigma:.2f}, {n_top} firms in top band):")
    print(f"  body (<£100k): £{body_nat/1e9:6.3f}bn   tail (£100k+): £{tail_nat/1e9:6.3f}bn   "
          f"tail share {100*tail_nat/tot if tot>0 else 0:.0f}%")

gt = grand_body + grand_tail
print("\n" + "=" * 78)
print(f"NATIONAL: body £{grand_body/1e9:.2f}bn ({100*grand_body/gt:.0f}%)  |  "
      f"tail £{grand_tail/1e9:.2f}bn ({100*grand_tail/gt:.0f}%)  |  total £{gt/1e9:.2f}bn")
print("=" * 78)
print("\nInterpretation key:")
print("  - High tail share => DEPTH (a few catastrophic £100k+ incidents drive the total)")
print("  - High body share => BREADTH (many ordinary sub-£100k incidents drive the total)")
print("  Note: the 'tail' here is only the OPEN top band as OBSERVED; within-band")
print("  lognormal tails also lift the body figures, so this is a lower bound on")
print("  how tail-driven the total really is.")
print("\nDone.")
