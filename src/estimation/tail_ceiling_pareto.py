"""EXPLORATORY — catastrophic-tail ceiling from the EMPTY high bands.

The survey offered loss bands above band 10 (11=£500k-1M, 12=£1M-5M, 13=£5M+)
and NO firm selected any of them. Those empty-but-available bands let us bound
how heavy the loss tail above £100k could plausibly be without contradicting
"we asked about bigger losses and found none", and then price the worst still-
allowed tail into the national total.

METHOD (plain):
 1. Tail shape: above u=£100k assume a power law, P(loss > x) = (x/u)^(-alpha).
    alpha is the single heaviness knob (small=fat tail, big=skinny).
 2. Anchor: 8 firms observed above £100k in the whole sample (all in band 10,
    £100k-500k), 0 above £500k. Condition on those 8 being the [100k,500k] firms.
 3. For a given alpha, expected # above £500k given the 8 = 8 * r/(1-r), r=5^-alpha
    (5 = 500k/100k). Observing 0 there has probability exp(-expected).
 4. Ceiling alpha* = heaviest (smallest) alpha with expected-above-£500k = 3
    (the ~5% "we'd probably have seen one" level — same spirit as rule-of-three,
    here derived from the tail model). Anything heavier is rejected by the zeros.
 5. Price it: mean loss of an above-£100k firm under Pareto(alpha*), CAPPED at C
    (a power law this heavy has a runaway mean, so the top must be capped and the
    total's sensitivity to C reported).
 6. Swap that tail mean in for the band-10 firms' cost, re-run the national total.

Everything below £100k is left exactly as the baseline. Deterministic point
estimate (= bootstrap mean); Impersonation at its range MIDPOINT to keep the
focus on the tail. Compare the ceiling to the ~£2.2bn baseline.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, brentq

sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}
SIZE_LABELS = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}
BAND_BOUNDS = {1: (0, 0), 2: (1, 100), 3: (100, 500), 4: (500, 1000),
               5: (1000, 5000), 6: (5000, 10000), 7: (10000, 20000),
               8: (20000, 50000), 9: (50000, 100000), 10: (100000, 500000)}
BRIDGE = {1: 2.21, 2: 1.93, 3: 0.36, 4: 1.01, 7: 1.01, 9: 1.01, 6: 1.02, 11: 5.50}
IMP_MID = (3.11 + 6.25) / 2
VALID_FREQ = {1, 2, 3, 4, 5, 6}
U = 100_000  # tail threshold (£100k, band-10 lower edge)


def fit_lognormal(bands):
    bands = bands[~np.isnan(bands)].astype(int)
    if len(bands) == 0:
        return None
    counts = np.array([np.sum(bands == b) for b in range(1, 11)], float)
    p0 = counts[0] / len(bands)
    cond = counts[1:]
    if cond.sum() < 5:
        return None

    def nll(p):
        mu, ls = p; s = np.exp(ls)
        pr = np.array([norm.cdf(np.log(BAND_BOUNDS[b][1]), mu, s)
                       - norm.cdf(np.log(max(BAND_BOUNDS[b][0], 1)), mu, s) for b in range(2, 11)])
        ss = pr.sum()
        if ss < 1e-9:
            return 1e10
        return -(cond * np.log(np.clip(pr / ss, 1e-12, 1))).sum()
    best = None
    for mu0 in (5, 7, 9, 11):
        for ls0 in (0.5, 1.0, 1.5):
            r = minimize(nll, [mu0, ls0], method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
            if best is None or r.fun < best.fun:
                best = r
    return p0, best.x[0], float(np.exp(best.x[1]))


def band_condmean(mu, sigma, b):
    if b == 1:
        return 0.0
    L, U_ = BAND_BOUNDS[b]
    a = (np.log(max(L, 1)) - mu) / sigma
    bU = (np.log(U_) - mu) / sigma
    denom = norm.cdf(bU) - norm.cdf(a)
    if denom < 1e-12:
        return np.exp(mu + sigma ** 2 / 2)
    num = norm.cdf(bU - sigma) - norm.cdf(a - sigma)
    return np.exp(mu + sigma ** 2 / 2) * num / denom


def pareto_capped_mean(alpha, u, C):
    """E[min(L, C) | L > u] for L ~ Pareto(alpha, u). Numeric (safe for alpha<1)."""
    xs = np.logspace(np.log10(u), np.log10(C), 20000)
    pdf = alpha * u ** alpha * xs ** (-alpha - 1)   # density on (u, inf)
    # integral of x*pdf over [u,C] + C * P(L>C)
    integ = np.trapezoid(xs * pdf, xs)
    tail_at_C = (u / C) ** alpha
    return integ + C * tail_at_C


# ---------------------------------------------------------------------------
df = get_business_data()
df = df[df['sizeb'].notna()].copy()
df['sizeb'] = df['sizeb'].astype(int)
df['attacked'] = df['freq'].isin(VALID_FREQ)
db = df['damage_bands']
df['band'] = np.where(df['attacked'] & db.notna() & (db < SPECIAL_CODE_THRESHOLD) & (db >= 1),
                      db, np.nan)
size_fit = {sz: fit_lognormal(df.loc[df['sizeb'] == sz, 'band'].values) for sz in (1, 2, 3, 4)}

# --- step 4: ceiling alpha from the empty bands -----------------------------
N_ABOVE_100K = int((df['band'] == 10).sum())   # = 8
EXPECTED_ABOVE_500K = 3.0                       # ~5% "would have seen one" level
RATIO = 500_000 / U                             # = 5


def expected_above_500k(alpha):
    r = RATIO ** (-alpha)
    return N_ABOVE_100K * r / (1 - r)


alpha_star = brentq(lambda a: expected_above_500k(a) - EXPECTED_ABOVE_500K, 0.05, 5.0)
print(f"Tail anchor: {N_ABOVE_100K} firms >£100k, 0 above £500k.")
print(f"Heaviest tail consistent with the empty bands: alpha* = {alpha_star:.3f}")
print(f"  (alpha<1 => the UNcapped tail has infinite mean; a cap is mandatory)")
print(f"  check: expected #>£500k at alpha* = {expected_above_500k(alpha_star):.2f} (target 3)")


def national_total(tail_cost=None):
    """Baseline if tail_cost is None; else band-10 firms valued at tail_cost."""
    total = 0.0
    per_size = {}
    for sz in (1, 2, 3, 4):
        sub = df[df['sizeb'] == sz]
        mu, sigma = size_fit[sz][1], size_fit[sz][2]
        w = sub['weight'].values
        att = sub['attacked'].values
        band = sub['band'].values
        freq = sub['freq'].values
        dis = sub['disrupta'].values
        cost = np.zeros(len(sub))
        for i in range(len(sub)):
            if not att[i] or np.isnan(band[i]):
                continue
            b = int(band[i])
            c = tail_cost if (b == 10 and tail_cost is not None) else band_condmean(mu, sigma, b)
            if freq[i] == 1:
                br = 1.0
            elif not pd.isna(dis[i]) and int(dis[i]) == 5:
                br = IMP_MID
            else:
                br = BRIDGE.get(int(dis[i]), 1.0) if not pd.isna(dis[i]) else 1.0
            cost[i] = c * br
        epb = np.average(cost, weights=w)
        c = N_BY_SIZE[sz] * epb
        per_size[sz] = c
        total += c
    return total, per_size


base_total, base_ps = national_total(None)
print(f"\nBASELINE (band 10 = bounded £100k-500k conditional mean): "
      f"£{base_total/1e9:,.2f}bn")

print("\nCEILING — band-10 firms valued at the fattest allowed Pareto tail mean,")
print("for several caps C on the largest single loss:")
print(f"  {'cap C':>10} {'tail mean/firm':>15} {'CEILING total':>15}  {'x baseline':>10}")
for C in (5e6, 50e6, 500e6):
    tmean = pareto_capped_mean(alpha_star, U, C)
    tot, _ = national_total(tmean)
    print(f"  £{C/1e6:>7,.0f}M {'£'+format(tmean,',.0f'):>15} "
          f"{'£'+format(tot/1e9,',.2f')+'bn':>15}  {tot/base_total:>9.2f}x")

print("\nRead: the empty high bands alone do NOT pin the tail — alpha* < 1, so the")
print("ceiling is driven by the assumed cap on the largest single loss, not by the")
print("data. Honest conclusion: with only 8 firms above £100k, 'we saw none above")
print("£500k' is weak evidence; bounding the catastrophic tail needs an EXTERNAL")
print("cap on the maximum credible single loss, not just this survey.")
print("Done.")
