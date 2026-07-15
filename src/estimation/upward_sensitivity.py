"""
Upward ("inflator") sensitivity scenarios (2026-07-15).

The headline ~£2.2bn (type-based) is deliberately built from deflationary
judgment calls (employer-only frame, hard £500k top-band cap, direct costs
only, midpoint bridges). This script quantifies the main arguments a
reasonable inflator could make, as explicit deterministic scenarios on top of
the corrected baseline, so the magnitudes are concrete rather than hand-waved.

Baseline: deterministic type-based national total (same construction as
body_vs_tail.py — per-size lognormal within-band conditional mean × per-type
bridge, Impersonation bridge at range midpoint), which matches the simulation
mean to ~2 s.f.

Scenarios:
  A. Sole-trader frame — include the 4.27M zero-employee businesses excluded by
     the employer frame, at an assumed fraction of Micro's per-business cost.
  B. Missing catastrophic (>£500k) tail — zero firms reported a most-disruptive
     incident above £500k, but a ~2,000-firm sample can only bound the true rate
     of such incidents, not rule them out. Add them back at a grid of
     (population rate, mean cost) consistent with the zero observation.
  C. Indirect-cost multiplier — apply a literature-style indirect:direct uplift.
  D. A combined "fair inflator" figure stacking A+B+C.

All scenarios are point/grid calculations, not distributions — the aim is
magnitude, not a new uncertainty band. They could later be folded into the
Squiggle simulation as additional model components.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}
N_EMPLOYERS = sum(N_BY_SIZE.values())
N_ZERO_EMPLOYEE = 4_272_535
SIZE_LABELS = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}
BAND_BOUNDS = {
    1: (0, 0), 2: (1, 100), 3: (100, 500), 4: (500, 1000), 5: (1000, 5000),
    6: (5000, 10000), 7: (10000, 20000), 8: (20000, 50000),
    9: (50000, 100000), 10: (100000, 500000),  # bounded per codebook
}
BEST_ESTIMATE_BRIDGE = {1: 2.21, 2: 1.93, 3: 0.36, 4: 1.01, 7: 1.01, 9: 1.01, 6: 1.02, 11: 5.50}
IMP_BRIDGE_MID = (3.11 + 6.25) / 2
VALID_FREQ = {1, 2, 3, 4, 5, 6}


def fit_lognormal(bands):
    bands = bands[~np.isnan(bands)].astype(int)
    counts = np.array([np.sum(bands == b) for b in range(1, 11)], float)
    cond = counts[1:]

    def nll(p):
        mu, ls = p
        sigma = np.exp(ls)
        pred = np.array([norm.cdf(np.log(BAND_BOUNDS[b][1]), mu, sigma)
                         - norm.cdf(np.log(max(BAND_BOUNDS[b][0], 1)), mu, sigma) for b in range(2, 11)])
        s = pred.sum()
        return 1e10 if s < 1e-9 else -(cond * np.log(np.clip(pred / s, 1e-12, 1))).sum()

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
    bU = (np.log(U) - mu) / sigma
    denom = norm.cdf(bU) - norm.cdf(a)
    if denom < 1e-12:
        return np.exp(mu + sigma ** 2 / 2)
    return np.exp(mu + sigma ** 2 / 2) * (norm.cdf(bU - sigma) - norm.cdf(a - sigma)) / denom


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

# ---- baseline type-based total + Micro per-business ----
cpb = {}
for sz in (1, 2, 3, 4):
    sub = df[df['sizeb'] == sz]
    mu, sigma = fit_lognormal(sub['band'].values)
    wsum = sub['weight'].sum()
    contrib = 0.0
    for _, r in sub.iterrows():
        if r['attacked'] and not np.isnan(r['band']):
            contrib += band_conditional_mean(mu, sigma, int(r['band'])) * type_bridge(r['disrupta'], r['freq']) * r['weight']
    cpb[sz] = contrib / wsum
baseline = sum(cpb[sz] * N_BY_SIZE[sz] for sz in (1, 2, 3, 4))
micro_cpb = cpb[1]

print("=" * 78)
print(f"BASELINE (type-based, deterministic): £{baseline/1e9:.2f}bn")
print(f"  Micro cost/business = £{micro_cpb:,.0f}")
print("=" * 78)

# ---- Scenario A: sole-trader frame ----
print("\nSCENARIO A — include 4.27M zero-employee businesses")
print(f"  (added = 4.27M × fraction × Micro £{micro_cpb:,.0f}/business)")
print(f"  {'frac of Micro cost':>20} {'added £bn':>12} {'new total £bn':>15}")
for frac in (0.2, 0.3, 0.4):
    added = N_ZERO_EMPLOYEE * frac * micro_cpb
    print(f"  {frac:>20.0%} {added/1e9:>12.2f} {(baseline+added)/1e9:>15.2f}")

# ---- Scenario B: missing catastrophic (>£500k) tail ----
n_could_report = int(df['band'].notna().sum())  # attacked firms with a valid band
rule_of_three = 3.0 / n_could_report  # upper ~95% bound on rate among these firms
# translate to a population rate: fraction of ALL employers with such an incident
# = (rate among attacked-with-band) × (share of employers that are attacked-with-band)
share_attacked = df['band'].notna().sum() / len(df)  # rough
pop_upper = rule_of_three * share_attacked
print("\nSCENARIO B — catastrophic (>£500k) incidents the sample cannot see")
print(f"  0 firms reported band >10 out of {n_could_report} with a valid band.")
print(f"  Rule-of-three upper 95% bound on the rate among them ≈ {100*rule_of_three:.2f}%")
print(f"  ⇒ population upper bound ≈ {100*pop_upper:.3f}% of employers "
      f"(~{int(pop_upper*N_EMPLOYERS):,} firms)")
print(f"  (added = rate × {N_EMPLOYERS:,} employers × mean catastrophic cost)")
print(f"  {'pop rate':>10} {'mean cost':>12} {'firms':>8} {'added £bn':>12} {'new total £bn':>15}")
for rate in (0.0005, 0.001, pop_upper):
    for mc in (1e6, 2e6, 5e6):
        firms = rate * N_EMPLOYERS
        added = firms * mc
        tag = " (Ro3 upper)" if abs(rate - pop_upper) < 1e-9 else ""
        print(f"  {100*rate:>9.3f}% £{mc/1e6:>9.1f}M {firms:>8.0f} {added/1e9:>12.2f} "
              f"{(baseline+added)/1e9:>15.2f}{tag}")

# ---- Scenario C: indirect-cost multiplier ----
print("\nSCENARIO C — indirect:direct cost uplift (applied to baseline)")
print(f"  {'multiplier':>12} {'new total £bn':>15}")
for m in (1.3, 1.5, 2.0):
    print(f"  {m:>12.1f} {baseline*m/1e9:>15.2f}")

# ---- Scenario D: combined 'fair inflator' ----
print("\nSCENARIO D — combined fair-inflator stack (central & aggressive)")
# central: sole traders @30% of Micro, catastrophic @0.05%×£2M, indirect ×1.3
a_c = N_ZERO_EMPLOYEE * 0.30 * micro_cpb
b_c = 0.0005 * N_EMPLOYERS * 2e6
central = (baseline + a_c + b_c) * 1.3
# aggressive: sole traders @40%, catastrophic @0.1%×£3M, indirect ×1.5
a_a = N_ZERO_EMPLOYEE * 0.40 * micro_cpb
b_a = 0.001 * N_EMPLOYERS * 3e6
aggressive = (baseline + a_a + b_a) * 1.5
print(f"  central   (soletrader 30%, cat 0.05%×£2M, indirect ×1.3): £{central/1e9:.2f}bn")
print(f"  aggressive(soletrader 40%, cat 0.10%×£3M, indirect ×1.5): £{aggressive/1e9:.2f}bn")

print("\n" + "=" * 78)
print("HARD LIMITS an honest inflator must respect (from our data):")
print("  - >£500k incident rate can't exceed ~the rule-of-three bound above")
print("    (else we'd likely have seen one) — caps Scenario B.")
print("  - prevalence already 40-69%, so undetected-breach uplift is small.")
print("  - phishing bridge ~1.02 is validated — the body (~36%) is well pinned.")
print("\nNote: Scenario B additions are ~fully additive (these firms are absent")
print("from the sample); A is additive; C multiplies. D stacks them.")
print("Done.")
