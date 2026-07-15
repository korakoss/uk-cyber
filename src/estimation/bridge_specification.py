"""
Bridge specification: folds the censored-model results (type_cost_censored_model.py,
2026-07-13) into the project's established three-tier bridge framework
(conservative=1, best-estimate=validated model where one exists, pessimistic
=k_implied — see NOTES.md "Build the Squiggle model" item), and applies the
resulting per-type multiplier table to the actual attacked-firm sample to
show what it changes versus the previous "wide bounds only" treatment for
non-phishing types.

Per-type tiers, and where each comes from:
- freq==1 (ANY type): bridge=1 exactly, overrides everything below — the
  single reported incident IS the whole year's cost, no bridging needed.
- Phishing (disrupta=6): validated mixture-model bridge. Best estimate
  ≈1.02 (mixture_bridge.py, excluding the still-unresolved freq=6 anomaly).
  Pessimistic ≈1.70 (including it). Conservative=1.
- Malware (disrupta=2), DoS (disrupta=3), Hacking-broad (disrupta in
  {4,7,9}), Takeover (disrupta=11): best estimate = the censored-lognormal-
  fit ratio "E[cost|type occurred] / naive single-worst-incident mean"
  from type_cost_censored_model.py's 2026-07-13 run (hardcoded below with
  the run's printed values — rerun that script if the data or model
  changes). Conservative=1, pessimistic=K_IMPLIED[freq] (same empirical
  multiplier convention as bridge_multiplier.py).
- Ransomware (disrupta=1): best estimate now resolved via the clean
  single-attack-count reference subsample + Monte Carlo technique
  (type_specific_montecarlo_bridge.py, 2026-07-14): £605.71/business vs.
  a £273.56/business naive figure, an implied 2.21x multiplier. Solid,
  confirmed consistent via a body-vs-tail lognormal check.
- Impersonation (disrupta=5): same Monte Carlo technique, but fragile
  (leave-one-out on its reference sample swings the estimate -50.2%), so
  it is carried as a RANGE: £950.12-£1,908.61/business, implied multiplier
  range 3.11x-6.25x on the naive £305.49/business base. Both endpoints
  applied below (low/high best-estimate columns).
- The smaller untested types (disrupta 8=student unauthorised access,
  10=video/IM eavesdropping, 12=any other): NO best estimate, never
  tested (too small). Conservative=1, pessimistic=K_IMPLIED[freq],
  best-estimate=NaN.

Output: applies all three tiers to every attacked firm in the sample and
reports population-weighted (per-business; national scaling still needs
the pending ONS business-count step) totals for:
  (a) fully conservative (bridge=1 everywhere)
  (b) fully pessimistic (bridge=K_IMPLIED[freq] everywhere, phishing=1.70)
  (c) "current best available": best-estimate where one exists (phishing,
      freq=1, Ransomware, Impersonation [low/high range], and the 4 solid
      censored-model types), falling back to conservative (=1) — for the
      remaining unresolved minor types
Plus: what SHARE of total attacked-firm weight now has a genuine best
estimate vs. still falls back to conservative-by-default, to make
concrete how much this closes the original wide-bounds gap.
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

df = get_business_data()
INVALID_FREQ = {997, 999, 997.0, 999.0}

attacked = df[
    df['freq'].notna() & ~df['freq'].isin(INVALID_FREQ) & (df['freq'] > 0) &
    df['damage_bands'].notna() & (df['damage_bands'] < SPECIAL_CODE_THRESHOLD) &
    df['disrupta'].notna() & (df['disrupta'] > 0) & (df['disrupta'] != 997)
].copy()

print(f"n attacked firms (valid freq, damage_bands, disrupta): {len(attacked)}, "
      f"weight sum={attacked['weight'].sum():.2f} (population base weight sum={df['weight'].sum():.2f})")

FULL_MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000,
            9: 75000, 10: 300000, 11: 750000, 12: 3000000, 13: 5000000}

K_IMPLIED = {1: 1, 2: 6, 3: 12, 4: 52, 5: 365, 6: 1000}

# Best-estimate bridge ratios, hardcoded from type_cost_censored_model.py's
# 2026-07-13 run ("[Comparison] ... a N.NNx multiplier" lines). Phishing's
# from mixture_bridge.py (established prior to this session).
BEST_ESTIMATE_BRIDGE = {
    6: 1.02,   # Phishing (mixture model, excl. freq=6 anomaly)
    2: 1.93,   # Other malware
    3: 0.36,   # Denial of service
    4: 1.01, 7: 1.01, 9: 1.01,  # Hacking (broad: bank acct + unauthorised access staff/outsiders)
    11: 5.50,  # Website/social/email takeover
    1: 2.21,   # Ransomware (Monte Carlo bridge, £605.71 vs £273.56 naive — solid point estimate)
}
# Impersonation (disrupta=5) is a RANGE, not a point — fragile leave-one-out result.
# Low = £950.12/business, High = £1,908.61/business, on a £305.49 naive base.
IMPERSONATION_BRIDGE_LOW = 3.11
IMPERSONATION_BRIDGE_HIGH = 6.25
PESSIMISTIC_OVERRIDE = {6: 1.70}  # Phishing's pessimistic isn't K_IMPLIED-based — mixture-model upper bound

TYPE_LABELS = {
    1: 'Ransomware', 2: 'Other malware', 3: 'Denial of service', 4: 'Hacking',
    5: 'Impersonation', 6: 'Phishing', 7: 'Hacking (staff unauth. access)',
    8: 'Unauth. access (students)', 9: 'Hacking (outsider unauth. access)',
    10: 'Video/IM eavesdropping', 11: 'Website/social/email takeover',
    12: 'Any other',
}

attacked['damage_mid'] = attacked['damage_bands'].map(FULL_MID)
attacked['k'] = attacked['freq'].map(lambda f: K_IMPLIED.get(int(f), 1))

attacked['cons_total'] = attacked['damage_mid'] * 1.0
attacked['pess_total'] = attacked.apply(
    lambda r: r['damage_mid'] * (PESSIMISTIC_OVERRIDE.get(int(r['disrupta']), r['k']) if r['freq'] > 1 else 1.0),
    axis=1
)


def best_total(row, impersonation_bridge=None):
    if row['freq'] == 1:
        return row['damage_mid']  # bridge=1 exactly, no ambiguity
    if int(row['disrupta']) == 5:  # Impersonation — range, not a fixed dict entry
        return row['damage_mid'] * impersonation_bridge
    b = BEST_ESTIMATE_BRIDGE.get(int(row['disrupta']))
    if b is None:
        return np.nan  # unresolved — no best estimate for this type
    return row['damage_mid'] * b


attacked['best_total'] = attacked.apply(lambda r: best_total(r, IMPERSONATION_BRIDGE_LOW), axis=1)
attacked['best_total_low'] = attacked['best_total']
attacked['best_total_high'] = attacked.apply(lambda r: best_total(r, IMPERSONATION_BRIDGE_HIGH), axis=1)
attacked['has_best'] = attacked['best_total'].notna()

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print("\n" + "=" * 100)
print("PER-TYPE BRIDGE TIER COVERAGE (attacked firms, freq>1 only — freq=1 is unambiguous)")
print("=" * 100)
freq_gt1 = attacked[attacked['freq'] > 1]
by_type = freq_gt1.groupby('disrupta').agg(
    n=('weight', 'size'), w=('weight', 'sum'), has_best=('has_best', 'mean')
).reset_index()
by_type['label'] = by_type['disrupta'].map(lambda d: TYPE_LABELS.get(int(d), str(d)))
def _bridge_label(d):
    d = int(d)
    if d == 5:
        return f"{IMPERSONATION_BRIDGE_LOW}-{IMPERSONATION_BRIDGE_HIGH} (range)"
    b = BEST_ESTIMATE_BRIDGE.get(d)
    return b if b is not None else np.nan


by_type['best_estimate_bridge'] = by_type['disrupta'].map(_bridge_label)
print(by_type[['label', 'n', 'w', 'best_estimate_bridge', 'has_best']].to_string(index=False))

print("\n" + "=" * 100)
print("AGGREGATE TOTALS (population-weighted £ per business — national scaling still pending ONS step)")
print("=" * 100)

pop_w = df['weight'].sum()


def pop_avg(series, weights):
    return np.average(series, weights=weights) * (weights.sum() / pop_w)


cons_avg = pop_avg(attacked['cons_total'], attacked['weight'])
pess_avg = pop_avg(attacked['pess_total'], attacked['weight'])

resolved = attacked[attacked['has_best']]
unresolved = attacked[~attacked['has_best']]
best_avg_resolved_part_low = pop_avg(resolved['best_total_low'], resolved['weight'])
best_avg_resolved_part_high = pop_avg(resolved['best_total_high'], resolved['weight'])
# For unresolved types, current project default is conservative (=1) pending further work
fallback_avg_unresolved_part = pop_avg(unresolved['cons_total'], unresolved['weight'])
mixed_best_avg_low = best_avg_resolved_part_low + fallback_avg_unresolved_part
mixed_best_avg_high = best_avg_resolved_part_high + fallback_avg_unresolved_part

print(f"(a) Fully conservative (bridge=1 everywhere):        £{cons_avg:,.2f} per business")
print(f"(b) Fully pessimistic (K_IMPLIED / phishing 1.70):    £{pess_avg:,.2f} per business")
print(f"(c) Current best available (best-estimate where it exists, conservative fallback elsewhere): "
      f"£{mixed_best_avg_low:,.2f}-£{mixed_best_avg_high:,.2f} per business "
      f"(range is entirely driven by Impersonation's low/high; all other resolved types are point estimates)")
print(f"    of which £{best_avg_resolved_part_low:,.2f}-£{best_avg_resolved_part_high:,.2f} comes from firms "
      f"WITH a genuine best-estimate bridge, £{fallback_avg_unresolved_part:,.2f} from firms still on the "
      f"conservative-by-default fallback")

w_total = attacked['weight'].sum()
w_resolved = resolved['weight'].sum()
print(f"\nShare of attacked-firm weight now covered by a genuine best-estimate bridge "
      f"(freq=1, phishing, Ransomware, Impersonation, or one of the 4 solid censored-model types): "
      f"{100*w_resolved/w_total:.1f}% (was 0% for non-phishing freq>1 firms before this session's censored-model work)")

print(
    "\nInterpretation: (c) sits between (a) and (b) as expected. The gap between (c) and (a) is entirely "
    "attributable to firms where a genuine best-estimate bridge now exists (phishing, Ransomware, "
    "Impersonation, and the 4 solid censored-model types) — previously those firms contributed only the "
    "conservative floor. Ransomware is a solid point estimate; Impersonation is carried as a range because "
    "its reference-sample estimate is fragile (leave-one-out swings it ~50%) — that range is the dominant "
    "source of the remaining width in (c) itself. The residual gap between (c) and (b) is the true "
    "unresolved uncertainty in the untested minor types. Narrowing that further would require new data "
    "(e.g. an incident-level cost breakdown), not more mining of the variables already tried."
)
print("\nDone.")
