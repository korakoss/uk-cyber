"""
Bridge calibration: damage_bands vs crimecost_bands.

Investigates the ~163 firms with both variables valid.

Checks run here:
1. Representativeness: does the crimecost subsample look like the full attacked population
   in terms of size band, freq band, and damage_bands distribution?
2. freq=1 consistency: for firms attacked exactly once, damage_bands and crimecost_bands
   should be the same (one incident = total cost). How often are they?
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

df = get_business_data()

INVALID_FREQ = {997, 999, 997.0, 999.0}

attacked = df[
    df['freq'].notna() &
    ~df['freq'].isin(INVALID_FREQ) &
    (df['freq'] > 0) &
    df['damage_bands'].notna() &
    (df['damage_bands'] < SPECIAL_CODE_THRESHOLD)
].copy()

has_crimecost = attacked[
    attacked['crimecost_bands'].notna() &
    (attacked['crimecost_bands'] < SPECIAL_CODE_THRESHOLD) &
    (attacked['crimecost_bands'] > 0)
].copy()

SIZE_LABELS = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}
FREQ_LABELS = {1: 'Once only', 2: '>once <monthly', 3: '~monthly',
               4: '~weekly', 5: '~daily', 6: 'Several/day'}
DAMAGE_BANDS = {
    1: 'No cost', 2: '<£100', 3: '£100-£500', 4: '£500-£1k',
    5: '£1k-£5k', 6: '£5k-£10k', 7: '£10k-£20k', 8: '£20k-£50k',
    9: '£50k-£100k', 10: '£100k-£500k'
}

n_all = len(attacked)
n_sub = len(has_crimecost)
print(f"Attacked businesses with valid damage_bands: {n_all}")
print(f"Subset with valid crimecost_bands:           {n_sub} ({100*n_sub/n_all:.1f}%)")

# ---------------------------------------------------------------------------
# 1. Representativeness check
# ---------------------------------------------------------------------------

print()
print("=" * 80)
print("CHECK 1: REPRESENTATIVENESS OF CRIMECOST SUBSAMPLE")
print("=" * 80)

for varname, labels in [('sizeb', SIZE_LABELS), ('freq', FREQ_LABELS), ('damage_bands', DAMAGE_BANDS)]:
    print(f"\n--- {varname} distribution ---")
    counts_all = attacked[varname].value_counts().sort_index()
    counts_sub = has_crimecost[varname].value_counts().sort_index()
    all_vals = sorted(set(counts_all.index) | set(counts_sub.index))

    rows = []
    for v in all_vals:
        n_a = counts_all.get(v, 0)
        n_s = counts_sub.get(v, 0)
        pct_a = 100 * n_a / n_all
        pct_s = 100 * n_s / n_sub
        label = labels.get(v, str(v))
        rows.append((label, n_a, pct_a, n_s, pct_s))

    header = f"  {'Category':<22} {'All n':>7} {'All %':>7} {'Sub n':>7} {'Sub %':>7}"
    print(header)
    for label, n_a, pct_a, n_s, pct_s in rows:
        print(f"  {label:<22} {n_a:>7} {pct_a:>6.1f}% {n_s:>7} {pct_s:>6.1f}%")

    # Chi-square test of equal proportions across categories
    # Use valid integer-coded values only
    valid_vals = [v for v in all_vals if v in counts_all.index and v in counts_sub.index]
    obs_all = np.array([counts_all.get(v, 0) for v in valid_vals])
    obs_sub = np.array([counts_sub.get(v, 0) for v in valid_vals])
    contingency = np.vstack([obs_all, obs_sub])
    if contingency.shape[1] >= 2 and contingency.min() >= 0:
        chi2, p, dof, _ = chi2_contingency(contingency)
        print(f"\n  Chi-square test (same distribution): χ²={chi2:.2f}, df={dof}, p={p:.4f}")

# ---------------------------------------------------------------------------
# 2. freq=1 consistency check
# ---------------------------------------------------------------------------

print()
print("=" * 80)
print("CHECK 2: freq=1 CONSISTENCY (damage_bands vs crimecost_bands)")
print("=" * 80)
print("For firms attacked exactly once, both variables should record the same incident.")

freq1_sub = has_crimecost[has_crimecost['freq'] == 1].copy()
print(f"\nfirms with freq=1 and valid crimecost_bands: {len(freq1_sub)}")

if len(freq1_sub) > 0:
    freq1_sub['band_diff'] = freq1_sub['crimecost_bands'] - freq1_sub['damage_bands']

    same = (freq1_sub['band_diff'] == 0).sum()
    crimecost_higher = (freq1_sub['band_diff'] > 0).sum()
    damage_higher = (freq1_sub['band_diff'] < 0).sum()

    print(f"  crimecost_bands == damage_bands : {same} ({100*same/len(freq1_sub):.1f}%)")
    print(f"  crimecost_bands >  damage_bands : {crimecost_higher} ({100*crimecost_higher/len(freq1_sub):.1f}%)")
    print(f"  crimecost_bands <  damage_bands : {damage_higher} ({100*damage_higher/len(freq1_sub):.1f}%)")

    print(f"\n  Band difference distribution (crimecost - damage):")
    print(f"  {freq1_sub['band_diff'].value_counts().sort_index().to_string()}")

    print(f"\n  Crosstab damage_bands x crimecost_bands (freq=1 only):")
    ct = pd.crosstab(
        freq1_sub['damage_bands'].map(DAMAGE_BANDS),
        freq1_sub['crimecost_bands'].map(DAMAGE_BANDS),
        rownames=['damage_bands'], colnames=['crimecost_bands']
    )
    print(ct)

print()
print("Done.")
