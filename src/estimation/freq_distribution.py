"""
Step 3: P(freq | size, attacked) — how the attack-frequency band distribution
shifts across firm size bands.

freq encodes periodicity bands among attacked firms (not literal attack counts):
1=once only, 2=>once<monthly, 3=~monthly, 4=~weekly, 5=~daily, 6=several/day.

This script is purely exploratory ("eyeballing"):
1. Raw (size x freq) cross-tab of counts, among attacked firms
2. Row proportions (unweighted and weighted) — does the shape shift with size?
3. Flag sparse cells that would be unsafe to resample from directly in the
   joint-empirical-sampling plan
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data

df = get_business_data()

VALID_FREQ = [1, 2, 3, 4, 5, 6]
FREQ_LABELS = {
    1: 'once only',
    2: '>once<monthly',
    3: '~monthly',
    4: '~weekly',
    5: '~daily',
    6: 'several/day',
}
SIZE_LABELS = {1: 'Micro (1-9)', 2: 'Small (10-49)', 3: 'Medium (50-249)', 4: 'Large (250+)'}

attacked = df[df['freq'].isin(VALID_FREQ)].copy()
sizes = sorted(attacked['sizeb'].dropna().unique())

# ---------------------------------------------------------------------------
# 1. Raw counts cross-tab
# ---------------------------------------------------------------------------

print("=" * 78)
print("RAW COUNTS: size band x freq band (attacked firms only)")
print("=" * 78)

counts = pd.crosstab(attacked['sizeb'], attacked['freq'])
counts = counts.reindex(index=sizes, columns=VALID_FREQ, fill_value=0)

header = f"{'Size':<20}" + "".join(f"{FREQ_LABELS[f]:>15}" for f in VALID_FREQ) + f"{'Total':>10}"
print(header)
for sz in sizes:
    row = counts.loc[sz]
    print(f"{SIZE_LABELS[sz]:<20}" + "".join(f"{row[f]:>15}" for f in VALID_FREQ) + f"{row.sum():>10}")

# Flag sparse cells
print("\n  Cells with n < 10 (unsafe to resample from directly):")
sparse_any = False
for sz in sizes:
    for f in VALID_FREQ:
        n = counts.loc[sz, f]
        if n < 10:
            sparse_any = True
            print(f"    {SIZE_LABELS[sz]:<20} {FREQ_LABELS[f]:<16} n={n}")
if not sparse_any:
    print("    (none)")

# ---------------------------------------------------------------------------
# 2. Row proportions — unweighted
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("ROW PROPORTIONS (unweighted): P(freq | size, attacked)")
print("=" * 78)

props_uw = counts.div(counts.sum(axis=1), axis=0)
print(header)
for sz in sizes:
    row = props_uw.loc[sz]
    print(f"{SIZE_LABELS[sz]:<20}" + "".join(f"{row[f]:>15.3f}" for f in VALID_FREQ) + f"{row.sum():>10.3f}")

# ---------------------------------------------------------------------------
# 3. Row proportions — weighted
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("ROW PROPORTIONS (weighted): P(freq | size, attacked)")
print("=" * 78)

w_counts = pd.DataFrame(index=sizes, columns=VALID_FREQ, dtype=float)
for sz in sizes:
    sub = attacked[attacked['sizeb'] == sz]
    total_w = sub['weight'].sum()
    for f in VALID_FREQ:
        w_counts.loc[sz, f] = sub.loc[sub['freq'] == f, 'weight'].sum() / total_w

print(header)
for sz in sizes:
    row = w_counts.loc[sz]
    print(f"{SIZE_LABELS[sz]:<20}" + "".join(f"{row[f]:>15.3f}" for f in VALID_FREQ) + f"{row.sum():>10.3f}")

# ---------------------------------------------------------------------------
# 4. Eyeball summary stats: mean freq band, % once-only, % weekly-or-more
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("SUMMARY STATS BY SIZE BAND")
print("=" * 78)
print(f"{'Size':<20}{'n':>8}{'mean freq':>12}{'% once-only':>14}{'% weekly+':>12}")
for sz in sizes:
    sub = attacked[attacked['sizeb'] == sz]
    n = len(sub)
    mean_freq = sub['freq'].mean()
    pct_once = (sub['freq'] == 1).mean()
    pct_weekly_plus = sub['freq'].isin([4, 5, 6]).mean()
    print(f"{SIZE_LABELS[sz]:<20}{n:>8}{mean_freq:>12.2f}{pct_once:>14.1%}{pct_weekly_plus:>12.1%}")

# ---------------------------------------------------------------------------
# 5. Chi-square test of independence: does freq distribution depend on size?
# ---------------------------------------------------------------------------

from scipy.stats import chi2_contingency

print("\n" + "=" * 78)
print("CHI-SQUARE TEST: independence of size band and freq band")
print("=" * 78)

chi2, p, dof, expected = chi2_contingency(counts.values)
print(f"  chi2 = {chi2:.2f}, dof = {dof}, p = {p:.4f}")
print(f"  Interpretation: {'REJECT independence (freq shifts with size)' if p < 0.05 else 'FAIL to reject independence'}")

# ---------------------------------------------------------------------------
# 6. Follow-up: pooled test, Large vs. (Micro+Small+Medium)
#    Row proportions looked flat across Micro/Small/Medium but distinct for
#    Large. The full 4x6 table dilutes that signal across 3 similar rows;
#    a 2x6 table isolating Large should be a cleaner test of the same pattern.
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("FOLLOW-UP CHI-SQUARE: Large vs. (Micro+Small+Medium pooled)")
print("=" * 78)

pooled_small = counts.loc[[1, 2, 3]].sum(axis=0)
pooled_counts = pd.DataFrame([pooled_small, counts.loc[4]], index=['Micro+Small+Medium', 'Large'])
print(pooled_counts.rename(columns=FREQ_LABELS))

chi2_p, p_p, dof_p, expected_p = chi2_contingency(pooled_counts.values)
print(f"\n  chi2 = {chi2_p:.2f}, dof = {dof_p}, p = {p_p:.4f}")
print(f"  Interpretation: {'REJECT independence (Large differs from the rest)' if p_p < 0.05 else 'FAIL to reject independence'}")

print("\nDone.")
