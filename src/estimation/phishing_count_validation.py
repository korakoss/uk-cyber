"""
Validate the SVD-inferred targeted/mass (F_T/F_M) phishing mixture (see
mixture_model.py) against real per-attack count data, and check whether
freq band 2 (">once<monthly", implied k=2-11) hides enough heterogeneity
to justify more continuous frequency resolution.

Real count variables (not previously loaded via proc.py — pulled directly
from data/proc/data.csv):
- phishcon_bands: "Number of specifically targeted phishing attacks"
  (bands: 1=None, 2=1, 3=2-3, 4=4-5, 5=6-10, 6=11-20, 7=21-50, 8=51-100, 9=100+)
  A firm with phishcon_bands==None (1) had zero targeted attacks — a direct,
  real-data proxy for "pure mass-market" (F_M), no SVD inference required.
- phisheng_bands: "Number of times someone engaged with a phishing attack"
  same band structure — a proxy for successful/impactful attacks.

Neither is a literal total-phishing-attack count (no such variable exists
in the survey), so this validates the targeted/mass split, not the full
per-attack cost model.
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')

df = pd.read_csv('data/proc/data.csv')
biz = df[df['typex'] == 1].copy()

VALID_FREQ = [1, 2, 3, 4, 5, 6]
FREQ_LABELS = {
    1: 'once only', 2: '>once<monthly', 3: '~monthly',
    4: '~weekly', 5: '~daily', 6: 'several/day',
}
COUNT_BAND_LABELS = {
    1: 'None', 2: '1', 3: '2-3', 4: '4-5', 5: '6-10',
    6: '11-20', 7: '21-50', 8: '51-100', 9: '100+',
}
VALID_COUNT_BANDS = list(COUNT_BAND_LABELS.keys())

phish = biz[(biz['type6'] == 1) & biz['freq'].isin(VALID_FREQ)].copy()
print(f"Phishing firms with valid freq: n={len(phish)}")

# ---------------------------------------------------------------------------
# 1. freq x phishcon_bands (targeted count) cross-tab
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("freq x phishcon_bands (# TARGETED phishing attacks): raw counts")
print("=" * 78)

con = phish[phish['phishcon_bands'].isin(VALID_COUNT_BANDS)].copy()
print(f"  n with valid phishcon_bands: {len(con)} (of {len(phish)} phishing firms)")

counts_con = pd.crosstab(con['freq'], con['phishcon_bands'])
counts_con = counts_con.reindex(index=VALID_FREQ, columns=VALID_COUNT_BANDS, fill_value=0)
header = f"{'freq band':<18}" + "".join(f"{COUNT_BAND_LABELS[b]:>8}" for b in VALID_COUNT_BANDS) + f"{'n':>8}"
print(header)
for f in VALID_FREQ:
    row = counts_con.loc[f]
    print(f"{FREQ_LABELS[f]:<18}" + "".join(f"{row[b]:>8}" for b in VALID_COUNT_BANDS) + f"{row.sum():>8}")

# ---------------------------------------------------------------------------
# 2. Direct "pure mass" fraction (phishcon_bands == None) per freq band,
#    vs. the SVD-inferred targeted mixing weight pi_f from mixture_model.py
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("P(at least one TARGETED attack | freq) — direct measure, vs. SVD pi_f")
print("=" * 78)

# From NOTES.md / mixture_model.py (phishing, all sizes pooled):
# once-only 12.9%, >once<monthly 5.7%, monthly/weekly/daily 0%, several/day 22.3%
svd_pi_f = {1: 0.129, 2: 0.057, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.223}

print(f"{'freq band':<18}{'n':>6}{'P(has targeted)':>18}{'SVD pi_f (T)':>15}")
for f in VALID_FREQ:
    sub = con[con['freq'] == f]
    n = len(sub)
    if n == 0:
        continue
    p_has_targeted = (sub['phishcon_bands'] != 1).mean()
    print(f"{FREQ_LABELS[f]:<18}{n:>6}{p_has_targeted:>18.3f}{svd_pi_f[f]:>15.3f}")

# ---------------------------------------------------------------------------
# 3. Same for phisheng_bands (# times someone engaged)
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("freq x phisheng_bands (# times someone ENGAGED): raw counts")
print("=" * 78)

eng = phish[phish['phisheng_bands'].isin(VALID_COUNT_BANDS)].copy()
print(f"  n with valid phisheng_bands: {len(eng)} (of {len(phish)} phishing firms)")

counts_eng = pd.crosstab(eng['freq'], eng['phisheng_bands'])
counts_eng = counts_eng.reindex(index=VALID_FREQ, columns=VALID_COUNT_BANDS, fill_value=0)
print(header)
for f in VALID_FREQ:
    row = counts_eng.loc[f]
    print(f"{FREQ_LABELS[f]:<18}" + "".join(f"{row[b]:>8}" for b in VALID_COUNT_BANDS) + f"{row.sum():>8}")

# ---------------------------------------------------------------------------
# 4. Zoom on freq=2 (">once<monthly", implied k=2-11): how much spread is
#    hidden inside this one band?
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("ZOOM: freq band 2 ('>once<monthly', implied k=2-11) — hidden spread")
print("=" * 78)

band2_con = con[con['freq'] == 2]
band2_eng = eng[eng['freq'] == 2]

print(f"\n  phishcon_bands (targeted count) within freq=2, n={len(band2_con)}:")
dist = band2_con['phishcon_bands'].value_counts(normalize=True).reindex(VALID_COUNT_BANDS, fill_value=0)
for b in VALID_COUNT_BANDS:
    print(f"    {COUNT_BAND_LABELS[b]:<8} {dist[b]:.3f}")

print(f"\n  phisheng_bands (engaged count) within freq=2, n={len(band2_eng)}:")
dist2 = band2_eng['phisheng_bands'].value_counts(normalize=True).reindex(VALID_COUNT_BANDS, fill_value=0)
for b in VALID_COUNT_BANDS:
    print(f"    {COUNT_BAND_LABELS[b]:<8} {dist2[b]:.3f}")

# ---------------------------------------------------------------------------
# 5. Reconciliation: does the literal "targeted attack" measure (phishcon_bands)
#    actually predict cost outcome (damage_bands) the way the SVD's F_T
#    component implies it should? If P(≥1 targeted) doesn't track SVD pi_f,
#    maybe it still tracks cost directly — that would justify using it as a
#    real covariate even if it isn't literally "type T membership".
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("RECONCILIATION: damage_bands by phishcon_bands / phisheng_bands level")
print("=" * 78)

SPECIAL_CODE_THRESHOLD = 100
valid_cost = phish['damage_bands'] < SPECIAL_CODE_THRESHOLD

print("\n  damage_bands by phishcon_bands (targeted count) level:")
print(f"  {'level':<10}{'n':>6}{'mean band':>12}{'% no-cost':>12}{'% band>=8 (20k+)':>18}")
sub = phish[valid_cost & phish['phishcon_bands'].isin(VALID_COUNT_BANDS)]
for b in VALID_COUNT_BANDS:
    g = sub[sub['phishcon_bands'] == b]
    if len(g) == 0:
        continue
    n = len(g)
    mean_band = g['damage_bands'].mean()
    pct_nocost = (g['damage_bands'] == 1).mean()
    pct_hi = (g['damage_bands'] >= 8).mean()
    print(f"  {COUNT_BAND_LABELS[b]:<10}{n:>6}{mean_band:>12.2f}{pct_nocost:>12.1%}{pct_hi:>18.1%}")

rho_con = sub['phishcon_bands'].corr(sub['damage_bands'], method='spearman')
print(f"\n  Spearman corr(phishcon_bands, damage_bands) = {rho_con:.3f}  (n={len(sub)})")

print("\n  damage_bands by phisheng_bands (engaged count) level:")
print(f"  {'level':<10}{'n':>6}{'mean band':>12}{'% no-cost':>12}{'% band>=8 (20k+)':>18}")
sub2 = phish[valid_cost & phish['phisheng_bands'].isin(VALID_COUNT_BANDS)]
for b in VALID_COUNT_BANDS:
    g = sub2[sub2['phisheng_bands'] == b]
    if len(g) == 0:
        continue
    n = len(g)
    mean_band = g['damage_bands'].mean()
    pct_nocost = (g['damage_bands'] == 1).mean()
    pct_hi = (g['damage_bands'] >= 8).mean()
    print(f"  {COUNT_BAND_LABELS[b]:<10}{n:>6}{mean_band:>12.2f}{pct_nocost:>12.1%}{pct_hi:>18.1%}")

rho_eng = sub2['phisheng_bands'].corr(sub2['damage_bands'], method='spearman')
print(f"\n  Spearman corr(phisheng_bands, damage_bands) = {rho_eng:.3f}  (n={len(sub2)})")

# ---------------------------------------------------------------------------
# 6. Direct comparison to SVD-extracted F_T / F_M cost profiles
#    (from mixture_model.py / NOTES.md): F_T mean band 4.0, 0% no-cost;
#    F_M mean band 2.0, 58.5% no-cost.
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("COMPARISON: 'has >=1 targeted attack' cost profile vs. SVD F_T/F_M")
print("=" * 78)

has_targeted = sub[sub['phishcon_bands'] != 1]
no_targeted = sub[sub['phishcon_bands'] == 1]

print(f"\n  {'group':<28}{'n':>6}{'mean band':>12}{'% no-cost':>12}")
print(f"  {'has >=1 targeted attack':<28}{len(has_targeted):>6}{has_targeted['damage_bands'].mean():>12.2f}{(has_targeted['damage_bands']==1).mean():>12.1%}")
print(f"  {'no targeted attacks':<28}{len(no_targeted):>6}{no_targeted['damage_bands'].mean():>12.2f}{(no_targeted['damage_bands']==1).mean():>12.1%}")
print(f"  {'SVD F_T (from mixture_model)':<28}{'--':>6}{4.00:>12.2f}{0.0:>12.1%}")
print(f"  {'SVD F_M (from mixture_model)':<28}{'--':>6}{2.00:>12.2f}{0.585:>12.1%}")

print("\nDone.")
