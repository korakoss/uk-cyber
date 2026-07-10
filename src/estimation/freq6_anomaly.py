"""
Investigate the freq=6 ("several times a day") targeted-attack anomaly in the
phishing mixture model (mixture_model.py / mixture_bridge.py).

Background: the SVD-based mixture model assigns 34.6% of phishing's "targeted"
(type T) mass to the freq=6 group (n=47), an outlier relative to the otherwise
declining once-only -> weekly trend. This drives most of the uncertainty in
the phishing bridge multiplier (overall ~1.02 if this is noise, ~1.7 if real).

This script checks whether the anomaly is visible in the raw data (not just
the SVD reconstruction), whether it's driven by a few outlier observations or
weight artefacts, and whether it's corroborated by real engagement/targeting
count data (phishcon_bands, phisheng_bands).
"""

import sys
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

df = get_business_data()
INVALID_FREQ = {997, 999, 997.0, 999.0}
INVALID_COUNT = {-9, -1, 997, 999, -9.0, -1.0, 997.0, 999.0}

attacked = df[
    df['freq'].notna() & ~df['freq'].isin(INVALID_FREQ) & (df['freq'] > 0) &
    df['damage_bands'].notna() & (df['damage_bands'] < SPECIAL_CODE_THRESHOLD) &
    df['disrupta'].notna() & (df['disrupta'] > 0) & (df['disrupta'] != 997)
].copy()
phishing = attacked[attacked['disrupta'] == 6].copy()

FREQ_LABELS = {1: 'once only', 2: '>once<monthly', 3: '~monthly',
               4: '~weekly', 5: '~daily', 6: 'several/day'}
COUNT_LABELS = {1: 'None', 2: '1', 3: '2-3', 4: '4-5', 5: '6-10',
                6: '11-20', 7: '21-50', 8: '51-100', 9: '100+'}

# ---------------------------------------------------------------------------
# 1. Raw (non-SVD) descriptive stats per freq group: is the anomaly visible
#    without any mixture-model machinery?
# ---------------------------------------------------------------------------

print("=" * 78)
print("1. RAW damage_bands stats by freq group (phishing, disrupta=6)")
print("=" * 78)
print(f"{'freq':<16}{'n':>5}{'mean band':>12}{'% no-cost':>12}{'% band>=8':>12}")
for f in [1, 2, 3, 4, 5, 6]:
    g = phishing[phishing['freq'] == f]
    n = len(g)
    mean_band = g['damage_bands'].mean()
    pct_nocost = (g['damage_bands'] == 1).mean()
    pct_hi = (g['damage_bands'] >= 8).mean()
    print(f"{FREQ_LABELS[f]:<16}{n:>5}{mean_band:>12.2f}{pct_nocost:>12.1%}{pct_hi:>12.1%}")

print("\n  damage_bands raw counts, freq=6 only:")
f6 = phishing[phishing['freq'] == 6]
print(f6['damage_bands'].value_counts().sort_index())

# ---------------------------------------------------------------------------
# 2. Are the high-cost outliers weight artefacts?
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("2. Outlier rows (band>=8) within freq=6, and weight comparison")
print("=" * 78)

hi = f6[f6['damage_bands'] >= 8]
print(hi[['sizeb', 'weight', 'freq', 'damage_bands']].to_string(index=False))

print("\n  Mean/max survey weight by freq group (phishing):")
for f in [1, 2, 3, 4, 5, 6]:
    g = phishing[phishing['freq'] == f]
    print(f"    freq={f} ({FREQ_LABELS[f]:<14}): mean={g['weight'].mean():.3f}  max={g['weight'].max():.3f}")

print("\n  Size band composition of freq=6 group:")
print(f6['sizeb'].value_counts().sort_index())

# ---------------------------------------------------------------------------
# 3. Does real engagement data corroborate the anomaly?
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("3. Real engagement rate (phisheng_bands) by freq group")
print("=" * 78)

for f in [1, 2, 3, 4, 5, 6]:
    g = phishing[(phishing['freq'] == f) & phishing['phisheng_bands'].isin(COUNT_LABELS.keys())]
    dist = g['phisheng_bands'].value_counts(normalize=True).reindex(COUNT_LABELS.keys(), fill_value=0)
    pct_engaged = 1 - dist[1]
    nonzero = ' '.join(f"{COUNT_LABELS[k]}:{dist[k]:.2f}" for k in COUNT_LABELS if dist[k] > 0)
    print(f"  {FREQ_LABELS[f]:<16} n={len(g):>4}  P(>=1 engaged)={pct_engaged:.1%}  dist={nonzero}")

print("\n  Cost outcome by engagement status, WITHIN freq=6 only:")
f6v = f6[f6['phisheng_bands'].isin(COUNT_LABELS.keys())]
for engaged, label in [(False, 'no engagement'), (True, '>=1 engagement')]:
    mask = (f6v['phisheng_bands'] != 1) if engaged else (f6v['phisheng_bands'] == 1)
    g = f6v[mask]
    if len(g) == 0:
        continue
    print(f"    {label:<18} n={len(g):>3}  mean band={g['damage_bands'].mean():.2f}"
          f"  %no-cost={(g['damage_bands']==1).mean():.1%}")

# ---------------------------------------------------------------------------
# 4. Follow-up: is "targeted" (phishcon_bands) elevated at freq=6 even though
#    engagement (phisheng_bands) isn't? These are different constructs
#    (targeted = attack *looked* personalised; engaged = someone acted on it).
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("4. Real TARGETED rate (phishcon_bands) by freq group")
print("=" * 78)

for f in [1, 2, 3, 4, 5, 6]:
    g = phishing[(phishing['freq'] == f) & phishing['phishcon_bands'].isin(COUNT_LABELS.keys())]
    pct_targeted = (g['phishcon_bands'] != 1).mean()
    print(f"  {FREQ_LABELS[f]:<16} n={len(g):>4}  P(>=1 targeted)={pct_targeted:.1%}")

print("\n  Cost outcome by targeted status, WITHIN freq=6 only:")
f6c = f6[f6['phishcon_bands'].isin(COUNT_LABELS.keys())]
for targeted, label in [(False, 'no targeted attack'), (True, '>=1 targeted attack')]:
    mask = (f6c['phishcon_bands'] != 1) if targeted else (f6c['phishcon_bands'] == 1)
    g = f6c[mask]
    if len(g) == 0:
        continue
    print(f"    {label:<22} n={len(g):>3}  mean band={g['damage_bands'].mean():.2f}"
          f"  %no-cost={(g['damage_bands']==1).mean():.1%}")

# ---------------------------------------------------------------------------
# 5. Multi-vector hypothesis: are freq=6 phishing firms actually under
#    chronic, multi-type attack (ransomware/hacking/DoS alongside phishing),
#    which would explain elevated cost through a mechanism that a
#    phishing-specific engagement/targeting variable can't capture?
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
print("5. Attack-type breadth by freq group (phishing firms, disrupta=6)")
print("=" * 78)

TYPE_NAMES = {
    'type1': 'Ransomware', 'type2': 'Other malware', 'type3': 'Denial of service',
    'type4': 'Hacking (bank accounts)', 'type5': 'Impersonation', 'type6': 'Phishing',
    'type7': 'Unauth. access (staff)', 'type8': 'Unauth. access (outsiders)',
    'type9': 'Other', 'type13': 'Unauth. access (students)',
    'type15': 'Video conf. eavesdropping', 'type16': 'Website/social media takeover',
}
type_cols = [c for c in TYPE_NAMES if c in phishing.columns]
types_bin = phishing[type_cols].apply(lambda s: (s == 1).astype(int))
phishing = phishing.assign(n_types=types_bin.sum(axis=1))

print(f"{'freq':<16}{'n':>5}{'mean # types':>14}{'% >1 type':>12}")
for f in [1, 2, 3, 4, 5, 6]:
    g = phishing[phishing['freq'] == f]
    print(f"{FREQ_LABELS[f]:<16}{len(g):>5}{g['n_types'].mean():>14.2f}{(g['n_types']>1).mean():>12.1%}")

print("\n  Which other attack types co-occur within freq=6 (n=47), vs. once-only (n=?):")
f6_types = phishing[phishing['freq'] == 6]
f1_types = phishing[phishing['freq'] == 1]
for tc, name in TYPE_NAMES.items():
    if tc == 'type6' or tc not in phishing.columns:
        continue
    p6 = (f6_types[tc] == 1).mean()
    p1 = (f1_types[tc] == 1).mean()
    print(f"    {name:<32} freq=6: {p6:>6.1%}   once-only: {p1:>6.1%}")

print("\n  Other attack types' cost variables, for the 3 high-cost (band>=8) freq=6 outliers:")
OTHER_COST_COLS = {'type1': 'ranscost_bands', 'type2': 'viruscost_bands',
                    'type3': 'doscost_bands', 'type4': 'hackcost_bands',
                    'type16': 'tkvrcost_bands'}
hi_cols = ['sizeb', 'damage_bands'] + type_cols + [c for c in OTHER_COST_COLS.values() if c in hi.columns]
print(hi[hi_cols].to_string(index=False))

print("\nDone.")
