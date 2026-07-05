"""
Avenue B: per-attack-type breakdown.

Hypothesis: the negative freq-cost correlation (monotonicity failure) is driven
by attack type composition — high-freq firms are dominated by cheap attacks
(phishing), low-freq firms by expensive ones (ransomware). If so, the i.i.d.
model may be better-behaved within a single attack type.

Analyses:
1. Attack type prevalence by freq group — tests the composition hypothesis directly.
2. Mean damage_band by freq group, restricted to firms experiencing each attack type —
   checks whether the monotonicity failure persists within type.
3. Single-type firms: firms that reported exactly one attack type — cleanest
   subgroup for within-type freq-cost analysis.
"""

import sys
import numpy as np
import pandas as pd
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

FREQ_LABELS = {
    1: 'Once only', 2: '>once <monthly', 3: '~monthly',
    4: '~weekly', 5: '~daily', 6: 'Several/day'
}
TYPE_NAMES = {
    'type1': 'Ransomware',
    'type2': 'Other malware',
    'type3': 'Denial of service',
    'type4': 'Hacking (bank accounts)',
    'type5': 'Impersonation',
    'type6': 'Phishing',
    'type7': 'Unauth. access (staff)',
    'type8': 'Unauth. access (outsiders)',
    'type9': 'Other',
    'type13': 'Unauth. access (students)',
    'type15': 'Video conf. eavesdropping',
    'type16': 'Website/social media takeover',
}

type_cols = [c for c in TYPE_NAMES if c in attacked.columns]

# Recode type variables: treat NaN and -1 as 0 (not experienced)
for c in type_cols:
    attacked[c] = attacked[c].fillna(0)
    attacked.loc[attacked[c] < 0, c] = 0
    attacked[c] = attacked[c].astype(int)

freq_groups = sorted(attacked['freq'].unique())

# ---------------------------------------------------------------------------
# 1. Attack type prevalence by freq group
# ---------------------------------------------------------------------------

print("=" * 80)
print("ANALYSIS 1: ATTACK TYPE PREVALENCE BY FREQ GROUP")
print("Proportion of firms in each freq group experiencing each attack type.")
print("Tests composition hypothesis: are high-freq firms dominated by cheap types?")
print("=" * 80)

prev_table = {}
for fg, flabel in FREQ_LABELS.items():
    sub = attacked[attacked['freq'] == fg]
    if len(sub) == 0:
        continue
    prev_table[flabel] = {TYPE_NAMES[c]: f"{100*sub[c].mean():.1f}%" for c in type_cols}

prev_df = pd.DataFrame(prev_table).T
print(prev_df.to_string())

# Also show group sizes
print("\nGroup sizes (n):")
print(attacked['freq'].map(FREQ_LABELS).value_counts().sort_index())

# ---------------------------------------------------------------------------
# 2. Mean damage_band by freq group, within each attack type
# ---------------------------------------------------------------------------

print()
print("=" * 80)
print("ANALYSIS 2: MEAN DAMAGE_BAND BY FREQ GROUP, WITHIN ATTACK TYPE")
print("For firms experiencing each type, does mean cost still decrease with freq?")
print("(Monotonicity failure = YES means pattern persists within type)")
print("=" * 80)

for col in type_cols:
    type_name = TYPE_NAMES[col]
    subset = attacked[attacked[col] == 1]
    if len(subset) < 20:
        print(f"\n{type_name}: n={len(subset)}, too small — skipping")
        continue

    print(f"\n--- {type_name} (n={len(subset)}) ---")
    print(f"  {'freq':<18} {'n':>5} {'mean_damage_band':>17} {'monotone?':>12}")

    prev_mean = -np.inf
    all_mono = True
    for fg in freq_groups:
        sub = subset[subset['freq'] == fg]
        if len(sub) < 3:
            continue
        mean_db = sub['damage_bands'].mean()
        mono = "YES" if mean_db >= prev_mean else "NO *"
        if mean_db < prev_mean:
            all_mono = False
        print(f"  {FREQ_LABELS.get(fg, str(fg)):<18} {len(sub):>5} {mean_db:>17.3f} {mono:>12}")
        prev_mean = mean_db
    print(f"  Overall monotone: {'YES' if all_mono else 'NO — violation(s) above'}")

# ---------------------------------------------------------------------------
# 3. Single-type firms
# ---------------------------------------------------------------------------

print()
print("=" * 80)
print("ANALYSIS 3: FIRMS REPORTING EXACTLY ONE ATTACK TYPE")
print("Cleanest subgroup: damage_bands is unambiguously attributable to one type.")
print("=" * 80)

attacked['n_types'] = attacked[type_cols].sum(axis=1)
single_type = attacked[attacked['n_types'] == 1].copy()
print(f"\nFirms with exactly one attack type: {len(single_type)} / {len(attacked)} ({100*len(single_type)/len(attacked):.1f}%)")

print("\nBreakdown by type (single-type firms):")
for col in type_cols:
    n = (single_type[col] == 1).sum()
    if n > 0:
        print(f"  {TYPE_NAMES[col]}: {n}")

print()
for col in type_cols:
    subset = single_type[single_type[col] == 1]
    if len(subset) < 10:
        continue
    type_name = TYPE_NAMES[col]
    print(f"--- {type_name} (single-type, n={len(subset)}) ---")
    print(f"  {'freq':<18} {'n':>5} {'mean_damage_band':>17} {'monotone?':>12}")
    prev_mean = -np.inf
    all_mono = True
    for fg in freq_groups:
        sub = subset[subset['freq'] == fg]
        if len(sub) < 3:
            continue
        mean_db = sub['damage_bands'].mean()
        mono = "YES" if mean_db >= prev_mean else "NO *"
        if mean_db < prev_mean:
            all_mono = False
        print(f"  {FREQ_LABELS.get(fg, str(fg)):<18} {len(sub):>5} {mean_db:>17.3f} {mono:>12}")
        prev_mean = mean_db
    print(f"  Overall monotone: {'YES' if all_mono else 'NO — violation(s) above'}")
    print()

print("Done.")

# ===========================================================================
# ANALYSIS 4: Full damage_bands distribution by freq group, within attack type
# Goal: determine whether freq=1 firms have a heavier upper tail, or whether
# the entire distribution shifts down for high-freq firms.
# Focus on largest types: phishing, impersonation, ransomware (for contrast).
# ===========================================================================

print()
print("=" * 80)
print("ANALYSIS 4: FULL damage_bands DISTRIBUTION BY FREQ GROUP, WITHIN TYPE")
print("Checks whether freq=1 upper tail is heavier (hypothesis A/C) or")
print("whether the whole distribution shifts down (uniform compression).")
print("=" * 80)

DAMAGE_BAND_LABELS = {
    1: 'No cost', 2: '<£100', 3: '£100-£500', 4: '£500-£1k',
    5: '£1k-£5k', 6: '£5k-£10k', 7: '£10k-£20k', 8: '£20k-£50k',
    9: '£50k-£100k', 10: '£100k-£500k'
}
all_bands = sorted(DAMAGE_BAND_LABELS.keys())

FOCUS_TYPES = {
    'type6': 'Phishing',
    'type5': 'Impersonation',
    'type1': 'Ransomware',
}

# Collapse freq into three groups for legibility
def freq_bucket(f):
    if f == 1: return 'Once only'
    if f in (2, 3): return '>once to monthly'
    return 'Weekly or more'

attacked['freq_bucket'] = attacked['freq'].apply(freq_bucket)
BUCKETS = ['Once only', '>once to monthly', 'Weekly or more']

for col, tname in FOCUS_TYPES.items():
    subset = attacked[attacked[col] == 1]
    print(f"\n--- {tname} (n={len(subset)}) ---")
    print(f"  Rows: damage_bands cost bands  |  Columns: freq group")
    print(f"  Values: % within each freq group")

    rows = []
    group_ns = {}
    for bucket in BUCKETS:
        grp = subset[subset['freq_bucket'] == bucket]
        group_ns[bucket] = len(grp)

    header = f"  {'Band':<14}" + "".join(f"  {b+' (n='+str(group_ns[b])+')':>22}" for b in BUCKETS)
    print(header)

    for b in all_bands:
        label = DAMAGE_BAND_LABELS[b]
        row = f"  {label:<14}"
        for bucket in BUCKETS:
            grp = subset[subset['freq_bucket'] == bucket]
            n = len(grp)
            pct = 100 * (grp['damage_bands'] == b).sum() / n if n > 0 else 0
            row += f"  {pct:>21.1f}%"
        rows.append(row)
    for r in rows:
        print(r)

    # Cumulative from top (P(damage >= b)) to highlight tail
    print(f"\n  Cumulative from top: P(damage_bands >= b)")
    header2 = f"  {'Band':<14}" + "".join(f"  {b:>22}" for b in BUCKETS)
    print(header2)
    for b in all_bands:
        label = DAMAGE_BAND_LABELS[b]
        row = f"  {label:<14}"
        for bucket in BUCKETS:
            grp = subset[subset['freq_bucket'] == bucket]
            n = len(grp)
            pct = 100 * (grp['damage_bands'] >= b).sum() / n if n > 0 else 0
            row += f"  {pct:>21.1f}%"
        print(row)

# ===========================================================================
# ANALYSIS 5: Type-specific total cost vs damage_bands for single-type firms
# Goal: empirical ratio of total annual cost to single-incident cost, by freq.
# Uses midpoints to convert both band systems to approximate £ values.
# Phishing has no type-specific cost variable — limited to other types.
# ===========================================================================

print()
print("=" * 80)
print("ANALYSIS 5: TYPE-SPECIFIC TOTAL COST vs damage_bands (single-type firms)")
print("For firms with exactly one attack type, compare total type cost to")
print("single-incident cost. Ratio > 1 means multi-attack costs accumulate.")
print("NOTE: phishing has no type-specific cost variable — excluded.")
print("=" * 80)

# Midpoints for damage_bands (£)
DAMAGE_MIDPOINTS = {
    1: 0, 2: 50, 3: 300, 4: 750, 5: 3000,
    6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000
}

# Midpoints for type-specific cost bands (£) — same structure across all types
TYPE_COST_MIDPOINTS = {
    1: 50, 2: 175, 3: 375, 4: 750, 5: 1500,
    6: 3500, 7: 7500, 8: 15000, 9: 35000, 10: 75000,
    11: 175000, 12: 500000
}

# Mapping: type indicator column → type-specific cost column
TYPE_COST_MAP = {
    'type1': 'ranscost_bands',
    'type2': 'viruscost_bands',
    'type3': 'doscost_bands',
    'type4': 'hackcost_bands',
    'type16': 'tkvrcost_bands',
}

single_type = attacked[attacked['n_types'] == 1].copy()

for type_col, cost_col in TYPE_COST_MAP.items():
    tname = TYPE_NAMES.get(type_col, type_col)
    subset = single_type[
        (single_type[type_col] == 1) &
        single_type[cost_col].notna() &
        (single_type[cost_col] < 100) &
        (single_type[cost_col] > 0) &
        single_type['damage_bands'].notna() &
        (single_type['damage_bands'] < SPECIAL_CODE_THRESHOLD)
    ].copy()

    if len(subset) < 3:
        print(f"\n{tname}: n={len(subset)} with both variables valid — too small, skipping")
        continue

    subset['damage_mid'] = subset['damage_bands'].map(DAMAGE_MIDPOINTS)
    subset['typecost_mid'] = subset[cost_col].map(TYPE_COST_MIDPOINTS)
    subset['ratio'] = subset.apply(
        lambda r: r['typecost_mid'] / r['damage_mid'] if r['damage_mid'] > 0 else np.nan, axis=1
    )

    print(f"\n--- {tname} (single-type, n={len(subset)}) ---")
    print(f"  cost_col: {cost_col}")
    print(f"  {'freq':<18} {'n':>5} {'mean damage_mid £':>18} {'mean typecost_mid £':>20} {'mean ratio':>12}")

    for fg in sorted(subset['freq'].unique()):
        grp = subset[subset['freq'] == fg]
        mean_dmg = grp['damage_mid'].mean()
        mean_tc = grp['typecost_mid'].mean()
        mean_ratio = grp['ratio'].dropna().mean()
        ratio_str = f"{mean_ratio:.2f}" if not np.isnan(mean_ratio) else "n/a"
        print(f"  {FREQ_LABELS.get(fg, str(fg)):<18} {len(grp):>5} {mean_dmg:>18,.0f} {mean_tc:>20,.0f} {ratio_str:>12}")

    print(f"\n  Overall: median ratio = {subset['ratio'].dropna().median():.2f}, "
          f"mean ratio = {subset['ratio'].dropna().mean():.2f} (n with ratio={subset['ratio'].notna().sum()})")
    print(f"  (ratio = type-specific total cost / single-incident cost midpoint)")
    print(f"  Note: ratio < 1 possible due to differing band structures and cost definitions")

print()
print("Done (analyses 4 & 5).")

# ===========================================================================
# ANALYSIS 6: disrupta — type of the MOST DISRUPTIVE attack, by freq group
#
# disrupta (Q64A) records which attack type was the single most disruptive,
# for all attacked firms (or the only attack for freq=1 firms). This is the
# cleanest test of the composition hypothesis: does the type of the most
# disruptive attack explain the negative freq-cost curve?
#
# Two sub-questions:
# 6a. Does disrupta composition vary by freq group? (high-freq → phishing?)
# 6b. Within each disrupta type, does the negative freq-cost pattern persist?
# ===========================================================================

print()
print("=" * 80)
print("ANALYSIS 6: disrupta — TYPE OF MOST DISRUPTIVE ATTACK")
print("=" * 80)

DISRUPTA_LABELS = {
    1: 'Ransomware',
    2: 'Other malware',
    3: 'Denial of service',
    4: 'Hacking (bank accts)',
    5: 'Impersonation',
    6: 'Phishing',
    7: 'Unauth access (staff)',
    8: 'Unauth access (students)',
    9: 'Unauth access (outsiders)',
    10: 'Video conf eavesdrop',
    11: 'Website/social/email takeover',
    12: 'Other',
}

# Reload attacked with disrupta now available
df2 = get_business_data()
INVALID_FREQ = {997, 999, 997.0, 999.0}
attacked2 = df2[
    df2['freq'].notna() &
    ~df2['freq'].isin(INVALID_FREQ) &
    (df2['freq'] > 0) &
    df2['damage_bands'].notna() &
    (df2['damage_bands'] < SPECIAL_CODE_THRESHOLD)
].copy()

# Filter disrupta: valid responses only (positive integers, not 997)
attacked2['disrupta_clean'] = attacked2['disrupta'].where(
    attacked2['disrupta'].notna() &
    (attacked2['disrupta'] > 0) &
    (attacked2['disrupta'] != 997)
)

n_with_disrupta = attacked2['disrupta_clean'].notna().sum()
print(f"\nAttacked firms with valid damage_bands: {len(attacked2)}")
print(f"Of which have valid disrupta:           {n_with_disrupta} ({100*n_with_disrupta/len(attacked2):.1f}%)")

# --- 6a: disrupta composition by freq group ---
print()
print("--- 6a: disrupta type composition by freq group ---")
print("(% of firms in each freq group where that type was most disruptive)")

has_dis = attacked2[attacked2['disrupta_clean'].notna()].copy()
has_dis['disrupta_clean'] = has_dis['disrupta_clean'].astype(int)

freq_groups_present = sorted(has_dis['freq'].unique())
disrupta_vals = sorted(has_dis['disrupta_clean'].unique())

# Header
header = f"  {'Type':<32}"
for fg in freq_groups_present:
    lbl = FREQ_LABELS.get(int(fg), str(fg))
    n_fg = (has_dis['freq'] == fg).sum()
    header += f"  {lbl[:14]+f'(n={n_fg})':>18}"
print(header)

for dv in disrupta_vals:
    lbl = DISRUPTA_LABELS.get(dv, str(dv))
    row = f"  {lbl:<32}"
    for fg in freq_groups_present:
        grp = has_dis[has_dis['freq'] == fg]
        n = len(grp)
        pct = 100 * (grp['disrupta_clean'] == dv).sum() / n if n > 0 else 0
        row += f"  {pct:>17.1f}%"
    print(row)

# --- 6b: within disrupta type, freq-cost pattern ---
print()
print("--- 6b: mean damage_band by freq group, within disrupta type ---")
print("Key question: does the negative freq-cost pattern persist once we")
print("condition on what the most disruptive attack actually was?")

FOCUS_DISRUPTA = [6, 5, 1, 2, 3, 11]  # phishing, impersonation, ransomware, malware, DoS, takeover

for dv in FOCUS_DISRUPTA:
    lbl = DISRUPTA_LABELS.get(dv, str(dv))
    subset = has_dis[has_dis['disrupta_clean'] == dv]
    if len(subset) < 10:
        print(f"\n{lbl}: n={len(subset)}, too small — skipping")
        continue

    print(f"\n--- Most disruptive = {lbl} (n={len(subset)}) ---")
    print(f"  {'freq':<18} {'n':>5} {'mean_damage_band':>17} {'monotone?':>12}")

    prev_mean = -np.inf
    all_mono = True
    for fg in freq_groups_present:
        sub = subset[subset['freq'] == fg]
        if len(sub) < 3:
            continue
        mean_db = sub['damage_bands'].mean()
        mono = "YES" if mean_db >= prev_mean else "NO *"
        if mean_db < prev_mean:
            all_mono = False
        print(f"  {FREQ_LABELS.get(int(fg), str(fg)):<18} {len(sub):>5} {mean_db:>17.3f} {mono:>12}")
        prev_mean = mean_db
    print(f"  Overall monotone: {'YES' if all_mono else 'NO — violation(s) above'}")

    # Also show full distribution for phishing and impersonation
    if dv in (6, 5):
        print(f"\n  Full damage_bands distribution (% within freq group):")
        freq_buckets_dis = {
            'Once only': subset[subset['freq'] == 1],
            '>once-monthly': subset[subset['freq'].isin([2, 3])],
            'Weekly+': subset[subset['freq'].isin([4, 5, 6])],
        }
        header2 = f"  {'Band':<14}"
        for bk, bsub in freq_buckets_dis.items():
            header2 += f"  {bk+f' (n={len(bsub)})':>20}"
        print(header2)
        for b in all_bands:
            blabel = DAMAGE_BAND_LABELS.get(b, str(b))
            row2 = f"  {blabel:<14}"
            for bk, bsub in freq_buckets_dis.items():
                n = len(bsub)
                pct = 100 * (bsub['damage_bands'] == b).sum() / n if n > 0 else 0
                row2 += f"  {pct:>19.1f}%"
            print(row2)

print()
print("Done (analysis 6).")
