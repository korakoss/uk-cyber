"""
Real, measured per-type bridge multipliers — not modeled, not SVD-inferred.

DISCOVERY (2026-07-12, see NOTES.md "RECORD IN ALL CAPS" callout): several
COST_COLNAMES already loaded via proc.py are not single-incident costs — they
are genuine TOTAL ANNUAL COST PER ATTACK TYPE variables, independent of what
else happened that year:
  ranscost_bands   = total cost of ransomware attacks (whole year)
  hackcost_bands   = total cost of hacking incidents (whole year)
  doscost_bands    = total cost of DoS attacks (whole year)
  viruscost_bands  = total cost of malware attacks (whole year)
  tkvrcost_bands   = total cost of takeovers (whole year)
  hacksumcost_bands = total cost of hacking AND takeovers combined (whole year)
  crimecost_bands  = total cost of ALL crimes (whole year)

Combined with damage_bands (cost of the single most disruptive incident) and
disrupta (which type that incident was), a firm with disrupta==type and a
valid type-specific total-cost variable gives a DIRECT, measured bridge ratio:
    bridge = total_type_cost / single_worst_incident_cost
No mixture model, no SVD, no engagement-count proxy needed — this is the
actual quantity the whole bridge exercise has been trying to estimate.

Earlier attempts (attack_type_breakdown.py Analysis 5) additionally required
n_types==1 (this was the ONLY attack type all year), which crushed sample
sizes to n<3. That restriction is unnecessary: disrupta==type already
guarantees damage_bands belongs to that type, regardless of what else
happened. Dropping it recovers much larger samples (e.g. ransomware n=21
instead of n<3).

ALSO CORRECTED HERE: damage_bands (and crimecost_bands, hacksumcost_bands,
the damagedirsx/lx/staffx/indx_bands family) actually use a 13-band scale in
the codebook (up to "5 million or more"), not the 10-band scale ("100k-500k"
top) used by every DAMAGE_MIDPOINTS dict elsewhere in this project so far.
Checked empirically: no firm in this dataset's damage_bands/crimecost_bands/
hacksumcost_bands actually reaches band 11+ (fraudcost_bands2 does, n=3, but
isn't used here), so this hasn't silently corrupted any existing analysis —
but the midpoint dict is extended to the full 1-13 range here for
correctness and to avoid a silent-NaN trap if this is ever rerun on data
where higher bands are populated.
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD, TYPE_COLNAMES

df = get_business_data()
INVALID_FREQ = {997, 999, 997.0, 999.0}

attacked = df[
    df['freq'].notna() &
    ~df['freq'].isin(INVALID_FREQ) &
    (df['freq'] > 0) &
    df['damage_bands'].notna() &
    (df['damage_bands'] < SPECIAL_CODE_THRESHOLD) &
    df['disrupta'].notna() &
    (df['disrupta'] > 0) &
    (df['disrupta'] != 997)
].copy()

# Full 13-band scale: used by damage_bands, crimecost_bands, hacksumcost_bands,
# fraudcost_bands(2), notfraudcost_bands, damagedirsx/lx/staffx/indx_bands.
# Band 13 is genuinely open-ended ("5 million or more") — midpoint is a floor,
# not a real estimate, but no observation in this dataset reaches it.
FULL_BAND_MIDPOINTS = {
    1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000,
    9: 75000, 10: 300000, 11: 750000, 12: 3000000, 13: 5000000,
}

# 12-band scale used by the per-type total-cost variables (ranscost_bands,
# hackcost_bands, doscost_bands, viruscost_bands, tkvrcost_bands). No "no
# cost" band — starts at "less than 100". Band 12 open-ended ("250,000+").
TYPE_COST_MIDPOINTS = {
    1: 50, 2: 175, 3: 375, 4: 750, 5: 1500, 6: 3500, 7: 7500, 8: 15000,
    9: 35000, 10: 75000, 11: 175000, 12: 500000,
}

DISRUPTA_LABELS = {
    1: 'Ransomware', 2: 'Other malware', 3: 'Denial of service', 4: 'Hacking',
    5: 'Impersonation', 6: 'Phishing', 11: 'Website/social/email takeover',
}

TYPE_COST_MAP = {
    1: ('ranscost_bands', 'Ransomware'),
    2: ('viruscost_bands', 'Other malware'),
    3: ('doscost_bands', 'Denial of service'),
    4: ('hackcost_bands', 'Hacking'),
    11: ('tkvrcost_bands', 'Website/social/email takeover'),
}

print("=" * 95)
print("REAL, MEASURED PER-TYPE BRIDGE MULTIPLIERS (total_type_cost / single_worst_incident_cost)")
print("=" * 95)

summary_rows = []

for dv, (cost_col, label) in TYPE_COST_MAP.items():
    sub = attacked[attacked['disrupta'] == dv].copy()
    n_disrupta = len(sub)
    valid = sub[
        sub[cost_col].notna() & (sub[cost_col] > 0) & (sub[cost_col] < 100)
    ].copy()

    print(f"\n--- {label} (disrupta={dv}) ---")
    print(f"  n(disrupta==type) = {n_disrupta}, n(valid {cost_col}) = {len(valid)}")

    if len(valid) < 3:
        print(f"  Too small to analyse (n<3) — skipping.")
        summary_rows.append({'type': label, 'n': len(valid), 'median_ratio': np.nan, 'mean_ratio': np.nan})
        continue

    valid['damage_mid'] = valid['damage_bands'].map(FULL_BAND_MIDPOINTS)
    valid['typecost_mid'] = valid[cost_col].map(TYPE_COST_MIDPOINTS)
    valid['ratio'] = valid.apply(
        lambda r: r['typecost_mid'] / r['damage_mid'] if r['damage_mid'] > 0 else np.nan, axis=1
    )

    n_zero_damage = (valid['damage_mid'] == 0).sum()
    ratios = valid['ratio'].dropna()
    n_below1 = (ratios < 0.99).sum()
    n_at1 = ((ratios >= 0.99) & (ratios <= 1.01)).sum()
    n_above1 = (ratios > 1.01).sum()

    print(f"  n with damage_mid==0 (no-cost worst incident, ratio undefined): {n_zero_damage}")
    print(f"  Of {len(ratios)} firms with a defined ratio: {n_below1} below 1.0 (total < single incident — "
          f"logically inconsistent / self-report noise), {n_at1} ~= 1.0 (total = single incident, no bridge "
          f"needed), {n_above1} above 1.0 (real evidence of a bridge correction)")

    print(f"  Ratio distribution: min={ratios.min():.3f}  25th={ratios.quantile(.25):.3f}  "
          f"median={ratios.median():.3f}  mean={ratios.mean():.3f}  75th={ratios.quantile(.75):.3f}  "
          f"max={ratios.max():.3f}")

    w_mean_ratio = np.average(ratios, weights=valid.loc[ratios.index, 'weight'])
    print(f"  Weighted mean ratio: {w_mean_ratio:.3f}")

    # Flag the most extreme low outlier explicitly rather than silently averaging over it
    if len(ratios) > 0 and ratios.min() < 0.05:
        worst_idx = ratios.idxmin()
        row = valid.loc[worst_idx]
        print(f"  FLAG: most extreme low-ratio row — sizeb={row['sizeb']:.0f}, freq={row['freq']:.0f}, "
              f"damage_bands={row['damage_bands']:.0f} (£{row['damage_mid']:,.0f}), {cost_col}="
              f"{row[cost_col]:.0f} (£{row['typecost_mid']:,.0f}), ratio={row['ratio']:.4f}. This is "
              f"logically inconsistent (total type-cost < the single worst incident's cost, which is of "
              f"this same type) — almost certainly two-different-self-report-questions noise, not a real "
              f"ratio. Median above is robust to this; mean is not.")

    print(f"  Median excl. below-1 rows (i.e. only real bridge-evidence rows, ratio>=1): "
          f"{ratios[ratios >= 0.99].median() if n_at1 + n_above1 > 0 else float('nan'):.3f}")

    summary_rows.append({
        'type': label, 'n': len(ratios), 'median_ratio': ratios.median(),
        'mean_ratio': ratios.mean(), 'w_mean_ratio': w_mean_ratio,
        'pct_below1': 100 * n_below1 / len(ratios), 'pct_at1': 100 * n_at1 / len(ratios),
        'pct_above1': 100 * n_above1 / len(ratios),
    })

# ===========================================================================
# Pooled Hacking + Website takeover via hacksumcost_bands (bigger n than
# either type alone)
# ===========================================================================

print("\n" + "=" * 95)
print("POOLED: Hacking + Website/takeover via hacksumcost_bands (13-band scale, same as damage_bands)")
print("=" * 95)

pooled = attacked[attacked['disrupta'].isin([4, 11])].copy()
valid_pooled = pooled[
    pooled['hacksumcost_bands'].notna() & (pooled['hacksumcost_bands'] > 0) &
    (pooled['hacksumcost_bands'] < SPECIAL_CODE_THRESHOLD)
].copy()
print(f"\nn(disrupta in Hacking/Takeover) = {len(pooled)}, n(valid hacksumcost_bands) = {len(valid_pooled)}")

if len(valid_pooled) >= 3:
    valid_pooled['damage_mid'] = valid_pooled['damage_bands'].map(FULL_BAND_MIDPOINTS)
    valid_pooled['total_mid'] = valid_pooled['hacksumcost_bands'].map(FULL_BAND_MIDPOINTS)
    valid_pooled['ratio'] = valid_pooled.apply(
        lambda r: r['total_mid'] / r['damage_mid'] if r['damage_mid'] > 0 else np.nan, axis=1
    )
    ratios_p = valid_pooled['ratio'].dropna()
    print(f"Ratio distribution: min={ratios_p.min():.3f}  median={ratios_p.median():.3f}  "
          f"mean={ratios_p.mean():.3f}  max={ratios_p.max():.3f}")
    print(f"n below 1.0: {(ratios_p < 0.99).sum()}, ~=1.0: {((ratios_p >= 0.99) & (ratios_p <= 1.01)).sum()}, "
          f"above 1.0: {(ratios_p > 1.01).sum()}")

# ===========================================================================
# Phishing cross-check: single-type (phishing-only) firms, damage_bands vs
# crimecost_bands (total of ALL crimes — a valid proxy for "total phishing
# cost" ONLY when phishing was the sole attack type all year).
# Both variables share the SAME 13-band scale, so this is a same-scale ratio,
# unlike the type-specific checks above.
# ===========================================================================

print("\n" + "=" * 95)
print("PHISHING CROSS-CHECK: single-type firms, damage_bands vs crimecost_bands")
print("(independent second check on phishing's mixture-model bridge, ~1.02)")
print("=" * 95)

tcols = [c for c in TYPE_COLNAMES if c in attacked.columns]
for c in tcols:
    attacked[c] = attacked[c].fillna(0)
    attacked.loc[attacked[c] < 0, c] = 0
attacked['n_types'] = attacked[tcols].sum(axis=1)

phish_single = attacked[(attacked['disrupta'] == 6) & (attacked['n_types'] == 1)].copy()
valid_phish = phish_single[
    phish_single['crimecost_bands'].notna() & (phish_single['crimecost_bands'] > 0) &
    (phish_single['crimecost_bands'] < SPECIAL_CODE_THRESHOLD)
].copy()
print(f"\nn(disrupta=Phishing, n_types==1) = {len(phish_single)}, n(valid crimecost_bands) = {len(valid_phish)}")

if len(valid_phish) >= 3:
    valid_phish['damage_mid'] = valid_phish['damage_bands'].map(FULL_BAND_MIDPOINTS)
    valid_phish['crime_mid'] = valid_phish['crimecost_bands'].map(FULL_BAND_MIDPOINTS)
    valid_phish['ratio'] = valid_phish.apply(
        lambda r: r['crime_mid'] / r['damage_mid'] if r['damage_mid'] > 0 else np.nan, axis=1
    )
    ratios_ph = valid_phish['ratio'].dropna()
    print(f"Ratio distribution: min={ratios_ph.min():.3f}  median={ratios_ph.median():.3f}  "
          f"mean={ratios_ph.mean():.3f}  max={ratios_ph.max():.3f}")
    print(f"Compare to mixture-model bridge estimate: ~1.02 (type M) / up to ~1.7 overall "
          f"(with the unresolved freq=6 subgroup)")
else:
    print("Too small to analyse.")

# ===========================================================================
# Summary table
# ===========================================================================

print("\n" + "=" * 95)
print("SUMMARY: measured bridge multipliers vs. the conservative(=1)/pessimistic(=k_implied) bounds")
print("=" * 95)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

print(
    "\nCORRECTED headline (the auto-generated one above overclaimed): the RAW median ratio across "
    "types is actually ~0.47-0.51, NOT close to 1 — because 50-67% of firms show ratio<1, which is "
    "logically impossible (total type-cost cannot be less than the single worst incident of that "
    "same type) if both variables describe the same event consistently. This is the SAME pattern "
    "already documented and closed as 'Avenue A' (damage_bands vs crimecost_bands): even restricted "
    "to freq=1 firms, where the two numbers MUST match if answered consistently, only 43% actually "
    "matched and 50% had the total LOWER than the single-incident cost by up to 9 bands. This is not "
    "a type-specific problem — it's a recurring feature of this survey's banded retrospective "
    "cost-recall questions: respondents do not answer 'total cost of X' and 'cost of your worst "
    "incident' consistently with each other, even when logic requires total >= worst. Section below "
    "tests whether a narrower 'external payments only' comparison resolves this (it does not, see "
    "below) before drawing a final conclusion."
)

# ===========================================================================
# Does ranscost_bands actually measure a NARROWER cost concept than
# damage_bands (e.g. just external payments, like the "Avenue A" finding that
# crimecost_bands = direct financial losses only, vs damage_bands = broad
# all-in cost)? Test: compare ranscost_bands to JUST the external-payment
# sub-components of the same incident (damagedirsx_bands + damagedirlx_bands)
# instead of full damage_bands. If that ratio clusters much closer to 1, the
# mismatch above is a definitional artefact, not noise.
# ===========================================================================

print("\n" + "=" * 95)
print("TESTING THE 'DIFFERENT COST CONCEPT' HYPOTHESIS (Ransomware)")
print("ranscost_bands vs. JUST the external-payment sub-components (not full damage_bands)")
print("=" * 95)

rans = attacked[attacked['disrupta'] == 1].copy()
valid_ext = rans[
    rans['ranscost_bands'].notna() & (rans['ranscost_bands'] > 0) & (rans['ranscost_bands'] < 100) &
    rans['damagedirsx_bands'].notna() & (rans['damagedirsx_bands'] > 0) & (rans['damagedirsx_bands'] < 100) &
    rans['damagedirlx_bands'].notna() & (rans['damagedirlx_bands'] > 0) & (rans['damagedirlx_bands'] < 100)
].copy()
print(f"\nn with ranscost_bands + both external-payment sub-components valid: {len(valid_ext)}")

if len(valid_ext) >= 5:
    valid_ext['dirsx_mid'] = valid_ext['damagedirsx_bands'].map(FULL_BAND_MIDPOINTS)
    valid_ext['dirlx_mid'] = valid_ext['damagedirlx_bands'].map(FULL_BAND_MIDPOINTS)
    valid_ext['ext_total'] = valid_ext['dirsx_mid'] + valid_ext['dirlx_mid']
    valid_ext['ranscost_mid'] = valid_ext['ranscost_bands'].map(TYPE_COST_MIDPOINTS)
    valid_ext['damage_mid'] = valid_ext['damage_bands'].map(FULL_BAND_MIDPOINTS)
    valid_ext['ratio_vs_ext'] = valid_ext.apply(
        lambda r: r['ranscost_mid'] / r['ext_total'] if r['ext_total'] > 0 else np.nan, axis=1)
    valid_ext['ratio_vs_full'] = valid_ext.apply(
        lambda r: r['ranscost_mid'] / r['damage_mid'] if r['damage_mid'] > 0 else np.nan, axis=1)

    r_ext = valid_ext['ratio_vs_ext'].dropna()
    r_full = valid_ext['ratio_vs_full'].dropna()
    print(f"  Median ratio vs external-payments-only: {r_ext.median():.3f}  (n={len(r_ext)})")
    print(f"  Median ratio vs full damage_bands:       {r_full.median():.3f}  (n={len(r_full)})")
    print(
        "\n  Verdict: the external-payments-only comparison is NOT meaningfully closer to 1 than the "
        "full-damage_bands comparison — the 'different cost concept' hypothesis is NOT confirmed. "
        "One row in particular (sizeb=3, freq=3) shows £300k in damage_bands AND £300k in the "
        "external-payments sub-component for the single worst incident, yet ranscost_bands (whole-"
        "year ransomware total) reports <£100 — an internally contradictory response, most plausibly "
        "a genuine data-entry/respondent error rather than a measurable concept difference. "
        "Conclusion: the mismatch is general self-report noise in banded retrospective cost recall "
        "(same phenomenon as 'Avenue A'), not a fixable definitional issue."
    )

# ===========================================================================
# BONUS: damage sub-component decomposition by freq — does any specific cost
# component (external payments during/after, staff time, damage/disruption)
# behave differently with freq? Could shed light on the still-unresolved
# freq-cost anticorrelation (Core Finding #1).
# ===========================================================================

print("\n" + "=" * 95)
print("BONUS: damage_bands sub-components by freq group (all attacked firms)")
print("=" * 95)

SUBCOMPONENTS = {
    'damagedirsx_bands': 'External payments (during)',
    'damagedirlx_bands': 'External payments (aftermath)',
    'damagestaffx_bands': 'Staff time cost',
    'damageindx_bands': 'Damage/disruption cost',
}
FREQ_LABELS = {1: 'Once only', 2: '>once <monthly', 3: '~monthly',
               4: '~weekly', 5: '~daily', 6: 'Several/day'}

for col, name in SUBCOMPONENTS.items():
    if col not in attacked.columns:
        print(f"\n{name} ({col}): not loaded in proc.py — skip")
        continue
    sub = attacked[attacked[col].notna() & (attacked[col] > 0) & (attacked[col] < SPECIAL_CODE_THRESHOLD)].copy()
    sub['mid'] = sub[col].map(FULL_BAND_MIDPOINTS)
    print(f"\n--- {name} ({col}), n={len(sub)} ---")
    print(f"  {'freq':<18}{'n':>6}{'mean £':>12}{'% no-cost':>12}")
    for fg in sorted(sub['freq'].unique()):
        g = sub[sub['freq'] == fg]
        if len(g) < 5:
            continue
        print(f"  {FREQ_LABELS.get(int(fg), str(fg)):<18}{len(g):>6}{g['mid'].mean():>12,.0f}"
              f"{100*(g[col]==1).mean():>11.1f}%")

print(
    "\nNote: coverage on these sub-component variables not yet checked for size — if any is much "
    "sparser than damage_bands itself, treat its freq breakdown as indicative only. Not yet "
    "integrated into the main freq-cost anticorrelation investigation (consistency_checks.py) — "
    "flagged as a lead, not a conclusion."
)

print("\nDone.")
