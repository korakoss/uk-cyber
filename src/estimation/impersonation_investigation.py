"""
Impersonation deep-dive: chasing the leads from cost_share_by_type.py.

Impersonation (disrupta=5) turned out to be the single largest weighted £
contributor (33.8% of total £ mass) despite only 18.3%/24.5% count share —
larger even than its unweighted £ share (18.3%) would suggest. Unlike
ransomware/hacking/DoS (n too small to ever model), Impersonation has n=241,
comparable to phishing's per-size-band samples — so, unlike those types, a
mixture-model-style approach isn't obviously infeasible on sample-size grounds
alone. This script investigates two questions:

1. Is there an impersonation-specific analog to phishing's phishcon_bands/
   phisheng_bands (a real per-attack severity/count proxy) that could ground
   a validated mixture split the way engagement did for phishing? And does a
   per-size-band rank-2 retry (never previously run for Impersonation, only
   for Phishing) do any better than the original pooled result?
2. What's actually driving the large weighted/unweighted £ gap? (Direct
   outlier drill-down, mirroring the freq6_anomaly.py approach for phishing.)

A third, broader finding fell out of (2): a general relationship between
damage_bands severity and attack-type breadth (n_types flagged), checked
across ALL attacked firms, not just impersonation.
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

imp = attacked[attacked['disrupta'] == 5].copy()

SIZE_LABELS = {1: 'Micro (1-9)', 2: 'Small (10-49)', 3: 'Medium (50-249)', 4: 'Large (250+)'}
FREQ_LABELS = {1: 'Once only', 2: '>once <monthly', 3: '~monthly',
               4: '~weekly', 5: '~daily', 6: 'Several/day'}
DAMAGE_BAND_LABELS = {
    1: 'No cost', 2: '<£100', 3: '£100-£500', 4: '£500-£1k',
    5: '£1k-£5k', 6: '£5k-£10k', 7: '£10k-£20k', 8: '£20k-£50k',
    9: '£50k-£100k', 10: '£100k-£500k'
}
DAMAGE_MIDPOINTS = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000}
bands = sorted(DAMAGE_BAND_LABELS.keys())

print("=" * 90)
print(f"IMPERSONATION (disrupta=5): n={len(imp)}")
print("=" * 90)

# ===========================================================================
# 1a. Is impersonationhack/impersonationtkvr a usable severity proxy?
# (Q53B/Q53C: did the impersonation instances involve unauthorised access, or
# a website/account takeover? Ordinal 1=Yes all, 2=Yes some, 3=No)
# ===========================================================================

print("\n--- 1a. impersonationhack / impersonationtkvr coverage & variation ---")
for col in ['impersonationhack', 'impersonationtkvr']:
    print(f"\n{col} value counts (n={len(imp)}):")
    print(imp[col].value_counts(dropna=False).sort_index())

print(
    "\nVerdict: essentially no variation — 230/241 (95.4%) 'No' on impersonationhack, "
    "234/241 (97.1%) 'No' on impersonationtkvr. Unlike phishing's phishcon_bands/"
    "phisheng_bands (which had real spread across hundreds of firms), these flags "
    "almost never fire. Not usable as a grouping variable for a rank-2 mixture "
    "retry — the groups would be ~10 and ~230, far too imbalanced. fraud4_comb1/2 "
    "(Q88A, downstream-impersonation-as-consequence-of-any-breach) is even sparser: "
    "only 14/241 (5.8%) non-missing. Neither candidate variable rescues the "
    "phishing-style validated-mixture approach for Impersonation."
)

# ===========================================================================
# 1b. Per-size-band rank-2 retry (never previously run for Impersonation —
# mixture_model.py only ran this breakdown for Phishing)
# ===========================================================================

def run_mixture_test(subset, label, group_col='freq', group_labels=None, min_per_freq=8):
    if group_labels is None:
        group_labels = FREQ_LABELS
    groups = sorted(subset[group_col].unique())
    rows, row_labels, ns = [], [], []
    for g in groups:
        sub = subset[subset[group_col] == g]
        if len(sub) < min_per_freq:
            continue
        counts = np.array([(sub['damage_bands'] == b).sum() for b in bands], dtype=float)
        if counts.sum() == 0:
            continue
        rows.append(counts / counts.sum())
        row_labels.append(group_labels.get(int(g), str(g)))
        ns.append(len(sub))

    if len(rows) < 3:
        print(f"  {label}: only {len(rows)} groups with n>={min_per_freq} — insufficient for rank test")
        return

    P = np.array(rows)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    total_var = (S ** 2).sum()
    cumvar = np.cumsum(S ** 2) / total_var
    var_2 = cumvar[1] if len(cumvar) > 1 else cumvar[0]
    gap = (cumvar[2] if len(cumvar) > 2 else 1.0) - var_2
    if var_2 > 0.95 and gap < 0.03:
        verdict = "CONSISTENT with rank-2"
    elif var_2 > 0.90:
        verdict = "MARGINAL"
    else:
        verdict = "NOT consistent with rank-2"
    print(f"  {label}: groups(n)={list(zip(row_labels, ns))}")
    print(f"    2-comp variance explained={var_2:.1%}, 3rd-comp gain={gap:.1%} -> {verdict}")

    # Extract F_T/F_M and check whether they're actually distinct (not just
    # whether rank-2 holds) — same extraction as mixture_model.py.
    direction = Vt[0]
    mean_p = P.mean(axis=0)

    def max_valid_extent(base, dir_vec):
        t_max, t_min = np.inf, -np.inf
        for i in range(len(base)):
            if dir_vec[i] > 1e-9:
                t_max = min(t_max, (1.0 - base[i]) / dir_vec[i])
                t_min = max(t_min, -base[i] / dir_vec[i])
            elif dir_vec[i] < -1e-9:
                t_min = max(t_min, (1.0 - base[i]) / dir_vec[i])
                t_max = min(t_max, -base[i] / dir_vec[i])
        return t_min, t_max

    t_min, t_max = max_valid_extent(mean_p, direction)
    F_a = np.clip(mean_p + t_max * direction, 0, None)
    F_b = np.clip(mean_p + t_min * direction, 0, None)
    if F_a.sum() > 0: F_a /= F_a.sum()
    if F_b.sum() > 0: F_b /= F_b.sum()
    mean_band_a, mean_band_b = np.dot(F_a, bands), np.dot(F_b, bands)
    if mean_band_a < mean_band_b:
        F_a, F_b, mean_band_a, mean_band_b = F_b, F_a, mean_band_b, mean_band_a
    print(f"    F_T (higher-cost) mean band={mean_band_a:.2f}, no-cost frac={F_a[0]:.3f}")
    print(f"    F_M (lower-cost)  mean band={mean_band_b:.2f}, no-cost frac={F_b[0]:.3f}")
    print(f"    Distinctness (mean band gap): {mean_band_a - mean_band_b:.2f} "
          f"{'(degenerate — components nearly identical)' if mean_band_a - mean_band_b < 0.5 else '(real separation)'}")


print("\n--- 1b. Per-size-band rank-2 retry for Impersonation ---")
for sz in sorted(imp['sizeb'].dropna().unique()):
    sz_label = SIZE_LABELS.get(int(sz), str(sz))
    run_mixture_test(imp[imp['sizeb'] == sz].copy(), f"Impersonation — {sz_label}")

# ===========================================================================
# 2. Outlier drill-down: what drives the weighted >> unweighted £ gap?
# ===========================================================================

print("\n" + "=" * 90)
print("2. OUTLIER DRILL-DOWN: weighted vs unweighted £ contribution")
print("=" * 90)

imp['damage_mid'] = imp['damage_bands'].map(DAMAGE_MIDPOINTS)
imp['contrib'] = imp['weight'] * imp['damage_mid']
w_total = imp['weight'].sum()
cost_total_w = imp['contrib'].sum()

print(f"\nUnweighted mean £: {imp['damage_mid'].mean():,.0f}")
print(f"Weighted mean £:   {cost_total_w / w_total:,.0f}")
print(f"corr(weight, damage_bands) = {imp['weight'].corr(imp['damage_bands']):.3f}  "
      f"(near zero — the gap is NOT a broad weight/cost relationship)")

TYPE_NAMES = {
    'type1': 'Ransomware', 'type2': 'OtherMalware', 'type3': 'DoS', 'type4': 'Hacking',
    'type5': 'Impersonation', 'type6': 'Phishing', 'type7': 'UnauthStaff',
    'type8': 'UnauthOutsiders', 'type9': 'Other', 'type13': 'UnauthStudents',
    'type15': 'VideoEavesdrop', 'type16': 'WebsiteTakeover',
}
tcols = [c for c in TYPE_NAMES if c in imp.columns]
for c in tcols:
    imp[c] = imp[c].fillna(0)
    imp.loc[imp[c] < 0, c] = 0

top = imp.sort_values('contrib', ascending=False).head(8).copy()
print(f"\nTop 8 rows by weighted £ contribution (of {len(imp)} total, "
      f"account for {100*top['contrib'].sum()/cost_total_w:.1f}% of weighted £ mass):")
print(f"  {'sizeb':>5} {'freq':>5} {'band':>5} {'weight':>8} {'contrib_£':>12}  types_flagged")
for _, row in top.iterrows():
    flagged = [TYPE_NAMES[c] for c in tcols if row[c] == 1]
    print(f"  {row['sizeb']:>5.0f} {row['freq']:>5.0f} {row['damage_bands']:>5.0f} "
          f"{row['weight']:>8.3f} {row['contrib']:>12,.0f}  {flagged}")

single_top = top.iloc[0]
print(f"\nSingle largest row alone: {100*single_top['contrib']/cost_total_w:.1f}% of the ENTIRE "
      f"weighted £ mass for Impersonation (weight={single_top['weight']:.3f} — unremarkable, "
      f"not a weight outlier; it's the £300k midpoint of a top-band (band=10) observation).")
print(
    "This single row also has type4/5/6/16 (Hacking/Impersonation/Phishing/WebsiteTakeover) all "
    "flagged. CORRECTION (see NOTES.md 2026-07-12 data-model clarification): type1-16 are whole-year "
    "presence flags (Q53A: 'did X happen at any point in the last 12 months'), not evidence these "
    "were the same incident — the data cannot distinguish 'one event that touched multiple systems' "
    "from 'this firm had several separate incidents of different types that year.' The correct claim "
    "is only: this firm experienced several different breach types during the year, of which "
    "Impersonation was the costliest. Its crimecost_bands (independent total-annual-cost variable, "
    "rarely populated) also = 10, matching damage_bands exactly — i.e. for this one firm, total "
    "annual cost really does equal the single max incident (the conservative bridge is exactly right "
    "here, not an underestimate)."
)

# ===========================================================================
# 3. Generalising: does severity correlate with attack-type breadth across
# ALL attacked firms (not just Impersonation)? This tests whether the outlier
# above is a one-off or part of a structural pattern.
# ===========================================================================

print("\n" + "=" * 90)
print("3. GENERAL PATTERN: mean # attack types flagged, by damage_bands level")
print("   (ALL attacked firms with valid damage_bands, n={})".format(
    len(df[df['freq'].notna() & ~df['freq'].isin(INVALID_FREQ) & (df['freq'] > 0) &
           df['damage_bands'].notna() & (df['damage_bands'] < SPECIAL_CODE_THRESHOLD)])))
print("=" * 90)

all_attacked = df[
    df['freq'].notna() & ~df['freq'].isin(INVALID_FREQ) & (df['freq'] > 0) &
    df['damage_bands'].notna() & (df['damage_bands'] < SPECIAL_CODE_THRESHOLD)
].copy()
tcols_all = [c for c in TYPE_COLNAMES if c in all_attacked.columns]
for c in tcols_all:
    all_attacked[c] = all_attacked[c].fillna(0)
    all_attacked.loc[all_attacked[c] < 0, c] = 0
all_attacked['n_types'] = all_attacked[tcols_all].sum(axis=1)

print(f"\n  {'band':>5} {'label':<14} {'n':>5} {'mean_n_types':>13} {'%multivector(>1 type)':>22}")
for b in bands:
    sub = all_attacked[all_attacked['damage_bands'] == b]
    if len(sub) == 0:
        continue
    print(f"  {b:>5} {DAMAGE_BAND_LABELS[b]:<14} {len(sub):>5} {sub['n_types'].mean():>13.2f} "
          f"{100*(sub['n_types'] > 1).mean():>21.1f}%")

print(
    "\nClean, monotonic pattern: mean # types flagged rises from 1.44 (no-cost) to 4.38 (top "
    "band), and % firms with >1 type flagged rises from 32.5% to 100%. CORRECTION: type1-16 are "
    "whole-year presence flags, not same-incident evidence — the correct reading is that firms with "
    "the costliest single worst incident also tend to have experienced MORE DIFFERENT KINDS of "
    "breaches somewhere during that year (a 'beleaguered firm had a rough year' pattern), not that "
    "the worst incident itself was a multi-vector attack. Still a real, useful finding — it says "
    "something about which firms end up with a high damage_bands (more broadly exposed/targeted "
    "firms), and still means disrupta's single 'most disruptive type' label describes less and less "
    "of that firm's full year as damage_bands rises — but it is NOT evidence that the worst incident "
    "itself was multi-vector, and the earlier framing overclaimed that."
)

# ===========================================================================
# 4. Does the Micro/Small 'real separation' found in step 1b actually reflect
# cross-type frequency contamination, rather than a genuine impersonation-
# specific mixture? freq (Q54) is defined as "how often did you experience
# ANY of the breaches/attacks you mentioned" — i.e. cross-type, not specific
# to Impersonation, even after filtering to disrupta==5. Unlike phishing
# (independently validated with phishing-specific phisheng_bands/phishcon_
# bands), no impersonation-specific frequency variable exists, so this can
# only be checked indirectly: does freq correlate with how many OTHER types
# the firm also experienced that year, and does that co-occurrence explain
# cost independent of freq?
# ===========================================================================

print("\n" + "=" * 90)
print("4. FREQ CROSS-TYPE CONTAMINATION CHECK (Impersonation, disrupta=5)")
print("=" * 90)

imp['n_other_types'] = imp[tcols].sum(axis=1) - imp['type5']

rho = imp['freq'].corr(imp['n_other_types'], method='spearman')
print(f"\nSpearman corr(freq, n_other_types) = {rho:.3f}  (n={len(imp)})")
print("(freq is the firm's whole-year, cross-type frequency; n_other_types = how many")
print(" OTHER attack types [besides Impersonation] this firm also experienced that year)")

print(f"\nMean n_other_types by freq group:")
print(f"  {'freq group':<18}{'n':>6}{'mean_n_other_types':>20}")
for fg in sorted(imp['freq'].unique()):
    sub = imp[imp['freq'] == fg]
    print(f"  {FREQ_LABELS.get(int(fg), str(fg)):<18}{len(sub):>6}{sub['n_other_types'].mean():>20.2f}")

print(f"\nWithin each freq group, does having other co-occurring types predict higher damage_bands")
print(f"independent of freq? (tests whether 'other types present' — not freq itself — is doing the work)")
print(f"  {'freq group':<18}{'n(0 other)':>11}{'mean_band(0)':>13}{'n(1+ other)':>12}{'mean_band(1+)':>14}")
for fg in sorted(imp['freq'].unique()):
    sub = imp[imp['freq'] == fg]
    no_other = sub[sub['n_other_types'] == 0]
    has_other = sub[sub['n_other_types'] >= 1]
    if len(no_other) < 3 or len(has_other) < 3:
        continue
    print(f"  {FREQ_LABELS.get(int(fg), str(fg)):<18}{len(no_other):>11}{no_other['damage_bands'].mean():>13.2f}"
          f"{len(has_other):>12}{has_other['damage_bands'].mean():>14.2f}")

print(
    "\nInterpretation: if corr(freq, n_other_types) is meaningfully positive and/or the "
    "'1+ other types' column runs consistently higher than the '0 other' column within the same "
    "freq group, that's direct evidence the Micro/Small 'real separation' found in step 1b is at "
    "least partly freq acting as a proxy for 'this firm had a broadly bad year across several attack "
    "types', not a genuine Impersonation-specific targeted/mass split. Given no impersonation-"
    "specific frequency variable exists (step 1a), this check cannot fully resolve the question — "
    "only bound how much it plausibly matters."
)

print("\nDone.")
