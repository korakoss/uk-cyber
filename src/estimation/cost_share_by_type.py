"""
First-pass £ share of total cost by attack type (disrupta).

Motivating question: phishing is ~57% of incidents by COUNT (disrupta = most
disruptive attack type). Is it also ~57% of total £ cost, or does it cover much
less than that because other types (ransomware, hacking, ...) have a much
fatter cost tail per incident?

Method (deliberately simple — a first-pass magnitude check, not a bridge model):
For each attacked firm, take damage_bands (cost of the single most disruptive
incident) and convert to a £ midpoint. Attribute that £ figure to whichever
type disrupta says was most disruptive for that firm. Sum (weighted and
unweighted) within each disrupta type and express as a % of the total.

This is exactly the CONSERVATIVE bridge tier (total = max, no multiplier for
repeat attacks) applied per type, so it understates whichever types have the
highest implied attack frequency — but that understatement is roughly uniform
across types being compared here (all still using only the reported max), so
the relative shares are informative even though the absolute £ figures are not
a real total-cost estimate. Does not attempt the mixture-bridge treatment
(phishing-only) or the k_implied pessimistic bound — see NOTES.md, "Pipeline
Design Discussion (2026-07-12)" for how this fits into the larger bridge
question.
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

df = get_business_data()
INVALID_FREQ = {997, 999, 997.0, 999.0}

attacked = df[
    df['freq'].notna() &
    ~df['freq'].isin(INVALID_FREQ) &
    (df['freq'] > 0) &
    df['damage_bands'].notna() &
    (df['damage_bands'] < SPECIAL_CODE_THRESHOLD)
].copy()

attacked['disrupta_clean'] = attacked['disrupta'].where(
    attacked['disrupta'].notna() &
    (attacked['disrupta'] > 0) &
    (attacked['disrupta'] != 997)
)

has_dis = attacked[attacked['disrupta_clean'].notna()].copy()
has_dis['disrupta_clean'] = has_dis['disrupta_clean'].astype(int)

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

# Same £ midpoints used in attack_type_breakdown.py analysis 5, for consistency.
DAMAGE_MIDPOINTS = {
    1: 0, 2: 50, 3: 300, 4: 750, 5: 3000,
    6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000
}

has_dis['damage_mid'] = has_dis['damage_bands'].map(DAMAGE_MIDPOINTS)

print("=" * 90)
print("£ SHARE OF TOTAL COST BY disrupta TYPE (conservative bridge: total = max incident)")
print("=" * 90)
print(f"\nAttacked firms with valid damage_bands: {len(attacked)}")
print(f"Of which have valid disrupta:           {len(has_dis)} ({100*len(has_dis)/len(attacked):.1f}%)")

n_total = len(has_dis)
w_total = has_dis['weight'].sum()
cost_total_unweighted = has_dis['damage_mid'].sum()
cost_total_weighted = (has_dis['weight'] * has_dis['damage_mid']).sum()

rows = []
for dv in sorted(has_dis['disrupta_clean'].unique()):
    sub = has_dis[has_dis['disrupta_clean'] == dv]
    lbl = DISRUPTA_LABELS.get(dv, str(dv))

    n = len(sub)
    w = sub['weight'].sum()
    mean_mid_unw = sub['damage_mid'].mean()
    mean_mid_w = np.average(sub['damage_mid'], weights=sub['weight'])

    cost_unw = sub['damage_mid'].sum()
    cost_w = (sub['weight'] * sub['damage_mid']).sum()

    rows.append({
        'type': lbl,
        'n': n,
        'count_share_%': 100 * n / n_total,
        'weighted_count_share_%': 100 * w / w_total,
        'mean_£_unw': mean_mid_unw,
        'mean_£_w': mean_mid_w,
        '£_share_unw_%': 100 * cost_unw / cost_total_unweighted,
        '£_share_w_%': 100 * cost_w / cost_total_weighted,
    })

res = pd.DataFrame(rows).sort_values('£_share_w_%', ascending=False)

print("\n--- Per-type breakdown, sorted by weighted £ share ---\n")
print(f"{'Type':<32}{'n':>5}{'count%':>9}{'wcount%':>9}{'mean£unw':>11}{'mean£w':>11}{'£share_unw%':>13}{'£share_w%':>11}")
for _, r in res.iterrows():
    print(f"{r['type']:<32}{r['n']:>5}{r['count_share_%']:>8.1f}%{r['weighted_count_share_%']:>8.1f}%"
          f"{r['mean_£_unw']:>11,.0f}{r['mean_£_w']:>11,.0f}{r['£_share_unw_%']:>12.1f}%{r['£_share_w_%']:>10.1f}%")

print(f"\nTOTAL (all types): n={n_total}, unweighted £ mass={cost_total_unweighted:,.0f}, "
      f"weighted £ mass={cost_total_weighted:,.0f}")

# ---------------------------------------------------------------------------
# Headline comparison: phishing vs. everything else
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("HEADLINE: PHISHING vs. NON-PHISHING SHARE OF COUNT vs. £")
print("=" * 90)

phish = has_dis[has_dis['disrupta_clean'] == 6]
nonphish = has_dis[has_dis['disrupta_clean'] != 6]

for label, sub in [('Phishing', phish), ('Non-phishing (all other types)', nonphish)]:
    n = len(sub)
    w = sub['weight'].sum()
    cost_unw = sub['damage_mid'].sum()
    cost_w = (sub['weight'] * sub['damage_mid']).sum()
    print(f"\n{label}:")
    print(f"  Count share (unweighted): {100*n/n_total:.1f}%   (weighted: {100*w/w_total:.1f}%)")
    print(f"  £ share (unweighted):     {100*cost_unw/cost_total_unweighted:.1f}%   "
          f"(weighted: {100*cost_w/cost_total_weighted:.1f}%)")
    print(f"  Mean £ per incident (unweighted): {sub['damage_mid'].mean():,.0f}   "
          f"(weighted: {np.average(sub['damage_mid'], weights=sub['weight']):,.0f})")

# ---------------------------------------------------------------------------
# Ransomware/hacking specifically, since these were flagged as suspected
# fat-tail types in NOTES.md
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("FAT-TAIL CHECK: ransomware / hacking vs. phishing, P(£ >= 100k) per incident")
print("=" * 90)

for dv, lbl in [(6, 'Phishing'), (1, 'Ransomware'), (4, 'Hacking (bank accts)')]:
    sub = has_dis[has_dis['disrupta_clean'] == dv]
    if len(sub) == 0:
        continue
    p_tail_unw = (sub['damage_bands'] == 10).mean()
    p_tail_w = np.average((sub['damage_bands'] == 10).astype(float), weights=sub['weight'])
    print(f"  {lbl:<24} n={len(sub):>4}  P(band=10, i.e. >=£100k) unweighted={100*p_tail_unw:.1f}%  "
          f"weighted={100*p_tail_w:.1f}%")

print()
print("Done.")
