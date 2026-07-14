"""
Raw-data survey of Ransomware and Impersonation (2026-07-13) — descriptive
only, no distribution fitting. Purpose: after two independent attempts to
model these types via censored lognormal MLE both failed (Ransomware
degenerate even after bias-correction; Impersonation's apparent fit turned
out to be propped up by a single outlier row — see type_cost_censored_model.py),
step back and actually look at what data exists before choosing another
parametric assumption.

This surfaced a previously-unloaded, richer variable family for Ransomware,
analogous to phishing's phishcon_bands/phisheng_bands:
  ranssoft_bands = COUNT of ransomware attacks this year (1=None ... 9=100+)
  ransdem_bands  = £ ransom DEMANDED (12-band scale, same as ranscost_bands)
  ranspay_bands  = £ actually PAID
  ranspayyn      = whether paid at all (totally/partially/no)
  ranscost_bands = £ total ORG cost (already used — broader than the ransom
                   itself, presumably includes recovery/downtime/etc.)

For Impersonation, no equivalent count variable exists. The closest thing
is the general fraud-consequence counts (fraud1/2/3, Q88A: money moved out /
card misuse / fraudulent-invoice payment) — not tagged by cause, but
plausibly Impersonation-heavy given fraud3 in particular reads like a
classic invoice/BEC impersonation scam. fraud4 (already loaded) is the one
consequence explicitly tied to impersonation.

Sections:
1. Ransomware: coverage of each Q83 variable among type1==1 firms; how
   ranssoft_bands (count), ransdem_bands (demand), ranspay_bands (paid),
   and ranscost_bands (org total) relate to each other and to damage_bands.
2. Impersonation: coverage of fraud1/2/3 among type5==1 firms; how they
   relate to damage_bands; cross-check against fraudcost_bands/crimecost_bands.
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

df = get_business_data()

COUNT_LABELS = {1: 'None', 2: '1', 3: '2-3', 4: '4-5', 5: '6-10', 6: '11-20',
                7: '21-50', 8: '51-100', 9: '100+'}

FULL_MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000,
            9: 75000, 10: 300000, 11: 750000, 12: 3000000, 13: 5000000}
TYPE_COST_MID = {1: 50, 2: 175, 3: 375, 4: 750, 5: 1500, 6: 3500, 7: 7500,
                  8: 15000, 9: 35000, 10: 75000, 11: 175000, 12: 500000}


def valid(s, lo=1, hi=12):
    return s.notna() & (s >= lo) & (s <= hi)


print("=" * 100)
print("SECTION 1: RANSOMWARE — full raw-data survey (type1==1 firms)")
print("=" * 100)

rans = df[df['type1'] == 1].copy()
print(f"n firms with type1==1 (ransomware occurred at all): {len(rans)}, weight sum={rans['weight'].sum():.2f}")

for col, hi in [('ranssoft_bands', 9), ('ransdem_bands', 12), ('ranspay_bands', 12),
                ('ranscost_bands', 12), ('ranspayyn', 3), ('ranschk', 2)]:
    v = valid(rans[col], 1, hi)
    print(f"  {col:<16} valid n={v.sum():>4} / {len(rans)}  ({100*v.sum()/len(rans):.1f}% coverage)")

print("\n--- ranssoft_bands: distribution of ransomware ATTACK COUNT (among valid) ---")
v = rans[valid(rans['ranssoft_bands'], 1, 9)]
print(v['ranssoft_bands'].map(COUNT_LABELS).value_counts().reindex(COUNT_LABELS.values(), fill_value=0))

print("\n--- Does ranscost_bands (org total cost) scale with ranssoft_bands (attack count)? ---")
both = rans[valid(rans['ranssoft_bands'], 1, 9) & valid(rans['ranscost_bands'], 1, 12)].copy()
print(f"n with BOTH valid: {len(both)}")
if len(both) >= 5:
    both['count_label'] = both['ranssoft_bands'].map(COUNT_LABELS)
    both['ranscost_mid'] = both['ranscost_bands'].map(TYPE_COST_MID)
    print(both.groupby('count_label')['ranscost_mid'].agg(['count', 'mean', 'median']).reindex(
        [c for c in COUNT_LABELS.values() if c in both['count_label'].unique()]))

print("\n--- ransdem_bands (ransom demanded) vs ranspay_bands (ransom paid) vs ranscost_bands (org total) ---")
print("(all three share the same 12-band £ scale — directly comparable)")
trio = rans[valid(rans['ransdem_bands'], 1, 12) & valid(rans['ranspay_bands'], 1, 12) &
            valid(rans['ranscost_bands'], 1, 12)].copy()
print(f"n with all three valid: {len(trio)}")
if len(trio) > 0:
    trio['dem_mid'] = trio['ransdem_bands'].map(TYPE_COST_MID)
    trio['pay_mid'] = trio['ranspay_bands'].map(TYPE_COST_MID)
    trio['cost_mid'] = trio['ranscost_bands'].map(TYPE_COST_MID)
    print(trio[['dem_mid', 'pay_mid', 'cost_mid', 'weight']].to_string(index=True))

print("\n--- ranspayyn: did firms pay? (among firms with a ransom demanded) ---")
demanded = rans[valid(rans['ransdem_bands'], 1, 12)]
print(demanded['ranspayyn'].value_counts(dropna=False))
print("(1=Yes totally, 2=Yes partially, 3=No, per codebook)")

print("\n--- damage_bands (single-worst-incident cost) among Ransomware-disrupta firms, vs ranscost_bands ---")
rans_disrupta = df[(df['disrupta'] == 1) & (df['damage_bands'] < SPECIAL_CODE_THRESHOLD)].copy()
print(f"n(disrupta==Ransomware): {len(rans_disrupta)}")
print(f"damage_bands distribution:\n{rans_disrupta['damage_bands'].value_counts().sort_index()}")

print("\n" + "=" * 100)
print("SECTION 2: IMPERSONATION — full raw-data survey (type5==1 firms)")
print("=" * 100)

imp = df[df['type5'] == 1].copy()
print(f"n firms with type5==1 (impersonation occurred at all): {len(imp)}, weight sum={imp['weight'].sum():.2f}")

for col in ['fraud1_comb1', 'fraud1_comb2', 'fraud2_comb1', 'fraud2_comb2',
            'fraud3_comb1', 'fraud3_comb2', 'fraud4_comb1', 'fraud4_comb2',
            'fraudcost_bands', 'notfraudcost_bands', 'crimecost_bands',
            'impersonationhack', 'impersonationtkvr']:
    if col not in imp.columns:
        print(f"  {col:<20} NOT LOADED")
        continue
    v = imp[col].notna() & ~imp[col].isin([997, 999, -9, -1])
    print(f"  {col:<20} valid n={v.sum():>4} / {len(imp)}  ({100*v.sum()/len(imp):.1f}% coverage)")

print("\n--- fraud1/2/3 (general fraud consequences of ANY breach) among Impersonation firms ---")
for col, name in [('fraud1_comb2', 'Money moved out (any)'), ('fraud2_comb2', 'Card misuse (any)'),
                   ('fraud3_comb2', 'Fraudulent-invoice payment (any)')]:
    v = imp[imp[col].isin([0, 1])]
    if len(v) > 0:
        print(f"  {name}: n valid={len(v)}, %Yes={100*(v[col]==1).mean():.1f}%")

print("\n--- fraudcost_bands (total cost of ALL frauds) among Impersonation-disrupta firms vs damage_bands ---")
imp_disrupta = df[(df['disrupta'] == 5) & (df['damage_bands'] < SPECIAL_CODE_THRESHOLD)].copy()
print(f"n(disrupta==Impersonation): {len(imp_disrupta)}")
v = imp_disrupta[valid(imp_disrupta['fraudcost_bands'], 1, 13)]
print(f"n with valid fraudcost_bands: {len(v)}")
if len(v) > 0:
    v = v.copy()
    v['fraud_mid'] = v['fraudcost_bands'].map(FULL_MID)
    v['damage_mid'] = v['damage_bands'].map(FULL_MID)
    v['ratio'] = v.apply(lambda r: r['fraud_mid']/r['damage_mid'] if r['damage_mid'] > 0 else np.nan, axis=1)
    print(v[['damage_bands', 'fraudcost_bands', 'damage_mid', 'fraud_mid', 'ratio', 'weight']].to_string(index=True))

print("\n--- damage_bands distribution among Impersonation-disrupta firms (n=241 in earlier work) ---")
print(imp_disrupta['damage_bands'].value_counts().sort_index())

print("\nDone. No modeling in this script — descriptive only, per request.")
