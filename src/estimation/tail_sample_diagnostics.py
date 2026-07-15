"""
Tail sample diagnostics (2026-07-15). Two questions raised in discussion:
1. Is £100k+ genuinely the TOP damage band, or is there resolution above it?
2. The tail is dominated by Micro — but Micro has a big sample. How well is the
   Micro top-band RATE actually estimated (vs. "only 2 firms" alarmism)?

Prints, per size band: total n, attacked n, the full band histogram, and the
top-band firms' weights / implied population representation.
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

df = get_business_data()
df = df[df['sizeb'].notna()].copy()
df['sizeb'] = df['sizeb'].astype(int)
VALID_FREQ = {1, 2, 3, 4, 5, 6}
df['attacked'] = df['freq'].isin(VALID_FREQ)
db = df['damage_bands']
df['band'] = np.where(df['attacked'] & db.notna() & (db < SPECIAL_CODE_THRESHOLD) & (db >= 1), db, np.nan)
N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}
SIZE_LABELS = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}

print("Q1: what is the maximum observed damage_bands value?")
print(f"   max band in data = {int(np.nanmax(df['band']))}  (codebook: 10 = £100k-£500k;")
print("   no firm selected any higher band, so band 10 is the top OBSERVED band.)")
print("   national_simulation treats band 10 as OPEN [£100k, inf) via the lognormal tail —")
print("   that is a MODELLING CHOICE, not a fact of the data. See note at bottom.\n")

print("Q2: per-size sample sizes and band histogram")
print("=" * 88)
for sz in (1, 2, 3, 4):
    sub = df[df['sizeb'] == sz]
    att = sub[sub['attacked']]
    hist = att['band'].value_counts().reindex(range(1, 11), fill_value=0).astype(int)
    n_top = int(hist[10])
    print(f"\n{SIZE_LABELS[sz]}: total n={len(sub)}, attacked n={len(att)}, "
          f"with valid band n={int(att['band'].notna().sum())}")
    print("  band:  " + " ".join(f"{b:>4}" for b in range(1, 11)))
    print("  count: " + " ".join(f"{hist[b]:>4}" for b in range(1, 11)))
    # top-band firms
    top = sub[sub['band'] == 10]
    if len(top):
        w = top['weight'].values
        wsum_all = sub['weight'].values.sum()
        share_w = w.sum() / wsum_all
        print(f"  top-band (£100k+) firms: n={n_top}, weights={np.round(w,3).tolist()}")
        print(f"    -> weighted top-band rate = {100*share_w:.3f}% of {SIZE_LABELS[sz]} firms "
              f"=> ~{int(round(share_w*N_BY_SIZE[sz])):,} businesses nationally")
        # crude sampling uncertainty on that rate (Poisson-ish on n_top counts)
        rel = 1/np.sqrt(n_top)
        print(f"    -> count is {n_top}; rough relative sampling error on the rate ~ 1/sqrt(n) "
              f"= {100*rel:.0f}%  (this is the thin-evidence concern)")

print("\n" + "=" * 88)
print("Reading: Micro's OVERALL sample is large, so its prevalence and body-band")
print("rates are well estimated. But the TOP-BAND rate specifically rests on the")
print("handful of top-band firms — that count, not the overall Micro n, is what")
print("bounds how well the tail (which drives ~74% of the total) is pinned.")
print("\nSeparately: whether band 10 is treated as £100k-£500k (bounded) or £100k+")
print("(open) changes the £ assigned to each top-band firm — a sensitivity worth running.")
print("\nDone.")
