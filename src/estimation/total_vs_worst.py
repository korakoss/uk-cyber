"""Is the worst incident (damage_bands) close to the annual total? Compare with crimecost_bands.

Both on the same 13-band cost scale (1 none ... 10 £100k-500k). Per firm with both valid: band difference
(crimecost - damage) and midpoint ratio. Tabulated overall, by frequency answer, and for single-channel groups.
Also: SPSS label of crimecost_bands and its coverage among attacked firms. Survey-weighted shares.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/total_vs_worst.py
"""

import glob

import numpy as np
import pandas as pd
import pyreadstat

import joint_five_channel as f
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load, SHORT

MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000,
       11: 750000, 12: 3000000, 13: 5000000}
FLAB = {1: "once", 2: "<monthly", 3: "monthly", 4: "weekly", 5: "daily", 6: "sev/day"}


def summary(label, cc, db, w, m):
    m = m & ~np.isnan(cc) & ~np.isnan(db)
    if m.sum() == 0:
        return
    diff = cc[m] - db[m]
    ww = w[m]
    lower = np.average(diff < 0, weights=ww)
    same = np.average(diff == 0, weights=ww)
    higher = np.average(diff > 0, weights=ww)
    both = m & (db > 1)
    ratio = np.array([MID[int(a)] / MID[int(b)] for a, b in zip(cc[both], db[both])]) if both.sum() else np.array([np.nan])
    tot_ratio = np.average([MID[int(a)] for a in cc[m]], weights=ww) / max(np.average([MID[int(b)] for b in db[m]], weights=ww), 1e-9)
    print(f"  {label:34s} n {m.sum():4d}  total<worst {lower:4.2f}  same band {same:4.2f}  total>worst {higher:4.2f}   "
          f"median ratio (worst>0) {np.median(ratio):5.2f}   ratio of weighted mean midpoints {tot_ratio:5.2f}")


def main():
    sav = glob.glob("/home/user/uk-cyber/data/raw/*.sav")[0]
    _, meta = pyreadstat.read_sav(sav, metadataonly=True)
    for c in ("crimecost_bands", "damage_bands"):
        print(f"{c}: {meta.column_names_to_labels.get(c)}")
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    cc = pd.to_numeric(r["crimecost_bands"], errors="coerce").values
    cc = np.where((cc >= 1) & (cc <= 13), cc, np.nan)
    db, fq = d["band"].values, d["freq"].values
    w = d["weight"].fillna(d["weight"].median()).values
    att = d["att"].values == 1
    print(f"\nattacked firms {att.sum()}: damage_bands valid {int((att & ~np.isnan(db)).sum())}, crimecost valid "
          f"{int((att & ~np.isnan(cc)).sum())}, both {int((att & ~np.isnan(db) & ~np.isnan(cc)).sum())}")
    print("crimecost band counts (both valid): " + str(pd.Series(cc[att & ~np.isnan(db) & ~np.isnan(cc)]).value_counts().sort_index().to_dict()))
    print("\nper firm, crimecost (total) vs damage (worst incident):")
    summary("all attacked", cc, db, w, att)
    for k in range(1, 7):
        summary(f"freq {FLAB[k]}", cc, db, w, att & (fq == k))
    only = X.sum(1) == 1
    summary("phishing-only", cc, db, w, att & only & (X[:, 0] == 1))
    summary("impersonation-only", cc, db, w, att & only & (X[:, SHORT.index("Imper")] == 1))
    summary("2+ types", cc, db, w, att & (X.sum(1) >= 2))
    summary("worst incident > 0 only", cc, db, w, att & (db > 1))


if __name__ == "__main__":
    main()
