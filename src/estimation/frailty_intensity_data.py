"""Coverage of per-firm attack COUNT variables usable as intensity measures for frailty shape.

Cyber crime module counts: Cybercrime_phishsum (phishing), Cybercrime_notphishsum (non-phishing
cyber crime), Cybercrime_allsum (all). Two counts per firm (phish + notphish) would separate a
shared intensity factor from channel-private noise. Here: coverage only, plus consistency
(allsum vs phish + notphish) and who reports both.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/frailty_intensity_data.py
"""

import numpy as np
import pandas as pd

import joint_four_channel as j
import joint_five_channel as f
from data import load_raw

VARS = ["Cybercrime_phishsum", "Cybercrime_notphishsum", "Cybercrime_allsum"]


def main():
    d = f.load_firms()
    raw = load_raw()
    keep = j.banded(raw["type_comb1"]).isin([0, 1])
    allg = np.column_stack([(j.banded(raw[c]) == 1).astype(int)[keep].values for c in j.GENUINE_TYPE_COLS])
    att0 = j.banded(raw["type_comb1"])[keep].values
    sel = (att0 == 0) | (allg.sum(1) > 0)
    C = raw.loc[keep, VARS].apply(pd.to_numeric, errors="coerce")[sel].reset_index(drop=True)
    C = C.where(C >= 0)
    assert len(C) == len(d)
    att = d["att"].values == 1
    nonphish = (d[["fI", "fR", "fS"]].values.sum(1) > 0)

    print("RAW VALUE SUMMARY (all rows in frame, valid >= 0)")
    for v in VARS:
        x = C[v]
        print(f"  {v:24s} valid {x.notna().sum():4d}  zeros {(x == 0).sum():4d}  >=1 {(x >= 1).sum():4d}"
              f"  median(>=1) {x[x >= 1].median() if (x >= 1).any() else np.nan}")

    ph, nph, al = (C[v].values for v in VARS)
    print(f"\nATTACKED firms n={att.sum()}; phishing-flagged {int((att & (d.fP == 1)).sum())};"
          f" non-phishing-flagged {int((att & nonphish).sum())}; both {int((att & (d.fP == 1) & nonphish).sum())}")
    grp = {"phish-flagged": att & (d.fP.values == 1), "non-phish-flagged": att & nonphish,
           "flagged both": att & (d.fP.values == 1) & nonphish}
    for lab, m in grp.items():
        print(f"  {lab:18s} n={m.sum():4d}:  phishsum {np.mean(~np.isnan(ph[m])):.2f}   notphishsum "
              f"{np.mean(~np.isnan(nph[m])):.2f}   allsum {np.mean(~np.isnan(al[m])):.2f}   "
              f"phish & notphish both {np.mean(~np.isnan(ph[m]) & ~np.isnan(nph[m])):.2f}")

    both = att & (d.fP.values == 1) & nonphish & ~np.isnan(ph) & ~np.isnan(nph)
    both_pos = both & (ph >= 1) & (nph >= 1)
    print(f"\n  firms flagged on both sides WITH both counts: {both.sum()}   (both counts >= 1: {both_pos.sum()})")
    print("  by size (flagged both / with both counts / both >=1):")
    for s in (1, 2, 3, 4):
        ms = d["sizeb"].values == s
        fb = att & (d.fP.values == 1) & nonphish & ms
        print(f"    size {s}: {fb.sum():4d} / {(both & ms).sum():4d} / {(both_pos & ms).sum():4d}")
    band = d["band"].values
    for lab, m in (("with both counts", both), ("flagged both, missing a count", att & (d.fP.values == 1) & nonphish & ~both)):
        b = band[m]
        print(f"  worst band, {lab}: n={m.sum()}, P(no cost)={np.nanmean(b == 1):.2f}, mean band={np.nanmean(b):.2f}")

    ok = ~np.isnan(ph) & ~np.isnan(nph) & ~np.isnan(al)
    print(f"\n  consistency allsum == phishsum + notphishsum: {np.mean(al[ok] == ph[ok] + nph[ok]):.2f} of {ok.sum()}")
    print(f"  notphishsum > 0 among firms with NO non-phishing flag (and count present): "
          f"{int(((nph > 0) & att & ~nonphish).sum())} of {int((~np.isnan(nph) & att & ~nonphish).sum())}")


if __name__ == "__main__":
    main()
