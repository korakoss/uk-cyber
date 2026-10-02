"""Ransomware tick (#9): what it contains, the reversed tier effect, and encryption without a demand.

Items: type1 (Q53A devices targeted with ransomware = our R channel); ranschk (Q83X other instances where devices were
targeted, even if unsuccessful); ranssoft (Q83E attacks where a financial ransom was demanded); disruptphish1 (worst
phishing incident resulted in ransomware); fraudconta (frauds resulting from ransomware).
Breached = W2; spread signature as in escalation.py; 'encryption-like' = systems corrupted OR files lost OR devices damaged
OR restore >= 1 day.
1. Composition of ransomware-ticked firms: demand >= 1 / attempt-only (no demand, not breached) / breached without demand.
2. By frailty tier and size: P(demand), P(breached), P(attempt-only), P(encryption-like) among ticked firms.
3. Breached without a demand: how many look encryption-like; their worst cost vs breached-with-demand.
4. Ransomware ticked as consequence of phishing (disruptphish1) and channels hit; costs by kind.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/ransomware.py
"""

import numpy as np
import pandas as pd

from breach_measures import num as cnum
from broad_interaction import channels, tightened
from counts_given_frailty import tier_posteriors, TIER
from latent_on_streams import aligned_raw
from material_breach import data

OUTC = (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13)


def main():
    X, size, band, D, oa, w = data()
    r = aligned_raw()
    nm = lambda c: pd.to_numeric(r[c], errors="coerce").values
    C = channels(X)
    nch = C.sum(1)
    lo, hi = tightened(band)
    mid = np.where(hi == 0, 0.0, (lo + hi) / 2)
    soft = np.isin(nm("restore"), [3, 4, 5, 6]) | (nm("impact1") == 1) | (nm("impact4") == 1) | (nm("impact2") == 1)
    br = ((oa == 1) | soft) & ~np.isnan(mid)
    enc = (nm("outcome1") == 1) | (nm("outcome3") == 1) | (nm("outcome12") == 1) | np.isin(nm("restore"), [3, 4, 5, 6])
    tier = tier_posteriors().argmax(1)
    R = (C[:, 2] == 1) & ~np.isnan(mid)
    dem = cnum(r, "ranssoft")
    chk = nm("ranschk")
    print(f"ransomware-ticked firms with cost data: {R.sum()} (weighted share of attacked {w[R].sum() / w[(nch >= 1) & ~np.isnan(mid)].sum():.3f})")
    print("  ranschk (other attempts, even unsuccessful): " + ", ".join(f"{int(k) if not np.isnan(k) else 'NaN'}: {v}" for k, v in pd.Series(chk[R]).value_counts(dropna=False).sort_index().items()))
    print("  ranssoft (ransom demanded, count): " + ", ".join(f"{int(k) if not np.isnan(k) else 'NaN'}: {v}" for k, v in pd.Series(np.minimum(dem[R], 5)).value_counts(dropna=False).sort_index().items()) + "  (5 = 5+)")
    d1 = R & (dem >= 1)
    att = R & ~(dem >= 1) & ~br
    bnd = R & ~(dem >= 1) & br

    def row(lab, m):
        if not m.any():
            print(f"   {lab:32s} n 0")
            return
        b, ww = mid[m], w[m]
        print(f"   {lab:32s} n {m.sum():3d}  wshare {w[m].sum() / w[R].sum():.2f}  breached {np.average(br[m], weights=ww):.2f}"
              f"  encryption-like {np.average(enc[m], weights=ww):.2f}  channels {np.average(nch[m], weights=ww):.2f}"
              f"  worst none/<500/500-5k/5k+ {np.average(b == 0, weights=ww):.2f}/{np.average((b > 0) & (b < 500), weights=ww):.2f}/"
              f"{np.average((b >= 500) & (b < 5000), weights=ww):.2f}/{np.average(b >= 5000, weights=ww):.2f}  mean £{np.average(b, weights=ww):,.0f}")
    print("\n1. composition")
    row("ransom demanded >= 1", d1)
    row("no demand, not breached (attempt)", att)
    row("no demand, breached", bnd)
    row("  of which encryption-like", bnd & enc)
    print("\n2. by tier / size (ticked firms): P(demand)  P(breached)  P(attempt-only)  P(encryption-like)  n")
    for lab, m in [(f"tier {t}", R & (tier == k)) for k, t in enumerate(TIER)] + [("Micro", R & (size == 1)), ("Small+", R & (size >= 2))]:
        ww = w[m]
        print(f"   {lab:9s} {np.average(dem[m] >= 1, weights=ww):.2f}  {np.average(br[m], weights=ww):.2f}  "
              f"{np.average(att[m], weights=ww):.2f}  {np.average(enc[m], weights=ww):.2f}  {m.sum()}")
    print("\n3. breached: with demand vs without demand but encryption-like")
    row("breached, demand", R & br & (dem >= 1))
    row("breached, no demand, encryption-like", bnd & enc)
    row("breached, no demand, not encryption-like", bnd & ~enc)
    print("\n4. ransomware as a consequence of phishing (disruptphish1 = 1)")
    cp = R & (nm("disruptphish1") == 1)
    print(f"   consequence-type ticks {cp.sum()} of {R.sum()};  among breached {int((cp & br).sum())} of {int((R & br).sum())};"
          f" among 1-2 channel firms {int((cp & (nch <= 2)).sum())} of {int((R & (nch <= 2)).sum())}")
    row("ransomware, 1-2 channels", R & (nch <= 2))
    row("ransomware, 3-4 channels", R & (nch >= 3))


if __name__ == "__main__":
    main()
