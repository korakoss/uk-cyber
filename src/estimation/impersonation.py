"""Impersonation (#7): what the ticks are, whether they are breaches, and whether they are co-labelled.

Survey items: type5 (Q53A impersonation in emails/online of the organisation or staff = our I channel);
  impersonationhack (Q53B: did any instance involve unauthorised access to files or networks?  1 all, 2 some, 3 no);
  impersonationtkvr (Q53C: did any involve taking over your own website / social media / email accounts?);
  disruptphish5 (Q64B: worst phishing incident was most disruptive because it resulted in impersonation);
  fraud4 (Q88A: times a breach resulted in impersonation using information obtained in the initial breach).
1. Coverage and answers of Q53B/C among impersonation-hit firms.
2. Split impersonation into 'external spoofing' (Q53B and Q53C both no) vs 'intrusive' (either yes): weighted shares;
   breached (W2) rate, worst-cost shares, channels hit, by group (and impersonation-only firms).
3. Co-labelling: impersonation ticked as a CONSEQUENCE of another breach (disruptphish5 = 1 or fraud4 >= 1); share among
   impersonation-hit firms by breadth and by costly / spread-signature status.
4. Breadth recount: drop impersonation ticks that are consequence-type (rule 3) or intrusive-only-via-takeover; how many
   broad firms (3+ channels) stay broad, by costly status.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/impersonation.py
"""

import numpy as np
import pandas as pd

from breach_measures import num as cnum
from broad_interaction import channels, tightened
from latent_on_streams import aligned_raw
from material_breach import data
from type_cooccurrence_structure import SHORT

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
    br = ((oa == 1) | soft) & ~np.isnan(mid) & (nch >= 1)
    nout = np.column_stack([nm(f"outcome{j}") == 1 for j in OUTC]).sum(1)
    sig = br & ((nm("outcome1") == 1) | (nm("outcome6") == 1) | (nout >= 3))
    imp = (C[:, 1] == 1) & ~np.isnan(mid)
    qh, qt = nm("impersonationhack"), nm("impersonationtkvr")
    print(f"1. impersonation-hit firms with cost data: {imp.sum()}")
    for lab, q in (("Q53B unauthorised access", qh), ("Q53C account/site takeover", qt)):
        vc = pd.Series(q[imp]).value_counts(dropna=False).sort_index()
        print(f"   {lab}: " + ", ".join(f"{int(k) if not np.isnan(k) else 'NaN'}: {v}" for k, v in vc.items()))
    yes = lambda q: np.isin(q, [1, 2])
    no = lambda q: q == 3
    intr = imp & (yes(qh) | yes(qt))
    spoof = imp & no(qh) & no(qt)
    unk = imp & ~intr & ~spoof

    def row(lab, m):
        if not m.any():
            print(f"   {lab:34s} n 0")
            return
        b = mid[m]
        ww = w[m]
        print(f"   {lab:34s} n {m.sum():4d}  wshare {w[m].sum() / w[imp].sum():.2f}  breached {np.average(br[m], weights=ww):.2f}"
              f"  spread-sig {np.average(sig[m], weights=ww):.2f}  worst none/<500/500-5k/5k+ "
              f"{np.average(b == 0, weights=ww):.2f}/{np.average((b > 0) & (b < 500), weights=ww):.2f}/"
              f"{np.average((b >= 500) & (b < 5000), weights=ww):.2f}/{np.average(b >= 5000, weights=ww):.2f}"
              f"  mean £{np.average(b, weights=ww):,.0f}  channels {np.average(nch[m], weights=ww):.2f}")
    print("\n2. intrusive vs external-spoofing impersonation")
    row("intrusive (access or takeover)", intr)
    row("  of which unauthorised access", imp & yes(qh))
    row("  of which account takeover", imp & yes(qt))
    row("external spoofing (both no)", spoof)
    row("unknown (not asked / don't know)", unk)
    only = nch == 1
    row("impersonation-only: intrusive", intr & only)
    row("impersonation-only: spoofing", spoof & only)

    print("\n3. impersonation ticked as a consequence of another breach (disruptphish5 = 1 or fraud4 >= 1)")
    cons = imp & ((nm("disruptphish5") == 1) | (cnum(r, "fraud4") >= 1))
    print(f"   answered disruptphish5 {int((imp & ~np.isnan(nm('disruptphish5')) & (nm('disruptphish5') >= 0)).sum())},"
          f" fraud4 {int((imp & ~np.isnan(cnum(r, 'fraud4'))).sum())};  consequence-type ticks {cons.sum()}")
    for lab, m in (("1-2 channels", imp & (nch <= 2)), ("3-4 channels", imp & (nch >= 3)),
                   ("3-4 ch, costly (>= £5k)", imp & (nch >= 3) & (mid >= 5000)), ("3-4 ch, cheap", imp & (nch >= 3) & (mid < 5000)),
                   ("spread signature", imp & sig), ("breached, no signature", imp & br & ~sig), ("not breached", imp & ~br)):
        print(f"   {lab:26s} n {m.sum():4d}  consequence-type {np.mean(cons[m]) if m.any() else np.nan:.2f}"
              f"  intrusive {np.mean(intr[m]) if m.any() else np.nan:.2f}  spoofing {np.mean(spoof[m]) if m.any() else np.nan:.2f}")

    print("\n4. breadth recount without consequence-type impersonation ticks")
    Xc = X.copy()
    Xc[cons, SHORT.index("Imper")] = 0
    ncc = channels(Xc).sum(1)
    for lab, m in (("broad, costly", (nch >= 3) & (mid >= 5000) & ~np.isnan(mid)), ("broad, cheap", (nch >= 3) & (mid < 5000)),
                   ("broad, spread signature", (nch >= 3) & sig)):
        print(f"   {lab:24s} n {m.sum():3d}  still broad {np.mean(ncc[m] >= 3):.2f}")


if __name__ == "__main__":
    main()
