"""Empirical per-type attack counts for the non-phishing channels.

Per-type count questions in the survey: ranssoft (ransomware), doscount (DoS), tkvrcount (account takeover),
hackcount (hacking / unauthorised access). Exact answer where >= 0, else the '_bands' answer (1=None,
2=1, 3=2-3, 4=4-5, 5=6-10, 6=11-20, 7=21-50, 8=51-100, 9=100+). Tabulated among firms flagged with the
matching type(s): coverage, exact value:#firms, band:#firms for the rest. Impersonation has no count
question: global freq answer among impersonation-only firms shown instead (1 once .. 6 several/day).
Unweighted firm counts.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/channel_counts_empirical.py
"""

import numpy as np
import pandas as pd

import joint_five_channel as f
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load, SHORT

VARS = [("ranssoft", ["Ransm"]), ("doscount", ["DoS"]), ("tkvrcount", ["Takov"]),
        ("hackcount", ["BankH", "AcOut", "AcStf"])]
BAND = {1: "0", 2: "1", 3: "2-3", 4: "4-5", 5: "6-10", 6: "11-20", 7: "21-50", 8: "51-100", 9: "100+"}


def main():
    X, w, size = load()
    r = aligned_raw()
    num = lambda c: pd.to_numeric(r[c], errors="coerce")
    for var, types in VARS:
        hit = X[:, [SHORT.index(t) for t in types]].max(1) == 1
        ex = num(var).where(lambda s: (s >= 0) & (s < 997)).values
        bd = num(var + "_bands").where(lambda s: s.between(1, 9)).values
        print(f"\n{var}  (flagged {'/'.join(types)}: {hit.sum()} firms; exact {int((hit & ~np.isnan(ex)).sum())}, "
              f"band only {int((hit & np.isnan(ex) & ~np.isnan(bd)).sum())}, missing {int((hit & np.isnan(ex) & np.isnan(bd)).sum())})")
        v = ex[hit & ~np.isnan(ex)]
        print("  exact:  " + "  ".join(f"{int(x)}:{int((v == x).sum())}" for x in np.unique(v)))
        b = bd[hit & np.isnan(ex) & ~np.isnan(bd)]
        if len(b):
            print("  bands:  " + "  ".join(f"{BAND[int(x)]}:{int((b == x).sum())}" for x in np.unique(b)))
        nh = (~hit) & ~np.isnan(ex) & (ex > 0)
        print(f"  (firms NOT flagged but reporting count>0: {int(nh.sum())})")

    d = f.load_firms()
    fq = d["freq"].values
    io = (X[:, SHORT.index("Imper")] == 1) & (X.sum(1) == 1) & ~np.isnan(fq)
    print(f"\nImpersonation-only firms, global freq answer (n={io.sum()}):")
    lab = {1: "once", 2: "<monthly", 3: "monthly", 4: "weekly", 5: "daily", 6: "several/day"}
    for g, gm in (("Micro", size == 1), ("Small+", size >= 2)):
        m = io & gm
        print(f"  {g:7s} n={m.sum():3d}  " + "  ".join(f"{lab[k]}:{int((fq[m] == k).sum())}" for k in range(1, 7)))


if __name__ == "__main__":
    main()
