"""How often do firms have MORE THAN ONE costly (breach) event in the year, and does it depend on exposure?

(1) crimecost_bands (total cost of all crimes incl. fraud) vs damage_bands (worst incident): share with the total in a
    higher band than the worst (-> at least a second costly event), by number of types hit and by frailty tier.
(2) Per-type annual cost variables (ransomware, hacking, malware, DoS, takeover, fraud): number of types with a
    NONZERO cost per firm. A firm with 2+ costly types had 2+ breach events. Share with 2+ among firms with any, by
    number of types hit and tier. Value labels printed first (to know which code is 'no cost').
Unweighted counts and survey-weighted shares.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/breach_multiplicity.py
"""

import glob

import numpy as np
import pandas as pd
import pyreadstat

import joint_five_channel as f
from counts_given_frailty import tier_posteriors, TIER
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load

TYPE_COST = ["ranscost_bands", "hackcost_bands", "viruscost_bands", "doscost_bands", "tkvrcost_bands", "fraudcost_bands"]


def main():
    sav = glob.glob("/home/user/uk-cyber/data/raw/*.sav")[0]
    _, meta = pyreadstat.read_sav(sav, metadataonly=True)
    for c in TYPE_COST[:1] + ["fraudcost_bands"]:
        print(f"{c}: {meta.column_names_to_labels.get(c)}\n  labels {meta.variable_value_labels.get(c)}")
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    w = d["weight"].fillna(d["weight"].median()).values
    tier = tier_posteriors().argmax(1)
    ntypes = X.sum(1)
    db = d["band"].values
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values

    groups = [("1 type", ntypes == 1), ("2 types", ntypes == 2), ("3+ types", ntypes >= 3)] + \
             [(f"tier {TIER[c]}", tier == c) for c in range(3)]

    print("\n(1) crimecost total vs worst incident: share with total in a HIGHER band than the worst")
    cc = num("crimecost_bands")
    cc = np.where((cc >= 1) & (cc <= 13), cc, np.nan)
    ok = ~np.isnan(cc) & ~np.isnan(db)
    for lab, m in [("all", np.ones(len(X), bool))] + groups:
        mm = ok & m
        if mm.sum():
            print(f"  {lab:10s} n {mm.sum():4d}  total>worst {np.average(cc[mm] > db[mm], weights=w[mm]):.3f}  "
                  f"(unweighted {int((cc[mm] > db[mm]).sum())})")

    print("\n(2) per-type annual costs: number of types with NONZERO cost")
    V = np.column_stack([num(c) for c in TYPE_COST])
    valid = (V >= 1) & (V <= 13)
    nz_code = 2   # set below from labels if 1 = no cost
    lab1 = meta.variable_value_labels.get("ranscost_bands", {}).get(1.0, "")
    no_cost_is_1 = "no" in str(lab1).lower() or "nothing" in str(lab1).lower() or "0" == str(lab1).strip()
    print(f"  code 1 label: {lab1!r} -> treated as {'no cost' if no_cost_is_1 else 'a positive amount'}")
    pos = valid & (V >= (2 if no_cost_is_1 else 1))
    npos = pos.sum(1)
    anyv = valid.any(1)
    print(f"  firms answering any type-cost question: {anyv.sum()}, with >= 1 nonzero type: {(npos >= 1).sum()}, "
          f"with 2+ nonzero types: {(npos >= 2).sum()}")
    for lab, m in [("all", np.ones(len(X), bool))] + groups:
        mm = m & (npos >= 1)
        if mm.sum():
            print(f"  {lab:10s} firms with >=1 costly type {mm.sum():4d}   share with 2+ {np.average(npos[mm] >= 2, weights=w[mm]):.3f}  "
                  f"(unweighted {int((npos[mm] >= 2).sum())})")
    both = npos >= 2
    if both.any():
        print("  firms with 2+ costly types (type-cost bands, then worst band / crimecost):")
        cols = [c.replace("cost_bands", "") for c in TYPE_COST]
        print(pd.DataFrame(np.where(valid, V, np.nan)[both], columns=cols).assign(worst=db[both], crimecost=cc[both],
              ntypes=ntypes[both], tier=[TIER[t] for t in tier[both]]).to_string())


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


TYPE_MID12 = {1: 50, 2: 175, 3: 375, 4: 750, 5: 1500, 6: 3500, 7: 7500, 8: 15000, 9: 35000, 10: 75000,
              11: 175000, 12: 400000}
FRAUD_MID13 = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000,
               11: 750000, 12: 3000000, 13: 5000000}


def corrected():
    """(2) redone with the right scales: type-cost vars are 12-band with 1 = '< £100' (always positive);
    fraudcost is 13-band with 1 = 'No cost'. Costly type = midpoint >= threshold (any / >= £100 / >= £500).
    Per firm: # costly types, and sum / max of type-cost midpoints (how much the sum exceeds the largest)."""
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    w = d["weight"].fillna(d["weight"].median()).values
    tier = tier_posteriors().argmax(1)
    ntypes = X.sum(1)
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values
    M = []
    for c in TYPE_COST:
        v = num(c)
        mp = TYPE_MID12 if c != "fraudcost_bands" else FRAUD_MID13
        M.append(np.array([mp.get(int(x), np.nan) if not np.isnan(x) else np.nan for x in v]))
    M = np.column_stack(M)
    groups = [("all", np.ones(len(X), bool)), ("1 type", ntypes == 1), ("2 types", ntypes == 2), ("3+ types", ntypes >= 3)] + \
             [(f"tier {TIER[c]}", tier == c) for c in range(3)]
    for thr in (1, 100, 500):
        costly = np.nan_to_num(M) >= thr
        n = costly.sum(1)
        print(f"\ncostly type = cost >= £{thr}:  firms with >=1 {int((n >= 1).sum())}, with 2+ {int((n >= 2).sum())}")
        for lab, m in groups:
            mm = m & (n >= 1)
            if mm.sum():
                s = np.nansum(np.where(costly, M, 0), 1)[mm]
                mx = np.nanmax(np.where(costly, M, 0), 1)[mm]
                print(f"  {lab:10s} n {mm.sum():4d}  share 2+ {np.average(n[mm] >= 2, weights=w[mm]):.3f} "
                      f"(unw {int((n[mm] >= 2).sum()):2d})   weighted sum/max of costs {np.average(s, weights=w[mm]) / np.average(mx, weights=w[mm]):.3f}")


if __name__ == "__main__" and "fix" in __import__("sys").argv:
    corrected()


def engaged_counts():
    """phisheng (Q89C, # phishing attacks someone engaged with) among phishing-flagged firms: coverage, share 0,
    positive values; by size group and most-likely tier; and vs the targeted count (phishcon)."""
    X, _, size = load()
    r = aligned_raw()
    tier = tier_posteriors().argmax(1)
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values
    e, t = num("phisheng"), num("phishcon")
    e = np.where(e >= 0, e, np.nan)
    t = np.where(t >= 0, t, np.nan)
    ph = X[:, 0] == 1
    print(f"phishing firms {ph.sum()}; phisheng answered {int((ph & ~np.isnan(e)).sum())}")

    def show(lab, m):
        v = e[m & ~np.isnan(e)]
        pos = v[v >= 1]
        print(f"  {lab:16s} n {len(v):4d}  P(eng>=1) {np.mean(v >= 1):.3f}  n+ {len(pos):3d}  "
              f"share1|+ {np.mean(pos == 1) if len(pos) else float('nan'):.2f}  "
              + ("quartiles|+ " + "/".join(f"{q:g}" for q in np.percentile(pos, [25, 50, 75])) if len(pos) else ""))

    show("all", ph)
    for g, gm in (("Micro", size == 1), ("Small+", size >= 2)):
        show(g, ph & gm)
        for c in range(3):
            show(f"  {g} {TIER[c]}", ph & gm & (tier == c))
    v = e[ph & (e >= 1)]
    print("\npositive values: " + "  ".join(f"{int(x)}:{int((v == x).sum())}" for x in np.unique(v)))
    both = ph & ~np.isnan(e) & ~np.isnan(t)
    print(f"\nvs targeted (both known, n {both.sum()}): P(eng>=1 | targeted 0) {np.mean(e[both & (t == 0)] >= 1):.3f}, "
          f"| targeted>=1 {np.mean(e[both & (t >= 1)] >= 1):.3f};  corr(log1p) {np.corrcoef(np.log1p(e[both]), np.log1p(t[both]))[0, 1]:.2f}; "
          f"eng > targeted in {int((e[both] > t[both]).sum())} firms")


if __name__ == "__main__" and "eng" in __import__("sys").argv:
    engaged_counts()
