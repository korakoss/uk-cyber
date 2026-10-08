"""Per-attack costs read directly off the data (no fitted cost law).

Observed: damage_bands = cost of the single most disruptive incident (worst), disrupta = its type (D: 1 phishing,
2 impersonation, 3 ransomware, 4 other serious). Bands: 1 none, 2 <£100, 3 £100-500, 4 £500-1k, 5 £1k-5k,
6 £5k-10k, 7 £10k-20k, 8 £20k-50k, 9 £50k-100k, 10 £100k-500k. Shown as none / <£500 / £500-5k / £5k-20k / £20k+.
Survey-weighted shares, unweighted n.

A. Phishing cost vs ENGAGEMENT (phisheng, Q89C: # phishing attacks someone engaged with). Phishing-only firms
   (no other type, so the worst incident is phishing): cost by engaged count 0 / 1 / 2-3 / 4+, and for engaged 0
   by targeted count (phishcon 0 vs >= 1). If cost arises only through engagement, engaged 0 -> ~no cost.
B. Single-attack cost distributions (worst incident = the one attack of that channel):
   phishing:      phishing-only firms with exactly 1 engaged attack (non-engaged attacks assumed costless)
   impersonation: impersonation-only firms answering freq 'once'
   ransomware:    worst incident ransomware and Cybercrime_ranssum == 1
   other serious: worst incident other serious and hacksum + virussum + dossum == 1
   each vs all firms whose worst incident is that channel.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/cost_per_attack.py
"""

import numpy as np
import pandas as pd

import joint_five_channel as f
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load, SHORT

GROUPS = [(1, 1), (2, 3), (4, 5), (6, 7), (8, 10)]
GLAB = "none / <£500 / £500-5k / £5k-20k / £20k+"


def shares(band, w, m):
    m = m & ~np.isnan(band)
    if m.sum() == 0:
        return m.sum(), None
    return m.sum(), [np.average((band[m] >= a) & (band[m] <= b), weights=w[m]) for a, b in GROUPS]


def show(label, band, w, m):
    n, s = shares(band, w, m)
    body = "  ".join(f"{v:5.2f}" for v in s) if s else "-"
    print(f"  {label:48s} n {n:4d}   {body}")


def main():
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values
    band, D, fq = d["band"].values, d["D"].values, d["freq"].values
    w = d["weight"].fillna(d["weight"].median()).values
    eng = num("phisheng")
    eng = np.where(eng >= 0, eng, np.nan)
    con = num("phishcon")
    con = np.where(con >= 0, con, np.nan)
    only = X.sum(1) == 1
    p_only = only & (X[:, 0] == 1)
    i_only = only & (X[:, SHORT.index("Imper")] == 1)

    print(f"shares: {GLAB}")
    print("\nA. PHISHING-ONLY FIRMS: worst-incident cost by engaged count")
    for lab, m in (("engaged 0", eng == 0), ("engaged 1", eng == 1), ("engaged 2-3", (eng >= 2) & (eng <= 3)),
                   ("engaged 4+", eng >= 4), ("engaged unknown", np.isnan(eng))):
        show(lab, band, w, p_only & m)
    print("  engaged 0, split by targeted count:")
    for lab, m in (("   targeted 0", con == 0), ("   targeted >= 1", con >= 1), ("   targeted unknown", np.isnan(con))):
        show(lab, band, w, p_only & (eng == 0) & m)
    for g, gm in (("Micro", size == 1), ("Small+", size >= 2)):
        show(f"{g}: engaged 0", band, w, p_only & gm & (eng == 0))
        show(f"{g}: engaged 1", band, w, p_only & gm & (eng == 1))

    rs = num("Cybercrime_ranssum")
    V = np.column_stack([num(c) for c in ("Cybercrime_hacksum", "Cybercrime_virussum", "Cybercrime_dossum")])
    V = np.where(V >= 1, V, np.nan)
    ossum = np.where(np.isnan(V).all(1), np.nan, np.nansum(V, 1))
    print("\nB. SINGLE-ATTACK COST DISTRIBUTIONS vs all firms whose worst incident is that channel")
    rows = [
        ("phishing: phishing-only, 1 engaged", p_only & (eng == 1), D == 1),
        ("impersonation: imp-only, freq once", i_only & (fq == 1), D == 2),
        ("ransomware: worst R, ranssum 1", (D == 3) & (rs == 1), D == 3),
        ("other serious: worst S, count 1", (D == 4) & (ossum == 1), D == 4),
    ]
    for lab, single, allw in rows:
        show(lab, band, w, single)
        show("   (all with this worst channel)", band, w, allw)
        for g, gm in (("Micro", size == 1), ("Small+", size >= 2)):
            show(f"   single, {g}", band, w, single & gm)


if __name__ == "__main__":
    main()
