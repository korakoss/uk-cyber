"""Escalation hypothesis: any material episode can escalate into a spreading intrusion, which costs far more AND makes
the firm tick extra attack types (co-labelling). So 'broad' (3+ channels) is partly an outcome, not exposure.

Breached = W2 (outcome flag OR restore >= 1 day / staff stopped / recovery costs / revenue loss).
Spread signature = systems corrupted OR money stolen OR 3+ distinct material outcomes.

1. signature across ALL breached firms by channel count: rate, and cost of signature vs non-signature events. Key
   check: do signature events cost the same whether the firm is labelled narrow or broad?
2. p_esc = signature firms / expected episodes (Lam from the W2 Poisson fit applied to each firm's hit channels),
   overall and by tier, size, entry channel (phishing hit or not; targeted phishing or not).
3. Attempt-backed ticks: takeover / hacking / DoS ticked with an answered attempt count of 0 = unbacked. Unbacked-tick
   rate by signature; channel breadth recounted without unbacked ticks; how many broad firms stay broad.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/escalation.py
"""

import os

import numpy as np
import pandas as pd

import poisson_cost_model as pcm
from breach_measures import num as cnum
from broad_interaction import channels, tightened
from counts_given_frailty import tier_posteriors, TIER
from latent_on_streams import aligned_raw
from material_breach import data
from type_cooccurrence_structure import SHORT

W2_PARAMS = "/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_params_w2.npy"
OUTC = (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13)


def w2_params():
    if os.path.exists(W2_PARAMS):
        return np.load(W2_PARAMS)
    D = pcm.setup()
    r = aligned_raw()
    nm = lambda c: pd.to_numeric(r[c], errors="coerce").values[D["idx"]]
    soft = np.isin(nm("restore"), [3, 4, 5, 6]) | (nm("impact1") == 1) | (nm("impact4") == 1) | (nm("impact2") == 1)
    D["oa"] = ((D["oa"] == 1) | soft).astype(float)
    p, _ = pcm.fit(D)
    np.save(W2_PARAMS, p)
    return p


def shares(mid, w, m):
    b = mid[m]
    ww = w[m]
    return [np.average(b == 0, weights=ww), np.average((b > 0) & (b < 500), weights=ww),
            np.average((b >= 500) & (b < 5000), weights=ww), np.average(b >= 5000, weights=ww)]


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
    tier = tier_posteriors().argmax(1)
    hitm = (nch >= 1) & ~np.isnan(mid)

    print("1. spread signature across breached firms, by channel count")
    print("   channels  n hit  n breached  P(sig | breached)  | worst none/<500/500-5k/5k+: signature | non-signature breached")
    for k in (1, 2, 3, 4):
        h = hitm & (nch == k)
        b = br & (nch == k)
        s, ns = sig & (nch == k), b & ~sig
        print(f"   {k}        {h.sum():5d}  {b.sum():5d}       {np.average(sig[b], weights=w[b]):.2f} ({s.sum():3d})        | " +
              " ".join(f"{x:.2f}" for x in shares(mid, w, s)) + " | " + (" ".join(f"{x:.2f}" for x in shares(mid, w, ns)) if ns.any() else "-"))
    for lab, a, b_ in (("narrow (1-2)", 1, 2), ("broad (3-4)", 3, 4)):
        s = sig & (nch >= a) & (nch <= b_)
        print(f"   signature events {lab:13s} n {s.sum():3d}  weighted mean worst £{np.average(mid[s], weights=w[s]):,.0f}"
              f"  median £{np.median(mid[s]):,.0f}  P(>=£5k) {np.average(mid[s] >= 5000, weights=w[s]):.2f}")
    costly = br & (mid >= 5000)
    print(f"   costly (>= £5k) breached firms {costly.sum()}: share with signature {np.average(sig[costly], weights=w[costly]):.2f};"
          f" share broad {np.average(nch[costly] >= 3, weights=w[costly]):.2f}")

    print("\n2. escalation probability per expected episode (W2 Poisson rates applied to each firm's hit channels)")
    p = w2_params()
    P = pcm.unpack(p)
    Lc = C * P["lam"][None] * np.exp(P["bt"][tier] + P["bs"] * (size >= 2))[:, None]
    L = Lc.sum(1)
    print("   rates per hit (low tier, Micro): " + ", ".join(f"{c} {v:.3f}" for c, v in zip("PIRS", P["lam"])) +
          f"; tier x{np.exp(P['bt'][1]):.2f}/x{np.exp(P['bt'][2]):.2f}; Small+ x{np.exp(P['bs']):.2f}")
    phish_t = cnum(r, "phishcon") >= 1
    groups = [("all hit firms", hitm), ("narrow", hitm & (nch <= 2)), ("broad", hitm & (nch >= 3))] + \
             [(f"tier {t}", hitm & (tier == k)) for k, t in enumerate(TIER)] + \
             [("Micro", hitm & (size == 1)), ("Small+", hitm & (size >= 2)),
              ("phishing hit", hitm & (C[:, 0] == 1)), ("no phishing", hitm & (C[:, 0] == 0)),
              ("targeted phishing >= 1", hitm & phish_t), ("phishing, 0 targeted", hitm & (C[:, 0] == 1) & (cnum(r, "phishcon") == 0))]
    for lab, m in groups:
        ns, ne = (w[m] * sig[m]).sum(), (w[m] * L[m]).sum()
        print(f"   {lab:24s} n {m.sum():4d}  signature firms {int(sig[m].sum()):3d}  expected episodes (weighted) {ne:7.1f}"
              f"  p_esc {ns / ne:.3f}  | breached firms {int(br[m].sum()):3d}, sig share of breached {np.average(sig[m & br], weights=w[m & br]):.2f}")

    print("\n3. attempt-backed ticks (takeover / hacking / DoS ticked, attempt count answered)")
    att = [("takeover", "Takov", "tkvrcount"), ("hacking", "BankH", "hackcount"), ("DoS", "DoS", "doscount")]
    unb = np.zeros(len(X), bool)
    Xc = X.copy()
    for lab, t, ac in att:
        A = cnum(r, ac)
        j = SHORT.index(t)
        tk = (X[:, j] == 1) & ~np.isnan(A)
        u = tk & (A == 0)
        unb |= u
        Xc[u, j] = 0
        cells = []
        for glab, g in (("signature", sig), ("breached, no sig", br & ~sig), ("not breached", hitm & ~br)):
            m = tk & g
            cells.append(f"{glab}: {m.sum():3d} ticked, unbacked {np.mean(A[m] == 0) if m.any() else np.nan:.2f}")
        print(f"   {lab:9s} " + " | ".join(cells))
    Cc = channels(Xc)
    ncc = Cc.sum(1)
    for glab, g in (("signature", sig), ("breached, no sig", br & ~sig), ("not breached", hitm & ~br)):
        m = g & (nch >= 3)
        print(f"   broad {glab:17s} n {m.sum():3d}: still broad after dropping unbacked ticks {np.mean(ncc[m] >= 3) if m.any() else np.nan:.2f}"
              f";  any unbacked tick {np.mean(unb[m]) if m.any() else np.nan:.2f}")
    print("   note: impersonation, ransomware, malware and the account/eavesdropping types have no attempt count, so"
          " co-labelling there is invisible to this check.")


if __name__ == "__main__":
    main()
