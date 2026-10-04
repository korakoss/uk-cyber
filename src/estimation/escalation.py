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


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


# --- Does frailty alone explain the steep big-breach-by-breadth curve? (2026-10-04) --------------------------------
# Breached firms (W2). 'Big' proxy = tightened worst-incident midpoint >= £5k (also spread signature).
# Frailty measured two ways: (a) per-band latent-class tier (generator.exposure; inferred from the same ticks - circular);
# (b) attack frequency Q54 (intensity; not built from the ticks). Within each frailty level, P(big | breached) by breadth;
# logistic regression big ~ breadth + frailty (+ Small+), weighted, robust se.
# Run: ... escalation.py within

def within():
    import joint_five_channel as f
    from generator import exposure
    from scipy.optimize import minimize as _min
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
    sig_ = (nm("outcome1") == 1) | (nm("outcome6") == 1) | (nout >= 3)
    big = mid >= 5000
    _, post, _, _, _ = exposure()
    tier = post.argmax(1)
    freq = f.load_firms()["freq"].values
    fg = np.where(freq <= 2, 0, np.where(freq <= 3, 1, np.where(freq <= 6, 2, -1)))   # rare / monthly / weekly+
    print("breached firms by breadth (1/2/3/4): P(worst >= £5k) [n];  P(spread signature) in brackets after")

    def table(lab, levels, g):
        print(f"\n{lab}")
        for k, name in levels:
            cells = []
            for b in (1, 2, 3, 4):
                m = br & (g == k) & (nch == b)
                cells.append(f"{np.average(big[m], weights=w[m]):.2f}/{np.average(sig_[m], weights=w[m]):.2f} [{m.sum():3d}]" if m.sum() >= 3 else f"   -      [{m.sum():3d}]")
            print(f"  {name:9s} " + "   ".join(cells))
    table("(a) latent-class tier", [(0, "low"), (1, "mid"), (2, "high")], tier)
    table("(b) attack frequency", [(0, "rare"), (1, "monthly"), (2, "weekly+")], fg)

    def logit(y, Xr, ww):
        Xc = np.column_stack([np.ones(len(y)), Xr])
        nll = lambda b: -(ww * (y * (Xc @ b) - np.logaddexp(0, Xc @ b))).sum()
        o = _min(nll, np.zeros(Xc.shape[1]), method="BFGS")
        p = 1 / (1 + np.exp(-(Xc @ o.x)))
        bread = np.linalg.inv((Xc * (ww * p * (1 - p))[:, None]).T @ Xc)
        g_ = Xc * (ww * (y - p))[:, None]
        return o.x, np.sqrt(np.diag(bread @ (g_.T @ g_) @ bread))
    for lab, fr, ok in (("tier", np.column_stack([tier == 1, tier == 2]), br),
                        ("frequency", np.column_stack([fg == 1, fg == 2]), br & (fg >= 0))):
        for ylab, y in (("worst >= £5k", big), ("spread signature", sig_)):
            Xr = np.column_stack([nch[ok] - 1, fr[ok], size[ok] >= 2]).astype(float)
            b, se = logit(y[ok].astype(float), Xr, w[ok] / w[ok].mean())
            print(f"\nlogit {ylab} ~ breadth + {lab} + Small+ (n {ok.sum()}): breadth per extra channel OR {np.exp(b[1]):.2f}"
                  f" [{np.exp(b[1] - 1.96 * se[1]):.2f}-{np.exp(b[1] + 1.96 * se[1]):.2f}];  {lab} level 2 OR {np.exp(b[2]):.2f}"
                  f" [{np.exp(b[2] - 1.96 * se[2]):.2f}-{np.exp(b[2] + 1.96 * se[2]):.2f}], level 3 OR {np.exp(b[3]):.2f}"
                  f" [{np.exp(b[3] - 1.96 * se[3]):.2f}-{np.exp(b[3] + 1.96 * se[3]):.2f}]")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "within":
    within()


# --- Attempt-backed breadth (2026-10-04) ---------------------------------------------------------------------------
# Rebuild breadth from ticks with evidence of an attempt:
#   phishing: tick counts (receiving phishing is an attempt, not something a breach paints);
#   impersonation: only external spoofing (Q53B and Q53C 'no') not reported as caused by a breach (disruptphish5, fraud4);
#   other serious: takeover / hacking / DoS only with attempts > 0 (unanswered attempt count -> counted);
#     malware and the access/eavesdropping types have no attempt question;
#   ransomware: no attempt question.
# Variant GENEROUS: unverifiable ticks (ransomware, malware, access types) counted; STRICT: dropped.
# Then P(big | breached) by backed breadth within attack-frequency levels; logit big ~ backed breadth + paint
# (observed minus backed channels) + frequency + Small+.  Run: ... escalation.py backed

def backed():
    import joint_five_channel as f
    from scipy.optimize import minimize as _min
    X, size, band, D, oa, w = data()
    r = aligned_raw()
    nm = lambda c: pd.to_numeric(r[c], errors="coerce").values
    C = channels(X)
    nch = C.sum(1)
    lo, hi = tightened(band)
    mid = np.where(hi == 0, 0.0, (lo + hi) / 2)
    soft = np.isin(nm("restore"), [3, 4, 5, 6]) | (nm("impact1") == 1) | (nm("impact4") == 1) | (nm("impact2") == 1)
    br = ((oa == 1) | soft) & ~np.isnan(mid) & (nch >= 1)
    big = mid >= 5000
    freq = f.load_firms()["freq"].values
    fg = np.where(freq <= 2, 0, np.where(freq <= 3, 1, np.where(freq <= 6, 2, -1)))
    T = lambda t: X[:, SHORT.index(t)] == 1
    cons_i = (nm("disruptphish5") == 1) | (cnum(r, "fraud4") >= 1)
    spoof = (nm("impersonationhack") == 3) & (nm("impersonationtkvr") == 3)
    I_b = T("Imper") & spoof & ~cons_i
    att = lambda t, c: T(t) & ~(cnum(r, c) == 0)
    S_checked = att("Takov", "tkvrcount") | att("BankH", "hackcount") | att("DoS", "doscount")
    S_unver = T("Malwr") | T("AcOut") | T("AcStf") | T("Eavsd")
    variants = {"GENEROUS": np.column_stack([T("Phish"), I_b, T("Ransm"), S_checked | S_unver]),
                "STRICT": np.column_stack([T("Phish"), I_b, np.zeros(len(X), bool), S_checked])}

    def logit(y, Xr, ww):
        Xc = np.column_stack([np.ones(len(y)), Xr])
        o = _min(lambda b: -(ww * (y * (Xc @ b) - np.logaddexp(0, Xc @ b))).sum(), np.zeros(Xc.shape[1]), method="BFGS")
        p = 1 / (1 + np.exp(-(Xc @ o.x)))
        bread = np.linalg.inv((Xc * (ww * p * (1 - p))[:, None]).T @ Xc)
        g_ = Xc * (ww * (y - p))[:, None]
        return o.x, np.sqrt(np.diag(bread @ (g_.T @ g_) @ bread))
    fmt = lambda b, s, j: f"{np.exp(b[j]):.2f} [{np.exp(b[j] - 1.96 * s[j]):.2f}-{np.exp(b[j] + 1.96 * s[j]):.2f}]"
    for vlab, Cb in variants.items():
        nb = Cb.sum(1)
        paint = nch - nb
        print(f"\n===== {vlab}: breached firms {br.sum()}; backed breadth 0/1/2/3/4: " +
              "/".join(str(int((br & (nb == k)).sum())) for k in range(5)) + f";  firms with paint >= 1: {int((br & (paint >= 1)).sum())}"
              f" (costly {int((br & big & (paint >= 1)).sum())} of {int((br & big).sum())})")
        print("  P(big | breached) by backed breadth 0/1/2/3/4 [n]:")
        for k, name in [(-9, "all"), (0, "rare"), (1, "monthly"), (2, "weekly+")]:
            g = br if k == -9 else br & (fg == k)
            cells = []
            for b in range(5):
                m = g & (nb == b)
                cells.append(f"{np.average(big[m], weights=w[m]):.2f} [{m.sum():3d}]" if m.sum() >= 3 else f"  -  [{m.sum():3d}]")
            print(f"    {name:8s} " + "  ".join(cells))
        ok = br & (fg >= 0)
        Xr = np.column_stack([nb[ok], paint[ok], fg[ok] == 1, fg[ok] == 2, size[ok] >= 2]).astype(float)
        b, s = logit(big[ok].astype(float), Xr, w[ok] / w[ok].mean())
        print(f"  logit big ~ backed breadth + paint + frequency + Small+ (n {ok.sum()}): backed breadth OR {fmt(b, s, 1)};"
              f" paint (per unbacked channel) OR {fmt(b, s, 2)}")
        Xr2 = np.column_stack([nb[ok], fg[ok] == 1, fg[ok] == 2, size[ok] >= 2]).astype(float)
        b2, s2 = logit(big[ok].astype(float), Xr2, w[ok] / w[ok].mean())
        print(f"  logit big ~ backed breadth + frequency + Small+: backed breadth OR {fmt(b2, s2, 1)}")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "backed":
    backed()


# --- Attempt-backed breadth, CORRECTED (2026-10-04) ----------------------------------------------------------------
# tkvrcount (Q86A: takeover of websites/social/email OR ONLINE BANK accounts) and hackcount (Q85A: unauthorised access to
# files/networks/IMs/calls) exclude attempts that led to fraud or ransomware; doscount (Q87A) has no exclusion.
# A checked tick is BACKED if its count > 0 or the firm had fraud (fraud1-3 >= 1) or ransomware (attempts counted
# elsewhere); POSSIBLY PAINT if count == 0 and no fraud/ransomware; UNVERIFIABLE if count missing.
#   takeover, bank hacking -> tkvrcount;  outsider access, staff access, eavesdropping -> hackcount;  DoS -> doscount.
# Phishing backed; impersonation backed unless intrusive or caused by a breach (follow-up items; partly by construction).
# Malware, ransomware: no attempt question -> GENEROUS counts them backed, STRICT unverifiable (dropped).
# Run: ... escalation.py backed2

def backed2():
    import joint_five_channel as f
    from scipy.optimize import minimize as _min
    X, size, band, D, oa, w = data()
    r = aligned_raw()
    nm = lambda c: pd.to_numeric(r[c], errors="coerce").values
    C = channels(X)
    nch = C.sum(1)
    lo, hi = tightened(band)
    mid = np.where(hi == 0, 0.0, (lo + hi) / 2)
    soft = np.isin(nm("restore"), [3, 4, 5, 6]) | (nm("impact1") == 1) | (nm("impact4") == 1) | (nm("impact2") == 1)
    br = ((oa == 1) | soft) & ~np.isnan(mid) & (nch >= 1)
    big = mid >= 5000
    freq = f.load_firms()["freq"].values
    fg = np.where(freq <= 2, 0, np.where(freq <= 3, 1, np.where(freq <= 6, 2, -1)))
    T = lambda t: X[:, SHORT.index(t)] == 1
    fraud = (np.column_stack([cnum(r, c) for c in ("fraud1", "fraud2", "fraud3")]) >= 1).any(1)
    elsewhere = fraud | T("Ransm")
    status = {}
    for t, c, excl in (("Takov", "tkvrcount", True), ("BankH", "tkvrcount", True), ("AcOut", "hackcount", True),
                       ("AcStf", "hackcount", True), ("Eavsd", "hackcount", True), ("DoS", "doscount", False)):
        n = cnum(r, c)
        tk = T(t)
        b_ = tk & ((n > 0) | (elsewhere if excl else False))
        p_ = tk & (n == 0) & ~(elsewhere if excl else False)
        u_ = tk & ~b_ & ~p_
        status[t] = (b_, p_, u_)
        print(f"  {t:5s} ticked {tk.sum():4d}: backed {b_.sum():4d}, possibly paint {p_.sum():3d}, unverifiable {u_.sum():3d}"
              f"  | among breached costly: ticked {(tk & br & big).sum()}, possibly paint {(p_ & br & big).sum()}")
    cons_i = (nm("disruptphish5") == 1) | (cnum(r, "fraud4") >= 1)
    intr = (np.isin(nm("impersonationhack"), [1, 2])) | (np.isin(nm("impersonationtkvr"), [1, 2]))
    I_b = T("Imper") & ~cons_i & ~intr
    print(f"  Imper ticked {T('Imper').sum():4d}: backed (spoofing, not breach-caused) {I_b.sum()}, consequence/intrusive {(T('Imper') & ~I_b).sum()}")
    S_checked_b = np.column_stack([status[t][0] for t in status]).any(1)
    S_checked_u = np.column_stack([status[t][2] for t in status]).any(1)
    variants = {"GENEROUS": np.column_stack([T("Phish"), I_b, T("Ransm"), S_checked_b | S_checked_u | T("Malwr")]),
                "STRICT": np.column_stack([T("Phish"), I_b, np.zeros(len(X), bool), S_checked_b])}

    def logit(y, Xr, ww):
        Xc = np.column_stack([np.ones(len(y)), Xr])
        o = _min(lambda b: -(ww * (y * (Xc @ b) - np.logaddexp(0, Xc @ b))).sum(), np.zeros(Xc.shape[1]), method="BFGS")
        p = 1 / (1 + np.exp(-(Xc @ o.x)))
        bread = np.linalg.inv((Xc * (ww * p * (1 - p))[:, None]).T @ Xc)
        g_ = Xc * (ww * (y - p))[:, None]
        return o.x, np.sqrt(np.diag(bread @ (g_.T @ g_) @ bread))
    fmt = lambda b, s, j: f"{np.exp(b[j]):.2f} [{np.exp(b[j] - 1.96 * s[j]):.2f}-{np.exp(b[j] + 1.96 * s[j]):.2f}]"
    for vlab, Cb in variants.items():
        nb = Cb.sum(1)
        rest = nch - nb
        print(f"\n===== {vlab}: backed breadth among breached 0/1/2/3/4: " + "/".join(str(int((br & (nb == k)).sum())) for k in range(5)) +
              f"; breached firms with any non-backed channel {int((br & (rest >= 1)).sum())} (costly {int((br & big & (rest >= 1)).sum())} of {int((br & big).sum())})")
        cells = []
        for b in range(5):
            m = br & (nb == b)
            cells.append(f"{np.average(big[m], weights=w[m]):.2f} [{m.sum():3d}]" if m.sum() >= 3 else f"  -  [{m.sum():3d}]")
        print("  P(big | breached) by backed breadth 0-4: " + "  ".join(cells))
        ok = br & (fg >= 0)
        b, s = logit(big[ok].astype(float), np.column_stack([nb[ok], rest[ok], fg[ok] == 1, fg[ok] == 2, size[ok] >= 2]).astype(float), w[ok] / w[ok].mean())
        print(f"  logit big ~ backed + non-backed channels + frequency + Small+ (n {ok.sum()}): backed OR {fmt(b, s, 1)}; non-backed OR {fmt(b, s, 2)}")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "backed2":
    backed2()
