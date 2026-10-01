"""Breach layer, channel by channel: is the survey's 'success' count the event that carries cost?

Per channel with a success count: among flagged firms,
  - coverage of the success question and its distribution (P(>=1), values);
  - the channel's own annual cost question (where it exists) by success count 0 / 1 / 2+: answered n and bands;
  - the worst-incident band by success count 0 / 1 / 2+ (firms hit only by that channel's group where possible).
Channels: phishing (phisheng, engaged), takeover (tkvrsuc), DoS (dossoft), malware (virussoft),
ransomware (ranssoft = ransom demanded). Unweighted.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/breach_layer.py
"""

import numpy as np
import pandas as pd

import joint_five_channel as f
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load, SHORT

CH = [("phishing", "Phish", "phisheng", None), ("takeover", "Takov", "tkvrsuc", "tkvrcost_bands"),
      ("DoS", "DoS", "dossoft", "doscost_bands"), ("malware", "Malwr", "virussoft", "viruscost_bands"),
      ("ransomware", "Ransm", "ranssoft", "ranscost_bands")]
WG = [(1, 1), (2, 3), (4, 5), (6, 13)]   # worst band: none / <£500 / £500-5k / £5k+
TG = [(1, 3), (4, 6), (7, 12)]           # type cost (12-band, 1 = <£100): <£500 / £500-5k / £5k+


def main():
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    band = d["band"].values
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values
    for lab, flag, sv, cv in CH:
        hit = X[:, SHORT.index(flag)] == 1
        only = hit & (X.sum(1) == 1)
        s = num(sv)
        s = np.where((s >= 0) & (s < 997), s, np.nan)
        ans = hit & ~np.isnan(s)
        pos = s[ans & (s >= 1)]
        print(f"\n{lab.upper()}: hit {hit.sum()}, success question answered {ans.sum()}, P(success>=1) "
              f"{np.mean(s[ans] >= 1):.3f}; values " + "  ".join(f"{int(x)}:{int((pos == x).sum())}" for x in np.unique(pos)))
        levels = [("succ 0", s == 0), ("succ 1", s == 1), ("succ 2+", s >= 2)]
        if cv:
            c = num(cv)
            c = np.where((c >= 1) & (c <= 12), c, np.nan)
            print(f"  own annual cost ({cv}) answered by hit firms: {int((hit & ~np.isnan(c)).sum())}")
            for ll, lm in levels:
                m = ans & lm
                cc = c[m & ~np.isnan(c)]
                print(f"    {ll:8s} n {m.sum():4d}  cost answered {len(cc):3d}  <£500 / £500-5k / £5k+  "
                      + " ".join(f"{np.mean((cc >= a) & (cc <= b)) if len(cc) else float('nan'):.2f}" for a, b in TG))
        for nm, base in (("all hit", ans), (f"{lab}-only", ans & only)):
            print(f"  worst-incident band by success, {nm}:  none / <£500 / £500-5k / £5k+")
            for ll, lm in levels:
                m = base & lm & ~np.isnan(band)
                b = band[m]
                print(f"    {ll:8s} n {m.sum():4d}  " + " ".join(f"{np.mean((b >= a) & (b <= c_)) if len(b) else float('nan'):.2f}" for a, c_ in WG))


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def rates_by_size_tier():
    """P(breach >= 1 | hit, success answered) per channel by size band, by tier, and by size group x tier.
    Plus likelihood-ratio tests from logistic regressions: size (4 bands) and tier (3), each added to the other."""
    from scipy.optimize import minimize
    from scipy.stats import chi2
    from counts_given_frailty import tier_posteriors, TIER
    X, _, size = load()
    r = aligned_raw()
    tier = tier_posteriors().argmax(1)
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values
    SL = {1: "Micro", 2: "Small", 3: "Medium", 4: "Large"}

    def fit(y, cols):
        Z = np.column_stack([np.ones(len(y))] + cols) if cols else np.ones((len(y), 1))
        def nll(b):
            eta = Z @ b
            return -(y * eta - np.logaddexp(0, eta)).sum()
        return -minimize(nll, np.zeros(Z.shape[1]), method="BFGS").fun

    for lab, flag, sv, _ in CH:
        s = num(sv)
        m = (X[:, SHORT.index(flag)] == 1) & (s >= 0) & (s < 997)
        y = (s[m] >= 1).astype(float)
        print(f"\n{lab.upper()}: n {m.sum()}, P(breach>=1) {y.mean():.3f}")
        print("  by size:  " + "   ".join(f"{SL[k]} {np.mean(y[size[m] == k]):.2f} (n {int((size[m] == k).sum())})"
                                         for k in range(1, 5) if (size[m] == k).any()))
        print("  by tier:  " + "   ".join(f"{TIER[c]} {np.mean(y[tier[m] == c]):.2f} (n {int((tier[m] == c).sum())})"
                                         for c in range(3) if (tier[m] == c).any()))
        for g, gm in (("Micro", size[m] == 1), ("Small+", size[m] >= 2)):
            print(f"  {g:6s} x tier: " + "   ".join(
                f"{TIER[c]} {np.mean(y[gm & (tier[m] == c)]):.2f} (n {int((gm & (tier[m] == c)).sum())})"
                for c in range(3) if (gm & (tier[m] == c)).any()))
        sz = [(size[m] == k).astype(float) for k in (2, 3, 4) if (size[m] == k).sum() > 0]
        tr = [(tier[m] == c).astype(float) for c in (1, 2) if (tier[m] == c).sum() > 0]
        try:
            l_both, l_s, l_t = fit(y, sz + tr), fit(y, sz), fit(y, tr)
            p_size = chi2.sf(2 * (l_both - l_t), len(sz))
            p_tier = chi2.sf(2 * (l_both - l_s), len(tr))
            print(f"  LR tests: size given tier p {p_size:.3f};  tier given size p {p_tier:.3f}")
        except Exception as e:
            print("  LR tests failed:", e)


if __name__ == "__main__" and "rates" in __import__("sys").argv:
    rates_by_size_tier()


def cost_concepts():
    """Per-type breach cost (12-band, asked only of breached firms) vs worst-incident cost (13-band) for the same
    firms. Labels of the per-type cost components first. Per channel: firms with both; midpoint ratio type/worst,
    band-level agreement (converted to £ midpoints), split by whether the worst incident (disrupta) was this channel."""
    import glob
    import pyreadstat
    sav = glob.glob("/home/user/uk-cyber/data/raw/*.sav")[0]
    _, meta = pyreadstat.read_sav(sav, metadataonly=True)
    for c in ("ranscosta", "ranscostb", "ranscost_bands", "tkvrcosta", "tkvrcostb", "damage_bands"):
        if c in meta.column_names_to_labels:
            print(f"{c:15s} {str(meta.column_names_to_labels[c])[:200]}")
    TM = {1: 50, 2: 175, 3: 375, 4: 750, 5: 1500, 6: 3500, 7: 7500, 8: 15000, 9: 35000, 10: 75000, 11: 175000, 12: 400000}
    WM = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000}
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    band, disr = d["band"].values, d["disrupta"].values
    DCODE = {"takeover": [11], "DoS": [3], "malware": [2], "ransomware": [1]}
    for lab, flag, sv, cv in CH[1:]:
        c = pd.to_numeric(r[cv], errors="coerce").values
        m = (c >= 1) & (c <= 12) & ~np.isnan(band)
        tm = np.array([TM[int(x)] for x in c[m]])
        wm = np.array([WM[int(x)] for x in band[m]])
        own = np.isin(disr[m], DCODE[lab])
        print(f"\n{lab.upper()}: firms with both {m.sum()} (worst incident = this channel: {own.sum()})")
        for nm, sel in (("worst = this channel", own), ("worst = other", ~own)):
            if sel.sum() == 0:
                continue
            ratio = tm[sel] / np.maximum(wm[sel], 25)
            print(f"  {nm:22s} n {sel.sum():3d}  type-cost > worst {np.mean(tm[sel] > wm[sel]):.2f}  "
                  f"median ratio {np.median(ratio):.2f}   sum type £{tm[sel].sum():,.0f} vs sum worst £{wm[sel].sum():,.0f}")
            print("      pairs (type £ / worst £): " + ", ".join(f"{a}/{b}" for a, b in zip(tm[sel], wm[sel])))


if __name__ == "__main__" and "concepts" in __import__("sys").argv:
    cost_concepts()


def cost_per_breach():
    """Cost per breach on the worst-incident concept (damage_bands), per channel, from firms where the worst incident
    is a single breach of that channel:
      phishing:      phishing-only firms with exactly 1 engaged attack
      ransomware:    worst incident ransomware (D 3) and exactly 1 ransom demand (ranssoft)
      other serious: worst incident other serious (D 4) and exactly 1 success summed over tkvrsuc + dossoft + virussoft
      impersonation: impersonation-only firms (collapsed: annual cost given hit, zeros included)
    Survey-weighted band shares, weighted mean at band midpoints (and without the top observed band), Micro vs Small+."""
    WM = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000}
    G = [(1, 1), (2, 3), (4, 5), (6, 7), (8, 13)]
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    band, D = d["band"].values, d["D"].values
    w = d["weight"].fillna(d["weight"].median()).values
    num = lambda c: np.where((pd.to_numeric(r[c], errors="coerce").values >= 0) & (pd.to_numeric(r[c], errors="coerce").values < 997),
                             pd.to_numeric(r[c], errors="coerce").values, np.nan)
    only = X.sum(1) == 1
    succ = np.column_stack([num(c) for c in ("tkvrsuc", "dossoft", "virussoft")])
    s_os = np.where(np.isnan(succ).all(1), np.nan, np.nansum(succ, 1))
    cells = {
        "phishing (1 engaged)": only & (X[:, 0] == 1) & (num("phisheng") == 1),
        "ransomware (1 demand)": (D == 3) & (num("ranssoft") == 1),
        "other serious (1 success)": (D == 4) & (s_os == 1),
        "impersonation (collapsed)": only & (X[:, SHORT.index("Imper")] == 1),
    }
    print("worst-incident cost, shares none / <£500 / £500-5k / £5k-20k / £20k+;  weighted mean £ (midpoints) [without top band]")
    for lab, base in cells.items():
        base = base & ~np.isnan(band)
        for g, gm in (("all", np.ones(len(X), bool)), ("Micro", size == 1), ("Small+", size >= 2)):
            m = base & gm
            if m.sum() == 0:
                continue
            b, ww = band[m], w[m]
            sh = [np.average((b >= a) & (b <= c), weights=ww) for a, c in G]
            mid = np.array([WM[int(x)] for x in b])
            notop = b < 10
            mean_nt = np.average(mid[notop], weights=ww[notop]) if notop.any() else float("nan")
            print(f"  {lab:27s} {g:6s} n {m.sum():3d}  " + " ".join(f"{v:4.2f}" for v in sh) +
                  f"   mean £{np.average(mid, weights=ww):8,.0f} [£{mean_nt:7,.0f}]  top-band firms {int((b == 10).sum())}")


if __name__ == "__main__" and "cpb" in __import__("sys").argv:
    cost_per_breach()


EDGES = {2: (1, 100), 3: (100, 500), 4: (500, 1000), 5: (1000, 5000), 6: (5000, 10000), 7: (10000, 20000),
         8: (20000, 50000), 9: (50000, 100000), 10: (100000, 500000), 11: (500000, 1e6), 12: (1e6, 5e6), 13: (5e6, 1e9)}


def single_breach_cells():
    """(channel label, band, weight, small+ flag) for the four single-breach cells of cost_per_breach()."""
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    band, D = d["band"].values, d["D"].values
    w = d["weight"].fillna(d["weight"].median()).values
    num = lambda c: np.where((pd.to_numeric(r[c], errors="coerce").values >= 0) & (pd.to_numeric(r[c], errors="coerce").values < 997),
                             pd.to_numeric(r[c], errors="coerce").values, np.nan)
    only = X.sum(1) == 1
    succ = np.column_stack([num(c) for c in ("tkvrsuc", "dossoft", "virussoft")])
    s_os = np.where(np.isnan(succ).all(1), np.nan, np.nansum(succ, 1))
    cells = [only & (X[:, 0] == 1) & (num("phisheng") == 1), (D == 3) & (num("ranssoft") == 1),
             (D == 4) & (s_os == 1), only & (X[:, SHORT.index("Imper")] == 1)]
    ch = np.full(len(X), -1)
    for k, m in enumerate(cells):
        ch[m & ~np.isnan(band) & (ch < 0)] = k
    keep = ch >= 0
    return ch[keep], band[keep].astype(int), w[keep], (size[keep] >= 2).astype(float)


def size_effect():
    """Is the size effect on cost per breach common across channels? Pooled single-breach firms.
    Two parts: (a) P(no cost): logistic, channel intercepts + size (common) vs channel x size; (b) positive costs:
    interval-censored lognormal on the band edges, channel-specific mu, common sigma, + size shift (common) vs
    channel-specific shifts. Weighted (normalised) pseudo-likelihood; LR tests approximate. Common shift as cost factor."""
    from scipy.optimize import minimize
    from scipy.stats import chi2, norm
    ch, band, w, sm = single_breach_cells()
    w = w / w.mean()
    K = 4
    names = ["phishing", "ransomware", "other serious", "impersonation"]
    print(f"single-breach firms: {len(ch)}  (" + ", ".join(f"{names[k]} {int((ch == k).sum())}" for k in range(K)) + ")")
    C = np.eye(K)[ch]

    y0 = (band == 1).astype(float)
    def logit_ll(Z):
        o = minimize(lambda b: -(w * (y0 * (Z @ b) - np.logaddexp(0, Z @ b))).sum(), np.zeros(Z.shape[1]), method="BFGS")
        return -o.fun, o.x
    keep_z = np.ones(len(ch), bool)
    lz0, _ = logit_ll(C)
    lz1, bz1 = logit_ll(np.column_stack([C, sm]))
    lz2, _ = logit_ll(np.column_stack([C, C * sm[:, None]]))
    print(f"\n(a) P(no cost): common size log-odds {bz1[-1]:+.2f} (odds x{np.exp(bz1[-1]):.2f});  "
          f"size p {chi2.sf(2 * (lz1 - lz0), 1):.3f};  separate vs common p {chi2.sf(2 * (lz2 - lz1), K - 1):.3f}")

    pos = band >= 2
    lo = np.log([EDGES[b][0] for b in band[pos]])
    hi = np.log([EDGES[b][1] for b in band[pos]])
    Cp, sp, wp = C[pos], sm[pos], w[pos]

    def ic_ll(mu, s):
        return (wp * np.log(np.maximum(norm.cdf((hi - mu) / s) - norm.cdf((lo - mu) / s), 1e-300))).sum()

    def fit(design):
        Z = design
        o = minimize(lambda p: -ic_ll(Z @ p[:-1], np.exp(p[-1])), np.r_[np.full(Z.shape[1], 6.0) * 0 + 6.0, 0.5],
                     method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(lambda p: -ic_ll(Z @ p[:-1], np.exp(p[-1])), o.x, method="BFGS")
        return -o.fun, o.x
    l0, _ = fit(Cp)
    l1, p1 = fit(np.column_stack([Cp, sp]))
    l2, p2 = fit(np.column_stack([Cp, Cp * sp[:, None]]))
    print(f"(b) positive costs (n {pos.sum()}): common size shift on log cost {p1[K]:+.2f} -> cost factor x{np.exp(p1[K]):.1f}, "
          f"sigma {np.exp(p1[-1]):.2f};  size p {chi2.sf(2 * (l1 - l0), 1):.4f};  separate vs common p {chi2.sf(2 * (l2 - l1), K - 1):.3f}")
    print("    per-channel shifts (separate model): " + ", ".join(f"{names[k]} {p2[K + k]:+.2f} (x{np.exp(p2[K + k]):.1f}, n+ {int((ch[pos] == k).sum())})"
                                                          for k in range(K)))
    print("    channel medians (Micro, common model): " + ", ".join(f"{names[k]} £{np.exp(p1[k]):,.0f}" for k in range(K)))


if __name__ == "__main__" and "size" in __import__("sys").argv:
    size_effect()


def tail_check():
    """How fragile are the per-breach means? Base model on the pooled single-breach firms: P(cost > 0) per channel
    (weighted share), positive costs interval-censored lognormal with channel mu, common sigma, common Small+ shift.
    Mean per breach = P(>0) * exp(mu + shift + sigma^2/2). Variants:
      drop-k   : drop each top-band (£100k-500k) firm, and all of them
      sep-sigma: channel-specific sigma
      cap500k  : lognormal renormalised below £500k (nothing above the offered-but-empty bands)
      sigma-all: sigma fixed at the value fitted on ALL attacked firms' worst incidents (channel of worst incident as mu)."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    names = ["phishing", "ransomware", "other serious", "impersonation"]
    K = 4
    ch, band, w, sm = single_breach_cells()
    w = w / w.mean()

    def fit(ch, band, w, sm, sep_sigma=False, fixed_sigma=None):
        pos = band >= 2
        lo = np.log([EDGES[b][0] for b in band[pos]])
        hi = np.log([EDGES[b][1] for b in band[pos]])
        C, s, ww = np.eye(K)[ch[pos]], sm[pos], w[pos]
        def nll(p):
            mu = C @ p[:K] + p[K] * s
            if fixed_sigma is not None:
                sg = np.full(len(mu), fixed_sigma)
            elif sep_sigma:
                sg = C @ np.exp(p[K + 1:K + 1 + K])
            else:
                sg = np.full(len(mu), np.exp(p[K + 1]))
            return -(ww * np.log(np.maximum(norm.cdf((hi - mu) / sg) - norm.cdf((lo - mu) / sg), 1e-300))).sum()
        npar = K + 1 + (0 if fixed_sigma is not None else (K if sep_sigma else 1))
        p0 = np.r_[np.full(K, 6.0), 0.5, np.full(npar - K - 1, 0.7)]
        o = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(nll, o.x, method="BFGS")
        p = o.x
        if fixed_sigma is not None:
            sig = np.full(K, fixed_sigma)
        elif sep_sigma:
            sig = np.exp(p[K + 1:K + 1 + K])
        else:
            sig = np.full(K, np.exp(p[K + 1]))
        ppos = np.array([np.average(band[ch == k] >= 2, weights=w[ch == k]) for k in range(K)])
        return p[:K], p[K], sig, ppos

    def means(mu, shift, sig, ppos, cap=None):
        out = []
        for k in range(K):
            row = []
            for sz in (0.0, 1.0):
                m = mu[k] + shift * sz
                e = np.exp(m + sig[k] ** 2 / 2)
                if cap is not None:
                    lu = np.log(cap)
                    e = e * norm.cdf((lu - m - sig[k] ** 2) / sig[k]) / norm.cdf((lu - m) / sig[k])
                row.append(ppos[k] * e)
            out.append(row)
        return np.array(out)

    def show(lab, M, extra=""):
        print(f"  {lab:28s} " + "  ".join(f"{names[k][:10]:>10s} £{M[k, 0]:>8,.0f}/{M[k, 1]:>8,.0f}" for k in range(K)) + extra)

    print("mean cost per breach, Micro / Small+")
    mu, sh, sig, pp = fit(ch, band, w, sm)
    show("base (common sigma)", means(mu, sh, sig, pp), f"   sigma {sig[0]:.2f}, size x{np.exp(sh):.1f}")
    show("cap at £500k", means(mu, sh, sig, pp, cap=5e5))
    top = np.where(band >= 10)[0]
    for i in top:
        keep = np.ones(len(ch), bool)
        keep[i] = False
        r = fit(ch[keep], band[keep], w[keep], sm[keep])
        show(f"drop top-band firm ({names[ch[i]][:10]})", means(*r), f"   sigma {r[2][0]:.2f}")
    keep = band < 10
    r = fit(ch[keep], band[keep], w[keep], sm[keep])
    show("drop all top-band firms", means(*r), f"   sigma {r[2][0]:.2f}")
    r = fit(ch, band, w, sm, sep_sigma=True)
    show("channel-specific sigma", means(*r), "   sigmas " + "/".join(f"{x:.2f}" for x in r[2]))

    X, _, size = load()
    d = f.load_firms()
    b_all, D = d["band"].values, d["D"].values
    w_all = d["weight"].fillna(d["weight"].median()).values
    lab_map = {1: 0, 3: 1, 4: 2, 2: 3}
    m = ~np.isnan(b_all) & ~np.isnan(D)
    ch_a = np.array([lab_map[int(x)] for x in D[m]])
    _, _, sig_all, _ = fit(ch_a, b_all[m].astype(int), w_all[m] / w_all[m].mean(), (size[m] >= 2).astype(float))
    print(f"  (sigma fitted on all {m.sum()} attacked firms' worst incidents: {sig_all[0]:.2f})")
    r = fit(ch, band, w, sm, fixed_sigma=sig_all[0])
    show("sigma from all firms", means(*r))


if __name__ == "__main__" and "tail" in __import__("sys").argv:
    tail_check()


def ransomware_fit():
    """Ransomware cost per breach using all ransomware-breach information (ranssoft >= 1 = ransom demanded).
    Cost per breach X: zero with prob p0, else lognormal(mu + delta*Small+, sigma). Likelihood pieces:
      A single: worst = ransomware (D 3), 1 demand        -> X in worst band
      B other : >= 1 demand but worst incident another channel -> X <= upper edge of worst band (censored)
      C multi : worst = ransomware, k >= 2 demands         -> max of k draws in worst band (k capped at 10)
    Fits A only, A+B, A+B+C. Mean per breach Micro / Small+. Then: the top-band firms' own ransomware-cost answers,
    and the A+B+C fit dropping each top-band firm."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    band, D = d["band"].values, d["D"].values
    w = d["weight"].fillna(d["weight"].median()).values
    k = pd.to_numeric(r["ranssoft"], errors="coerce").values
    k = np.where((k >= 0) & (k < 997), k, np.nan)
    rc = pd.to_numeric(r["ranscost_bands"], errors="coerce").values
    ok = ~np.isnan(band) & (k >= 1)
    sA, sB, sC = ok & (D == 3) & (k == 1), ok & (D != 3) & ~np.isnan(D), ok & (D == 3) & (k >= 2)
    sm = (size >= 2).astype(float)
    print(f"A single {sA.sum()}, B worst-other {sB.sum()}, C multi-demand {sC.sum()}")

    def F(x, mu, s, p0):          # cdf of X at x (x > 0)
        return p0 + (1 - p0) * norm.cdf((np.log(x) - mu) / s)

    def loglik(p, A, B, C, wn):
        m0, delta, ls, lp = p
        s, p0 = np.exp(ls), 1 / (1 + np.exp(-lp))
        ll = 0.0
        for sel, kind in ((A, "A"), (B, "B"), (C, "C")):
            idx = np.where(sel)[0]
            if len(idx) == 0:
                continue
            mu = m0 + delta * sm[idx]
            b = band[idx].astype(int)
            hi = np.array([EDGES[x][1] if x >= 2 else 1e-9 for x in b])
            lo = np.array([EDGES[x][0] if x >= 2 else 1e-9 for x in b])
            if kind == "A":
                pr = np.where(b == 1, p0, F(hi, mu, s, p0) - F(lo, mu, s, p0))
            elif kind == "B":
                pr = np.where(b == 1, p0, F(hi, mu, s, p0))
            else:
                kk = np.minimum(k[idx], 10)
                pr = np.where(b == 1, p0 ** kk, F(hi, mu, s, p0) ** kk - F(lo, mu, s, p0) ** kk)
            ll += (wn[idx] * np.log(np.maximum(pr, 1e-300))).sum()
        return ll

    wn = w / w[ok].mean()

    def fit(A, B, C):
        o = minimize(lambda p: -loglik(p, A, B, C, wn), [7.0, 0.5, 0.7, -2.0], method="Nelder-Mead",
                     options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
        m0, delta, ls, lp = o.x
        s, p0 = np.exp(ls), 1 / (1 + np.exp(-lp))
        mean = [(1 - p0) * np.exp(m0 + delta * z + s ** 2 / 2) for z in (0, 1)]
        return m0, delta, s, p0, mean

    for lab, (A, B, C) in (("A only", (sA, sA & False, sA & False)), ("A + B", (sA, sB, sA & False)), ("A + B + C", (sA, sB, sC))):
        m0, delta, s, p0, mean = fit(A, B, C)
        print(f"  {lab:10s} median £{np.exp(m0):6,.0f}  size x{np.exp(delta):.1f}  sigma {s:.2f}  p0 {p0:.2f}   "
              f"mean per breach Micro £{mean[0]:7,.0f}  Small+ £{mean[1]:7,.0f}")
    top = np.where(ok & (band >= 10))[0]
    print("\ntop-band ransomware-breach firms: group (A/B/C), worst band, own ransomware cost band (12-band, 1=<£100 ... 10=£50-100k), types hit")
    for i in top:
        g = "A" if sA[i] else ("B" if sB[i] else "C")
        print(f"  firm {i}: {g}  worst {int(band[i])}  ranscost {rc[i] if rc[i] < 997 else 'NA'}  types {int(X[i].sum())}  size {int(size[i])}")
    for i in top:
        keep = np.ones(len(X), bool)
        keep[i] = False
        m0, delta, s, p0, mean = fit(sA & keep, sB & keep, sC & keep)
        print(f"  A+B+C without firm {i}: sigma {s:.2f}  mean Micro £{mean[0]:7,.0f}  Small+ £{mean[1]:7,.0f}")


if __name__ == "__main__" and "rans" in __import__("sys").argv:
    ransomware_fit()


def ransomware_fit_constrained():
    """Ransomware A+B+C fit with the size shift fixed at the common factor (x1.7 from size_effect) and firm 505 removed
    (worst £100k-500k but own ransomware cost < £100, 6 types hit: a broad compromise, not a ransomware cost);
    with and without firm 1431 (the other top-band firm, micro, 3 types, no ransomware-cost answer)."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    band, D = d["band"].values, d["D"].values
    w = d["weight"].fillna(d["weight"].median()).values
    k = pd.to_numeric(r["ranssoft"], errors="coerce").values
    k = np.where((k >= 0) & (k < 997), k, np.nan)
    ok = ~np.isnan(band) & (k >= 1)
    sm = (size >= 2).astype(float)
    delta = np.log(1.7)
    wn = w / w[ok].mean()

    def F(x, mu, s, p0):
        return p0 + (1 - p0) * norm.cdf((np.log(x) - mu) / s)

    def run(excl):
        keep = ok.copy()
        keep[list(excl)] = False
        A, B, C = keep & (D == 3) & (k == 1), keep & (D != 3) & ~np.isnan(D), keep & (D == 3) & (k >= 2)
        def nll(p):
            m0, ls, lp = p
            s, p0 = np.exp(ls), 1 / (1 + np.exp(-lp))
            ll = 0.0
            for sel, kind in ((A, "A"), (B, "B"), (C, "C")):
                idx = np.where(sel)[0]
                mu = m0 + delta * sm[idx]
                b = band[idx].astype(int)
                hi = np.array([EDGES[x][1] if x >= 2 else 1e-9 for x in b])
                lo = np.array([EDGES[x][0] if x >= 2 else 1e-9 for x in b])
                if kind == "A":
                    pr = np.where(b == 1, p0, F(hi, mu, s, p0) - F(lo, mu, s, p0))
                elif kind == "B":
                    pr = np.where(b == 1, p0, F(hi, mu, s, p0))
                else:
                    kk = np.minimum(k[idx], 10)
                    pr = np.where(b == 1, p0 ** kk, F(hi, mu, s, p0) ** kk - F(lo, mu, s, p0) ** kk)
                ll += (wn[idx] * np.log(np.maximum(pr, 1e-300))).sum()
            return -ll
        o = minimize(nll, [7.0, 0.7, -2.0], method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
        m0, s, p0 = o.x[0], np.exp(o.x[1]), 1 / (1 + np.exp(-o.x[2]))
        return m0, s, p0, [(1 - p0) * np.exp(m0 + delta * z + s ** 2 / 2) for z in (0, 1)]

    for lab, ex in (("all firms", []), ("without 505", [505]), ("without 505 and 1431", [505, 1431])):
        m0, s, p0, mean = run(ex)
        print(f"  {lab:22s} median £{np.exp(m0):6,.0f}  sigma {s:.2f}  p0 {p0:.2f}  mean per breach Micro £{mean[0]:7,.0f}  Small+ £{mean[1]:7,.0f}")


if __name__ == "__main__" and "ransc" in __import__("sys").argv:
    ransomware_fit_constrained()


def cost_components():
    """What are the cheap vs expensive worst incidents made of? damage_bands is built from four components:
    external payments during (damagedirsx) and after (damagedirlx) the incident, staff time (damagestaffx), and damage /
    disruption (damageindx). Per worst-incident channel (D) and cost group (<£500 / £500-5k / £5k+): share of firms
    with each component > 0, mean component midpoints, restoration time (Q71), and for ransomware ransom paid (Q83J)."""
    import glob
    import pyreadstat
    sav = glob.glob("/home/user/uk-cyber/data/raw/*.sav")[0]
    _, meta = pyreadstat.read_sav(sav, metadataonly=True)
    print("restore labels:", meta.variable_value_labels.get("restore"))
    print("ranspayyn labels:", meta.variable_value_labels.get("ranspayyn"))
    print("component scale (damagestaffx):", meta.variable_value_labels.get("damagestaffx_bands"))
    MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000, 11: 750000,
           12: 3e6, 13: 5e6}
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    band, D = d["band"].values, d["D"].values
    comp = ["damagedirsx_bands", "damagedirlx_bands", "damagestaffx_bands", "damageindx_bands"]
    cn = ["ext-during", "ext-after", "staff", "disruption"]
    V = np.column_stack([pd.to_numeric(r[c], errors="coerce").values for c in comp])
    V = np.where((V >= 1) & (V <= 13), V, np.nan)
    rest = pd.to_numeric(r["restore"], errors="coerce").values
    pay = pd.to_numeric(r["ranspayyn"], errors="coerce").values
    chn = {1: "phishing", 2: "impersonation", 3: "ransomware", 4: "other serious"}
    for code in (3, 4, 2, 1):
        print(f"\nWORST INCIDENT = {chn[code].upper()}")
        for gl, lo, hi in (("<£500", 2, 3), ("£500-5k", 4, 5), ("£5k+", 6, 13)):
            m = (D == code) & (band >= lo) & (band <= hi)
            if m.sum() == 0:
                continue
            Vm = V[m]
            has = [np.mean(Vm[:, j][~np.isnan(Vm[:, j])] > 1) if (~np.isnan(Vm[:, j])).any() else float("nan") for j in range(4)]
            mean = [np.nanmean([MID[int(x)] for x in Vm[:, j] if not np.isnan(x)]) if (~np.isnan(Vm[:, j])).any() else float("nan") for j in range(4)]
            rs = rest[m]
            rs = rs[(rs >= 1) & (rs < 997)]
            line = (f"  {gl:8s} n {m.sum():3d}  share >0: " + " ".join(f"{n} {h:.2f}" for n, h in zip(cn, has)) +
                    "   mean £: " + " ".join(f"{n} {v:,.0f}" for n, v in zip(cn, mean)) +
                    "   restore codes: " + " ".join(f"{int(k)}:{int((rs == k).sum())}" for k in np.unique(rs)))
            if code == 3:
                p = pay[m]
                line += f"   ransom paid (Q83J codes): " + str(pd.Series(p[(p > 0) & (p < 997)]).value_counts().to_dict())
            print(line)


if __name__ == "__main__" and "comp" in __import__("sys").argv:
    cost_components()


def cost_pairs_pattern():
    """Per-type breach cost vs worst-incident cost, pooled over takeover / DoS / malware / ransomware, for firms whose
    worst incident was that channel. Lower bound / upper bound of each band; classify each pair as clearly lower
    (type band entirely below worst band), overlapping, clearly higher. Also vs worst-incident components:
    external payments (during + after), staff time, disruption: which component does the type cost track?"""
    TB = {1: (0, 100), 2: (100, 250), 3: (250, 500), 4: (500, 1000), 5: (1000, 2000), 6: (2000, 5000), 7: (5000, 10000),
          8: (10000, 20000), 9: (20000, 50000), 10: (50000, 100000), 11: (100000, 250000), 12: (250000, 1e9)}
    WB = {1: (0, 0), 2: (0, 100), 3: (100, 500), 4: (500, 1000), 5: (1000, 5000), 6: (5000, 10000), 7: (10000, 20000),
          8: (20000, 50000), 9: (50000, 100000), 10: (100000, 500000), 11: (500000, 1e6), 12: (1e6, 5e6), 13: (5e6, 1e9)}
    MID = {k: (a + b) / 2 if b < 1e9 else a for k, (a, b) in WB.items()}
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    band, disr = d["band"].values, d["disrupta"].values
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values
    comp = {c: num(c) for c in ("damagedirsx_bands", "damagedirlx_bands", "damagestaffx_bands", "damageindx_bands")}
    rows = []
    for lab, cv, codes in (("takeover", "tkvrcost_bands", [11]), ("DoS", "doscost_bands", [3]),
                           ("malware", "viruscost_bands", [2]), ("ransomware", "ranscost_bands", [1])):
        c = num(cv)
        for i in np.where((c >= 1) & (c <= 12) & ~np.isnan(band) & np.isin(disr, codes))[0]:
            cm = {k: (MID[int(v[i])] if 1 <= v[i] <= 13 else np.nan) for k, v in comp.items()}
            rows.append((lab, TB[int(c[i])], WB[int(band[i])], cm))
    lower = sum(t[1] <= w[0] and not (t[1] == w[0] == 0) for _, t, w, _ in rows)
    higher = sum(t[0] >= w[1] and w[1] > 0 for _, t, w, _ in rows)
    print(f"pairs {len(rows)}: per-type clearly LOWER {lower}, overlapping {len(rows) - lower - higher}, clearly HIGHER {higher}")
    for lab in ("takeover", "DoS", "malware", "ransomware"):
        rr = [x for x in rows if x[0] == lab]
        lo = sum(t[1] <= w[0] and not (t[1] == w[0] == 0) for _, t, w, _ in rr)
        hi = sum(t[0] >= w[1] and w[1] > 0 for _, t, w, _ in rr)
        print(f"  {lab:10s} n {len(rr):2d}  lower {lo:2d}  overlap {len(rr) - lo - hi:2d}  higher {hi:2d}")
    print("\nwhich worst-incident component does the per-type cost (midpoint) sit closest to? (log distance, pairs with all components)")
    tmid = lambda t: (t[0] + t[1]) / 2 if t[1] < 1e9 else t[0]
    names = {"ext payments (during+after)": lambda cm: cm["damagedirsx_bands"] + cm["damagedirlx_bands"],
             "staff time": lambda cm: cm["damagestaffx_bands"], "disruption": lambda cm: cm["damageindx_bands"],
             "worst total": lambda cm: sum(cm.values())}
    full = [x for x in rows if not any(np.isnan(v) for v in x[3].values())]
    for nm, fn in names.items():
        dist = [abs(np.log10(max(tmid(t), 10)) - np.log10(max(fn(cm), 10))) for _, t, _, cm in full]
        print(f"  {nm:28s} median |log10 gap| {np.median(dist):.2f}  (n {len(full)})")
    print("\npairs (type band £ range | worst band £ range | ext / staff / disruption midpoints):")
    for lab, t, w, cm in rows:
        print(f"  {lab:10s} {t[0]:>7,.0f}-{t[1]:<9,.0f} | {w[0]:>7,.0f}-{w[1]:<9,.0f} | "
              f"{cm['damagedirsx_bands'] + cm['damagedirlx_bands']:>9,.0f} / {cm['damagestaffx_bands']:>9,.0f} / {cm['damageindx_bands']:>9,.0f}")


if __name__ == "__main__" and "pairs2" in __import__("sys").argv:
    cost_pairs_pattern()
