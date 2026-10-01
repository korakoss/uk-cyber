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
