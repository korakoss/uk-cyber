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
