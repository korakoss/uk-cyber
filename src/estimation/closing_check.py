"""Closing check: counts x per-attack costs -> worst incident.

If attacks within a channel have iid costs G, a firm with K attacks has worst-incident cost = max of K draws from G.
G = empirical per-attack cost band distribution from single-attack firms (survey-weighted); predict the worst band of
multi-attack firms and compare with what they report. Bands grouped none / <£500 / £500-5k / £5k-20k / £20k+.

  impersonation: impersonation-only firms. G from freq 'once'. K for the others from their freq band, drawn
                 log-uniform within band edges (scenario B: 2-8, 9-30, 31-150, 151-500, 501-1000).
  phishing, no engagement: phishing-only firms with 0 engaged attacks (handling costs). G from freq 'once'; K as above.
  ransomware:    worst incident ransomware. G from ranssum == 1; K = ranssum for ranssum >= 2 (exact).
Uncertainty: 1000 bootstrap resamples of the single-attack firms (G) -> 90% band for each predicted share.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/closing_check.py
"""

import numpy as np
import pandas as pd

import joint_five_channel as f
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load, SHORT

GROUPS = [(1, 1), (2, 3), (4, 5), (6, 7), (8, 13)]
GLAB = "none / <£500 / £500-5k / £5k-20k / £20k+"
FREQ_K = {2: (2, 8), 3: (9, 30), 4: (31, 150), 5: (151, 500), 6: (501, 1000)}
FLAB = {2: "<monthly", 3: "monthly", 4: "weekly", 5: "daily", 6: "sev/day"}


def grp(b):
    out = np.zeros(len(b), int)
    for i, (a, c) in enumerate(GROUPS):
        out[(b >= a) & (b <= c)] = i
    return out


def wshare(g, w):
    return np.array([w[g == i].sum() for i in range(len(GROUPS))]) / w.sum()


def max_of_k(p, K):
    """Distribution over groups of the max of K iid draws from group distribution p (vectorised over firms)."""
    c = np.cumsum(p)
    cdf = c[None, :] ** K[:, None]
    return np.diff(np.concatenate([np.zeros((len(K), 1)), cdf], 1), axis=1)


def predict(gs, ws, K, wk, rng, reps=1000):
    """Weighted predicted group shares for firms with counts K (weights wk); G from singles gs (weights ws)."""
    point = (max_of_k(wshare(gs, ws), K) * wk[:, None]).sum(0) / wk.sum()
    boots = []
    for _ in range(reps):
        i = rng.integers(0, len(gs), len(gs))
        boots.append((max_of_k(wshare(gs[i], ws[i]), K) * wk[:, None]).sum(0) / wk.sum())
    lo, hi = np.percentile(boots, [5, 95], axis=0)
    return point, lo, hi


def line(label, obs, n, pred=None):
    s = f"  {label:34s} n {n:4d}  obs " + " ".join(f"{v:4.2f}" for v in obs)
    if pred is not None:
        p, lo, hi = pred
        s += "   pred " + " ".join(f"{a:4.2f}[{b:.2f},{c:.2f}]" for a, b, c in zip(p, lo, hi))
    print(s)


def band_channel(label, band, w, fq, base, rng):
    """Single-attack G from freq 'once'; predicted worst band per freq band via log-uniform K within band."""
    single = base & (fq == 1)
    gs, ws = grp(band[single]), w[single]
    print(f"\n{label}")
    line("G: single attack (freq once)", wshare(gs, ws), single.sum())
    for k, (a, b) in FREQ_K.items():
        m = base & (fq == k)
        if m.sum() < 3:
            continue
        K = np.exp(rng.uniform(np.log(a), np.log(b + 1), (m.sum(), 50))).astype(int)  # 50 K draws per firm
        Kf = K.ravel()
        wk = np.repeat(w[m], 50)
        line(f"freq {FLAB[k]} (K {a}-{b})", wshare(grp(band[m]), w[m]), m.sum(), predict(gs, ws, Kf, wk, rng, 300))


def main():
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values
    band, D, fq = d["band"].values, d["D"].values, d["freq"].values
    w = d["weight"].fillna(d["weight"].median()).values
    ok = ~np.isnan(band)
    only = X.sum(1) == 1
    eng = num("phisheng")
    rng = np.random.default_rng(0)
    print(f"shares: {GLAB}; pred = max of K iid per-attack draws, [5%, 95%] over bootstrap of the single-attack firms")

    band_channel("IMPERSONATION-ONLY FIRMS", band, w, fq, ok & only & (X[:, SHORT.index("Imper")] == 1), rng)
    band_channel("PHISHING-ONLY FIRMS, 0 ENGAGED (handling cost)", band, w, fq, ok & only & (X[:, 0] == 1) & (eng == 0), rng)

    rs = num("Cybercrime_ranssum")
    single = ok & (D == 3) & (rs == 1)
    multi = ok & (D == 3) & (rs >= 2)
    gs, ws = grp(band[single]), w[single]
    print("\nRANSOMWARE (worst incident ransomware)")
    line("G: ranssum 1", wshare(gs, ws), single.sum())
    line("ranssum >= 2 (K exact: " + ",".join(str(int(x)) for x in np.sort(rs[multi])) + ")",
         wshare(grp(band[multi]), w[multi]), multi.sum(), predict(gs, ws, rs[multi].astype(int), w[multi], rng))


if __name__ == "__main__":
    main()
