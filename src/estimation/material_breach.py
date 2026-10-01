"""Material-breach cost layer.

Split of the worst-incident cost into two event kinds:
  handling: incidents with no material outcome (Q56A outcome_any = 0) -> staff-time handling cost;
  material: incidents with a material outcome (outcome_any = 1) -> the costly events.
(1) Handling cost by worst-incident channel (D): weighted band shares and midpoint mean.
(2) Material-event cost: interval-censored lognormal on the worst band of material firms (zero cost as a separate
    point mass), channel shifts (D) + common Small+ shift, common sigma. Mean per material event by channel x size;
    sensitivity: drop each top-band firm, all top-band firms, cap at £500k, channel-specific sigma.
(3) Material-breach rate given hit, by channel: single-channel firms, and all hit firms.
Weights normalised over the fitted sample.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/material_breach.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import joint_five_channel as f
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load, SHORT

EDGES = {2: (1, 100), 3: (100, 500), 4: (500, 1000), 5: (1000, 5000), 6: (5000, 10000), 7: (10000, 20000),
         8: (20000, 50000), 9: (50000, 100000), 10: (100000, 500000), 11: (500000, 1e6), 12: (1e6, 5e6), 13: (5e6, 1e9)}
MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000}
NAMES = ["phishing", "impersonation", "ransomware", "other serious"]   # D = 1..4
K = 4


def data():
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    oa = pd.to_numeric(r["outcome_any"], errors="coerce").values
    oa = np.where(np.isin(oa, [0, 1]), oa, np.nan)
    w = d["weight"].fillna(d["weight"].median()).values
    return X, size, d["band"].values, d["D"].values, oa, w


def fit_material(ch, band, w, sm, sep_sigma=False):
    """Zero mass per channel (weighted share) + interval-censored lognormal on positives."""
    pos = band >= 2
    lo = np.log([EDGES[b][0] for b in band[pos]])
    hi = np.log([EDGES[b][1] for b in band[pos]])
    C, s, ww = np.eye(K)[ch[pos]], sm[pos], w[pos]

    def nll(p):
        mu = C @ p[:K] + p[K] * s
        sg = C @ np.exp(p[K + 1:K + 1 + K]) if sep_sigma else np.exp(p[K + 1])
        return -(ww * np.log(np.maximum(norm.cdf((hi - mu) / sg) - norm.cdf((lo - mu) / sg), 1e-300))).sum()
    npar = K + 1 + (K if sep_sigma else 1)
    p0 = np.r_[np.full(K, 7.0), 0.5, np.full(npar - K - 1, 0.7)]
    o = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-6, "fatol": 1e-8})
    o = minimize(nll, o.x, method="BFGS")
    p = o.x
    sig = np.exp(p[K + 1:K + 1 + K]) if sep_sigma else np.full(K, np.exp(p[K + 1]))
    pz = np.array([np.average(band[ch == k] == 1, weights=w[ch == k]) if (ch == k).any() else np.nan for k in range(K)])
    return p[:K], p[K], sig, pz


def means(mu, shift, sig, pz, cap=None):
    out = np.zeros((K, 2))
    for k in range(K):
        for j, z in enumerate((0.0, 1.0)):
            m = mu[k] + shift * z
            e = np.exp(m + sig[k] ** 2 / 2)
            if cap is not None:
                lu = np.log(cap)
                e *= norm.cdf((lu - m - sig[k] ** 2) / sig[k]) / norm.cdf((lu - m) / sig[k])
            out[k, j] = (1 - pz[k]) * e
    return out


def show(lab, M, extra=""):
    print(f"  {lab:30s} " + "  ".join(f"{NAMES[k][:9]:>9s} £{M[k, 0]:>7,.0f}/{M[k, 1]:>7,.0f}" for k in range(K)) + extra)


def main():
    X, size, band, D, oa, w = data()
    ok = ~np.isnan(band) & ~np.isnan(D) & ~np.isnan(oa)
    print("(1) HANDLING: worst incidents without a material outcome, by worst channel; none / <£500 / £500-5k / £5k+; mean £")
    for k in range(K):
        m = ok & (oa == 0) & (D == k + 1)
        b, ww = band[m], w[m]
        print(f"  {NAMES[k]:14s} n {m.sum():4d}  " + " ".join(f"{np.average((b >= a) & (b <= c), weights=ww):.2f}" for a, c in ((1, 1), (2, 3), (4, 5), (6, 13)))
              + f"   mean £{np.average([MID[int(x)] for x in b], weights=ww):,.0f}")
    mm = ok & (oa == 1)
    ch, b, ww, sm = (D[mm] - 1).astype(int), band[mm].astype(int), w[mm] / w[mm].mean(), (size[mm] >= 2).astype(float)
    print(f"\n(2) MATERIAL EVENTS: {mm.sum()} firms (" + ", ".join(f"{NAMES[k]} {int((ch == k).sum())}" for k in range(K)) +
          f"); top-band firms {int((b >= 10).sum())}")
    for k in range(K):
        bk, wk = b[ch == k], ww[ch == k]
        print(f"  {NAMES[k]:14s} shares none / <£500 / £500-5k / £5k-20k / £20k+: " +
              " ".join(f"{np.average((bk >= a) & (bk <= c), weights=wk):.2f}" for a, c in ((1, 1), (2, 3), (4, 5), (6, 7), (8, 13))))
    print("  mean cost per material event, Micro / Small+")
    mu, shf, sig, pz = fit_material(ch, b, ww, sm)
    show("base (common sigma)", means(mu, shf, sig, pz), f"  sigma {sig[0]:.2f} size x{np.exp(shf):.1f}")
    show("cap at £500k", means(mu, shf, sig, pz, cap=5e5))
    for i in np.where(b >= 10)[0]:
        keep = np.arange(len(b)) != i
        r_ = fit_material(ch[keep], b[keep], ww[keep], sm[keep])
        show(f"drop top-band firm ({NAMES[ch[i]][:9]})", means(*r_), f"  sigma {r_[2][0]:.2f}")
    keep = b < 10
    r_ = fit_material(ch[keep], b[keep], ww[keep], sm[keep])
    show("drop all top-band firms", means(*r_), f"  sigma {r_[2][0]:.2f}")
    r_ = fit_material(ch, b, ww, sm, sep_sigma=True)
    show("channel-specific sigma", means(*r_), "  sigmas " + "/".join(f"{x:.2f}" for x in r_[2]))

    print("\n(3) MATERIAL-BREACH RATE given hit: single-channel firms | all hit firms")
    nch = X.sum(1)
    flags = {"phishing": X[:, 0] == 1, "impersonation": X[:, SHORT.index("Imper")] == 1,
             "ransomware": X[:, SHORT.index("Ransm")] == 1,
             "other serious": np.delete(X, [0, SHORT.index("Imper"), SHORT.index("Ransm")], axis=1).max(1) == 1}
    for k, nm in enumerate(NAMES):
        hit = flags[nm] & ~np.isnan(oa)
        single = hit & (nch == 1) if nm != "other serious" else hit & (X.sum(1) - flags["phishing"] - flags["impersonation"] - flags["ransomware"] == X.sum(1))
        print(f"  {nm:14s} single n {single.sum():4d} P {np.average(oa[single], weights=w[single]) if single.any() else float('nan'):.3f}   |   "
              f"all hit n {hit.sum():4d} P {np.average(oa[hit], weights=w[hit]):.3f}")


if __name__ == "__main__":
    main()
