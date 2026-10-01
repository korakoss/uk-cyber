"""Do broad compromises (firms hit by 3+ channels) cost more than the per-channel model predicts? Two parts.

(a) Breach occurrence. Channels P, I, R, S (flags). Material breach = Q56A any outcome. Independence across channels:
    P(no material breach) = prod over hit channels (1 - q_c). Fit q_c by ML on firms hit by 1 or 2 channels, predict
    P(any material) for 3- and 4-channel firms; compare with observed (also by tier). Weighted.
(b) Breach size. Material firms' worst-incident cost on TIGHTENED intervals (total band intersected with the summed
    component range); zero cost as point mass; positive part interval-censored lognormal with shifts for the worst
    channel (D), Small+, and broad (3+ channels); common sigma. LR test and size of the broad shift.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/broad_interaction.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from counts_given_frailty import tier_posteriors, TIER
from latent_on_streams import aligned_raw
from material_breach import data, WB, NAMES
from type_cooccurrence_structure import SHORT


def channels(X):
    P = X[:, 0] == 1
    I = X[:, SHORT.index("Imper")] == 1
    R = X[:, SHORT.index("Ransm")] == 1
    S = np.delete(X, [0, SHORT.index("Imper"), SHORT.index("Ransm")], axis=1).max(1) == 1
    return np.column_stack([P, I, R, S]).astype(float)


def tightened(band):
    r = aligned_raw()
    comp = ["damagedirsx_bands", "damagedirlx_bands", "damagestaffx_bands", "damageindx_bands"]
    V = np.column_stack([pd.to_numeric(r[c], errors="coerce").values for c in comp])
    full = ((V >= 1) & (V <= 13)).all(1) & ~np.isnan(band)
    lo = np.array([WB[int(b)][0] if not np.isnan(b) else np.nan for b in band])
    hi = np.array([WB[int(b)][1] if not np.isnan(b) else np.nan for b in band])
    for i in np.where(full)[0]:
        slo = sum(WB[int(V[i, j])][0] for j in range(4))
        shi = sum(WB[int(V[i, j])][1] for j in range(4))
        if slo <= hi[i] and shi >= lo[i]:
            lo[i], hi[i] = max(lo[i], slo), min(hi[i], shi)
    return lo, hi


def part_a(C, oa, w, tier):
    nch = C.sum(1)
    ok = ~np.isnan(oa) & (nch >= 1)
    fitm = ok & (nch <= 2)
    y, Cf, wf = oa[fitm], C[fitm], w[fitm] / w[fitm].mean()

    def nll(z):
        q = 1 / (1 + np.exp(-z))
        p_none = np.exp(Cf @ np.log(1 - q))
        return -(wf * (y * np.log(np.maximum(1 - p_none, 1e-12)) + (1 - y) * np.log(np.maximum(p_none, 1e-12)))).sum()
    z = minimize(nll, np.full(4, -2.0), method="BFGS").x
    q = 1 / (1 + np.exp(-z))
    print("(a) material-breach rate per channel (fit on 1-2 channel firms): " + ", ".join(f"{n} {v:.3f}" for n, v in zip("PIRS", q)))
    pred = 1 - np.exp(C @ np.log(1 - q))
    print("    by number of channels hit: observed vs predicted P(any material breach)")
    for k in (1, 2, 3, 4):
        m = ok & (nch == k)
        print(f"      {k} channel(s)  n {m.sum():4d}  obs {np.average(oa[m], weights=w[m]):.3f}  pred {np.average(pred[m], weights=w[m]):.3f}"
              + ("   (fit sample)" if k <= 2 else "   (out of sample)"))
    print("    3+ channels by tier:")
    for c in range(3):
        m = ok & (nch >= 3) & (tier == c)
        if m.sum():
            print(f"      tier {TIER[c]:4s} n {m.sum():3d}  obs {np.average(oa[m], weights=w[m]):.3f}  pred {np.average(pred[m], weights=w[m]):.3f}")
    # bootstrap the 3+ gap
    rng = np.random.default_rng(0)
    idx3 = np.where(ok & (nch >= 3))[0]
    gaps = []
    for _ in range(500):
        b = rng.choice(idx3, len(idx3))
        gaps.append(np.average(oa[b], weights=w[b]) - np.average(pred[b], weights=w[b]))
    lo, hi = np.percentile(gaps, [5, 95])
    print(f"    3+ channels: obs - pred {np.mean(gaps):+.3f}  90% [{lo:+.3f}, {hi:+.3f}] (resampling firms; q held fixed)")


def part_b(C, band, D, oa, w, size):
    lo, hi = tightened(band)
    nch = C.sum(1)
    m = (oa == 1) & ~np.isnan(band) & ~np.isnan(D)
    pos = m & (band >= 2)
    ch = (D[pos] - 1).astype(int)
    L, H = np.log(np.maximum(lo[pos], 1)), np.log(hi[pos])
    sm, br, ww = (size[pos] >= 2).astype(float), (nch[pos] >= 3).astype(float), w[pos] / w[pos].mean()
    E = np.eye(4)[ch]

    def fit(use_broad):
        def nll(p):
            mu = E @ p[:4] + p[4] * sm + (p[5] * br if use_broad else 0)
            s = np.exp(p[6])
            return -(ww * np.log(np.maximum(norm.cdf((H - mu) / s) - norm.cdf((L - mu) / s), 1e-300))).sum()
        o = minimize(nll, np.r_[np.full(4, 7.0), 0.3, 0.0, 0.7], method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(nll, o.x, method="BFGS")
        return -o.fun, o.x
    l0, p0 = fit(False)
    l1, p1 = fit(True)
    print(f"\n(b) material breaches with positive cost: {pos.sum()} (broad 3+ channels: {int(br.sum())})")
    print(f"    broad shift on log cost {p1[5]:+.2f} -> cost x{np.exp(p1[5]):.2f};  LR p {chi2.sf(2 * (l1 - l0), 1):.3f};  sigma {np.exp(p1[6]):.2f}")
    print("    channel medians (Micro, non-broad): " + ", ".join(f"{n} £{np.exp(v):,.0f}" for n, v in zip(NAMES, p1[:4])))
    zero = m & (band == 1)
    print(f"    zero-cost share among material breaches: broad {np.average(zero[m & (nch >= 3)], weights=w[m & (nch >= 3)]):.2f}  "
          f"non-broad {np.average(zero[m & (nch < 3)], weights=w[m & (nch < 3)]):.2f}")


def main():
    X, size, band, D, oa, w = data()
    C = channels(X)
    tier = tier_posteriors().argmax(1)
    part_a(C, oa, w, tier)
    part_b(C, band, D, oa, w, size)


if __name__ == "__main__":
    main()
