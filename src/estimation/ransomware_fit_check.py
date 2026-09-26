"""Is ransomware's data consistent with a lognormal, and how well pinned is its sigma?

Uses the five-channel frailty fits with the ransomware count (joint_five_channel.py ... rcount).
1. Fit check: simulate from the free (and tied) fit; compare the worst-band distribution among
   firms whose most disruptive attack was ransomware (disrupta=R), and P(disrupta=R | R flag).
2. Profile likelihood over sigma_R: fix sigma_R on a grid, re-optimise (p_R, mu_R) with all other
   parameters held at the free optimum (partial profile -> interval slightly too narrow).

Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src:src/estimation \
     python3 src/estimation/ransomware_fit_check.py
"""

import json
import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

import joint_four_channel as j
import joint_five_channel as f

HERE = os.path.dirname(os.path.abspath(__file__))
COARSE = [(1, 1, "no cost"), (2, 3, "£1-500"), (4, 5, "£500-5k"), (6, 7, "£5k-20k"),
          (8, 10, "£20k-500k"), (11, 13, ">£500k")]


def simulate(q, n, rng):
    exposed = rng.random(n) < q["pi"]
    Y, K = [], {}
    for ch in f.CH:
        rate = q[f.RATE[ch]] * np.where(exposed, np.exp(q["d" + ch]), 1.0)
        if ch in ("T", "I"):
            k = rng.poisson(rate)
        else:
            r = q["r" + ch]
            k = rng.negative_binomial(r, r / (r + rate))
        K[ch] = k
        G = j.band_cdf(q["p" + ch], q["mu" + ch], q["s" + ch])
        Gk = G[None, :] ** np.minimum(k, j.KMAX)[:, None]
        y = (Gk < rng.random(n)[:, None]).sum(1)
        Y.append(np.where(k == 0, 0, y))
    Y = np.stack(Y, 1)
    M = Y.max(1)
    lab = np.array([f.LAB[ch] for ch in f.CH])
    tie = (Y == M[:, None]) & (M[:, None] > 0)
    D = np.where(M > 0, lab[np.argmax(tie * rng.random(Y.shape), 1)], 0)
    return M, D, K["R"] > 0


def fit_check(d, w, label, q):
    M, D, fR = simulate(q, 2_000_000, np.random.default_rng(3))
    obs = (d["D"] == 3) & d["band"].notna()
    b, bw = d.loc[obs, "band"].values, w[obs.values]
    sb = M[D == 3]
    print(f"\n  {label}: worst band | disrupta=R   (obs n={obs.sum()})")
    x2, rows = 0.0, []
    for lo, hi, name in COARSE:
        o = np.average((b >= lo) & (b <= hi), weights=bw)
        s = np.mean((sb >= lo) & (sb <= hi))
        ocount = int(((b >= lo) & (b <= hi)).sum())
        ecount = s * obs.sum()
        x2 += (ocount - ecount) ** 2 / max(ecount, 0.5)
        print(f"    {name:>10s}  obs {o:5.2f} ({ocount:2d})   sim {s:5.3f} (exp {ecount:4.1f})")
    print(f"    Pearson X2 = {x2:.1f} on ~{len(COARSE) - 1} df (raw counts, sparse -> rough)")
    fr_obs = d["fR"].values == 1
    print(f"    P(disrupta=R | R flag): obs {np.average(d.loc[fr_obs, 'D'].values == 3, weights=w[fr_obs]):.2f}"
          f"   sim {np.mean(D[fR] == 3):.2f}")


def main():
    d = f.load_firms()
    F, _, _ = j.calibrate_freq(d)
    w = d["weight"].fillna(d["weight"].median()).values
    w = w / w.mean()
    qf = json.load(open(os.path.join(HERE, "build", "joint_five_channel_free_rcount_weighted.json")))
    qt = json.load(open(os.path.join(HERE, "build", "joint_five_channel_tied_rcount_weighted.json")))

    print("=" * 78)
    print("1. FIT CHECK: ransomware worst-band distribution, observed vs simulated")
    print("=" * 78)
    fit_check(d, w, "FREE fit", qf)
    fit_check(d, w, "TIED fit", qt)

    print(f"\n{'=' * 78}")
    print("2. PROFILE LIKELIHOOD over sigma_R (others fixed at free optimum)")
    print("=" * 78)
    lik = f.Lik(d, F, w, tied=False, use_rcount=True)
    x_opt = f.pack(qf, False)
    nll_opt = lik.nll(x_opt)
    rows = []
    xs = x_opt.copy()
    for sR in [1.6, 1.9, 2.2, 2.5, 2.7, 2.93, 3.2, 3.5, 3.8, 4.2, 4.6]:
        def g(z):
            x = xs.copy()
            x[20], x[21], x[22] = z[0], z[1], np.log(sR)
            return lik.nll(x)
        res = minimize(g, [xs[20], xs[21]], method="Nelder-Mead",
                       options=dict(xatol=1e-3, fatol=1e-3, maxiter=400))
        xs[20], xs[21] = res.x
        q = f.unpack(np.r_[xs[:22], np.log(sR), xs[23:]], False)
        ek = (1 - q["pi"]) * q["mR"] + q["pi"] * q["mR"] * np.exp(q["dR"])
        ec = j.trunc_mean(q["muR"], sR)
        rows.append((sR, res.fun - nll_opt, q["pR"], q["muR"], ec, ek * q["pR"] * ec))
        print(f"  sigma_R {sR:4.2f}: dNLL {res.fun - nll_opt:6.2f}  p_R {q['pR']:.3f}  mu_R {q['muR']:5.2f}"
              f"  median £{np.exp(q['muR']):6,.0f}  E[C|succ] £{ec:9,.0f}  R £/firm {ek * q['pR'] * ec:7,.0f}",
              flush=True)
    cut = chi2.ppf(0.95, 1) / 2
    inside = [r for r in rows if r[1] <= cut]
    print(f"\n  95% profile interval (dNLL <= {cut:.2f}): sigma_R in [{inside[0][0]:.2f}, {inside[-1][0]:.2f}]"
          f"  -> R £/firm in [{min(r[5] for r in inside):,.0f}, {max(r[5] for r in inside):,.0f}]"
          f"  (grid resolution; partial profile)")


if __name__ == "__main__":
    main()
