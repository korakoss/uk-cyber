"""Count law of TARGETED phishing given the latent (and size).

Targeted count t: phishcon (Q89E exact) else phishcon_bands midpoint ('don't know' -> approx. band);
0 for firms not hit by phishing. Exposure tiers from the 3-class flag model (10 survey types), within
Micro and within Small+Medium+Large; firms assigned to their most likely tier.
Per tier: p = P(t >= 1) (firms with a known t). Poisson given the tier fixes the mean: lambda = -ln(1-p),
predicting the distribution of t among firms with t >= 1 with nothing fitted to counts.
If Poisson fails: NegBin with P(t >= 1) pinned to p, shape r shared across tiers (the latent scales the
mean, not the shape); goodness of fit on coarse bins; r compared across the two size groups.
Survey-weighted (normalised within size group). Counts heap at round numbers; coarse bins.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/targeted_count_law.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, nbinom, poisson

from latent_on_streams import aligned_raw, DK_MID
from type_cooccurrence_structure import load
from frailty_shape_flags import lca, patterns

BINS = [(1, 1), (2, 3), (4, 10), (11, 50), (51, 10**9)]
K = np.arange(0, 5001)
TIER = ["low", "mid", "high"]


def tiers_for(X, w, rng):
    P, c = patterns(X, w)
    _, pi, th = lca(P, c, 3, rng, starts=30)
    order = np.argsort(th.mean(1))
    pi, th = pi[order], th[order]
    lp = np.log(pi)[None] + X @ np.log(th).T + (1 - X) @ np.log(1 - th).T
    post = np.exp(lp - lp.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    return post.argmax(1), post.max(1), pi


def binned(pk):
    pk = pk.copy()
    pk[0] = 0
    pk /= pk.sum()
    return np.array([pk[(K >= a) & (K <= min(b, K[-1]))].sum() for a, b in BINS])


def nb_given_hit(p, r):
    """NegBin with shape r and P(>=1) = p: solve for the mean."""
    target = np.log(1 - p)
    g = lambda lm: r * np.log(r / (r + np.exp(lm))) - target
    if g(-20) * g(25) > 0:
        return None
    m = np.exp(brentq(g, -20, 25))
    return nbinom.pmf(K, r, r / (r + m)), m


def main():
    X, w, size = load()
    r = aligned_raw()
    con = pd.to_numeric(r["phishcon"], errors="coerce").where(lambda s: s >= 0)
    band = pd.to_numeric(r["phishcon_bands"], errors="coerce").where(lambda s: s.between(1, 9))
    t = con.fillna(band.map(DK_MID)).values
    phish = X[:, 0] == 1
    t = np.where(phish, t, 0.0)                         # not hit by phishing -> 0 targeted
    rng = np.random.default_rng(0)

    fits = {}
    for glab, gm in (("Micro", size == 1), ("Small+Medium+Large", size >= 2)):
        ww = w[gm] / w[gm].mean()
        tier, cert, pi = tiers_for(X[gm], ww, rng)
        tt = t[gm]
        known = ~np.isnan(tt)
        print("\n" + "=" * 104)
        print(f"{glab}: firms {gm.sum()}, known targeted count {known.sum()}; tier shares "
              + " ".join(f"{s:.2f}" for s in pi))
        print("=" * 104)
        print(f"  {'tier':>5s} {'n':>5s} {'cert':>5s} {'p=P(t>=1)':>9s} {'lambda':>7s} {'n t>=1':>6s}   "
              f"{'1':>11s} {'2-3':>11s} {'4-10':>11s} {'11-50':>11s} {'51+':>11s}   {'mean t|t>=1':>13s}   (obs / Poisson)")
        data = []
        for k in range(3):
            m = (tier == k) & known
            p = np.average(tt[m] >= 1, weights=ww[m])
            lam = -np.log(1 - p)
            pos = m & (tt >= 1)
            x = np.round(tt[pos]).astype(int)
            wx = ww[pos]
            obs = np.array([np.average((x >= a) & (x <= b), weights=wx) for a, b in BINS])
            pred = binned(poisson.pmf(K, lam))
            print(f"  {TIER[k]:>5s} {m.sum():5d} {cert[m].mean():5.2f} {p:9.3f} {lam:7.3f} {pos.sum():6d}   " +
                  " ".join(f"{o:5.2f}/{q:5.2f}" for o, q in zip(obs, pred)) +
                  f"   {np.average(x, weights=wx):6.1f}/{lam / (1 - np.exp(-lam)):5.2f}")
            data.append((p, x, wx))
        fits[glab] = data

    print("\n" + "=" * 104)
    print("NegBin with P(t>=1) pinned per tier, shape r shared across tiers")
    print("=" * 104)

    def nll_group(r, data):
        tot = 0.0
        for p, x, wx in data:
            res = nb_given_hit(p, r)
            if res is None:
                return 1e12
            pk = res[0]
            tot -= (wx * (np.log(np.maximum(pk[np.minimum(x, K[-1])], 1e-300)) - np.log(p))).sum()
        return tot

    best_r, nll_sep = {}, 0.0
    for glab, data in fits.items():
        res = minimize_scalar(lambda lr: nll_group(np.exp(lr), data), bounds=(np.log(1e-3), np.log(50)), method="bounded")
        best_r[glab] = np.exp(res.x)
        nll_sep += res.fun
        print(f"\n  {glab}: shape r = {best_r[glab]:.3f}")
        for k, (p, x, wx) in enumerate(data):
            pk, mean = nb_given_hit(p, best_r[glab])
            obs = np.array([np.average((x >= a) & (x <= b), weights=wx) for a, b in BINS])
            exp = binned(pk)
            n_eff = wx.sum() ** 2 / (wx ** 2).sum()
            x2 = n_eff * ((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
            print(f"    {TIER[k]:>5s} n={len(x):3d} mean {mean:7.2f}  " +
                  " ".join(f"{o:.2f}/{e:.2f}" for o, e in zip(obs, exp)) +
                  f"   GOF p = {chi2.sf(x2, len(BINS) - 2):.2f}")
    alld = fits["Micro"] + fits["Small+Medium+Large"]
    res = minimize_scalar(lambda lr: nll_group(np.exp(lr), alld), bounds=(np.log(1e-3), np.log(50)), method="bounded")
    lr_stat = 2 * (res.fun - nll_sep)
    print(f"\n  one shape for both size groups: r = {np.exp(res.x):.3f};  vs separate: pseudo-LR {lr_stat:.2f} on 1 df, "
          f"p = {chi2.sf(lr_stat, 1):.2f}")


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def positive_part():
    """Hurdle view: counts among firms with t >= 1 only. Zero-truncated NegBin per tier (own mean), shape r
    shared across tiers within size group; GOF on coarse bins. Also shared r across size groups."""
    from scipy.optimize import minimize
    X, w, size = load()
    r = aligned_raw()
    con = pd.to_numeric(r["phishcon"], errors="coerce").where(lambda s: s >= 0)
    band = pd.to_numeric(r["phishcon_bands"], errors="coerce").where(lambda s: s.between(1, 9))
    t = np.where(X[:, 0] == 1, con.fillna(band.map(DK_MID)).values, 0.0)
    rng = np.random.default_rng(0)
    groups = {}
    for glab, gm in (("Micro", size == 1), ("Small+Medium+Large", size >= 2)):
        ww = w[gm] / w[gm].mean()
        tier, _, _ = tiers_for(X[gm], ww, rng)
        tt = t[gm]
        groups[glab] = [(np.round(tt[(tier == k) & (tt >= 1)]).astype(int), ww[(tier == k) & (tt >= 1)]) for k in range(3)]

    def ztnb_logpmf(x, m, rr):
        pr = rr / (rr + m)
        return nbinom.logpmf(x, rr, pr) - np.log1p(-nbinom.pmf(0, rr, pr))

    def nll(v, data):
        rr = np.exp(v[0])
        return -sum((wx * ztnb_logpmf(x, np.exp(v[1 + i]), rr)).sum() for i, (x, wx) in enumerate(data))

    print("\n" + "=" * 104)
    print("POSITIVE PART ONLY: zero-truncated NegBin per tier (own mean), shape r shared across tiers")
    print("=" * 104)
    tot_sep = 0.0
    for glab, data in groups.items():
        best = min((minimize(nll, np.r_[np.log(r0), [np.log(max(np.average(x, weights=wx), 1.1)) for x, wx in data]],
                             args=(data,), method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-8))
                    for r0 in (0.05, 0.2, 1.0)), key=lambda z: z.fun)
        tot_sep += best.fun
        rr = np.exp(best.x[0])
        print(f"\n  {glab}: shared shape r = {rr:.3f}")
        for i, (x, wx) in enumerate(data):
            m = np.exp(best.x[1 + i])
            pk = nbinom.pmf(K, rr, rr / (rr + m))
            obs = np.array([np.average((x >= a) & (x <= b), weights=wx) for a, b in BINS])
            exp = binned(pk)
            n_eff = wx.sum() ** 2 / (wx ** 2).sum()
            x2 = n_eff * ((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
            print(f"    {TIER[i]:>5s} n={len(x):3d} NB mean {m:7.1f}   " +
                  " ".join(f"{o:.2f}/{e:.2f}" for o, e in zip(obs, exp)) + f"   GOF p = {chi2.sf(x2, len(BINS) - 2):.2f}")
    alld = groups["Micro"] + groups["Small+Medium+Large"]
    best = min((minimize(nll, np.r_[np.log(r0), [np.log(max(np.average(x, weights=wx), 1.1)) for x, wx in alld]],
                         args=(alld,), method="Nelder-Mead", options=dict(maxiter=40000, xatol=1e-6, fatol=1e-8))
                for r0 in (0.05, 0.2, 1.0)), key=lambda z: z.fun)
    lr = 2 * (best.fun - tot_sep)
    print(f"\n  one shape for both size groups: r = {np.exp(best.x[0]):.3f}; vs separate pseudo-LR {lr:.2f} on 1 df, "
          f"p = {chi2.sf(lr, 1):.2f}")


if __name__ == "__main__" and "positive" in __import__("sys").argv:
    positive_part()


def logseries_part():
    """Positive part as a logarithmic-series law (the r -> 0 limit of zero-truncated NegBin):
    P(k) = -theta^k / (k ln(1 - theta)), k >= 1. theta per tier; GOF on coarse bins; fitted mean
    and the observed weighted mean (the log-series mean is -theta / ((1 - theta) ln(1 - theta)))."""
    from scipy.stats import logser
    X, w, size = load()
    r = aligned_raw()
    con = pd.to_numeric(r["phishcon"], errors="coerce").where(lambda s: s >= 0)
    band = pd.to_numeric(r["phishcon_bands"], errors="coerce").where(lambda s: s.between(1, 9))
    t = np.where(X[:, 0] == 1, con.fillna(band.map(DK_MID)).values, 0.0)
    rng = np.random.default_rng(0)
    Kp = np.arange(1, 5001)
    print("\n" + "=" * 104)
    print("POSITIVE PART as logarithmic series, theta per tier")
    print("=" * 104)
    for glab, gm in (("Micro", size == 1), ("Small+Medium+Large", size >= 2)):
        ww = w[gm] / w[gm].mean()
        tier, _, _ = tiers_for(X[gm], ww, rng)
        tt = t[gm]
        print(f"\n  {glab}")
        for k in range(3):
            m = (tier == k) & (tt >= 1)
            x, wx = np.round(tt[m]).astype(int), ww[m]
            res = minimize_scalar(lambda z: -(wx * logser.logpmf(x, 1 / (1 + np.exp(-z)))).sum(),
                                  bounds=(-5, 15), method="bounded")
            th = 1 / (1 + np.exp(-res.x))
            pk = np.r_[0.0, logser.pmf(Kp, th)]
            obs = np.array([np.average((x >= a) & (x <= b), weights=wx) for a, b in BINS])
            exp = binned(pk)
            n_eff = wx.sum() ** 2 / (wx ** 2).sum()
            x2 = n_eff * ((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
            fmean = -th / ((1 - th) * np.log(1 - th))
            print(f"    {TIER[k]:>5s} n={len(x):3d} theta {th:.4f} fitted mean {fmean:7.1f} (obs {np.average(x, weights=wx):6.1f})  " +
                  " ".join(f"{o:.2f}/{e:.2f}" for o, e in zip(obs, exp)) + f"   GOF p = {chi2.sf(x2, len(BINS) - 2):.2f}")


if __name__ == "__main__" and "logser" in __import__("sys").argv:
    logseries_part()
