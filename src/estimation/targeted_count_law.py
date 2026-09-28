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


def empirical():
    """Raw empirical distribution of the targeted count, per size group x tier: exact values of t
    (exact phishcon answers only, no band midpoints), counts of firms and weighted shares; plus quantiles."""
    X, w, size = load()
    r = aligned_raw()
    con = pd.to_numeric(r["phishcon"], errors="coerce").where(lambda s: s >= 0).values
    phish = X[:, 0] == 1
    t = np.where(phish, con, 0.0)
    rng = np.random.default_rng(0)
    for glab, gm in (("Micro", size == 1), ("Small+Medium+Large", size >= 2)):
        ww = w[gm] / w[gm].mean()
        tier, _, _ = tiers_for(X[gm], ww, rng)
        tt = t[gm]
        print("\n" + "=" * 90)
        print(f"{glab}  (exact answers only; 'dk band' firms excluded: {int((phish[gm] & np.isnan(tt)).sum())})")
        print("=" * 90)
        for k in range(3):
            m = (tier == k) & ~np.isnan(tt)
            v, wv = tt[m], ww[m]
            print(f"\n  tier {TIER[k]}: n={m.sum()}, zeros {int((v == 0).sum())} (w-share {wv[v == 0].sum() / wv.sum():.2f})")
            vals = np.unique(v[v > 0])
            pos = v > 0
            line = [f"{int(x)}:{int((v == x).sum())}" for x in vals]
            print("    value:#firms  " + "  ".join(line))
            if pos.sum():
                q = np.percentile(v[pos], [10, 25, 50, 75, 90])
                print(f"    t>=1 quantiles p10/25/50/75/90: " + " / ".join(f"{x:g}" for x in q) + f"   max {v.max():g}")


if __name__ == "__main__" and "empirical" in __import__("sys").argv:
    empirical()


def tier_profiles():
    """What the tiers are: 3-class flag model per size group; class share, firms assigned (most likely tier),
    per-type hit probability, and mean # types hit among assigned firms."""
    from type_cooccurrence_structure import SHORT
    X, w, size = load()
    rng = np.random.default_rng(0)
    for glab, gm in (("Micro", size == 1), ("Small+Medium+Large", size >= 2)):
        Xg, ww = X[gm], w[gm] / w[gm].mean()
        P, c = patterns(Xg, ww)
        _, pi, th = lca(P, c, 3, rng, starts=30)
        order = np.argsort(th.mean(1))
        pi, th = pi[order], th[order]
        tier, _, _ = tiers_for(Xg, ww, np.random.default_rng(0))
        print(f"\n{glab}")
        print(f"  {'tier':>5s} {'share':>6s} {'n':>5s} {'mean#types':>10s}  " + " ".join(f"{s:>6s}" for s in SHORT))
        for k in range(3):
            m = tier == k
            print(f"  {TIER[k]:>5s} {pi[k]:6.2f} {m.sum():5d} {np.average(Xg[m].sum(1), weights=ww[m]):10.2f}  "
                  + " ".join(f"{v:6.2f}" for v in th[k]))


if __name__ == "__main__" and "tiers" in __import__("sys").argv:
    tier_profiles()


def zipf_check():
    """Positive part of the targeted count (t >= 1, exact phishcon answers only), unweighted: log-series vs
    Zipf vs one-inflated log-series, per size group (all tiers pooled) and per size group x tier.
    MLE; obs vs expected on BINS; AIC; parametric bootstrap p for Pearson X2 (refit per replicate, 300 reps)."""
    from scipy.optimize import minimize, minimize_scalar
    from scipy.stats import logser, zipf
    tf = lambda u: 1 / (1 + np.exp(-u))

    def fit(name, x):
        if name == "zipf":
            a = minimize_scalar(lambda a: -zipf.logpmf(x, a).sum(), bounds=(1.01, 10), method="bounded").x
            return (a,), zipf.logpmf(x, a).sum()
        if name == "logser":
            u = minimize_scalar(lambda u: -logser.logpmf(x, tf(u)).sum(), bounds=(-10, 15), method="bounded").x
            return (tf(u),), logser.logpmf(x, tf(u)).sum()
        f = lambda z: -np.log(np.where(x == 1, tf(z[0]) + (1 - tf(z[0])) * logser.pmf(x, tf(z[1])),
                                       (1 - tf(z[0])) * logser.pmf(x, tf(z[1])))).sum()
        o = minimize(f, [0.0, 4.0], method="Nelder-Mead")
        return (tf(o.x[0]), tf(o.x[1])), -o.fun

    def cdf(name, k, p):
        if name == "zipf":
            return zipf.cdf(k, *p)
        if name == "logser":
            return logser.cdf(k, *p)
        return p[0] + (1 - p[0]) * logser.cdf(k, p[1]) if k >= 1 else 0.0

    def rvs(name, p, n, rng):
        if name == "zipf":
            return zipf.rvs(*p, size=n, random_state=rng)
        if name == "logser":
            return logser.rvs(*p, size=n, random_state=rng)
        return np.where(rng.random(n) < p[0], 1, logser.rvs(p[1], size=n, random_state=rng))

    def exp_obs(name, p, x):
        e = np.array([len(x) * ((cdf(name, b, p) if b < 10**6 else 1.0) - cdf(name, a - 1, p)) for a, b in BINS])
        o = np.array([((x >= a) & (x <= b)).sum() for a, b in BINS])
        return e, o

    def report(label, x, rng):
        print(f"\n  {label}: n(t>=1)={len(x)}, median {np.median(x):g}, max {x.max():g}")
        print(f"    {'obs':>34s} " + " ".join(f"{o:6d}" for o in exp_obs('zipf', (2.0,), x)[1])
              + "   (1 / 2-3 / 4-10 / 11-50 / 51+)")
        for name, k in (("logser", 1), ("zipf", 1), ("oils", 2)):
            p, ll = fit(name, x)
            e, o = exp_obs(name, p, x)
            x2 = ((o - e) ** 2 / e).sum()
            sx = []
            for _ in range(300):
                s = rvs(name, p, len(x), rng)
                ps, _ = fit(name, s)
                es, os_ = exp_obs(name, ps, s)
                sx.append(((os_ - es) ** 2 / es).sum())
            par = ",".join(f"{v:.3f}" for v in p)
            print(f"    {name:6s} ({par:>11s}) AIC {2 * k - 2 * ll:7.1f} p {np.mean(np.array(sx) >= x2):.3f} "
                  + " ".join(f"{v:6.1f}" for v in e))

    X, w, size = load()
    r = aligned_raw()
    con = pd.to_numeric(r["phishcon"], errors="coerce").where(lambda s: s >= 0).values
    t = np.where(X[:, 0] == 1, con, 0.0)
    rng = np.random.default_rng(0)
    for glab, gm in (("Micro", size == 1), ("Small+Medium+Large", size >= 2)):
        tier, _, _ = tiers_for(X[gm], w[gm] / w[gm].mean(), np.random.default_rng(0))
        tt = t[gm]
        print("\n" + "=" * 100 + f"\n{glab}\n" + "=" * 100)
        report("all tiers", tt[tt >= 1].astype(int), rng)
        for k in range(3):
            x = tt[(tier == k) & (tt >= 1)].astype(int)
            report(f"tier {TIER[k]}", x, rng)


if __name__ == "__main__" and "zipf" in __import__("sys").argv:
    zipf_check()


def log_shape():
    """Detailed shape of the positive targeted count (exact answers, unweighted): firms per doubling bin
    [1], [2], [3-4], [5-8], ... and firms per unit of k in each bin (density on the count scale), per size group
    (all tiers) and Small+ mid tier; plus share of answers at round values."""
    X, w, size = load()
    r = aligned_raw()
    con = pd.to_numeric(r["phishcon"], errors="coerce").where(lambda s: s >= 0).values
    t = np.where(X[:, 0] == 1, con, 0.0)
    edges = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025]
    for glab, gm, tk in (("Micro all tiers", size == 1, None), ("Small+ all tiers", size >= 2, None),
                         ("Small+ mid tier", size >= 2, 1)):
        x = t[gm]
        if tk is not None:
            tier, _, _ = tiers_for(X[gm], w[gm] / w[gm].mean(), np.random.default_rng(0))
            x = x[tier == tk]
        x = x[x >= 1]
        print(f"\n{glab}: n={len(x)}")
        print(f"  {'bin':>9s} {'firms':>5s} {'share':>6s} {'per unit k':>10s}")
        for a, b in zip(edges[:-1], edges[1:]):
            m = (x >= a) & (x < b)
            print(f"  {f'{a}-{b - 1}' if b - 1 > a else str(a):>9s} {m.sum():5d} {m.mean():6.2f} {m.sum() / (b - a):10.2f}")
        rnd = np.isin(x % 10, [0]) | np.isin(x, [5, 12, 15, 24, 25, 52, 365])
        print(f"  round answers (x10, 5, 12, 15, 24, 25, 52, 365): {rnd.mean():.2f}; "
              f"most common values: " + ", ".join(f"{int(v)}:{c}" for v, c in
                                                   sorted(zip(*np.unique(x, return_counts=True)), key=lambda z: -z[1])[:10]))


if __name__ == "__main__" and "logshape" in __import__("sys").argv:
    log_shape()


def zipf_body(caps=(16, 32)):
    """Zipf fitted to the body only: right-truncated Zipf P(k) = k^-a / sum_{j<=cap} j^-a on firms with
    1 <= t <= cap (exact answers, unweighted). Obs vs expected per doubling bin, bootstrap X2 p (refit, 1000 reps),
    and share of hit firms beyond the cap."""
    from scipy.optimize import minimize_scalar
    X, w, size = load()
    r = aligned_raw()
    con = pd.to_numeric(r["phishcon"], errors="coerce").where(lambda s: s >= 0).values
    t = np.where(X[:, 0] == 1, con, 0.0)
    rng = np.random.default_rng(0)
    for cap in caps:
        ks = np.arange(1, cap + 1)
        bins = [(a, min(b, cap)) for a, b in [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32)] if a <= cap]
        pmf = lambda a: ks ** -a / (ks ** -a).sum()
        fit = lambda x: minimize_scalar(lambda a: -(np.log(pmf(a))[x - 1]).sum(), bounds=(0.01, 6), method="bounded").x
        e_o = lambda a, x: (np.array([len(x) * pmf(a)[lo - 1:hi].sum() for lo, hi in bins]),
                            np.array([((x >= lo) & (x <= hi)).sum() for lo, hi in bins]))
        print(f"\n===== cap {cap} =====")
        for glab, gm, tk in (("Micro all", size == 1, None), ("Small+ all", size >= 2, None),
                             ("Micro low", size == 1, 0), ("Micro mid", size == 1, 1), ("Micro high", size == 1, 2),
                             ("Small+ low", size >= 2, 0), ("Small+ mid", size >= 2, 1), ("Small+ high", size >= 2, 2)):
            x = t[gm]
            if tk is not None:
                tier, _, _ = tiers_for(X[gm], w[gm] / w[gm].mean(), np.random.default_rng(0))
                x = x[tier == tk]
            pos = x[x >= 1].astype(int)
            xb = pos[pos <= cap]
            a = fit(xb)
            e, o = e_o(a, xb)
            x2 = ((o - e) ** 2 / e).sum()
            sx = []
            for _ in range(1000):
                s = rng.choice(ks, len(xb), p=pmf(a))
                es, os_ = e_o(fit(s), s)
                sx.append(((os_ - es) ** 2 / es).sum())
            print(f"  {glab:11s} n body {len(xb):3d} (beyond cap {np.mean(pos > cap):.2f})  alpha {a:.2f}  "
                  f"p {np.mean(np.array(sx) >= x2):.3f}   obs " + " ".join(f"{v:3d}" for v in o)
                  + "   exp " + " ".join(f"{v:5.1f}" for v in e))


if __name__ == "__main__" and "zipfbody" in __import__("sys").argv:
    zipf_body()


def zipf_body_extrapolate(cap=16):
    """Extrapolate the body-fitted Zipf (right-truncated at cap, fitted on 1 <= t <= cap) past the cap with the
    body's normalisation: expected firms at k = n_body * k^-a / sum_{j<=cap} j^-a. Compare with observed firms
    per doubling bin beyond cap, up to 1024 (exact answers, unweighted)."""
    from scipy.optimize import minimize_scalar
    X, w, size = load()
    r = aligned_raw()
    con = pd.to_numeric(r["phishcon"], errors="coerce").where(lambda s: s >= 0).values
    t = np.where(X[:, 0] == 1, con, 0.0)
    ks = np.arange(1, cap + 1)
    tail_bins = [(a, 2 * a - 2) for a in (17, 33, 65, 129, 257, 513) if a > cap]
    print(f"\ncap {cap}; tail bins " + " ".join(f"{a}-{b}" for a, b in tail_bins) + " | total beyond cap")
    for glab, gm, tk in (("Micro all", size == 1, None), ("Small+ all", size >= 2, None),
                         ("Micro low", size == 1, 0), ("Micro mid", size == 1, 1), ("Micro high", size == 1, 2),
                         ("Small+ low", size >= 2, 0), ("Small+ mid", size >= 2, 1), ("Small+ high", size >= 2, 2)):
        x = t[gm]
        if tk is not None:
            tier, _, _ = tiers_for(X[gm], w[gm] / w[gm].mean(), np.random.default_rng(0))
            x = x[tier == tk]
        pos = x[x >= 1].astype(int)
        xb = pos[pos <= cap]
        a = minimize_scalar(lambda a: -(np.log(ks ** -a / (ks ** -a).sum())[xb - 1]).sum(),
                            bounds=(0.01, 6), method="bounded").x
        z = (ks ** -a).sum()
        ex = [len(xb) * (np.arange(lo, hi + 1) ** -a).sum() / z for lo, hi in tail_bins]
        ob = [int(((pos >= lo) & (pos <= hi)).sum()) for lo, hi in tail_bins]
        print(f"  {glab:11s} a {a:.2f}  obs " + " ".join(f"{v:5d}" for v in ob) + f" | {sum(ob):4d}"
              + "    extrap " + " ".join(f"{v:5.1f}" for v in ex) + f" | {sum(ex):6.1f}")


if __name__ == "__main__" and "zipfextrap" in __import__("sys").argv:
    zipf_body_extrapolate(16)
    zipf_body_extrapolate(32)


def dlognorm_check():
    """Discretised lognormal for the positive targeted count (exact answers, unweighted):
    k = round(X), X ~ lognormal(mu, sigma), conditioned on k >= 1:
    P(k) = [Phi((ln(k+.5)-mu)/s) - Phi((ln(k-.5)-mu)/s)] / [1 - Phi((ln .5 - mu)/s)].
    Per size group (all tiers) and size x tier: MLE, obs vs expected per doubling bin to 1024, AIC vs log-series,
    parametric bootstrap X2 p (refit, 300 reps), fitted vs empirical mean given t >= 1."""
    from scipy.optimize import minimize, minimize_scalar
    from scipy.stats import logser, norm
    DB = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 10**6)]

    def lp(x, mu, s):
        lo = np.where(x == 1, -np.inf, (np.log(np.maximum(x - 0.5, 1e-12)) - mu) / s)
        num = norm.cdf((np.log(x + 0.5) - mu) / s) - norm.cdf(lo)
        return np.log(np.maximum(num, 1e-300)) - np.log(norm.sf((np.log(0.5) - mu) / s))

    def fit(x):
        lx = np.log(x)
        o = minimize(lambda z: -lp(x, z[0], np.exp(z[1])).sum(), [lx.mean(), np.log(lx.std() + 0.3)],
                     method="Nelder-Mead")
        return o.x[0], np.exp(o.x[1]), -o.fun

    def binprob(mu, s):
        c = lambda k: norm.cdf((np.log(k + 0.5) - mu) / s)
        z = norm.sf((np.log(0.5) - mu) / s)
        return np.array([((c(b) if b < 10**6 else 1.0) - (c(a - 1) if a > 1 else norm.cdf((np.log(0.5) - mu) / s))) / z
                         for a, b in DB])

    def rvs(mu, s, n, rng):
        out = np.empty(0, int)
        while len(out) < n:
            k = np.rint(np.exp(rng.normal(mu, s, 4 * n))).astype(np.int64)
            out = np.concatenate([out, k[k >= 1]])
        return out[:n]

    def x2(o, e):
        m = e > 0
        return ((o[m] - e[m]) ** 2 / e[m]).sum()

    X, w, size = load()
    r = aligned_raw()
    con = pd.to_numeric(r["phishcon"], errors="coerce").where(lambda s: s >= 0).values
    t = np.where(X[:, 0] == 1, con, 0.0)
    rng = np.random.default_rng(0)
    print("bins: " + " ".join(f"{a}" if a == b else (f"{a}-{b}" if b < 10**6 else f"{a}+") for a, b in DB))
    for glab, gm, tk in (("Micro all", size == 1, None), ("Small+ all", size >= 2, None),
                         ("Micro low", size == 1, 0), ("Micro mid", size == 1, 1), ("Micro high", size == 1, 2),
                         ("Small+ low", size >= 2, 0), ("Small+ mid", size >= 2, 1), ("Small+ high", size >= 2, 2)):
        x = t[gm]
        if tk is not None:
            tier, _, _ = tiers_for(X[gm], w[gm] / w[gm].mean(), np.random.default_rng(0))
            x = x[tier == tk]
        x = x[x >= 1].astype(np.int64)
        n = len(x)
        mu, s, ll = fit(x)
        o = np.array([((x >= a) & (x <= b)).sum() for a, b in DB])
        e = n * binprob(mu, s)
        stat = x2(o, e)
        sx = []
        for _ in range(300):
            y = rvs(mu, s, n, rng)
            m2, s2, _ = fit(y)
            sx.append(x2(np.array([((y >= a) & (y <= b)).sum() for a, b in DB]), n * binprob(m2, s2)))
        th = minimize_scalar(lambda u: -logser.logpmf(x, 1 / (1 + np.exp(-u))).sum(), bounds=(-10, 15),
                             method="bounded").x
        ll_ls = logser.logpmf(x, 1 / (1 + np.exp(-th))).sum()
        ks = np.arange(1, 200001)
        pk = np.exp(lp(ks, mu, s))
        print(f"\n  {glab:11s} n={n:3d}  mu {mu:5.2f} sigma {s:4.2f}  AIC dlnorm {4 - 2 * ll:7.1f} vs logser "
              f"{2 - 2 * ll_ls:7.1f}  p {np.mean(np.array(sx) >= stat):.3f}   mean|t>=1: fit {(ks * pk).sum():6.1f} "
              f"(to 1e3: {(ks[:1000] * pk[:1000]).sum() / pk[:1000].sum():5.1f})  emp {x.mean():6.1f}")
        print("     obs " + " ".join(f"{v:5d}" for v in o))
        print("     exp " + " ".join(f"{v:5.1f}" for v in e))


if __name__ == "__main__" and "dlnorm" in __import__("sys").argv:
    dlognorm_check()
