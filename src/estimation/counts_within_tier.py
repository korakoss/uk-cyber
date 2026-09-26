"""Step 4 of the counts argument: channel count shape once the shared latent is factored out.

Tiers come from the count-agnostic flag model of Step 3 (3-class latent class model,
frailty_shape_flags.py), fitted within size group. Leave-one-out: for phishing, tiers use the
OTHER types' flags only, excluding phishing AND impersonation (co-labelled with phishing); for
ransomware, excluding ransomware AND malware. Within each tier x size group, the exact count
distribution (phishing count N; ransomware count) among firms that report it:
  mean, variance, dispersion (var/mean), quantiles, and zero-truncated Poisson vs NegBin fits
  (counts are only observed for flagged firms, so N >= 1).
If the latent explained all spread, within-tier counts would look Poisson. Tier assignment is
uncertain (reported as mean posterior certainty); misassignment mixes tiers and inflates spread,
so leftover overdispersion is an upper bound on channel-private spread.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/counts_within_tier.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

import joint_five_channel as f
from type_cooccurrence_structure import load
from frailty_shape_flags import lca, patterns

TIER = ["low", "mid", "high"]


def posteriors(Xo, w, rng, K=3):
    P, c = patterns(Xo, w)
    _, pi, th = lca(P, c, K, rng, starts=20)
    order = np.argsort(th.mean(1))
    pi, th = pi[order], th[order]
    lp = np.log(pi)[None] + Xo @ np.log(th).T + (1 - Xo) @ np.log(1 - th).T
    post = np.exp(lp - lp.max(1, keepdims=True))
    return post / post.sum(1, keepdims=True), pi


def fit_trunc(x, w):
    x = x.astype(int)

    def nll_p(v):
        lam = np.exp(v[0])
        lp = poisson.logpmf(x, lam) - np.log1p(-np.exp(-lam))
        return -(w * lp).sum()

    def nll_nb(v):
        m, r = np.exp(v)
        p = r / (r + m)
        lp = nbinom.logpmf(x, r, p) - np.log1p(-np.clip(nbinom.pmf(0, r, p), 0, 1 - 1e-12))
        return -(w * lp).sum()

    rp = minimize(nll_p, [np.log(max(np.average(x, weights=w), 1.01))], method="Nelder-Mead")
    rn = min((minimize(nll_nb, [np.log(max(np.average(x, weights=w), 1.01)), np.log(r0)], method="Nelder-Mead")
              for r0 in (0.1, 0.5, 2.0)), key=lambda r: r.fun)
    return rp.fun, rn.fun, np.exp(rn.x[1])


def analyse(d, X, w, size, target, exclude, label, rng):
    keep_cols = [i for i in range(X.shape[1]) if i not in exclude]
    print("\n" + "=" * 96)
    print(label)
    print("=" * 96)
    print(f"  {'size':>6s} {'tier':>5s} {'tier share':>10s} {'n':>4s} {'certainty':>9s} {'mean':>7s} {'var/mean':>9s}"
          f" {'q25/50/75/90':>18s}   {'NB r':>6s} {'LL gain NB vs Pois':>18s}")
    for slab, sizes in (("Micro", [1]), ("Rest", [2, 3, 4])):
        m = np.isin(size, sizes)
        ww = w[m] / w[m].mean()
        post, pi = posteriors(X[m][:, keep_cols], ww, rng)
        tier = post.argmax(1)
        cert = post.max(1)
        cnt = target[m]
        for t in range(3):
            sel = (tier == t) & ~np.isnan(cnt)
            if sel.sum() < 4:
                print(f"  {slab:>6s} {TIER[t]:>5s} {pi[t]:10.2f} {sel.sum():4d}   (too few)")
                continue
            x, wx = cnt[sel], ww[sel]
            mu = np.average(x, weights=wx)
            var = np.average((x - mu) ** 2, weights=wx)
            q = np.percentile(np.repeat(x, np.maximum(1, np.round(wx * 10).astype(int))), [25, 50, 75, 90])
            llp, lln, r = fit_trunc(x, wx)
            print(f"  {slab:>6s} {TIER[t]:>5s} {pi[t]:10.2f} {sel.sum():4d} {cert[sel].mean():9.2f} {mu:7.1f} {var / mu:9.1f}"
                  f"   {'/'.join(f'{v:.0f}' for v in q):>16s}   {r:6.2f} {llp - lln:18.1f}")


def main():
    X, w, size = load()
    d = f.load_firms()
    rng = np.random.default_rng(0)
    N = np.where(d["fP"].values == 1, d["N"].values, np.nan)
    analyse(d, X, w, size, N, exclude=[0, 1],
            label="PHISHING COUNT within tiers (tiers from flags excluding phishing + impersonation)", rng=rng)
    NR = d["NR"].values
    analyse(d, X, w, size, NR, exclude=[2, 3],
            label="RANSOMWARE COUNT within tiers (tiers from flags excluding ransomware + malware)", rng=rng)


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def poisson_prediction_test():
    """Sharper Step-4 test. Tiers from the Step-3 flag LCA on ALL 10 types (within size).
    Under Poisson-within-tier, a tier's hit probability p pins its mean: lambda = -ln(1-p), so
    counts among hit firms follow zero-truncated Poisson(lambda) -- a prediction with nothing
    fitted to counts. Compare with observed counts (phishing N; ransomware count).
    Co-labelling (more phishing incidents -> more impersonation ticks -> pushed to higher tiers)
    biases the low tier towards FEWER high counts, i.e. against finding overdispersion."""
    X, w, size = load()
    d = f.load_firms()
    rng = np.random.default_rng(0)
    for chan, col, cnt in (("PHISHING", 0, np.where(d["fP"].values == 1, d["N"].values, np.nan)),
                           ("RANSOMWARE", 2, d["NR"].values)):
        print("\n" + "=" * 100)
        print(f"{chan}: observed counts among hit firms vs Poisson prediction lambda = -ln(1 - p_tier)")
        print("=" * 100)
        print(f"  {'size':>6s} {'tier':>5s} {'share':>6s} {'p_hit':>6s} {'lambda':>7s} {'n':>4s} {'cert':>5s}"
              f"  {'P(1)':>11s} {'P(2-3)':>11s} {'P(4-10)':>11s} {'P(>10)':>11s} {'mean':>13s}   (obs / Poisson)")
        for slab, sizes in (("Micro", [1]), ("Rest", [2, 3, 4])):
            m = np.isin(size, sizes)
            ww = w[m] / w[m].mean()
            P, c = patterns(X[m], ww)
            _, pi, th = lca(P, c, 3, rng, starts=20)
            order = np.argsort(th.mean(1))
            pi, th = pi[order], th[order]
            Xm = X[m]
            lp = np.log(pi)[None] + Xm @ np.log(th).T + (1 - Xm) @ np.log(1 - th).T
            post = np.exp(lp - lp.max(1, keepdims=True))
            post /= post.sum(1, keepdims=True)
            tier, cert = post.argmax(1), post.max(1)
            x_all = cnt[m]
            for t in range(3):
                p = th[t, col]
                lam = -np.log(1 - p)
                sel = (tier == t) & ~np.isnan(x_all)
                if sel.sum() == 0:
                    continue
                x, wx = x_all[sel], ww[sel]
                k = np.arange(0, 5000)
                pk = poisson.pmf(k, lam)
                pk[0] = 0
                pk /= pk.sum()
                bins = [(1, 1), (2, 3), (4, 10), (11, 10**9)]
                obs = [np.average((x >= a) & (x <= b), weights=wx) for a, b in bins]
                pred = [pk[(k >= a) & (k <= b)].sum() for a, b in bins]
                print(f"  {slab:>6s} {TIER[t]:>5s} {pi[t]:6.2f} {p:6.3f} {lam:7.3f} {sel.sum():4d} {cert[sel].mean():5.2f}  " +
                      " ".join(f"{o:5.2f}/{q:5.2f}" for o, q in zip(obs, pred)) +
                      f"  {np.average(x, weights=wx):6.1f}/{(k * pk).sum():5.2f}")


if __name__ == "__main__" and "poisson" in __import__("sys").argv:
    poisson_prediction_test()
