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


if __name__ == "__main__":
    main()
