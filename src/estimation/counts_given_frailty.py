"""Count laws CONDITIONAL on frailty.

Frailty = 3-class latent class model on the 10 type flags, fitted within Micro and within Small+Medium+Large
(as in targeted_count_law.tiers_for); each firm carries its full posterior over tiers (low/mid/high), not a hard
assignment. A channel's positive count k (>= 1) then has likelihood sum_c post_ic f(k | tier c).
Candidate conditional laws, compared by AIC:
  zipf      - Zipf alpha, tier-free (scale-free: a rate multiplier does not change the shape of k >= 1)
  ztpois    - zero-truncated Poisson, lambda per tier (frailty = rate multiplier, no within-tier spread)
  dln_pool  - discretised lognormal, one (mu, sigma)
  dln_tier  - discretised lognormal, mu per tier (log of the tier multiplier), shared sigma
Expected counts per bin = sum_i sum_c post_ic P(bin | tier c). Unweighted.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/counts_given_frailty.py rans
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, poisson, zipf

from frailty_shape_flags import lca, patterns
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "..", "build", "tier_posteriors.npy")
TIER = ["low", "mid", "high"]
BINS = [(1, 1), (2, 2), (3, 4), (5, 10), (11, 10**6)]


def tier_posteriors():
    """(n, 3) posterior over tiers ordered by mean hit probability, per size group; cached."""
    if os.path.exists(CACHE):
        return np.load(CACHE)
    X, w, size = load()
    post = np.zeros((len(X), 3))
    for gm in (size == 1, size >= 2):
        Xg, wg = X[gm], w[gm] / w[gm].mean()
        P, c = patterns(Xg, wg)
        _, pi, th = lca(P, c, 3, np.random.default_rng(0), starts=30)
        o = np.argsort(th.mean(1))
        pi, th = pi[o], th[o]
        lp = np.log(pi)[None] + Xg @ np.log(th).T + (1 - Xg) @ np.log(1 - th).T
        p = np.exp(lp - lp.max(1, keepdims=True))
        post[gm] = p / p.sum(1, keepdims=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.save(CACHE, post)
    return post


def dln_logpmf(k, mu, s):
    lo = (np.log(np.maximum(k - 0.5, 0.5)) - mu) / s
    hi = (np.log(k + 0.5) - mu) / s
    a, b = norm.logsf(lo), norm.logsf(hi)
    return a + np.log(np.maximum(-np.expm1(b - a), 1e-300)) - norm.logsf((np.log(0.5) - mu) / s)


def dln_bin(a, b, mu, s):
    z = norm.logsf((np.log(0.5) - mu) / s)
    top = lambda k: np.exp(norm.logsf((np.log(k - 0.5) - mu) / s) - z) if k > 1 else 1.0
    return top(a) - (top(b + 1) if b < 10**6 else 0.0)


def ztp_logpmf(k, lam):
    return poisson.logpmf(k, lam) - np.log(-np.expm1(-lam))


def ztp_bin(a, b, lam):
    hi = poisson.cdf(b, lam) if b < 10**6 else 1.0
    return (hi - poisson.cdf(a - 1, lam)) / (-np.expm1(-lam))


MODELS = {
    # name: (n params, init, per-tier logpmf(k, z) -> (n, 3), per-tier bin prob(a, b, z) -> (3,))
    "zipf": (1, [0.0],
             lambda k, z: np.repeat(zipf.logpmf(k, 1 + np.exp(z[0]))[:, None], 3, 1),
             lambda a, b, z: np.repeat((zipf.cdf(b, 1 + np.exp(z[0])) if b < 10**6 else 1.0)
                                       - zipf.cdf(a - 1, 1 + np.exp(z[0])), 3)),
    "ztpois": (3, [0.0, 0.5, 1.0],
               lambda k, z: np.column_stack([ztp_logpmf(k, np.exp(z[c])) for c in range(3)]),
               lambda a, b, z: np.array([ztp_bin(a, b, np.exp(z[c])) for c in range(3)])),
    "dln_pool": (2, [0.0, 0.5],
                 lambda k, z: np.repeat(dln_logpmf(k, z[0], np.exp(z[1]))[:, None], 3, 1),
                 lambda a, b, z: np.repeat(dln_bin(a, b, z[0], np.exp(z[1])), 3)),
    "dln_tier": (4, [-1.0, 0.0, 1.0, 0.5],
                 lambda k, z: np.column_stack([dln_logpmf(k, max(z[c], -40), np.exp(z[3])) for c in range(3)]),
                 lambda a, b, z: np.array([dln_bin(a, b, max(z[c], -40), np.exp(z[3])) for c in range(3)])),
}


def fit(k, post, name):
    npar, z0, lpm, _ = MODELS[name]
    nll = lambda z: -np.log(np.maximum((post * np.exp(lpm(k, z))).sum(1), 1e-300)).sum()
    starts = [z0] + [list(np.array(z0) + d) for d in (-2.0, 2.0, -6.0)]
    best = min((minimize(nll, s, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
                for s in starts), key=lambda o: o.fun)
    return best.x, -best.fun


def report(label, k, post, bins=None):
    bins = bins or BINS
    k = k.astype(np.int64)
    obs = np.array([((k >= a) & (k <= b)).sum() for a, b in bins])
    print(f"\n{label}: n={len(k)}; tier posterior mass low/mid/high "
          + " / ".join(f"{v:.1f}" for v in post.sum(0)) + "   values "
          + "  ".join(f"{int(v)}:{int((k == v).sum())}" for v in np.unique(k)))
    print(f"  {'obs':>52s} " + " ".join(f"{o:5d}" for o in obs) + "   (" + " / ".join(f"{a}" if a == b else (f"{a}-{b}" if b < 10**6 else f"{a}+") for a, b in bins) + ")")
    for name, (npar, _, _, binf) in MODELS.items():
        z, ll = fit(k, post, name)
        e = np.array([(post * binf(a, b, z)[None]).sum() for a, b in bins])
        if name == "zipf":
            par = f"alpha {1 + np.exp(z[0]):.2f}"
        elif name == "ztpois":
            par = "lambda " + "/".join(f"{np.exp(v):.2f}" for v in z)
        elif name == "dln_pool":
            par = f"mu {z[0]:.2f} sigma {np.exp(z[1]):.2f}"
        else:
            par = "mu " + "/".join(f"{max(v, -40):.1f}" for v in z[:3]) + f" sigma {np.exp(z[3]):.2f}"
        print(f"  {name:8s} {par:34s} AIC {2 * npar - 2 * ll:6.1f}   " + " ".join(f"{v:5.1f}" for v in e))


def rans():
    post = tier_posteriors()
    r = aligned_raw()
    v = pd.to_numeric(r["Cybercrime_ranssum"], errors="coerce").values
    m = v >= 1
    report("Ransomware (ranssum)", v[m], post[m])
    m2 = m & (v != 100)
    report("Ransomware, 100 dropped", v[m2], post[m2])


if __name__ == "__main__" and "rans" in sys.argv:
    rans()


def rans_by_tier():
    """Ransomware counts listed by most likely tier (and its posterior), per size group."""
    X, w, size = load()
    post = tier_posteriors()
    v = pd.to_numeric(aligned_raw()["Cybercrime_ranssum"], errors="coerce").values
    for g, gm in (("Micro", size == 1), ("Small+", size >= 2)):
        for c in range(3):
            m = gm & (v >= 1) & (post.argmax(1) == c)
            print(f"  {g:6s} {TIER[c]:4s} n={m.sum():2d}  values " + " ".join(str(int(x)) for x in np.sort(v[m]))
                  + f"   mean post {post[m, c].mean() if m.any() else float('nan'):.2f}")


if __name__ == "__main__" and "ranstier" in sys.argv:
    rans_by_tier()


def targeted():
    """Targeted phishing (exact phishcon >= 1, phishing-flagged firms), per size group, with tier posteriors."""
    X, w, size = load()
    post = tier_posteriors()
    v = pd.to_numeric(aligned_raw()["phishcon"], errors="coerce").values
    v = np.where(X[:, 0] == 1, v, np.nan)
    db = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 10**6)]
    for g, gm in (("Micro", size == 1), ("Small+", size >= 2)):
        m = gm & (v >= 1)
        report(f"Targeted phishing, {g}", v[m], post[m], db)


if __name__ == "__main__" and "targeted" in sys.argv:
    targeted()
