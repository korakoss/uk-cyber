"""Shape of the shared latent from occurrence flags only (no intra-channel count assumptions).

Given the latent, the chance a firm is hit at least once by type j is a free per-type parameter,
whatever the count distribution; types are independent given the latent (the model's assumption).
So flag patterns alone compare frailty SHAPES:
  - latent class (LCA), K = 1..5 discrete classes, free hit probability per class x type
  - latent trait: one continuous N(0,1) score, P(hit_j | z) = logistic(a_j + b_j z)  (2PL)
Items: the 10 survey types (A), and a version with the co-labelling pairs merged (B):
phishing+impersonation -> one item, ransomware+malware -> one item.
Within Micro and within Small+Medium+Large (so size cannot pose as tiers); all firms incl.
non-attacked; survey weights normalised to mean 1 within group (pseudo-likelihood; BIC uses raw n).
Diagnostics for the best models: class profiles (are classes ordered along one axis?), observed vs
expected distribution of # types hit, and pairwise co-occurrence residuals (local dependence).

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/frailty_shape_flags.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from type_cooccurrence_structure import load, SHORT

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(41)
GH_W = GH_W / GH_W.sum()


def patterns(X, w):
    keys, inv = np.unique(X, axis=0, return_inverse=True)
    return keys, np.bincount(inv.ravel(), weights=w)


def lca(P, c, K, rng, starts=30, iters=2000):
    n, J = P.shape
    best = None
    for _ in range(starts):
        pi = rng.dirichlet(np.ones(K))
        th = rng.uniform(0.02, 0.6, (K, J))
        ll_old = -np.inf
        for _ in range(iters):
            lp = np.log(pi)[None] + P @ np.log(th).T + (1 - P) @ np.log(1 - th).T
            m = lp.max(1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(lp - m).sum(1))
            ll = (c * lse).sum()
            R = np.exp(lp - lse[:, None]) * c[:, None]
            pi = R.sum(0) / R.sum()
            th = np.clip((R.T @ P) / R.sum(0)[:, None], 1e-6, 1 - 1e-6)
            if ll - ll_old < 1e-8:
                break
            ll_old = ll
        if best is None or ll > best[0]:
            best = (ll, pi.copy(), th.copy())
    return best


def lca_expected(P, pi, th):
    return np.exp(np.log(pi)[None] + P @ np.log(th).T + (1 - P) @ np.log(1 - th).T).sum(1)


def trait_probs(P, x):
    J = P.shape[1]
    a, b = x[:J], x[J:]
    pr = np.clip(expit(a[None, :] + b[None, :] * GH_X[:, None]), 1e-9, 1 - 1e-9)   # (Q, J)
    L = np.exp(P @ np.log(pr).T + (1 - P) @ np.log(1 - pr).T)                     # (patterns, Q)
    return L @ GH_W


def trait(P, c, rng):
    J = P.shape[1]
    p = np.clip((c[:, None] * P).sum(0) / c.sum(), 1e-3, 1 - 1e-3)
    best = None
    for s in range(4):
        x0 = np.r_[np.log(p / (1 - p)) - 1.0, np.full(J, 1.5)] + (rng.normal(0, 0.3, 2 * J) if s else 0)
        res = minimize(lambda x: -(c * np.log(np.maximum(trait_probs(P, x), 1e-300))).sum(), x0,
                       method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    return -best.fun, best.x


def diagnostics(P, c, expected, names, label):
    n = c.sum()
    k_obs = P.sum(1)
    print(f"    [{label}] # types hit, observed vs expected share:  " +
          "  ".join(f"{k}:{c[k_obs == k].sum() / n:.3f}/{expected[k_obs == k].sum():.3f}"
                    for k in range(0, min(6, P.shape[1] + 1))) +
          f"  6+:{c[k_obs >= 6].sum() / n:.4f}/{expected[k_obs >= 6].sum():.4f}")
    J = P.shape[1]
    res = []
    for a in range(J):
        for b in range(a + 1, J):
            m = (P[:, a] == 1) & (P[:, b] == 1)
            o, e = c[m].sum(), expected[m].sum() * n
            res.append(((o - e) / np.sqrt(max(e, 0.5)), names[a], names[b], o, e))
    res.sort(key=lambda t: -abs(t[0]))
    print(f"    [{label}] largest pairwise residuals (obs vs expected joint count, weighted):  " +
          "  ".join(f"{a}-{b} {o:.0f}/{e:.0f} (z {z:+.1f})" for z, a, b, o, e in res[:4]))


def run(X, w, names, label, rng):
    P, c = patterns(X, w)
    n_raw = len(X)
    J = X.shape[1]
    print("\n" + "=" * 100)
    print(f"{label}: n={n_raw}, items={J}, distinct patterns={len(P)}")
    print("=" * 100)
    fits = {}
    for K in range(1, 6):
        ll, pi, th = lca(P, c, K, rng)
        npar = (K - 1) + K * J
        fits[f"LCA{K}"] = (ll, npar, (pi, th))
    llt, xt = trait(P, c, rng)
    fits["trait"] = (llt, 2 * J, xt)
    print(f"  {'model':>7s} {'loglik':>10s} {'params':>6s} {'BIC':>10s} {'AIC':>10s}")
    for k, (ll, npar, _) in fits.items():
        print(f"  {k:>7s} {ll:10.2f} {npar:6d} {-2 * ll + npar * np.log(n_raw):10.1f} {-2 * ll + 2 * npar:10.1f}")
    best_lca = min((k for k in fits if k.startswith("LCA")),
                   key=lambda k: -2 * fits[k][0] + fits[k][1] * np.log(n_raw))
    ll, npar, (pi, th) = fits[best_lca]
    order = np.argsort((th * pi[:, None] * 0 + th).mean(1))
    print(f"\n  best LCA by BIC: {best_lca}. Class profiles (hit probability per type), classes sorted by mean rate:")
    print("    share  " + " ".join(f"{s:>6s}" for s in names))
    for k in order:
        print(f"    {pi[k]:.3f}  " + " ".join(f"{v:6.3f}" for v in th[k]))
    mono = np.mean([np.all(np.diff(th[order, j]) >= -0.01) for j in range(J)])
    print(f"    share of types whose hit probability rises with class rank: {mono:.2f}  (1.0 = classes ordered on one axis)")
    diagnostics(P, c, lca_expected(P, pi, th), names, best_lca)
    diagnostics(P, c, trait_probs(P, xt), names, "trait")
    print("    trait slopes b_j: " + " ".join(f"{s} {v:.2f}" for s, v in zip(names, xt[J:])))


def main():
    X, w, size = load()
    rng = np.random.default_rng(0)
    merged_names = ["Ph/Imp", "Rn/Mal", "DoS", "BankH", "Takov", "AcOut", "AcStf", "Eavsd"]
    Xm = np.column_stack([X[:, 0] | X[:, 1], X[:, 2] | X[:, 3], X[:, 4:]])
    for glabel, m in (("MICRO", size == 1), ("SMALL+MEDIUM+LARGE", np.isin(size, [2, 3, 4]))):
        ww = w[m] / w[m].mean()
        run(X[m], ww, SHORT, f"{glabel} / A: 10 types", rng)
        run(Xm[m], ww, merged_names, f"{glabel} / B: co-labelling pairs merged", rng)


if __name__ == "__main__":
    main()
