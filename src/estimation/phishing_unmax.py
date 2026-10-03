"""Is phishing internally homogeneous? Per-attack cost distributions recovered by un-maxing.

Phishing-only firms with an exact phishing count k (worst incident = worst phishing attack).
Assuming attacks iid given the firm's attack mix, worst-band CDF = G(b)^k, G = per-attack CDF.
Per count bucket, G is estimated by weighted ML using each firm's own k:
    L_i = G(c_i)^k_i - G(c_i - 1)^k_i       (coarse cost categories c)
1. Homogeneity: one G for all buckets vs one per bucket (pseudo-LR; also firm bootstrap).
   A fixed mix of attack types across firms is indistinguishable from a single type; types are
   only visible if their mix varies across firms, which shows as G shifting with k.
2. Number of types: if firms differ only in their mix of T fixed types, each bucket's per-attack
   distribution is a blend of T profiles -> the centred bucket x category matrix has rank T-1.
   SVD (rows weighted by sqrt(n), CA-style column scaling) vs a firm-bootstrap noise floor.
Variants: survey-weighted; + inverse-probability weighting for count reporting by worst band;
Micro vs Small+Medium+Large.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/phishing_unmax.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

import joint_five_channel as f

CATS = ["no cost", "£1-500", "£500-5k", "£5k+"]
EDGES = [2, 4, 6]                      # band 1 | 2-3 | 4-5 | 6+
BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10**6)]


def fit_G(c, k, w):
    """Per-attack category probabilities q (softmax) maximising sum w log(G_c^k - G_{c-1}^k)."""
    C = len(CATS)

    def nll(v):
        q = np.exp(np.r_[0.0, v])
        q /= q.sum()
        G = np.cumsum(q)
        Gm = np.r_[0.0, G[:-1]]
        lik = G[c] ** k - Gm[c] ** k
        return -(w * np.log(np.maximum(lik, 1e-300))).sum()

    best = min((minimize(nll, v0, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-9, maxiter=20000))
                for v0 in (np.zeros(C - 1), np.array([-2.0, -3.0, -4.0]), np.array([1.0, 0.0, -1.0]))),
               key=lambda r: r.fun)
    q = np.exp(np.r_[0.0, best.x])
    return q / q.sum(), -best.fun


def bucket_fits(c, k, w, b):
    Q, LL, N = [], 0.0, []
    for i in range(len(BUCKETS)):
        m = b == i
        q, ll = fit_G(c[m], k[m], w[m])
        Q.append(q)
        LL += ll
        N.append(m.sum())
    return np.array(Q), LL, np.array(N)


def svd_rows(Q, N, qbar):
    M = (Q - qbar) * np.sqrt(N)[:, None] / np.sqrt(qbar)[None, :]
    return np.linalg.svd(M, compute_uv=False), M


def analyse(c, k, w, label, rng, B=500):
    b = np.array([next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= kk <= hi) for kk in k])
    w = w / w.mean()
    Q, LLsep, N = bucket_fits(c, k, w, b)
    qpool, LLpool = fit_G(c, k, w)
    lr = 2 * (LLsep - LLpool)
    dof = (len(BUCKETS) - 1) * (len(CATS) - 1)
    deff = (w ** 2).mean()
    print("\n" + "=" * 96)
    print(f"{label}: n={len(c)}  (Kish deff {deff:.2f})")
    print("=" * 96)
    print("  per-attack cost distribution by count bucket:   " + "  ".join(f"{s:>8s}" for s in CATS))
    for (lo, hi), q, n in zip(BUCKETS, Q, N):
        lab = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f"{lo}+")
        print(f"    k = {lab:>6s}  (n={n:3d})                        " + "  ".join(f"{v:8.3f}" for v in q))
    print(f"    pooled (one G for all)                          " + "  ".join(f"{v:8.3f}" for v in qpool))
    print(f"  homogeneity: pseudo-LR = {lr:.1f} on {dof} df, deff-adjusted p = {chi2.sf(lr / deff, dof):.2g}")

    sv, M = svd_rows(Q, N, qpool)
    boot_sv, boot_lr = [], []
    idx_by_b = [np.where(b == i)[0] for i in range(len(BUCKETS))]
    for _ in range(B):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_by_b])
        Qb, _, _ = bucket_fits(c[idx], k[idx], w[idx], b[idx])
        boot_sv.append(svd_rows(Qb - Q + qpool, N, qpool)[0])     # noise around a homogeneous centre
    noise = np.percentile(np.array(boot_sv), 95, axis=0)
    print("  SVD of per-attack distributions (centred, sqrt(n)-weighted):")
    print("    singular values      " + " ".join(f"{v:7.3f}" for v in sv))
    print("    noise floor (95%)    " + " ".join(f"{v:7.3f}" for v in noise))
    print(f"    dimensions above noise: {int((sv > noise).sum())}  -> implied number of types: {int((sv > noise).sum()) + 1}")
    return Q, N


def main():
    d = f.load_firms()
    w = d["weight"].fillna(d["weight"].median()).values
    band, N = d["band"].values, d["N"].values
    fl = {x: d[x].values == 1 for x in ("fP", "fI", "fR", "fS")}
    base = fl["fP"] & ~fl["fI"] & ~fl["fR"] & ~fl["fS"] & ~np.isnan(band)
    m = base & ~np.isnan(N)
    c = np.digitize(band[m], EDGES)
    k = N[m].astype(int)
    rng = np.random.default_rng(0)

    analyse(c, k, w[m], "PHISHING-ONLY FIRMS, survey-weighted", rng)

    ipw = np.ones(len(d))
    for bb in np.unique(band[base]):
        mm = base & (band == bb)
        ipw[mm] = 1 / max(np.average(~np.isnan(N[mm]), weights=w[mm]), 0.05)
    analyse(c, k, (w * ipw)[m], "PHISHING-ONLY, survey-weighted x inverse reporting rate by worst band", rng)

    big = d["sizeb"].values[m] >= 2
    for lab, mm in (("MICRO", ~big), ("SMALL+MEDIUM+LARGE", big)):
        analyse(c[mm], k[mm], w[m][mm], f"PHISHING-ONLY, {lab}", rng, B=300)


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def blend_test():
    """Step 2 by likelihood instead of SVD: per-attack distribution in bucket b is a blend
    q_b = (1 - t_b) A + t_b B of two fixed profiles A, B (t_b free per bucket). Compared with free
    per-bucket distributions: pseudo-LR on (buckets x (cats-1)) - (2 (cats-1) + buckets) df.
    Not rejected -> the buckets lie on a line -> two attack types suffice."""
    d = f.load_firms()
    w = d["weight"].fillna(d["weight"].median()).values
    band, N = d["band"].values, d["N"].values
    fl = {x: d[x].values == 1 for x in ("fP", "fI", "fR", "fS")}
    base = fl["fP"] & ~fl["fI"] & ~fl["fR"] & ~fl["fS"] & ~np.isnan(band)
    ipw = np.ones(len(d))
    for bb in np.unique(band[base]):
        mm = base & (band == bb)
        ipw[mm] = 1 / max(np.average(~np.isnan(N[mm]), weights=w[mm]), 0.05)
    m0 = base & ~np.isnan(N)
    C = len(CATS)
    for label, wt, sel in (("survey-weighted", w, m0), ("+ IPW for reporting", w * ipw, m0),
                           ("Micro", w, m0 & (d["sizeb"].values == 1)),
                           ("Small+Medium+Large", w, m0 & (d["sizeb"].values >= 2))):
        c = np.digitize(band[sel], EDGES)
        k = N[sel].astype(int)
        ww = wt[sel] / wt[sel].mean()
        b = np.array([next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= kk <= hi) for kk in k])
        nb = len(BUCKETS)
        _, LLsep, _ = bucket_fits(c, k, ww, b)

        def sm(v):
            e = np.exp(np.r_[0.0, v])
            return e / e.sum()

        def nll(v):
            A, B = sm(v[:C - 1]), sm(v[C - 1:2 * (C - 1)])
            t = 1 / (1 + np.exp(-v[2 * (C - 1):]))
            q = (1 - t)[:, None] * A[None] + t[:, None] * B[None]
            G = np.cumsum(q, 1)
            Gm = np.c_[np.zeros(nb), G[:, :-1]]
            lik = G[b, c] ** k - Gm[b, c] ** k
            return -(ww * np.log(np.maximum(lik, 1e-300))).sum()

        rng = np.random.default_rng(0)
        best = None
        for s in range(40):
            v0 = np.r_[rng.normal(0, 1.5, C - 1), rng.normal(0, 1.5, C - 1), rng.normal(0, 1.5, nb)]
            r = minimize(nll, v0, method="L-BFGS-B")
            r = minimize(nll, r.x, method="Nelder-Mead", options=dict(maxiter=40000, maxfev=40000, xatol=1e-7, fatol=1e-9))
            best = r if best is None or r.fun < best.fun else best
        LL2 = -best.fun
        dof = nb * (C - 1) - (2 * (C - 1) + nb)
        deff = (ww ** 2).mean()
        lr = 2 * (LLsep - LL2)
        A, B = sm(best.x[:C - 1]), sm(best.x[C - 1:2 * (C - 1)])
        t = 1 / (1 + np.exp(-best.x[2 * (C - 1):]))
        if A[0] > B[0]:
            A, B, t = B, A, 1 - t
        print(f"\n  {label}: two-profile blend vs free buckets: pseudo-LR {lr:.1f} on {dof} df, "
              f"deff-adj p = {chi2.sf(lr / deff, dof):.2f}")
        print("    profile A (costlier):  " + "  ".join(f"{s} {v:.3f}" for s, v in zip(CATS, A)))
        print("    profile B (harmless):  " + "  ".join(f"{s} {v:.3f}" for s, v in zip(CATS, B)))
        print("    share of B-type attacks by count bucket: " +
              "  ".join(f"{('%d' % lo if lo == hi else ('%d-%d' % (lo, hi) if hi < 10**6 else '%d+' % lo))}: {v:.2f}"
                        for (lo, hi), v in zip(BUCKETS, t)))


if __name__ == "__main__" and "blend" in __import__("sys").argv:
    blend_test()
