"""Nonparametric dimensionality check (DETECT-style conditional covariances) on the attack-type flags.

Assumes only: one latent, monotone (higher latent never lowers a type's hit probability). No normality,
no link shape. For each pair (i, j): firms stratified by rest score S = # of OTHER types hit (excluding
i and j), covariance of the two flags within each stratum, averaged with stratum weight shares.
Under one latent, conditioning on S absorbs the shared factor -> leftover ~ 0 (with few, mostly rare
types S is a coarse read of the latent, so leftovers skew somewhat positive across the board).
Pairs standing out positively share something extra (sub-cluster or co-labelling); negative leftovers
across groups would indicate separate dimensions. Reported as conditional correlation (cov scaled by
the pair's average within-stratum SDs) with firm-bootstrap z-scores.
Groups: Micro; Small+Medium+Large. Survey-weighted.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/detect_conditional_cov.py
"""

import numpy as np

from type_cooccurrence_structure import load, SHORT

S_CAP = 3            # rest-score strata 0, 1, 2, 3+


def cond_cov(X, w):
    n, k = X.shape
    C = np.zeros((k, k))
    Rn = np.zeros((k, k))
    tot = X.sum(1)
    for i in range(k):
        for j in range(i + 1, k):
            s = np.minimum(tot - X[:, i] - X[:, j], S_CAP)
            cv, sd, wsum = 0.0, 0.0, 0.0
            for v in range(S_CAP + 1):
                m = s == v
                ws = w[m].sum()
                if m.sum() < 5 or ws == 0:
                    continue
                a, b, ww = X[m, i], X[m, j], w[m]
                ma, mb = np.average(a, weights=ww), np.average(b, weights=ww)
                c = np.average((a - ma) * (b - mb), weights=ww)
                cv += ws * c
                sd += ws * np.sqrt(ma * (1 - ma) * mb * (1 - mb))
                wsum += ws
            C[i, j] = C[j, i] = cv / wsum
            Rn[i, j] = Rn[j, i] = cv / sd if sd > 0 else 0.0
    return C, Rn


def analyse(X, w, names, label, rng, B=400):
    keep = X.sum(0) > 0
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    w = w / w.mean()
    C, R = cond_cov(X, w)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(X), len(X))
        boots.append(cond_cov(X[idx], w[idx])[1])
    se = np.array(boots).std(0)
    k = len(names)
    iu = np.triu_indices(k, 1)
    print("\n" + "=" * 100)
    print(f"{label}: n={len(X)}, types={k}")
    print("=" * 100)
    print("  conditional correlation given rest score (upper) / bootstrap z (lower):")
    print("          " + " ".join(f"{s:>7s}" for s in names))
    for i in range(k):
        cells = [f"{R[i, j]:7.2f}" if j > i else (f"{R[i, j] / max(se[i, j], 1e-9):7.1f}" if j < i else "      -")
                 for j in range(k)]
        print(f"  {names[i]:>7s} " + " ".join(cells))
    vals = R[iu]
    print(f"\n  across all {len(vals)} pairs: median {np.median(vals):.3f}, mean {vals.mean():.3f}, "
          f"share negative {np.mean(vals < 0):.2f}")
    z = vals / np.maximum(se[iu], 1e-9)
    order = np.argsort(-vals)
    print("  largest positive leftovers: " + ", ".join(
        f"{names[iu[0][o]]}-{names[iu[1][o]]} {vals[o]:+.2f} (z {z[o]:+.1f})" for o in order[:5]))
    print("  most negative leftovers:    " + ", ".join(
        f"{names[iu[0][o]]}-{names[iu[1][o]]} {vals[o]:+.2f} (z {z[o]:+.1f})" for o in order[::-1][:5]))
    raw = np.corrcoef(X.T)[iu]
    print(f"  for scale: unconditional (phi) correlations median {np.median(raw):.3f}")


def main():
    X, w, size = load()
    rng = np.random.default_rng(0)
    analyse(X[size == 1], w[size == 1], SHORT, "MICRO", rng)
    m = np.isin(size, [2, 3, 4])
    analyse(X[m], w[m], SHORT, "SMALL+MEDIUM+LARGE", rng)


if __name__ == "__main__":
    main()
