"""Model-free look at frailty structure from the channel flags (P, I, R, S) on ALL firms.

1. Weighted 16-cell co-occurrence table vs independence (non-attacked firms included, so no
   conditioning-on-attacked selection effect).
2. Pairwise covariances + tetrad check. If firms fall into TWO latent classes and channels are
   independent within class, cov(i,j) = pi(1-pi) d_i d_j for i != j: the off-diagonal covariance
   matrix is rank 1, so the tetrad differences
       t1 = c_PI c_RS - c_PR c_IS,  t2 = c_PI c_RS - c_PS c_IR,  t3 = c_PR c_IS - c_PS c_IR
   are exactly 0. Three+ classes with non-proportional channel profiles (or >1 exposure
   dimension) make them nonzero. Firm-weighted bootstrap for intervals. Run pooled and within
   size groups, since size is itself a source of heterogeneity.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/frailty_flag_structure.py
"""

from itertools import product
import numpy as np

import joint_five_channel as f

CH = ["fP", "fI", "fR", "fS"]
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def wcov(X, w):
    m = np.average(X, axis=0, weights=w)
    Xc = X - m
    return (Xc * w[:, None]).T @ Xc / w.sum()


def tetrads(C):
    PI, PR, PS, IR, IS, RS = (C[a, b] for a, b in PAIRS)
    return np.array([PI * RS - PR * IS, PI * RS - PS * IR, PR * IS - PS * IR])


def analyse(X, w, label, rng, B=2000):
    n = len(X)
    print("\n" + "=" * 78)
    print(f"{label}: n={n}, weighted share attacked on >=1 channel "
          f"{np.average(X.any(1), weights=w):.3f}")
    print("=" * 78)
    p = np.average(X, axis=0, weights=w)
    print("  marginals: " + "  ".join(f"{c[1]} {v:.3f}" for c, v in zip(CH, p)))

    print("\n  pattern (P I R S)   raw n   obs share   indep share   obs/indep")
    for pat in product([0, 1], repeat=4):
        m = (X == pat).all(1)
        o = np.average(m, weights=w)
        ind = np.prod([pi if s else 1 - pi for pi, s in zip(p, pat)])
        print(f"    {pat}   {m.sum():6d}   {o:9.4f}   {ind:11.4f}   {o / ind if ind > 0 else np.nan:9.2f}")
    k = X.sum(1)
    print("\n  # channels hit:  " + "  ".join(f"{i}: obs {np.average(k == i, weights=w):.3f}" for i in range(5)))

    C = wcov(X, w)
    T = tetrads(C)
    boot_c, boot_t = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Cb = wcov(X[idx], w[idx])
        boot_c.append([Cb[a, b] for a, b in PAIRS])
        boot_t.append(tetrads(Cb))
    boot_c, boot_t = np.array(boot_c), np.array(boot_t)
    names = ["PI", "PR", "PS", "IR", "IS", "RS"]
    print("\n  pairwise covariance (x1000) [95% boot]   |  corr")
    for i, (a, b) in enumerate(PAIRS):
        lo, hi = np.percentile(boot_c[:, i], [2.5, 97.5]) * 1000
        r = C[a, b] / np.sqrt(C[a, a] * C[b, b])
        print(f"    {names[i]}: {C[a, b] * 1000:7.2f}  [{lo:6.2f}, {hi:6.2f}]   |  {r:.3f}")
    # scale tetrads relative to the size of the products involved
    scale = np.array([abs(C[0, 1] * C[2, 3]), abs(C[0, 1] * C[2, 3]), abs(C[0, 2] * C[1, 3])])
    print("\n  tetrad differences (x1e6) [95% boot]  (relative to first product)  -> 0 under 2 classes")
    for i, lab in enumerate(["PI.RS - PR.IS", "PI.RS - PS.IR", "PR.IS - PS.IR"]):
        lo, hi = np.percentile(boot_t[:, i], [2.5, 97.5]) * 1e6
        print(f"    {lab}: {T[i] * 1e6:8.2f}  [{lo:7.2f}, {hi:7.2f}]   rel {T[i] / scale[i]:+.2f}")


def main():
    d = f.load_firms()
    X = d[CH].values.astype(int)
    w = d["weight"].fillna(d["weight"].median()).values
    rng = np.random.default_rng(0)
    analyse(X, w / w.mean(), "POOLED (survey-weighted)", rng)
    for lab, sizes in (("MICRO", [1]), ("SMALL+MEDIUM+LARGE", [2, 3, 4]), ("SMALL", [2]),
                       ("MEDIUM", [3]), ("LARGE", [4])):
        m = d["sizeb"].isin(sizes).values
        ww = w[m] / w[m].mean()
        analyse(X[m], ww, f"{lab} (weighted within group)", rng)


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def once_only_coflags():
    """Sanity check: firms reporting breaches only ONCE in the year (freq=1) — how often do they
    flag 2+ channels? One incident should map to one channel unless categories overlap."""
    d = f.load_firms()
    w = d["weight"].fillna(d["weight"].median()).values
    att = d["att"].values == 1
    X = d[CH].values.astype(int)
    k = X.sum(1)
    names = ["PI", "PR", "PS", "IR", "IS", "RS"]
    print("\nCO-FLAGGING BY REPORTED FREQUENCY (attacked firms)")
    print(f"  {'freq':>12s} {'n':>5s} {'P(2+ ch) w':>11s} {'raw 2+':>7s}   " +
          " ".join(f"{p:>6s}" for p in names) + "   (weighted P(both) among group)")
    labels = {1: "once", 2: "less monthly", 3: "monthly", 4: "weekly", 5: "daily", 6: "several/day"}
    for fq in range(1, 7):
        m = att & (d["freq"].values == fq)
        if m.sum() == 0:
            continue
        pair = [np.average(X[m, a] & X[m, b], weights=w[m]) for a, b in PAIRS]
        print(f"  {labels[fq]:>12s} {m.sum():5d} {np.average(k[m] >= 2, weights=w[m]):11.3f} {(k[m] >= 2).sum():7d}   " +
              " ".join(f"{v:6.3f}" for v in pair))
    m = att & (d["freq"].values == 1) & (k >= 2)
    print("\n  once-only firms with 2+ channels, pattern counts (P I R S):")
    pats, cnt = np.unique(X[m], axis=0, return_counts=True)
    for p_, c in sorted(zip(map(tuple, pats), cnt), key=lambda t: -t[1]):
        print(f"    {p_}: {c}")
    n1 = att & (d["N"].values == 1)
    print(f"\n  phishing count N=1 firms (n={n1.sum()}): share also flagging I {np.average(X[n1, 1], weights=w[n1]):.3f},"
          f" R {np.average(X[n1, 2], weights=w[n1]):.3f}, S {np.average(X[n1, 3], weights=w[n1]):.3f}")


if __name__ == "__main__" and "once" in __import__("sys").argv[1:]:
    once_only_coflags()


def once_only_counts():
    """Do firms answering freq='once' really have one incident? Their reported phishing counts."""
    d = f.load_firms()
    att = d["att"].values == 1
    k = d[CH].values.sum(1)
    for lab, m in (("once, any", att & (d["freq"].values == 1)),
                   ("once, 2+ channels", att & (d["freq"].values == 1) & (k >= 2))):
        N = d.loc[m, "N"].dropna().astype(int)
        print(f"  {lab}: n={m.sum()}, with phishing count {len(N)}; N=1 {np.mean(N == 1):.2f}, "
              f"N=2-5 {np.mean(N.between(2, 5)):.2f}, N>5 {np.mean(N > 5):.2f}; values {sorted(N.tolist())[:40]}")


if __name__ == "__main__" and "oncecounts" in __import__("sys").argv:
    once_only_counts()
