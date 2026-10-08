"""Does the exposure model (latent-class frailty on the 10 attack-type flags, fitted within Micro and within
Small+Medium+Large) reproduce the breadth that drives cost?

Model: firm in size group g has frailty class k with prob pi_gk; given k, each of the 10 types is hit independently
with prob th_gkj. Channels: phishing, impersonation, ransomware = their own type; other serious = any of the other 7.
Exact model-implied distributions (no simulation): breadth = number of the 4 channels hit (0-4), per class a sum of
independent Bernoullis, mixed over classes.
Compared with the survey (weighted) - within the two fitted size groups, and within Small / Medium / Large separately
(the model pools them). Also: channel hit rates, pairwise channel co-occurrence, number of raw types hit (0-5+).
Effective-n Pearson X2 for the breadth distribution. K = 3 (the tiers in use) and K = 4 for comparison.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/exposure_check.py
"""

import itertools

import numpy as np

from frailty_shape_flags import lca, patterns
from type_cooccurrence_structure import load, SHORT

CH = ["phishing", "impersonation", "ransomware", "other serious"]
IP, II, IR = SHORT.index("Phish"), SHORT.index("Imper"), SHORT.index("Ransm")
OTHER = [j for j in range(len(SHORT)) if j not in (IP, II, IR)]


def chan_probs(th):
    """(K, 4) channel hit probabilities per class."""
    s = 1 - np.prod(1 - th[:, OTHER], axis=1)
    return np.column_stack([th[:, IP], th[:, II], th[:, IR], s])


def bern_sum(p):
    """Distribution of the number of successes for independent Bernoullis p (1-D)."""
    d = np.array([1.0])
    for q in p:
        d = np.convolve(d, [1 - q, q])
    return d


def model_dists(pi, th):
    cp = chan_probs(th)
    breadth = sum(pi[k] * bern_sum(cp[k]) for k in range(len(pi)))
    ntypes = sum(pi[k] * bern_sum(th[k]) for k in range(len(pi)))
    rates = pi @ cp
    pairs = {(a, b): float(pi @ (cp[:, a] * cp[:, b])) for a, b in itertools.combinations(range(4), 2)}
    return breadth, ntypes, rates, pairs


def obs_dists(X, w):
    C = np.column_stack([X[:, IP], X[:, II], X[:, IR], X[:, OTHER].max(1)])
    nb = C.sum(1).astype(int)
    breadth = np.array([w[nb == k].sum() for k in range(5)]) / w.sum()
    nt = np.minimum(X.sum(1), 10).astype(int)
    ntypes = np.array([w[nt == k].sum() for k in range(11)]) / w.sum()
    rates = (w[:, None] * C).sum(0) / w.sum()
    pairs = {(a, b): float((w * C[:, a] * C[:, b]).sum() / w.sum()) for a, b in itertools.combinations(range(4), 2)}
    return breadth, ntypes, rates, pairs, w.sum() ** 2 / (w ** 2).sum()


def x2(o, e, neff):
    e = np.maximum(e, 1e-6)
    return neff * ((o - e) ** 2 / e).sum()


def main():
    X, w, size = load()
    groups = {"Micro": size == 1, "Small+": size >= 2}
    for K in (3, 4):
        print(f"\n===== latent classes K = {K}")
        fits = {}
        for g, gm in groups.items():
            P, c = patterns(X[gm], w[gm] / w[gm].mean())
            ll, pi, th = lca(P, c, K, np.random.default_rng(0), starts=30)
            o = np.argsort(th.mean(1))
            fits[g] = (pi[o], th[o])
            print(f"  {g}: class shares " + " / ".join(f"{v:.2f}" for v in pi[o]) + ";  channel hit prob by class: " +
                  "; ".join("/".join(f"{v:.2f}" for v in row) for row in chan_probs(th[o])))
        cells = [("Micro", "Micro", size == 1), ("Small+", "Small+", size >= 2),
                 ("Small", "Small+", size == 2), ("Medium", "Small+", size == 3), ("Large", "Small+", size == 4)]
        print("  breadth 0/1/2/3/4 channels: observed | model;  X2 (df 4) with effective n")
        for lab, g, m in cells:
            mb, mt, mr, mp = model_dists(*fits[g])
            ob, ot, orr, op, neff = obs_dists(X[m], w[m])
            print(f"    {lab:7s} n {m.sum():4d}  " + " ".join(f"{v:.3f}" for v in ob) + " | " + " ".join(f"{v:.3f}" for v in mb) +
                  f"   X2 {x2(ob, mb, neff):6.1f}")
        if K == 3:
            print("  channel hit rates (observed | model):")
            for lab, g, m in cells:
                mb, mt, mr, mp = model_dists(*fits[g])
                ob, ot, orr, op, neff = obs_dists(X[m], w[m])
                print(f"    {lab:7s} " + "  ".join(f"{CH[j][:5]} {orr[j]:.3f}|{mr[j]:.3f}" for j in range(4)))
            print("  pairwise co-occurrence P(both) (observed | model; ratio):")
            for lab, g, m in cells[:2]:
                mb, mt, mr, mp = model_dists(*fits[g])
                ob, ot, orr, op, neff = obs_dists(X[m], w[m])
                print(f"    {lab:7s} " + "  ".join(f"{CH[a][:3]}&{CH[b][:3]} {op[(a, b)]:.3f}|{mp[(a, b)]:.3f} ({op[(a, b)] / mp[(a, b)]:.2f})"
                                                  for a, b in mp))
            print("  number of raw types hit 0/1/2/3/4/5+ (observed | model):")
            for lab, g, m in cells[:2]:
                mb, mt, mr, mp = model_dists(*fits[g])
                ob, ot, orr, op, neff = obs_dists(X[m], w[m])
                agg = lambda d: np.r_[d[:5], d[5:].sum()]
                print(f"    {lab:7s} " + " ".join(f"{v:.3f}" for v in agg(ot)) + " | " + " ".join(f"{v:.3f}" for v in agg(mt)))


if __name__ == "__main__":
    main()
