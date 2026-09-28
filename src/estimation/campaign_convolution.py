"""Campaign model check for targeted phishing.

K = B_1 + ... + B_N, N ~ Poisson(lambda) (campaigns), B_j iid campaign sizes, frailty acts on lambda only.
Campaign-size law B := empirical positive counts of LOW-tier firms (lambda small -> hit firms had ~1 campaign),
per size group and pooled. For each mid / high cell: lambda = -ln(1 - P(K >= 1)) from the cell's hit share;
predicted K | K >= 1 = sum over N ~ zero-truncated Poisson(lambda) of B's (Monte Carlo, resampling B).
Compare observed vs predicted quantiles and doubling-bin shares; p-value: X2 of observed vs predicted bins,
referred to X2 of simulated samples of the same size, each simulated with a bootstrap-resampled B (so the
uncertainty in B is included). Exact phishcon answers, unweighted, most-likely tier.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/campaign_convolution.py
"""

import numpy as np
import pandas as pd

from counts_given_frailty import tier_posteriors, TIER
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load

BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 10**9)]
LAB = "1 / 2 / 3-4 / 5-8 / 9-16 / 17-32 / 33-64 / 65-128 / 129+"


def ztp(lam, n, rng):
    out = rng.poisson(lam, 4 * n + 50)
    out = out[out >= 1]
    while len(out) < n:
        x = rng.poisson(lam, 4 * n + 50)
        out = np.concatenate([out, x[x >= 1]])
    return out[:n]


def simulate(B, lam, n, rng):
    N = ztp(lam, n, rng)
    idx = rng.integers(0, len(B), N.sum())
    return np.add.reduceat(B[idx], np.concatenate([[0], np.cumsum(N)[:-1]]))


def binshare(x):
    return np.array([((x >= a) & (x <= b)).mean() for a, b in BINS])


def check(label, B, hit, pos, rng, reps=1000, big=200000):
    p = hit.mean()
    lam = -np.log(1 - p)
    pred = simulate(B, lam, big, rng)
    e = binshare(pred) * len(pos)
    o = binshare(pos) * len(pos)
    m = e > 0
    x2 = ((o[m] - e[m]) ** 2 / e[m]).sum()
    sims = []
    for _ in range(reps):
        Bb = rng.choice(B, len(B))
        eb = binshare(simulate(Bb, lam, 20000, rng)) * len(pos)
        s = simulate(Bb, lam, len(pos), rng)
        ob = binshare(s) * len(pos)
        mb = eb > 0
        sims.append(((ob[mb] - eb[mb]) ** 2 / eb[mb]).sum())
    qo = np.percentile(pos, [25, 50, 75])
    qp = np.percentile(pred, [25, 50, 75])
    print(f"\n  {label}: P(K>=1) {p:.3f} -> lambda {lam:.2f} campaigns (E[N | N>=1] {lam / p:.2f}); n+ {len(pos)}")
    print(f"    quartiles obs {qo[0]:g} / {qo[1]:g} / {qo[2]:g}    pred {qp[0]:g} / {qp[1]:g} / {qp[2]:g}    "
          f"X2 {x2:.1f}  p {np.mean(np.array(sims) >= x2):.3f}")
    print("    obs  " + " ".join(f"{v:5.1f}" for v in o) + f"   ({LAB})")
    print("    pred " + " ".join(f"{v:5.1f}" for v in e))


def main():
    X, w, size = load()
    tier = tier_posteriors().argmax(1)
    t = pd.to_numeric(aligned_raw()["phishcon"], errors="coerce").values
    t = np.where(X[:, 0] == 1, np.where(t >= 0, t, np.nan), 0.0)
    known = ~np.isnan(t)
    rng = np.random.default_rng(0)
    grp = {"Micro": size == 1, "Small+": size >= 2}
    low_all = (tier == 0) & known & (t >= 1)
    for bsrc in ("own size group", "pooled sizes"):
        print("\n" + "=" * 100 + f"\ncampaign size B = low-tier positive counts, {bsrc}\n" + "=" * 100)
        for g, gm in grp.items():
            B = t[low_all & gm] if bsrc == "own size group" else t[low_all]
            B = B.astype(np.int64)
            print(f"\n{g}: B n={len(B)}, median {np.median(B):g}, mean {B.mean():.1f}")
            for c in (1, 2):
                mc = gm & known & (tier == c)
                check(f"{g} {TIER[c]}", B, t[mc] >= 1, t[mc & (t >= 1)].astype(np.int64), rng)


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def ztnb(r, mu, n, rng):
    """Zero-truncated NegBin(mean mu, shape r) draws; r = inf -> Poisson."""
    draw = (lambda k: rng.poisson(mu, k)) if np.isinf(r) else (lambda k: rng.negative_binomial(r, r / (r + mu), k))
    out = np.empty(0, np.int64)
    while len(out) < n:
        x = draw(4 * n + 50)
        out = np.concatenate([out, x[x >= 1]])
    return out[:n]


def nb_mean(p, r):
    """Mean of NegBin(r) with P(N >= 1) = p."""
    return -np.log(1 - p) if np.isinf(r) else r * ((1 - p) ** (-1 / r) - 1)


def simulate_nb(B, r, mu, n, rng):
    N = np.minimum(ztnb(r, mu, n, rng), 130)   # N >= 130 puts K in the top bin (129+) anyway; caps memory
    idx = rng.integers(0, len(B), N.sum())
    return np.add.reduceat(B[idx], np.concatenate([[0], np.cumsum(N)[:-1]]))


def negbin_campaigns(shapes=(0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf)):
    """Campaign count N ~ NegBin (overdispersed campaign rate within tier) instead of Poisson; shape r shared by all
    cells, mean per cell from its hit share. B = pooled low-tier positive counts. X2 per mid/high cell and total over
    a grid of r; bootstrap p (incl. B uncertainty) per cell at the best r and at r = 1 (geometric)."""
    X, w, size = load()
    tier = tier_posteriors().argmax(1)
    t = pd.to_numeric(aligned_raw()["phishcon"], errors="coerce").values
    t = np.where(X[:, 0] == 1, np.where(t >= 0, t, np.nan), 0.0)
    known = ~np.isnan(t)
    rng = np.random.default_rng(0)
    B = t[(tier == 0) & known & (t >= 1)].astype(np.int64)
    cells = []
    for g, gm in (("Micro", size == 1), ("Small+", size >= 2)):
        for c in (1, 2):
            mc = gm & known & (tier == c)
            cells.append((f"{g} {TIER[c]}", (t[mc] >= 1).mean(), t[mc & (t >= 1)].astype(np.int64)))

    def x2(pos, pred):
        e, o = binshare(pred) * len(pos), binshare(pos) * len(pos)
        m = e > 0
        return ((o[m] - e[m]) ** 2 / e[m]).sum()

    print(f"B: pooled low-tier positives n={len(B)}, median {np.median(B):g}")
    print(f"\n{'shape r':>8s} " + " ".join(f"{lab:>13s}" for lab, _, _ in cells) + "   total X2   (per cell: X2 | mean campaigns among hit)")
    tot = {}
    for r in shapes:
        row, s = [], 0.0
        for lab, p, pos in cells:
            mu = nb_mean(p, r)
            pred = simulate_nb(B, r, mu, 100000, rng)
            v = x2(pos, pred)
            s += v
            en = mu / p
            row.append(f"{v:6.1f}|{en:6.1f}")
        tot[r] = s
        print(f"{r:8.1f} " + " ".join(f"{x:>13s}" for x in row) + f"   {s:7.1f}")
    best = min(tot, key=tot.get)
    for r in sorted({best, 1.0}):
        print(f"\nshape r = {r}: per-cell quartiles and bootstrap p (1000 reps, B resampled)")
        for lab, p, pos in cells:
            mu = nb_mean(p, r)
            pred = simulate_nb(B, r, mu, 200000, rng)
            obs_x2 = x2(pos, pred)
            sims = []
            for _ in range(1000):
                Bb = rng.choice(B, len(B))
                eb = simulate_nb(Bb, r, mu, 20000, rng)
                sims.append(x2(simulate_nb(Bb, r, mu, len(pos), rng), eb))
            qo, qp = np.percentile(pos, [25, 50, 75]), np.percentile(pred, [25, 50, 75])
            print(f"  {lab:12s} obs {qo[0]:g}/{qo[1]:g}/{qo[2]:g}  pred {qp[0]:g}/{qp[1]:g}/{qp[2]:g}  "
                  f"X2 {obs_x2:5.1f} p {np.mean(np.array(sims) >= obs_x2):.3f}")
            print("    obs  " + " ".join(f"{v:5.1f}" for v in binshare(pos) * len(pos)) + f"   ({LAB})")
            print("    pred " + " ".join(f"{v:5.1f}" for v in binshare(pred) * len(pos)))


if __name__ == "__main__" and "negbin" in __import__("sys").argv:
    negbin_campaigns()


def one_campaign_test():
    """'One campaign per hit firm' for low and mid tiers: are positive targeted counts of each mid / high cell drawn
    from the same distribution as the pooled low-tier positives? Two-sample tests: permutation test on the binned X2
    (bins BINS) and on the difference in median log count, 10000 permutations; plus Mann-Whitney p."""
    from scipy.stats import mannwhitneyu
    X, w, size = load()
    tier = tier_posteriors().argmax(1)
    t = pd.to_numeric(aligned_raw()["phishcon"], errors="coerce").values
    t = np.where(X[:, 0] == 1, np.where(t >= 0, t, np.nan), 0.0)
    known = ~np.isnan(t)
    rng = np.random.default_rng(0)
    B = t[(tier == 0) & known & (t >= 1)]

    def chi(a, b):
        o = np.array([[((x >= lo) & (x <= hi)).sum() for lo, hi in BINS] for x in (a, b)], float)
        e = o.sum(1, keepdims=True) * o.sum(0, keepdims=True) / o.sum()
        m = e > 0
        return ((o[m] - e[m]) ** 2 / e[m]).sum()

    print(f"low-tier positives (pooled sizes) n={len(B)}, quartiles " + "/".join(f"{v:g}" for v in np.percentile(B, [25, 50, 75])))
    for g, gm in (("Micro", size == 1), ("Small+", size >= 2)):
        for c in (1, 2):
            pos = t[gm & known & (tier == c) & (t >= 1)]
            both = np.concatenate([B, pos])
            x0 = chi(B, pos)
            d0 = abs(np.median(np.log(pos)) - np.median(np.log(B)))
            px = pd_ = 0
            for _ in range(10000):
                perm = rng.permutation(both)
                a, b = perm[:len(B)], perm[len(B):]
                px += chi(a, b) >= x0
                pd_ += abs(np.median(np.log(b)) - np.median(np.log(a))) >= d0
            q = np.percentile(pos, [25, 50, 75])
            print(f"  {g:6s} {TIER[c]:4s} n={len(pos):3d} quartiles {q[0]:g}/{q[1]:g}/{q[2]:g}   "
                  f"binned X2 perm p {px / 10000:.3f}   median perm p {pd_ / 10000:.3f}   "
                  f"Mann-Whitney p {mannwhitneyu(pos, B).pvalue:.3f}")


if __name__ == "__main__" and "onecamp" in __import__("sys").argv:
    one_campaign_test()
