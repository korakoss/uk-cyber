"""F2 with the global frequency answer: does the latent scale attack intensity, beyond mechanics?

Frequency answer -> literal incidents/year: once 1, <monthly 6, monthly 12, weekly 52, daily 365,
several/day 730. It totals ALL breaches, so it rises with the number of types hit by construction.
Mechanical null ("the latent acts on which types hit, not on how hard"): each hit type contributes
its own typical incidents, whatever else hit the firm. Per-type means mu_j from firms hit by ONLY
type j; a multi-type firm's null expectation = sum of mu_j over its types. Test: observed / expected
by number of types hit (m = 1 is ~1 by construction). Ratio > 1 rising with m -> latent scales
intensity. Co-labelling (one incident ticked twice) makes the sum over-predict -> conservative;
top band capped at 730 -> also conservative.
Types with < 5 single-type firms borrow the pooled mean of single-type firms of the rare serious
subtypes. Survey-weighted, within Micro and Small+Medium+Large; firm bootstrap.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/latent_intensity_freq.py
"""

import numpy as np

import joint_five_channel as f
from type_cooccurrence_structure import load, SHORT

LIT = {1: 1, 2: 6, 3: 12, 4: 52, 5: 365, 6: 730}


def ratios(X, K, w, mcap=4):
    m = X.sum(1)
    single = m == 1
    k = X.shape[1]
    mu = np.full(k, np.nan)
    for j in range(k):
        s = single & (X[:, j] == 1)
        if s.sum() >= 5:
            mu[j] = np.average(K[s], weights=w[s])
    rare = np.isnan(mu)
    s_rare = single & (X[:, rare].sum(1) == 1) if rare.any() else np.zeros(len(X), bool)
    fill = np.average(K[s_rare], weights=w[s_rare]) if s_rare.sum() else np.nanmean(mu)
    mu = np.where(rare, fill, mu)
    exp = X @ mu
    out = []
    for v in range(1, mcap + 1):
        g = (m == v) if v < mcap else (m >= v)
        out.append((g.sum(), np.average(K[g], weights=w[g]), np.average(exp[g], weights=w[g])))
    return mu, out


def main():
    X, w, size = load()
    d = f.load_firms()
    fq = d["freq"].values
    att = d["att"].values == 1
    ok = att & ~np.isnan(fq)
    K = np.array([LIT.get(int(v), np.nan) if not np.isnan(v) else np.nan for v in fq])
    rng = np.random.default_rng(0)
    for lab, sizes in (("MICRO", [1]), ("SMALL+MEDIUM+LARGE", [2, 3, 4])):
        m = ok & np.isin(size, sizes)
        Xg, Kg, wg = X[m], K[m], w[m] / w[m].mean()
        mu, out = ratios(Xg, Kg, wg)
        boots = []
        for _ in range(1000):
            idx = rng.integers(0, len(Xg), len(Xg))
            _, ob = ratios(Xg[idx], Kg[idx], wg[idx])
            boots.append([o / e for _, o, e in ob])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        print("\n" + "=" * 90)
        print(f"{lab}: attacked firms with a frequency answer, n={m.sum()}")
        print("=" * 90)
        print("  per-type mean incidents when hit alone: " +
              "  ".join(f"{s} {v:.0f}" for s, v in zip(SHORT, mu)))
        print(f"  {'# types hit':>11s} {'firms':>6s} {'observed mean':>14s} {'null (sum of mu)':>17s} {'ratio':>6s}  {'95% CI':>14s}")
        for (n, o, e), l, h, lab2 in zip(out, lo, hi, ["1", "2", "3", "4+"]):
            print(f"  {lab2:>11s} {n:6d} {o:14.1f} {e:17.1f} {o / e:6.2f}  [{l:5.2f}, {h:5.2f}]")


if __name__ == "__main__":
    main()
