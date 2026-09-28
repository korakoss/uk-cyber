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


if __name__ == "__main__":
    main()
