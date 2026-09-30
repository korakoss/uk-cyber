"""Are channel costs independent within a firm? Max-of-independent test on multi-channel firms.

Channels from the 10 flags: P phishing, I impersonation, R ransomware, S other serious (any of the remaining types).
Pure cells: firms hit by exactly one channel -> that channel's annual worst-cost distribution (incl. zero cost),
survey-weighted, on groups none / <£500 / £500-5k / £5k-20k / £20k+.
Multi-channel profile (e.g. P+I): if channel costs are independent, worst = max of one draw from each pure
distribution. Predicted vs observed group shares per profile (R-containing profiles only described: pure R is too
thin). Uncertainty: 2000 bootstrap resamples of the pure cells -> 90% band per predicted share; plus a test on the
upper tail P(worst >= £5k) and on the mean group index. Also split by frailty tier for the P+I+S-type broad profile.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/cost_independence.py
"""

import numpy as np

import joint_five_channel as f
from counts_given_frailty import tier_posteriors, TIER
from type_cooccurrence_structure import load, SHORT

GROUPS = [(1, 1), (2, 3), (4, 5), (6, 7), (8, 13)]
GLAB = "none / <£500 / £500-5k / £5k-20k / £20k+"


def grp(b):
    out = np.zeros(len(b), int)
    for i, (a, c) in enumerate(GROUPS):
        out[(b >= a) & (b <= c)] = i
    return out


def dist(g, w):
    return np.array([w[g == i].sum() for i in range(len(GROUPS))]) / w.sum()


def max_dist(ps):
    cdf = np.ones(len(GROUPS))
    for p in ps:
        cdf = cdf * np.cumsum(p)
    return np.diff(np.concatenate([[0], cdf]))


def main():
    X, _, size = load()
    d = f.load_firms()
    band = d["band"].values
    w = d["weight"].fillna(d["weight"].median()).values
    tier = tier_posteriors().argmax(1)
    ok = ~np.isnan(band)
    P = X[:, 0] == 1
    I = X[:, SHORT.index("Imper")] == 1
    R = X[:, SHORT.index("Ransm")] == 1
    S = np.delete(X, [0, SHORT.index("Imper"), SHORT.index("Ransm")], axis=1).max(1) == 1
    ch = {"P": P, "I": I, "S": S, "R": R}
    nch = P.astype(int) + I + S + R
    g = grp(np.nan_to_num(band, nan=1))
    pure = {k: ok & v & (nch == 1) for k, v in ch.items()}
    rng = np.random.default_rng(0)
    print(f"shares: {GLAB}\n\npure cells:")
    for k, m in pure.items():
        print(f"  {k}  n {m.sum():4d}  " + " ".join(f"{v:.2f}" for v in dist(g[m], w[m])))

    def boot_pure(k):
        idx = np.where(pure[k])[0]
        j = rng.choice(idx, len(idx))
        return dist(g[j], w[j])

    profiles = [("P+I", P & I & ~S & ~R), ("P+S", P & S & ~I & ~R), ("I+S", I & S & ~P & ~R), ("P+I+S", P & I & S & ~R)]
    print("\nmulti-channel profiles without ransomware: observed vs max-of-independent prediction")
    for lab, m in profiles:
        m = m & ok
        keys = [c for c in lab.split("+")]
        obs = dist(g[m], w[m])
        pred = max_dist([dist(g[pure[k]], w[pure[k]]) for k in keys])
        bs = np.array([max_dist([boot_pure(k) for k in keys]) for _ in range(2000)])
        lo, hi = np.percentile(bs, [5, 95], axis=0)
        tail_obs, tail_bs = obs[3:].sum(), bs[:, 3:].sum(1)
        mean_obs = (obs * np.arange(5)).sum()
        mean_bs = (bs * np.arange(5)).sum(1)
        print(f"\n  {lab:6s} n {m.sum():4d}")
        print("    observed  " + " ".join(f"{v:5.2f}" for v in obs))
        print("    predicted " + " ".join(f"{v:5.2f}" for v in pred) + "   90%: " +
              " ".join(f"[{a:.2f},{b:.2f}]" for a, b in zip(lo, hi)))
        print(f"    P(worst >= £5k): obs {tail_obs:.3f}  pred {pred[3:].sum():.3f} (share of bootstrap preds >= obs "
              f"{np.mean(tail_bs >= tail_obs):.3f});  mean group: obs {mean_obs:.2f}  pred {(pred * np.arange(5)).sum():.2f} "
              f"(share >= obs {np.mean(mean_bs >= mean_obs):.3f})")
        for c in range(3):
            mc = m & (tier == c)
            if mc.sum() >= 10:
                print(f"      tier {TIER[c]:4s} n {mc.sum():3d}  obs " + " ".join(f"{v:4.2f}" for v in dist(g[mc], w[mc])))
    print("\nprofiles with ransomware (described only; pure R n above):")
    for lab, m in (("R + anything", R & (nch >= 2) & ok), ("R + 2+ others", R & (nch >= 3) & ok)):
        print(f"  {lab:14s} n {m.sum():4d}  " + " ".join(f"{v:.2f}" for v in dist(g[m], w[m])))


if __name__ == "__main__":
    main()
