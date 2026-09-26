"""Per-channel cost-shape checks and a screen for frailty acting on costs.

Uses the five-channel frailty fit with ransomware count (joint_five_channel.py free rcount).
1. Lognormal check per channel: worst band among firms whose MOST DISRUPTIVE attack was that
   channel (disrupta label), observed (weighted) vs simulated from the fit.
2. Cost-frailty screen: among firms with a costly worst incident from channel L, compare the
   number of OTHER channel groups flagged for high-cost (£5k+) vs low-cost (£1-5k) firms.
   The fit has frailty on counts only, which already induces some such association; if the
   observed gap is much larger than the simulated one, costs likely share the firm-level factor.
   Caveat: the fit pools sizes; bigger firms have both more attack types and costlier incidents,
   so observed gaps are also shown within Micro and within Rest.

Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src:src/estimation \
     python3 src/estimation/channel_cost_checks.py
"""

import json
import os
import numpy as np

import joint_four_channel as j
import joint_five_channel as f

HERE = os.path.dirname(os.path.abspath(__file__))
COARSE = [(1, 1, "no cost"), (2, 3, "£1-500"), (4, 5, "£500-5k"), (6, 7, "£5k-20k"),
          (8, 10, "£20k-500k"), (11, 13, ">£500k")]
LABELS = {1: "phishing", 2: "impersonation", 3: "ransomware", 4: "other serious"}
GROUP_OF_LABEL = {1: 0, 2: 1, 3: 2, 4: 3}      # flag groups: P, I, R, S


def simulate(q, n, rng):
    exposed = rng.random(n) < q["pi"]
    Y, K = [], {}
    for ch in f.CH:
        rate = q[f.RATE[ch]] * np.where(exposed, np.exp(q["d" + ch]), 1.0)
        if ch in ("T", "I"):
            k = rng.poisson(rate)
        else:
            r = q["r" + ch]
            k = rng.negative_binomial(r, r / (r + rate))
        K[ch] = k
        G = j.band_cdf(q["p" + ch], q["mu" + ch], q["s" + ch])
        Gk = G[None, :] ** np.minimum(k, j.KMAX)[:, None]
        Y.append(np.where(k == 0, 0, (Gk < rng.random(n)[:, None]).sum(1)))
    Y = np.stack(Y, 1)
    M = Y.max(1)
    lab = np.array([f.LAB[ch] for ch in f.CH])
    tie = (Y == M[:, None]) & (M[:, None] > 0)
    D = np.where(M > 0, lab[np.argmax(tie * rng.random(Y.shape), 1)], 0)
    flags = np.stack([(K["T"] + K["M"]) > 0, K["I"] > 0, K["R"] > 0, K["S"] > 0], 1)
    return M, D, flags


def main():
    d = f.load_firms()
    w = d["weight"].fillna(d["weight"].median()).values
    w = w / w.mean()
    q = json.load(open(os.path.join(HERE, "build", "joint_five_channel_free_rcount_weighted.json")))
    M, D, SF = simulate(q, 3_000_000, np.random.default_rng(11))
    OF = d[["fP", "fI", "fR", "fS"]].values.astype(bool)
    band = d["band"].values
    dis = d["D"].values

    print("=" * 80)
    print("1. WORST BAND | disrupta = channel: observed (weighted) vs simulated")
    print("=" * 80)
    for L, name in LABELS.items():
        m = (dis == L) & ~np.isnan(band)
        b, bw, sb = band[m], w[m], M[D == L]
        x2 = 0.0
        n_eff = bw.sum() ** 2 / (bw ** 2).sum()
        print(f"\n  {name} (obs n={m.sum()}, Kish n_eff={n_eff:.0f})")
        cells = 0
        for lo, hi, lab in COARSE:
            o = np.average((b >= lo) & (b <= hi), weights=bw)
            s = np.mean((sb >= lo) & (sb <= hi))
            oc = int(((b >= lo) & (b <= hi)).sum())
            if s * n_eff >= 0.5:
                x2 += n_eff * (o - s) ** 2 / s
                cells += 1
            print(f"    {lab:>10s}  obs {o:5.2f} (raw {oc:3d})   sim {s:5.3f}")
        from scipy.stats import chi2
        print(f"    weighted X2 = {x2:.1f} on {cells - 1} df  p={chi2.sf(x2, cells - 1):.3f}"
              f"  (n_eff-scaled weighted shares; rough)")

    print(f"\n{'=' * 80}")
    print("2. COST-FRAILTY SCREEN: # OTHER channel groups flagged, high (£5k+) vs low (£1-5k) cost")
    print("=" * 80)
    size = d["sizeb"].values
    print(f"  {'channel':>14s} {'n hi/lo':>8s} {'obs hi':>7s} {'obs lo':>7s} {'obs gap':>8s}"
          f" {'sim gap':>8s} | {'Micro gap':>9s} {'Rest gap':>9s}")
    for L, name in LABELS.items():
        g = GROUP_OF_LABEL[L]
        other = [k for k in range(4) if k != g]
        n_other_o = OF[:, other].sum(1)
        n_other_s = SF[:, other].sum(1)
        hi_o = (dis == L) & (band >= 6)
        lo_o = (dis == L) & (band >= 4) & (band <= 5)
        hi_s = (D == L) & (M >= 6)
        lo_s = (D == L) & (M >= 4) & (M <= 5)
        avg = lambda m: np.average(n_other_o[m], weights=w[m]) if m.sum() else np.nan
        gap_o = avg(hi_o) - avg(lo_o)
        gap_s = n_other_s[hi_s].mean() - n_other_s[lo_s].mean()
        gaps = []
        for sz in ([1], [2, 3, 4]):
            ms = np.isin(size, sz)
            gaps.append(avg(hi_o & ms) - avg(lo_o & ms))
        print(f"  {name:>14s} {hi_o.sum():3d}/{lo_o.sum():<4d} {avg(hi_o):7.2f} {avg(lo_o):7.2f} {gap_o:8.2f}"
              f" {gap_s:8.2f} | {gaps[0]:9.2f} {gaps[1]:9.2f}")


if __name__ == "__main__" and "coverage" not in __import__("sys").argv:
    main()


def per_type_cost_coverage():
    """Cyber crime module per-type annual cost vars: how many firms report 2+ types?"""
    from data import load_raw
    raw = load_raw()
    cols = ["ranscost_bands", "hackcost_bands", "viruscost_bands", "doscost_bands", "tkvrcost_bands"]
    v = raw[cols].apply(lambda s: j.banded(s).where(j.banded(s).between(1, 13)))
    att = j.banded(raw["type_comb1"]) == 1
    print("\nPER-TYPE ANNUAL COST VARIABLES (cyber crime module), attacked firms n =", int(att.sum()))
    for c in cols:
        x = v.loc[att, c]
        print(f"  {c:18s} reported {x.notna().sum():3d}  of which >no-cost {(x > 1).sum():3d}")
    k = v.loc[att].notna().sum(1)
    k2 = (v.loc[att] > 1).sum(1)
    print("  firms reporting k types:        " + "  ".join(f"k={i}: {(k == i).sum()}" for i in range(1, 6)))
    print("  firms with k types costing >0:  " + "  ".join(f"k={i}: {(k2 == i).sum()}" for i in range(1, 6)))


if __name__ == "__main__" and "coverage" in __import__("sys").argv:
    per_type_cost_coverage()
