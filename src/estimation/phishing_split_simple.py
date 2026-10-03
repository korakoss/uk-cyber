"""Simple, frailty-free signatures of two kinds of phishing attack. Phishing-only firms (worst incident is
phishing), survey-weighted.

1. P(no cost at all | exact count N) vs the one-homogeneous-type prediction: with per-attack no-cost
   probability q, P(no cost | N) = q^N; q estimated from N = 1 firms; prediction per bucket averages q^N_i
   over the bucket's firms.
2. Per-attack no-cost probability per count bucket (un-maxed by ML using each firm's own N):
   maximise sum w [ no cost * N log q + costly * log(1 - q^N) ].
3. Survey labels: phishcon_bands (# attacks that looked specifically targeted) and phisheng_bands
   (# someone engaged with). Cost profile by none vs >= 1; and share of attacks targeted (band midpoint / N)
   by count bucket.
Firm bootstrap intervals.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/phishing_split_simple.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import joint_four_channel as j
import joint_five_channel as f
from data import load_raw

BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10**6)]
MID = {1: 0, 2: 1, 3: 2.5, 4: 4.5, 5: 8, 6: 15.5, 7: 35.5, 8: 75.5, 9: 150}


def blab(lo, hi):
    return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f"{lo}+")


def per_attack_q(nocost, N, w):
    def nll(q):
        return -(w * np.where(nocost, N * np.log(q), np.log(np.maximum(1 - q ** N, 1e-300)))).sum()
    return minimize_scalar(nll, bounds=(1e-6, 1 - 1e-9), method="bounded").x


def main():
    d = f.load_firms()
    raw = load_raw()
    keep = j.banded(raw["type_comb1"]).isin([0, 1])
    allg = np.column_stack([(j.banded(raw[c]) == 1).astype(int)[keep].values for c in j.GENUINE_TYPE_COLS])
    a0 = j.banded(raw["type_comb1"])[keep].values
    sel = (a0 == 0) | (allg.sum(1) > 0)
    lab = raw.loc[keep][sel].reset_index(drop=True)
    tcon = pd.to_numeric(lab["phishcon_bands"], errors="coerce").where(lambda s: s.between(1, 9)).values
    teng = pd.to_numeric(lab["phisheng_bands"], errors="coerce").where(lambda s: s.between(1, 9)).values

    w = d["weight"].fillna(d["weight"].median()).values
    band, N = d["band"].values, d["N"].values
    fl = {x: d[x].values == 1 for x in ("fP", "fI", "fR", "fS")}
    ponly = fl["fP"] & ~fl["fI"] & ~fl["fR"] & ~fl["fS"] & ~np.isnan(band)
    rng = np.random.default_rng(0)

    print("=" * 100)
    print("1 + 2. PHISHING-ONLY FIRMS BY EXACT COUNT: P(no cost at all) vs one-type prediction; per-attack P(no cost)")
    print("=" * 100)
    for glab, gm in (("All", np.ones(len(d), bool)), ("Micro", d["sizeb"].values == 1), ("Small+Medium+Large", d["sizeb"].values >= 2)):
        m = ponly & ~np.isnan(N) & gm
        nc, Nm, wm = band[m] == 1, N[m], w[m]
        q1 = np.average(nc[Nm == 1], weights=wm[Nm == 1])
        print(f"\n  {glab}: n={m.sum()}, per-attack no-cost from N=1 firms q = {q1:.2f}")
        print(f"  {'N':>6s} {'n':>4s}  {'observed P(no cost)':>22s}  {'one-type pred.':>14s}   {'per-attack P(no cost) [95%]':>28s}")
        for lo, hi in BUCKETS:
            b = (Nm >= lo) & (Nm <= hi)
            if b.sum() == 0:
                continue
            obs = np.average(nc[b], weights=wm[b])
            pred = np.average(q1 ** Nm[b], weights=wm[b])
            qa = per_attack_q(nc[b], Nm[b], wm[b])
            idx = np.where(b)[0]
            bo, bq = [], []
            for _ in range(500):
                r = rng.choice(idx, len(idx))
                bo.append(np.average(nc[r], weights=wm[r]))
                bq.append(per_attack_q(nc[r], Nm[r], wm[r]))
            lo_o, hi_o = np.percentile(bo, [2.5, 97.5])
            lo_q, hi_q = np.percentile(bq, [2.5, 97.5])
            print(f"  {blab(lo, hi):>6s} {b.sum():4d}  {obs:6.2f} [{lo_o:.2f}, {hi_o:.2f}]     {pred:14.3f}   "
                  f"{qa:10.3f} [{lo_q:.3f}, {hi_q:.3f}]")

    print("\n" + "=" * 100)
    print("3. SURVEY LABELS (phishing-only firms): cost profile by targeted / engaged")
    print("=" * 100)
    for name, v in (("targeted (phishcon)", tcon), ("engaged (phisheng)", teng)):
        for gl, gmask in (("none", v == 1), (">= 1", v >= 2)):
            m = ponly & gmask
            s = [np.average(band[m] == 1, weights=w[m]), np.average((band[m] >= 2) & (band[m] <= 3), weights=w[m]),
                 np.average(band[m] >= 4, weights=w[m])]
            print(f"  {name:20s} {gl:>5s}: n={m.sum():4d}   no cost {s[0]:.2f}  £1-500 {s[1]:.2f}  £500+ {s[2]:.2f}")
    print("\n  share of attacks that looked targeted (band midpoint / N), by phishing count (phishing-only, both known):")
    for lo, hi in BUCKETS:
        m = ponly & ~np.isnan(N) & ~np.isnan(tcon) & (N >= lo) & (N <= hi)
        if m.sum() < 3:
            continue
        share = np.minimum(np.array([MID[int(x)] for x in tcon[m]]) / N[m], 1)
        anyt = np.average(tcon[m] >= 2, weights=w[m])
        print(f"    N {blab(lo, hi):>6s}: n={m.sum():3d}  mean targeted share {np.average(share, weights=w[m]):.2f}"
              f"   P(any targeted) {anyt:.2f}")


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def count_vs_targeted_band():
    """Is Cybercrime_phishsum (N) the same quantity as the targeted-attack count (phishcon_bands)?
    Compare N's band (same cut points) with the reported targeted band."""
    d = f.load_firms()
    raw = load_raw()
    keep = j.banded(raw["type_comb1"]).isin([0, 1])
    allg = np.column_stack([(j.banded(raw[c]) == 1).astype(int)[keep].values for c in j.GENUINE_TYPE_COLS])
    a0 = j.banded(raw["type_comb1"])[keep].values
    sel = (a0 == 0) | (allg.sum(1) > 0)
    lab = raw.loc[keep][sel].reset_index(drop=True)
    tcon = pd.to_numeric(lab["phishcon_bands"], errors="coerce").where(lambda s: s.between(1, 9)).values
    N = d["N"].values
    m = ~np.isnan(N) & ~np.isnan(tcon)
    nb = np.digitize(N[m], [1, 2, 4, 6, 11, 21, 51, 101]) + 1        # N -> same band codes (2 = '1', ...)
    print(f"\nfirms with both N and targeted band: {m.sum()}")
    print(f"  targeted band == band of N: {np.mean(nb == tcon[m]):.2f}   targeted band < band of N: "
          f"{np.mean(tcon[m] < nb):.2f}   targeted band > band of N: {np.mean(tcon[m] > nb):.2f}")
    print(f"  phishing firms WITHOUT a count: {int(((d['fP'].values == 1) & np.isnan(N)).sum())}; of those with a targeted "
          f"band: {int(((d['fP'].values == 1) & np.isnan(N) & ~np.isnan(tcon)).sum())}, "
          f"reporting none targeted: {int(((d['fP'].values == 1) & np.isnan(N) & (tcon == 1)).sum())}")


if __name__ == "__main__" and "same" in __import__("sys").argv:
    count_vs_targeted_band()
