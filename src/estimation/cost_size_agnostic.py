"""Model-free check that per-attack cost does not depend on firm size.

Worst-incident costs can't be compared by size directly (larger firms get more attacks, so their
worst incident is mechanically higher). Single-attack firms avoid this: their worst incident IS the
one attack's cost (incl. whether it cost anything). Slices:
  - phishing: phishing-only firms with exact phishing count N = 1
  - impersonation: impersonation-only firms answering freq = 'once'
  - other serious: other-serious-only (no ransomware) firms answering 'once'
  - ransomware: exact ransomware count 1 and ransomware the most disruptive attack
Plus phishing matched on count: phishing-only firms by N bucket, size compared within bucket.
Cost bins: no cost | £1-500 | £500+. Micro vs Small+Medium+Large (groups are thin).
Test: survey-weighted shares; X2 on weighted table divided by Kish design effect, and a
permutation p-value on raw counts.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/cost_size_agnostic.py
"""

import numpy as np
from scipy.stats import chi2

import joint_five_channel as f

BIN_EDGES = [2, 4]           # band 1 = no cost, 2-3 = £1-500, 4+ = £500+
BIN_LAB = ["no cost", "£1-500", "£500+"]


def table(cb, grp, wt, G):
    return np.array([[wt[(grp == g) & (cb == k)].sum() for k in range(3)] for g in range(G)])


def x2(T):
    E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
    return ((T - E) ** 2 / np.where(E > 0, E, 1)).sum()


def compare(band, big, w, label, rng):
    cb = np.digitize(band, BIN_EDGES)
    grp = big.astype(int)
    wn = w / w.mean()
    Tw = table(cb, grp, wn, 2)
    Tr = table(cb, grp, np.ones(len(cb)), 2)
    deff = (wn ** 2).mean()
    stat = x2(Tr)
    perm = np.array([x2(table(cb, rng.permutation(grp), np.ones(len(cb)), 2)) for _ in range(5000)])
    print(f"\n  {label}")
    for g, name in enumerate(["Micro", "Small+Med+Large"]):
        sh = Tw[g] / Tw[g].sum() if Tw[g].sum() else np.full(3, np.nan)
        print(f"    {name:>16s} n={int(Tr[g].sum()):3d}   " + "  ".join(f"{l} {v:.2f}" for l, v in zip(BIN_LAB, sh)))
    print(f"    weighted X2/deff={x2(Tw) / deff:.2f} (2 df) p={chi2.sf(x2(Tw) / deff, 2):.2f}   "
          f"permutation p (raw)={np.mean(perm >= stat):.2f}")
    return Tw, Tr


def main():
    d = f.load_firms()
    w = d["weight"].fillna(d["weight"].median()).values
    rng = np.random.default_rng(0)
    band, N, NR, freq = d["band"].values, d["N"].values, d["NR"].values, d["freq"].values
    big = d["sizeb"].values >= 2
    fl = {k: d[k].values == 1 for k in ("fP", "fI", "fR", "fS")}
    ok = ~np.isnan(band)
    print("=" * 96)
    print("SINGLE-ATTACK FIRMS: per-attack cost by size (survey-weighted shares)")
    print("=" * 96)
    slices = {
        "Phishing: phishing-only, exact count N = 1":
            ok & fl["fP"] & ~fl["fI"] & ~fl["fR"] & ~fl["fS"] & (N == 1),
        "Impersonation: impersonation-only, answered 'once'":
            ok & fl["fI"] & ~fl["fP"] & ~fl["fR"] & ~fl["fS"] & (freq == 1),
        "Other serious: other-serious-only, answered 'once'":
            ok & fl["fS"] & ~fl["fP"] & ~fl["fI"] & ~fl["fR"] & (freq == 1),
        "Ransomware: exact count 1, ransomware most disruptive":
            ok & (NR == 1) & (d["D"].values == 3),
    }
    for lab, m in slices.items():
        compare(band[m], big[m], w[m], lab, rng)

    print("\n" + "=" * 96)
    print("PHISHING-ONLY FIRMS MATCHED ON COUNT: size compared within each count bucket")
    print("=" * 96)
    base = ok & fl["fP"] & ~fl["fI"] & ~fl["fR"] & ~fl["fS"] & ~np.isnan(N)
    tot_stat, tot_df, tot_perm = 0.0, 0, 0.0
    for lo, hi in ((1, 1), (2, 3), (4, 10), (11, 50), (51, 10**9)):
        m = base & (N >= lo) & (N <= hi)
        lab = f"N = {lo}" if lo == hi else (f"N = {lo}-{hi}" if hi < 10**9 else f"N >= {lo}")
        Tw, Tr = compare(band[m], big[m], w[m], lab, rng)
        wn = w[m] / w[m].mean()
        tot_stat += x2(Tw) / (wn ** 2).mean()
        tot_df += 2
    print(f"\n  combined across count buckets: X2/deff = {tot_stat:.1f} on {tot_df} df, p = {chi2.sf(tot_stat, tot_df):.2f}")


if __name__ == "__main__":
    main()
