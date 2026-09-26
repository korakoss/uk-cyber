"""Is 'serious' one channel? Descriptive profile of its 9 subtypes.

Per subtype: weighted prevalence, overlap with other serious subtypes, and — among firms
where that subtype was the MOST DISRUPTIVE attack (disrupta) — P(no cost) and the
cost-band profile given cost. Also the per-type count variables where they exist.
If subtypes share a cost profile, lumping them into one lognormal channel is harmless;
if not, the lumped lognormal is fitting a mixture.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/serious_subtypes.py
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from data import load_raw, BAND_BOUNDS
import joint_four_channel as j

SUB = {  # type col -> (label, disrupta code)
    "type1": ("Ransomware", 1), "type2": ("Other malware", 2), "type3": ("DoS", 3),
    "type4": ("Hacking bank accts", 4), "type7": ("Unauth. access: staff", 7),
    "type13": ("Unauth. access: students", 8), "type8": ("Unauth. access: outsiders", 9),
    "type15": ("Eavesdropping VC/IM", 10), "type16": ("Takeover web/social/email", 11),
}
COUNTS = {"type1": "Cybercrime_ranssum", "type2": "Cybercrime_virussum",
          "type4": "Cybercrime_hacksum", "type3": "Cybercrime_dossum"}
MID = {b: np.sqrt(max(lo, 1) * hi) if b > 1 else 0 for b, (lo, hi) in BAND_BOUNDS.items() if b <= 10}


def main():
    raw = load_raw()
    d = j.load_firms()
    base = raw.loc[raw["type_comb1"].isin([0, 1])].copy()
    # align raw rows to load_firms() rows by re-applying the same filter
    t = {c: (pd.to_numeric(raw[c], errors="coerce") == 1) for c in SUB}
    keep = raw["type_comb1"].isin([0, 1])
    fl = pd.DataFrame(t)[keep]
    anyflag = fl.any(axis=1) | (pd.to_numeric(raw.loc[keep, "type5"], errors="coerce") == 1) \
        | (pd.to_numeric(raw.loc[keep, "type6"], errors="coerce") == 1)
    sel = (raw.loc[keep, "type_comb1"] == 0) | anyflag
    fl = fl[sel].reset_index(drop=True)
    r = raw.loc[keep][sel].reset_index(drop=True)
    assert len(r) == len(d)
    w = d["weight"].fillna(d["weight"].median()).values
    att = d["att"].values == 1
    ser = d["fS"].values == 1
    band = d["band"].values
    dis = d["disrupta"].values

    print("=" * 100)
    print(f"SERIOUS SUBTYPES  (attacked firms n={att.sum()}, serious-flagged n={ser.sum()})")
    print("=" * 100)
    print(f"  {'subtype':26s} {'n':>4s} {'w.prev':>7s} {'%ser':>5s} {'only-ser':>8s}"
          f" | {'disr n':>6s} {'P(0cost)':>8s} {'mean band|>0':>12s} {'P(b>=7|>0)':>10s} {'geo£|>0':>8s}")
    rows = []
    for c, (lab, code) in SUB.items():
        f = fl[c].values
        n = f.sum()
        prev = np.average(f, weights=w)
        share = f[ser].sum() / ser.sum()
        only = (f & (fl.drop(columns=c).sum(1).values == 0)).sum()
        m = (dis == code) & ~np.isnan(band)
        b = band[m]
        bw = w[m]
        pc = np.average(b == 1, weights=bw) if m.sum() else np.nan
        pos = b > 1
        mb = np.average(b[pos], weights=bw[pos]) if pos.any() else np.nan
        p7 = np.average(b[pos] >= 7, weights=bw[pos]) if pos.any() else np.nan
        geo = np.exp(np.average(np.log([MID[int(x)] for x in b[pos]]), weights=bw[pos])) if pos.any() else np.nan
        rows.append((lab, b))
        print(f"  {lab:26s} {n:4d} {prev:7.3f} {share:5.2f} {only:8d}"
              f" | {m.sum():6d} {pc:8.2f} {mb:12.2f} {p7:10.2f} {geo:8.0f}")

    for lab, code in (("Impersonation (ref)", 5), ("Phishing (ref)", 6), ("Any other (12)", 12)):
        m = (dis == code) & ~np.isnan(band)
        b, bw = band[m], w[m]
        pos = b > 1
        print(f"  {lab:26s} {'':4s} {'':7s} {'':5s} {'':8s} | {m.sum():6d} {np.average(b == 1, weights=bw):8.2f}"
              f" {np.average(b[pos], weights=bw[pos]):12.2f} {np.average(b[pos] >= 7, weights=bw[pos]):10.2f}"
              f" {np.exp(np.average(np.log([MID[int(x)] for x in b[pos]]), weights=bw[pos])):8.0f}")

    # homogeneity of cost-band profile across subtypes with enough disrupta firms
    print("\n  Cost-band homogeneity across serious subtypes (disrupta firms, bands coarsened"
          " to {0, 2-3, 4-5, 6-7, 8-10}):")
    coarse = lambda b: np.digitize(b, [2, 4, 6, 8])
    big = [(lab, b) for lab, b in rows if len(b) >= 15]
    tab = np.array([[np.sum(coarse(b) == k) for k in range(5)] for _, b in big])
    stat, p, dof, _ = chi2_contingency(tab + 0.5)
    for (lab, _), row in zip(big, tab):
        print(f"    {lab:26s} " + " ".join(f"{v:4d}" for v in row))
    print(f"    chi2={stat:.1f} dof={dof} p={p:.3g}  (unweighted counts)")

    # co-occurrence among serious subtypes
    print("\n  Among serious-flagged firms: number of serious subtypes flagged")
    ns = fl.values[ser].sum(1)
    for k in range(1, 6):
        print(f"    {k}{'+' if k == 5 else ''}: {np.sum(ns == k) if k < 5 else np.sum(ns >= 5):4d}")
    print("\n  Pairwise co-flag share among serious firms (P(both)/P(either)):")
    cols = list(SUB)
    short = [SUB[c][0][:10] for c in cols]
    print("    " + " ".join(f"{s:>10s}" for s in short))
    X = fl.values[ser].astype(bool)
    for i, c in enumerate(cols):
        print(f"    " + " ".join(f"{(X[:, i] & X[:, k]).sum() / max((X[:, i] | X[:, k]).sum(), 1):10.2f}"
                                 if k != i else f"{'-':>10s}" for k in range(len(cols))) + f"  {short[i]}")

    # counts
    print("\n  Per-type count variables (firms flagged for that subtype):")
    for c, v in COUNTS.items():
        x = pd.to_numeric(r[v], errors="coerce").values
        m = fl[c].values & (x >= 1)
        cov = m.sum() / max(fl[c].values.sum(), 1)
        if m.sum():
            q = np.quantile(x[m], [0.25, 0.5, 0.75, 0.9])
            print(f"    {SUB[c][0]:26s} coverage {cov:.2f} (n={m.sum()})  Q25/50/75/90 = "
                  + "/".join(f"{v:.0f}" for v in q) + f"  mean {x[m].mean():.1f}")


if __name__ == "__main__":
    main()


def ranssum_profile():
    """Cybercrime_ranssum among firms by ransomware flag: coding, coverage, values."""
    import joint_five_channel as f
    raw = load_raw()
    d = f.load_firms()
    x = pd.to_numeric(raw["Cybercrime_ranssum"], errors="coerce")
    print("\nCybercrime_ranssum raw value counts (all rows):")
    print(x.value_counts(dropna=False).sort_index().to_string())
    nr = d["NR"]
    print(f"\nfR=1 firms: {int(d.fR.sum())}; with NR>=1: {int(nr.notna().sum())}")
    print("NR values:", nr.dropna().astype(int).value_counts().sort_index().to_dict())
    for lab, m in (("disrupta=R", d["D"] == 3), ("not disrupta=R", (d["D"] != 3) & (d.fR == 1))):
        print(f"  {lab}: n={int(m.sum())}, NR coverage {nr[m].notna().mean():.2f}")


if __name__ == "__main__" and "ranssum" in __import__("sys").argv:
    ranssum_profile()


def other_serious_homogeneity():
    """Cost-profile homogeneity among the populated OTHER-serious subtypes (ransomware excluded).
    Firms whose most disruptive attack was the subtype; worst band coarsened. Chi-square with a
    permutation p-value (small expected counts), unweighted and survey-weighted (Kish-scaled)."""
    from scipy.stats import chi2 as chi2d
    d = j.load_firms()
    w = d["weight"].fillna(d["weight"].median()).values
    subs = {2: "Other malware", 3: "DoS", 4: "Bank hacking", 11: "Takeover"}
    m = d["disrupta"].isin(list(subs)).values & d["band"].notna().values
    lab, band, ww = d["disrupta"].values[m].astype(int), d["band"].values[m], w[m]
    for name, edges in (("5 bins {no cost, £1-500, £500-5k, £5k-20k, £20k+}", [2, 4, 6, 8]),
                        ("3 bins {no cost, £1-500, £500+}", [2, 4])):
        cb = np.digitize(band, edges)
        K = cb.max() + 1

        def stat(labels, weights=None):
            wt = np.ones(len(labels)) if weights is None else weights
            T = np.array([[wt[(labels == s) & (cb == k)].sum() for k in range(K)] for s in subs])
            E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
            return ((T - E) ** 2 / np.where(E > 0, E, 1)).sum(), T

        x2, T = stat(lab)
        rng = np.random.default_rng(0)
        perm = np.array([stat(rng.permutation(lab))[0] for _ in range(5000)])
        wn = ww / ww.mean()
        x2w, Tw = stat(lab, wn)
        deff = (wn ** 2).mean()                     # Kish design effect (mean-1 weights)
        dof = (len(subs) - 1) * (K - 1)
        print(f"\nOTHER-SERIOUS HOMOGENEITY (ransomware excluded), {name}")
        for (code, s), row, roww in zip(subs.items(), T, Tw):
            print(f"  {s:14s} n={int(row.sum()):3d}  raw " + " ".join(f"{int(v):3d}" for v in row) +
                  "   weighted shares " + " ".join(f"{v / roww.sum():.2f}" for v in roww))
        print(f"  unweighted X2={x2:.1f}, dof={dof}: asymptotic p={chi2d.sf(x2, dof):.3f}, "
              f"permutation p={np.mean(perm >= x2):.3f}")
        print(f"  weighted X2/deff={x2w / deff:.1f} (deff {deff:.2f}): p={chi2d.sf(x2w / deff, dof):.3f}")


if __name__ == "__main__" and "homog" in __import__("sys").argv:
    other_serious_homogeneity()
