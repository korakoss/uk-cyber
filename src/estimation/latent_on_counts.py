"""Step 2 of the counts argument: does the shared latent (Step 1: one-dimensional co-occurrence
factor) also scale attack INTENSITY, and multiplicatively?

Intensity = exact phishing count N (Cybercrime_phishsum). Read-out of the latent = number of OTHER
channels hit (impersonation, ransomware, other serious flags). These are different measurements,
so an association can only run through a shared firm-level factor.
For phishing-flagged attacked firms, by # other channels (0 / 1 / 2+), within Micro, within
Small+Medium+Large, and pooled (survey-weighted within each):
  - reporting rate of N (costly firms report more; check it doesn't differ wildly by group)
  - weighted quantiles of N and their ratio to the 0-other-channels group: similar ratios across
    quantiles => the whole distribution scales (multiplicative), not just a fatter tail
  - same quantiles after inverse-probability weighting for reporting by worst band
  - bootstrap 90% interval for the shift in median log N (2+ vs 0)

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/latent_on_counts.py
"""

import numpy as np

import joint_five_channel as f

QS = [0.25, 0.5, 0.75, 0.9]


def wq(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, c, x)


def analyse(d, w, label, rng):
    fP = d["fP"].values == 1
    k_other = d[["fI", "fR", "fS"]].values.sum(1)
    grp = np.where(k_other >= 2, 2, k_other)
    N = d["N"].values
    band = d["band"].values
    base = fP & (d["att"].values == 1)
    have = base & ~np.isnan(N)

    # IPW for reporting, by worst band (pooled over groups within this size set)
    ipw = np.ones(len(d))
    for b in np.unique(band[base & ~np.isnan(band)]):
        m = base & (band == b)
        rate = np.average(~np.isnan(N[m]), weights=w[m])
        ipw[m] = 1 / max(rate, 0.05)

    print("\n" + "=" * 92)
    print(f"{label}")
    print("=" * 92)
    print(f"  {'other ch.':>9s} {'firms':>6s} {'w/ N':>5s} {'report':>7s}   "
          f"{'N quantiles 25/50/75/90':>26s}   {'ratio to 0-group at each quantile':>34s}")
    ref = None
    rows = {}
    for g in (0, 1, 2):
        m = base & (grp == g)
        mh = have & (grp == g)
        rep = np.average(~np.isnan(N[m]), weights=w[m]) if m.sum() else np.nan
        qv = np.array([wq(N[mh], w[mh], q) for q in QS]) if mh.sum() >= 5 else np.full(4, np.nan)
        rows[g] = qv
        if g == 0:
            ref = qv
        print(f"  {['0', '1', '2+'][g]:>9s} {m.sum():6d} {mh.sum():5d} {rep:7.2f}   " +
              " ".join(f"{v:6.1f}" for v in qv) + "   " + " ".join(f"{v / r:7.2f}" for v, r in zip(qv, ref)))
    print("  IPW-adjusted (reporting by worst band):")
    ref2 = None
    for g in (0, 1, 2):
        mh = have & (grp == g) & ~np.isnan(band)
        qv = np.array([wq(N[mh], (w * ipw)[mh], q) for q in QS]) if mh.sum() >= 5 else np.full(4, np.nan)
        if g == 0:
            ref2 = qv
        print(f"  {['0', '1', '2+'][g]:>9s} {'':6s} {mh.sum():5d} {'':7s}   " +
              " ".join(f"{v:6.1f}" for v in qv) + "   " + " ".join(f"{v / r:7.2f}" for v, r in zip(qv, ref2)))

    a, b = have & (grp == 0), have & (grp == 2)
    ia, ib = np.where(a)[0], np.where(b)[0]
    boots = []
    for _ in range(2000):
        sa, sb = rng.choice(ia, len(ia)), rng.choice(ib, len(ib))
        boots.append(np.log(wq(N[sb], w[sb], 0.5)) - np.log(wq(N[sa], w[sa], 0.5)))
    lo, hi = np.percentile(boots, [5, 95])
    print(f"  median shift 2+ vs 0: x{np.exp(np.log(rows[2][1]) - np.log(rows[0][1])):.2f}"
          f"  (90% boot x{np.exp(lo):.2f} to x{np.exp(hi):.2f})")


def main():
    d = f.load_firms()
    w = d["weight"].fillna(d["weight"].median()).values
    rng = np.random.default_rng(0)
    for label, sizes in (("MICRO", [1]), ("SMALL+MEDIUM+LARGE", [2, 3, 4]), ("POOLED", [1, 2, 3, 4])):
        m = d["sizeb"].isin(sizes).values
        dd = d[m].reset_index(drop=True)
        ww = w[m] / w[m].mean()
        analyse(dd, ww, label, rng)


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def one_other_split():
    """Split the '1 other channel' group by WHICH channel: impersonation (often co-labelled with
    phishing for one incident) vs ransomware / other serious."""
    d = f.load_firms()
    w = d["weight"].fillna(d["weight"].median()).values
    base = (d["fP"].values == 1) & (d["att"].values == 1)
    fI, fR, fS = (d[c].values == 1 for c in ("fI", "fR", "fS"))
    N = d["N"].values
    groups = {"0 other": base & ~fI & ~fR & ~fS,
              "only +impersonation": base & fI & ~fR & ~fS,
              "only +ransom/serious": base & ~fI & (fR ^ fS),
              "2+ other": base & (fI.astype(int) + fR + fS >= 2)}
    print("\nPHISHING COUNT BY WHICH OTHER CHANNEL (pooled, survey-weighted)")
    ref = None
    for lab, m in groups.items():
        mh = m & ~np.isnan(N)
        qv = np.array([wq(N[mh], w[mh], q) for q in QS])
        ref = qv if ref is None else ref
        print(f"  {lab:22s} firms {m.sum():4d}  w/N {mh.sum():4d}  q25/50/75/90 " +
              " ".join(f"{v:6.1f}" for v in qv) + "   ratio " + " ".join(f"{v / r:5.2f}" for v, r in zip(qv, ref)))


if __name__ == "__main__" and "split" in __import__("sys").argv:
    one_other_split()
