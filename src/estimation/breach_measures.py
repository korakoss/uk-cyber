"""Breach measures for the channels without a success count.

fraud:    Can the fraud-outcome counts (Q88A fraud1-4) serve as impersonation's breach count?
          Labels of fraud1-4 and the Q88D attribution items (fraudconta-i); P(any fraud) and count values among
          impersonation-hit vs other attacked firms (impersonation-only, phishing-only, both); attribution of frauds
          to causes among firms with fraud; worst-incident cost with vs without fraud among impersonation-only firms.
subtract: Targeted-phishing breaches by subtraction. Engagement (phisheng) among firms with 0 targeted (= mass
          breaches only) vs >= 1 targeted (mass + targeted), by size group x tier. Implied targeted breach probability
          1 - (1 - p_both) / (1 - p_mass) under 'mass engagement rate is the same whether or not targeted'.
Unweighted.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/breach_measures.py fraud|subtract
"""

import glob
import sys

import numpy as np
import pandas as pd
import pyreadstat

import joint_five_channel as f
from counts_given_frailty import tier_posteriors, TIER
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load, SHORT

FRAUD = ["fraud1", "fraud2", "fraud3", "fraud4"]
CONT = ["fraudconta", "fraudcontb", "fraudcontc", "fraudcontd", "fraudconte", "fraudcontf", "fraudcontg",
        "fraudconth", "fraudconti"]


def num(r, c):
    v = pd.to_numeric(r[c], errors="coerce").values
    return np.where((v >= 0) & (v < 997), v, np.nan)


def fraud():
    sav = glob.glob("/home/user/uk-cyber/data/raw/*.sav")[0]
    _, meta = pyreadstat.read_sav(sav, metadataonly=True)
    for c in FRAUD + CONT:
        print(f"{c:11s} {str(meta.column_names_to_labels.get(c))[-120:]}")
    X, _, size = load()
    r = aligned_raw()
    d = f.load_firms()
    band = d["band"].values
    F = np.column_stack([num(r, c) for c in FRAUD])
    imp, ph = X[:, SHORT.index("Imper")] == 1, X[:, 0] == 1
    only = X.sum(1) == 1
    att = X.sum(1) >= 1
    groups = [("impersonation-only", only & imp), ("phishing-only", only & ph), ("imp + phishing (+others)", imp & ph),
              ("imp, no phishing", imp & ~ph), ("phishing, no imp", ph & ~imp), ("attacked, neither", att & ~imp & ~ph)]
    print("\nfraud outcome counts (Q88A): P(>=1) and positive values, per group [answered n]")
    for lab, g in groups:
        cells = []
        for j, c in enumerate(FRAUD):
            v = F[g & ~np.isnan(F[:, j]), j]
            pos = v[v >= 1]
            cells.append(f"{c} n{len(v):4d} P {np.mean(v >= 1) if len(v) else float('nan'):.3f} "
                         f"[{' '.join(str(int(x)) for x in np.sort(pos))}]")
        print(f"  {lab}:")
        for cc in cells:
            print(f"      {cc}")
    tot = np.nansum(F[:, :3], 1)
    anyf = tot >= 1
    C = np.column_stack([num(r, c) for c in CONT])
    print(f"\nattribution (Q88D) among firms with any fraud1-3 (n {anyf.sum()}); answered any Q88D item: "
          f"{int((anyf & ~np.isnan(C).all(1)).sum())}")
    lab = ["ransom", "malware", "DoS", "hacking", "phishing", "unauth-a", "unauth-b", "unauth-c", "takeover"]
    s = np.nansum(C[anyf], 0)
    print("  frauds attributed: " + ", ".join(f"{l} {int(x)}" for l, x in zip(lab, s)) +
          f";  total frauds (fraud1-3) {int(tot[anyf].sum())};  unattributed residual {int(tot[anyf].sum() - s.sum())}")
    for gl, g in (("impersonation-hit", imp), ("not impersonation-hit", ~imp)):
        m = anyf & g
        print(f"  {gl}: firms with fraud {m.sum()}, frauds {int(tot[m].sum())}, attributed {int(np.nansum(C[m]))}")
    io = only & imp & ~np.isnan(band)
    fa = np.nansum(F, 1) >= 1
    print("\nimpersonation-only firms, worst-incident band shares none / <£500 / £500-5k / £5k+:")
    for gl, m in (("any fraud outcome", io & fa), ("no fraud outcome", io & ~fa & ~np.isnan(F).all(1))):
        b = band[m]
        print(f"  {gl:18s} n {m.sum():3d}  " + " ".join(f"{np.mean((b >= a) & (b <= c)):.2f}" for a, c in ((1, 1), (2, 3), (4, 5), (6, 13))))


def subtract():
    X, _, size = load()
    r = aligned_raw()
    tier = tier_posteriors().argmax(1)
    e, t = num(r, "phisheng"), num(r, "phishcon")
    cb = pd.to_numeric(r["phishcon_bands"], errors="coerce").values
    t0 = np.where(~np.isnan(t), t == 0, cb == 1)
    t1 = np.where(~np.isnan(t), t >= 1, (cb >= 2) & (cb <= 9))
    ph = (X[:, 0] == 1) & ~np.isnan(e)
    print("engagement among phishing firms: mass-only (0 targeted) vs targeted >= 1")
    print(f"  {'cell':16s} {'n mass':>6s} {'P eng mass':>10s} {'n both':>6s} {'P eng both':>10s} {'implied P targeted':>18s}"
          f"   mean eng|>=1 mass / both")
    cells = [("all", np.ones(len(X), bool))] + [(g, gm) for g, gm in (("Micro", size == 1), ("Small+", size >= 2))] + \
            [(f"{g} {TIER[c]}", gm & (tier == c)) for g, gm in (("Micro", size == 1), ("Small+", size >= 2)) for c in range(3)]
    rng = np.random.default_rng(0)
    for lab, m in cells:
        a, b = ph & m & t0, ph & m & t1
        if a.sum() < 5 or b.sum() < 5:
            print(f"  {lab:16s} too few ({a.sum()}, {b.sum()})")
            continue
        pa, pb = np.mean(e[a] >= 1), np.mean(e[b] >= 1)
        imp = 1 - (1 - pb) / (1 - pa)
        bs = []
        ia, ib = np.where(a)[0], np.where(b)[0]
        for _ in range(2000):
            xa, xb = e[rng.choice(ia, len(ia))], e[rng.choice(ib, len(ib))]
            qa = np.mean(xa >= 1)
            bs.append(1 - (1 - np.mean(xb >= 1)) / (1 - qa) if qa < 1 else np.nan)
        lo, hi = np.nanpercentile(bs, [5, 95])
        ma = e[a & (e >= 1)].mean() if (a & (e >= 1)).any() else float("nan")
        mb = e[b & (e >= 1)].mean() if (b & (e >= 1)).any() else float("nan")
        print(f"  {lab:16s} {a.sum():6d} {pa:10.3f} {b.sum():6d} {pb:10.3f} {imp:8.3f} [{lo:.2f},{hi:.2f}]   {ma:.1f} / {mb:.1f}")


if __name__ == "__main__":
    {"fraud": fraud, "subtract": subtract}[sys.argv[1]]()
