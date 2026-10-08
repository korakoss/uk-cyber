"""One-law test: counts K ~ Poisson(lambda), log lambda ~ Normal(m, sigma), sigma common to everything.

For fixed sigma, a cell's P(K >= 1) pins its centre m, and then the whole conditional distribution of
K | K >= 1 is predicted -- so all cells should fall on ONE curve in the plane
(P(K >= 1), median of K | K >= 1). Per cell: observed P(K >= 1) and observed median (and quartiles) of positive
counts; predicted median / quartiles for sigma in SIGMAS at that cell's P. Medians rather than means (means are
driven by top-codes 500/999). Unweighted.

Cells (channel x group):
  targeted phishing: exact phishcon (0 for firms not hit by phishing); size group x most-likely tier, and pooled.
      P = share with t >= 1 among firms with a known t.
  ransomware:  P = ransomware flag share among all firms; positives = Cybercrime_ranssum.
  other serious: P = share flagged with any of malware / DoS / bank hacking / unauthorised access / takeover;
      positives = per-firm sum of Cybercrime_hacksum + virussum + dossum.
  ransomware and other serious by size group (Micro / Small+) and by tier (sizes pooled).

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/lognormal_exposure_curve.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import poisson

from counts_given_frailty import tier_posteriors, TIER
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load, SHORT

SIGMAS = [1.5, 2.0, 2.5, 3.0]
Z, WZ = np.polynomial.hermite_e.hermegauss(80)
WZ = WZ / WZ.sum()
KS = np.arange(0, 20001)


def p_hit(m, s):
    lam = np.exp(m + s * Z)
    return (WZ * -np.expm1(-lam)).sum()


def positive_quantiles(m, s, qs=(0.25, 0.5, 0.75)):
    lam = np.exp(m + s * Z)
    pk = (WZ[:, None] * np.exp(poisson.logpmf(KS[None, :], lam[:, None]))).sum(0)
    pk[0] = 0
    c = np.cumsum(pk) / pk.sum()
    return [int(KS[np.searchsorted(c, q)]) for q in qs]


def predict(p, s):
    m = brentq(lambda m: p_hit(m, s) - p, -30, 15)
    return m, positive_quantiles(m, s)


def cell(label, hit, pos):
    """hit: boolean array (K >= 1) over the cell's firms; pos: positive counts observed in the cell."""
    p = hit.mean()
    q = np.percentile(pos, [25, 50, 75]) if len(pos) else [np.nan] * 3
    preds = []
    if not 0 < p < 0.995:
        print(f"  {label:28s} n {len(hit):5d}  P {p:5.3f}  (P at boundary, no prediction)")
        return
    for s in SIGMAS:
        m, pq = predict(p, s)
        preds.append(f"{pq[1]:>4d} [{pq[0]},{pq[2]}]")
    print(f"  {label:28s} n {len(hit):5d}  P {p:5.3f}  n+ {len(pos):4d}  obs med {q[1]:6.1f} "
          f"[{q[0]:g},{q[2]:g}]   pred " + "   ".join(preds))


def main():
    X, w, size = load()
    tier = tier_posteriors().argmax(1)
    r = aligned_raw()
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values
    grp = {"Micro": size == 1, "Small+": size >= 2}
    print("pred = median [q25, q75] of K | K >= 1 for sigma " + " / ".join(str(s) for s in SIGMAS))

    print("\nTARGETED PHISHING")
    t = num("phishcon")
    t = np.where(X[:, 0] == 1, np.where(t >= 0, t, np.nan), 0.0)
    known = ~np.isnan(t)
    for g, gm in grp.items():
        m = gm & known
        cell(f"{g} all tiers", t[m] >= 1, t[m & (t >= 1)])
        for c in range(3):
            mc = m & (tier == c)
            cell(f"{g} {TIER[c]}", t[mc] >= 1, t[mc & (t >= 1)])

    rs = num("Cybercrime_ranssum")
    fr = X[:, SHORT.index("Ransm")] == 1
    os_types = ["Malwr", "DoS", "BankH", "AcOut", "AcStf", "Takov"]
    fs = X[:, [SHORT.index(x) for x in os_types]].max(1) == 1
    V = np.column_stack([num(c) for c in ("Cybercrime_hacksum", "Cybercrime_virussum", "Cybercrime_dossum")])
    V = np.where(V >= 1, V, np.nan)
    ossum = np.where(np.isnan(V).all(1), np.nan, np.nansum(V, 1))
    for name, flag, cnt in (("RANSOMWARE", fr, rs), ("OTHER SERIOUS", fs, ossum)):
        print(f"\n{name}")
        allf = np.ones(len(X), bool)
        cell("all firms", flag[allf], cnt[flag & (cnt >= 1)])
        for g, gm in grp.items():
            cell(g, flag[gm], cnt[gm & flag & (cnt >= 1)])
        for c in range(3):
            mc = tier == c
            cell(f"tier {TIER[c]} (sizes pooled)", flag[mc], cnt[mc & flag & (cnt >= 1)])


if __name__ == "__main__":
    main()
