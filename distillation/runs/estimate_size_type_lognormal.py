"""Approach B — fully parametric size x type lognormal cell-mean estimate.

The national total under a fully parametric cost model: fit a zero-inflated
lognormal per (size, type) cell, collapse each cell to a single analytic mean
single-incident cost (bounded, band 10 = £100k-£500k), and let the fitted shape
set BOTH the band mix and the within-band value. The firm's own observed band is
discarded and replaced by its cell's mean. Attack TYPE thus moves the cost
distribution, not just the bridge. Contrast with the semiparametric baseline
("A"), which keeps each firm's empirical band and only models within-band value.

    Total = Sum_size N(size) * E[annual cost per firm | size]
    E[.. | size] = weighted mean over firms of
                     attacked * cellmean(size, type) * bridge(type, freq)

Thin (size,type) cells (few cost>0 obs or implausible sigma) fall back to the
well-populated per-SIZE shape. Impersonation reported at both range ends.
Deterministic point estimate (= the bootstrap mean).

Result: ~£1.0-1.4bn, roughly HALF the ~£2.2bn baseline. The gap is the tail:
the fitted lognormal shrinks the handful of empirically-observed top-band firms
(esp. 2 Micro firms it deems a ~5% fluctuation) down toward its smooth curve, so
this is best read as a tail-SHRUNK LOWER anchor, not a truer central estimate.
See docs/NARRATIVE.md.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import load_business_data, N_BY_SIZE, SIZE_LABELS, DISRUPTA_LABELS
from fitting import fit_lognormal, bounded_cell_mean
from bridge import bridge, IMPERSONATION_LOW, IMPERSONATION_HIGH

MIN_NONZERO = 10   # size x type cells are inherently thin; looser than per-(size,freq)
SIGMA_CAP = 3.5

OUT = os.path.join(os.path.dirname(__file__), "results", "estimate_size_type_lognormal.txt")


def main():
    df = load_business_data()
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    # per-size fallback shapes and their bounded cell means
    size_fit = {sz: fit_lognormal(df.loc[df["sizeb"] == sz, "band"].values) for sz in (1, 2, 3, 4)}
    size_mean = {sz: bounded_cell_mean(size_fit[sz]) for sz in (1, 2, 3, 4)}

    # per (size, type) bounded cell means, with thin-cell fallback to per-size
    types = sorted(int(t) for t in df.loc[df["attacked"], "disrupta"].dropna().unique())
    cellmean, cellsrc = {}, {}
    emit("Per (size, type) bounded cell-mean single-incident cost (£):")
    emit(f"  {'size':<7} {'type':<26} {'n_nz':>5} {'mu':>6} {'sig':>5} {'cellmean£':>11}  src")
    for sz in (1, 2, 3, 4):
        for t in types:
            sub = df[(df["sizeb"] == sz) & df["attacked"] & (df["disrupta"] == t)]
            fit = fit_lognormal(sub["band"].values)
            if fit is not None and fit.n_nonzero >= MIN_NONZERO and fit.sigma <= SIGMA_CAP:
                cellmean[(sz, t)] = bounded_cell_mean(fit)
                cellsrc[(sz, t)] = "cell"
                emit(f"  {SIZE_LABELS[sz]:<7} {DISRUPTA_LABELS.get(t, str(t)):<26} "
                     f"{fit.n_nonzero:>5} {fit.mu:>6.2f} {fit.sigma:>5.2f} "
                     f"{cellmean[(sz, t)]:>11,.0f}  cell")
            else:
                cellmean[(sz, t)] = size_mean[sz]
                cellsrc[(sz, t)] = "size-pooled"
    n_cell = sum(v == "cell" for v in cellsrc.values())
    emit(f"  -> {n_cell}/{len(cellsrc)} (size,type) cells use own shape; rest fall back to per-size")

    def national_total(imp):
        total, per_size = 0.0, {}
        for sz in (1, 2, 3, 4):
            sub = df[df["sizeb"] == sz]
            w = sub["weight"].values
            contrib = np.zeros(len(sub))
            for i, (_, row) in enumerate(sub.iterrows()):
                if not row["attacked"]:
                    continue
                t = row["disrupta"]
                cm = cellmean.get((sz, int(t)), size_mean[sz]) if not pd.isna(t) else size_mean[sz]
                contrib[i] = cm * bridge(t, row["freq"], impersonation=imp)
            epb = np.average(contrib, weights=w)
            c = N_BY_SIZE[sz] * epb
            per_size[sz] = (epb, c)
            total += c
        return total, per_size

    emit("")
    emit("=" * 70)
    emit("APPROACH B — national annual total (point estimate)")
    emit("=" * 70)
    for label, imp in [("Impersonation LOW  (3.11x)", IMPERSONATION_LOW),
                       ("Impersonation HIGH (6.25x)", IMPERSONATION_HIGH)]:
        total, per_size = national_total(imp)
        emit(f"\n{label}:  TOTAL = £{total / 1e9:,.2f}bn")
        for sz in (1, 2, 3, 4):
            epb, c = per_size[sz]
            emit(f"    {SIZE_LABELS[sz]:<7} £{c / 1e9:>6.2f}bn   "
                 f"(£{epb:,.0f}/business x {N_BY_SIZE[sz]:,})")

    emit("\nBaseline (semiparametric A, empirical band mix): ~£2.26bn.")
    emit("B is ~half — the fitted lognormal shrinks the observed top-band firms.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[written to {OUT}]")


if __name__ == "__main__":
    main()
