"""Correct E[T] for max-order-statistic bias.

We observe the max peak band per firm, not a random draw from T.
Under Poisson(λ) peaks, the observed-max CDF G(t) relates to the
true per-draw CDF F(t) via:

    G(t) = (exp(λ·F(t)) - 1) / (exp(λ) - 1)

Inverting:

    F(t) = log(1 + G(t)·(exp(λ) - 1)) / λ

We apply this per size_group (Micro / Rest), using the group-average λ
among peak firms to match how T is pooled.

Run: ``python src/estimation/max_bias_correction.py``
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BAND_BOUNDS, SIZE_LABELS, load

PEAK_CUT = 3
N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}


def band_to_gbp(band, tail_multiple=2.0):
    lo, hi = BAND_BOUNDS.get(band, (0, 0))
    if band == 1: return 0.0
    if hi == np.inf: return lo * tail_multiple
    return (lo + hi) / 2


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)

    attacked = df[df["attacked"]].copy()
    v = attacked[attacked["band"].notna() & attacked["freq"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["peak"] = v["band"] >= PEAK_CUT
    v["size_group"] = np.where(v["sizeb"] == 1, "Micro", "Rest")
    v["cost"] = v["band"].map(band_to_gbp)

    # Peak bands present in data
    peak_bands = sorted(v[v["peak"]]["band"].unique())
    print(f"Peak bands observed: {peak_bands}")
    print(f"Band midpoints: { {b: band_to_gbp(b) for b in peak_bands} }")

    # --- 1. Compute group-average λ ---
    print("\n" + "=" * 74)
    print("1. GROUP-AVERAGE λ (among all attacked firms with valid data)")
    print("=" * 74)

    group_lambda = {}
    for grp in ["Micro", "Rest"]:
        grp_firms = v[v["size_group"] == grp]
        p_peak = grp_firms["peak"].mean()
        lam = -np.log(1 - p_peak) if p_peak < 1 else 5.0
        group_lambda[grp] = lam
        print(f"  {grp}: P(peak)={p_peak:.4f}, λ={lam:.4f}")

    # Also compute cell-level lambdas and their weighted average
    print("\n  Cell-level λ (for reference):")
    for grp in ["Micro", "Rest"]:
        sz_list = [1] if grp == "Micro" else [2, 3, 4]
        lams, weights = [], []
        for sz in sz_list:
            for nt in sorted(v["n_types"].unique()):
                cell = v[(v["sizeb"] == sz) & (v["n_types"] == nt)]
                if len(cell) < 5:
                    continue
                p = cell["peak"].mean()
                lam = -np.log(1 - p) if p < 1 else 5.0
                lams.append(lam)
                weights.append(len(cell))
        avg_lam = np.average(lams, weights=weights)
        print(f"  {grp}: weighted-avg cell λ = {avg_lam:.4f}")

    # --- 2. Observed max CDF G(t) per group ---
    print("\n" + "=" * 74)
    print("2. OBSERVED MAX CDF G(t) AND CORRECTED PER-DRAW CDF F(t)")
    print("=" * 74)

    for grp in ["Micro", "Rest"]:
        lam = group_lambda[grp]
        peaks = v[(v["peak"]) & (v["size_group"] == grp)]
        n_peaks = len(peaks)

        print(f"\n  --- {grp} (n={n_peaks}, λ={lam:.4f}) ---")
        print(f"  {'band':>5s} {'£':>10s} {'n':>6s} {'g(t)':>8s} {'G(t)':>8s} {'F(t)':>8s} {'f(t)':>8s}")

        band_counts = peaks["band"].value_counts().sort_index()
        cum = 0
        G_vals = {}
        for b in peak_bands:
            cum += band_counts.get(b, 0)
            G_vals[b] = cum / n_peaks

        # Invert: F(t) = log(1 + G(t)*(exp(λ) - 1)) / λ
        F_vals = {}
        for b in peak_bands:
            G = G_vals[b]
            F_vals[b] = np.log(1 + G * (np.exp(lam) - 1)) / lam

        # Print table
        prev_G = 0
        prev_F = 0
        for b in peak_bands:
            g = G_vals[b] - prev_G  # PMF of observed max
            f = F_vals[b] - prev_F  # PMF of true per-draw
            cost = band_to_gbp(b)
            print(f"  {b:>5d} {cost:>10,.0f} {band_counts.get(b, 0):>6d} "
                  f"{g:>8.4f} {G_vals[b]:>8.4f} {F_vals[b]:>8.4f} {f:>8.4f}")
            prev_G = G_vals[b]
            prev_F = F_vals[b]

        # E[T] under observed max vs corrected per-draw
        ET_observed = peaks["cost"].mean()

        # E[T] corrected: use the per-draw PMF
        ET_corrected = 0
        prev_F = 0
        for b in peak_bands:
            f = F_vals[b] - prev_F
            ET_corrected += f * band_to_gbp(b)
            prev_F = F_vals[b]

        print(f"\n  E[T] observed (max):     £{ET_observed:>10,.0f}")
        print(f"  E[T] corrected (draw):   £{ET_corrected:>10,.0f}")
        print(f"  Ratio observed/corrected: {ET_observed/ET_corrected:.3f}x")
        print(f"  Overcount:                {ET_observed/ET_corrected - 1:+.1%}")

    # --- 3. National total with corrected E[T] ---
    print("\n" + "=" * 74)
    print("3. NATIONAL TOTAL: OBSERVED vs CORRECTED E[T]")
    print("=" * 74)

    # Get prevalence
    prev = {}
    for sz in SIZE_LABELS:
        sub = df[df["sizeb"] == sz]
        prev[sz] = (sub["attacked"].astype(int) * sub["weight"]).sum() / sub["weight"].sum()

    # Compute E[T] both ways per group
    ET = {}
    for grp in ["Micro", "Rest"]:
        lam = group_lambda[grp]
        peaks = v[(v["peak"]) & (v["size_group"] == grp)]

        ET[(grp, "observed")] = peaks["cost"].mean()

        # Corrected
        band_counts = peaks["band"].value_counts().sort_index()
        n_peaks = len(peaks)
        cum = 0
        G_vals = {}
        for b in peak_bands:
            cum += band_counts.get(b, 0)
            G_vals[b] = cum / n_peaks
        F_vals = {}
        for b in peak_bands:
            F_vals[b] = np.log(1 + G_vals[b] * (np.exp(lam) - 1)) / lam

        et_corr = 0
        prev_F = 0
        for b in peak_bands:
            f = F_vals[b] - prev_F
            et_corr += f * band_to_gbp(b)
            prev_F = F_vals[b]
        ET[(grp, "corrected")] = et_corr

    print(f"\n  E[T] lookup:")
    for grp in ["Micro", "Rest"]:
        print(f"    {grp}: observed £{ET[(grp, 'observed')]:,.0f} → corrected £{ET[(grp, 'corrected')]:,.0f} "
              f"({ET[(grp, 'observed')]/ET[(grp, 'corrected')] - 1:+.1%})")

    # Run the estimate both ways
    for label, t_key in [("OBSERVED (current)", "observed"), ("CORRECTED (max-bias removed)", "corrected")]:
        print(f"\n  --- {label} ---")
        total = 0
        for sz, sz_label in SIZE_LABELS.items():
            grp = "Micro" if sz == 1 else "Rest"
            et = ET[(grp, t_key)]

            cell_cost = 0
            cell_n = 0
            for nt in sorted(v["n_types"].unique()):
                cell = v[(v["sizeb"] == sz) & (v["n_types"] == nt)]
                if len(cell) < 5:
                    continue
                p_peak = cell["peak"].mean()
                lam = -np.log(1 - p_peak) if p_peak < 1 else 5.0
                cell_cost += len(cell) * lam * et
                cell_n += len(cell)

            e_cost = cell_cost / cell_n if cell_n > 0 else 0
            nat = N_BY_SIZE[sz] * prev[sz] * e_cost
            total += nat
            print(f"    {sz_label:<10s} £{nat/1e9:.3f}bn")
        print(f"    {'TOTAL':<10s} £{total/1e9:.3f}bn")

    # --- 4. Sensitivity: correction at different λ levels ---
    print("\n" + "=" * 74)
    print("4. SENSITIVITY: CORRECTION FACTOR vs λ")
    print("=" * 74)
    print("  (Using Micro's observed band distribution as reference)")

    peaks_micro = v[(v["peak"]) & (v["size_group"] == "Micro")]
    band_counts = peaks_micro["band"].value_counts().sort_index()
    n_peaks = len(peaks_micro)

    # Observed PMF (of the max)
    obs_pmf = {}
    for b in peak_bands:
        obs_pmf[b] = band_counts.get(b, 0) / n_peaks

    ET_obs = sum(obs_pmf[b] * band_to_gbp(b) for b in peak_bands)

    print(f"\n  {'λ':>6s} {'E[T] obs':>10s} {'E[T] corr':>10s} {'ratio':>8s} {'overcount':>10s}")
    for lam in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        cum = 0
        G_vals = {}
        for b in peak_bands:
            cum += obs_pmf[b]
            G_vals[b] = cum

        F_vals = {}
        for b in peak_bands:
            F_vals[b] = np.log(1 + G_vals[b] * (np.exp(lam) - 1)) / lam

        et_corr = 0
        prev_F = 0
        for b in peak_bands:
            f = F_vals[b] - prev_F
            et_corr += f * band_to_gbp(b)
            prev_F = F_vals[b]

        ratio = ET_obs / et_corr if et_corr > 0 else float('inf')
        print(f"  {lam:>6.2f} {ET_obs:>10,.0f} {et_corr:>10,.0f} {ratio:>8.3f} {ratio-1:>+10.1%}")


if __name__ == "__main__":
    main()
