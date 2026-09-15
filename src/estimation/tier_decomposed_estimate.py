"""Tier-decomposed estimate: independent Poisson per type tier.

Assign each peak firm's observed band to a tier via `disrupta` (which type
was the most disruptive). Then estimate λ_tier and E[T_tier] separately,
apply max-bias correction per tier, and sum via Wald:
    E[total] = Σ_tier λ_tier · E[T_tier]

Check: Σ λ_tier vs λ_total to see how much cross-tier peak mass is missed.

Run: ``python src/estimation/tier_decomposed_estimate.py``
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BAND_BOUNDS, SIZE_LABELS, load

PEAK_CUT = 3
N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}

# disrupta codes → tiers
TIER_MAP = {
    1: "expensive",   # Ransomware
    3: "expensive",   # DoS
    4: "expensive",   # Hacking
    2: "mid",         # Malware
    5: "mid",         # Impersonation
    7: "mid",         # Unauth access (staff)
    8: "mid",         # Unauth access (other)
    11: "mid",        # Website/takeover
    12: "mid",        # Eavesdropping
    9: "mid",         # Other
    6: "cheap",       # Phishing
}
TIER_NAMES = ["expensive", "mid", "cheap"]


def band_to_gbp(band, tail_multiple=2.0):
    lo, hi = BAND_BOUNDS.get(band, (0, 0))
    if band == 1: return 0.0
    if hi == np.inf: return lo * tail_multiple
    return (lo + hi) / 2


def max_bias_correct_ET(peak_bands_series, all_peak_bands, lam):
    if lam < 0.01 or len(peak_bands_series) < 3:
        return peak_bands_series.map(band_to_gbp).mean()

    band_counts = peak_bands_series.value_counts().sort_index()
    n = len(peak_bands_series)

    cum = 0
    G = {}
    for b in all_peak_bands:
        cum += band_counts.get(b, 0)
        G[b] = cum / n

    exp_lam_m1 = np.exp(lam) - 1
    F = {}
    for b in all_peak_bands:
        F[b] = np.log(1 + G[b] * exp_lam_m1) / lam

    et = 0.0
    prev_F = 0.0
    for b in all_peak_bands:
        f = F[b] - prev_F
        et += f * band_to_gbp(b)
        prev_F = F[b]
    return et


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)

    att = df[df["attacked"]].copy()
    v = att[att["band"].notna() & att["freq"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["peak"] = v["band"] >= PEAK_CUT
    v["size_group"] = np.where(v["sizeb"] == 1, "Micro", "Rest")

    # Assign tier via disrupta
    v["tier"] = v["disrupta"].map(TIER_MAP)

    peak_bands = sorted(v[v["peak"]]["band"].unique())

    # Prevalence
    prev = {}
    for sz in SIZE_LABELS:
        sub = df[df["sizeb"] == sz]
        prev[sz] = (sub["attacked"].astype(int) * sub["weight"]).sum() / sub["weight"].sum()

    # ── 1. Coverage check: how many peak firms have a valid disrupta tier? ──
    peaks = v[v["peak"]]
    print("=" * 74)
    print("1. DISRUPTA COVERAGE AMONG PEAK FIRMS")
    print("=" * 74)
    n_peak = len(peaks)
    n_with_tier = peaks["tier"].notna().sum()
    print(f"  Peak firms: {n_peak}")
    print(f"  With valid disrupta → tier: {n_with_tier} ({n_with_tier/n_peak:.1%})")
    print(f"  Missing disrupta: {n_peak - n_with_tier}")
    print(f"\n  Tier breakdown among peak firms:")
    for tier in TIER_NAMES:
        n = (peaks["tier"] == tier).sum()
        print(f"    {tier:<12s} n={n:>4d} ({n/n_peak:.1%})")

    # ── 2. λ_tier and E[T_tier] by size_group ──
    print(f"\n{'=' * 74}")
    print("2. PER-TIER λ AND E[T] BY SIZE GROUP")
    print("=" * 74)

    # λ_tier = -log(1 - P(peak AND disrupta ∈ tier | attacked))
    # P(peak AND disrupta ∈ tier) = fraction of attacked firms whose worst
    # incident was a peak from this tier
    results = {}
    for grp in ["Micro", "Rest"]:
        grp_data = v[v["size_group"] == grp]
        n_grp = len(grp_data)

        # Total λ for reference
        p_peak_total = grp_data["peak"].mean()
        lam_total = -np.log(1 - p_peak_total) if p_peak_total < 1 else 5.0

        print(f"\n  --- {grp} (n={n_grp}) ---")
        print(f"  Total: P(peak)={p_peak_total:.4f}, λ_total={lam_total:.4f}")
        print(f"\n  {'tier':<12s} {'P(peak∧tier)':>12s} {'λ_tier':>8s} {'n_peaks':>8s} "
              f"{'E[T] obs':>10s} {'E[T] corr':>10s} {'bias':>7s}")

        lam_sum = 0
        for tier in TIER_NAMES:
            tier_peaks = grp_data[(grp_data["peak"]) & (grp_data["tier"] == tier)]
            p = len(tier_peaks) / n_grp
            lam = -np.log(1 - p) if p < 1 else 5.0
            lam_sum += lam

            if len(tier_peaks) >= 3:
                et_obs = tier_peaks["band"].map(band_to_gbp).mean()
                et_corr = max_bias_correct_ET(tier_peaks["band"], peak_bands, lam)
                bias = et_obs / et_corr - 1 if et_corr > 0 else 0
            else:
                et_obs = et_corr = 0
                bias = 0

            results[(grp, tier)] = {"lam": lam, "et_obs": et_obs, "et_corr": et_corr,
                                    "n_peaks": len(tier_peaks)}

            print(f"  {tier:<12s} {p:>12.4f} {lam:>8.4f} {len(tier_peaks):>8d} "
                  f"£{et_obs:>9,.0f} £{et_corr:>9,.0f} {bias:>+6.1%}")

        print(f"\n  Σ λ_tier = {lam_sum:.4f} vs λ_total = {lam_total:.4f} "
              f"(gap = {lam_total - lam_sum:.4f}, {(lam_total - lam_sum)/lam_total:.1%} of total)")

    # ── 3. National total via Σ λ_tier · E[T_tier] ──
    print(f"\n{'=' * 74}")
    print("3. NATIONAL TOTAL — TIER-DECOMPOSED")
    print("=" * 74)

    for t_variant, label in [("et_obs", "uncorrected"), ("et_corr", "max-bias corrected")]:
        print(f"\n  --- {label} ---")
        total = 0
        for sz, sz_label in SIZE_LABELS.items():
            grp = "Micro" if sz == 1 else "Rest"
            sz_data = v[v["sizeb"] == sz]

            # For each n_types cell, compute Σ_tier λ_tier(cell) · E[T_tier(grp)]
            cell_cost = 0
            cell_wt = 0
            for nt in sorted(v["n_types"].unique()):
                cell = sz_data[sz_data["n_types"] == nt]
                if len(cell) < 5:
                    continue
                n_cell = len(cell)

                tier_cost = 0
                for tier in TIER_NAMES:
                    tier_peaks_in_cell = cell[(cell["peak"]) & (cell["tier"] == tier)]
                    p = len(tier_peaks_in_cell) / n_cell
                    lam = -np.log(1 - p) if p < 1 else 5.0
                    et = results[(grp, tier)][t_variant]
                    tier_cost += lam * et

                cell_cost += n_cell * tier_cost
                cell_wt += n_cell

            e_cost = cell_cost / cell_wt if cell_wt > 0 else 0
            nat = N_BY_SIZE[sz] * prev[sz] * e_cost
            total += nat
            print(f"    {sz_label:<10s} £{nat/1e9:.3f}bn")
        print(f"    {'TOTAL':<10s} £{total/1e9:.3f}bn")

    # ── 4. Variant: tier-only λ (no n_types) ──
    print(f"\n{'=' * 74}")
    print("4. SIMPLER VARIANT — tier-only λ (no n_types conditioning)")
    print("=" * 74)

    for t_variant, label in [("et_obs", "uncorrected"), ("et_corr", "max-bias corrected")]:
        print(f"\n  --- {label} ---")
        total = 0
        for sz, sz_label in SIZE_LABELS.items():
            grp = "Micro" if sz == 1 else "Rest"

            e_cost = 0
            for tier in TIER_NAMES:
                e_cost += results[(grp, tier)]["lam"] * results[(grp, tier)][t_variant]

            nat = N_BY_SIZE[sz] * prev[sz] * e_cost
            total += nat
            print(f"    {sz_label:<10s} £{nat/1e9:.3f}bn")
        print(f"    {'TOTAL':<10s} £{total/1e9:.3f}bn")

    # ── 5. Per-tier contribution breakdown ──
    print(f"\n{'=' * 74}")
    print("5. PER-TIER CONTRIBUTION (corrected, tier-only λ)")
    print("=" * 74)
    for grp in ["Micro", "Rest"]:
        print(f"\n  {grp}:")
        grp_total = sum(results[(grp, t)]["lam"] * results[(grp, t)]["et_corr"]
                        for t in TIER_NAMES)
        for tier in TIER_NAMES:
            contrib = results[(grp, tier)]["lam"] * results[(grp, tier)]["et_corr"]
            share = contrib / grp_total if grp_total > 0 else 0
            print(f"    {tier:<12s} λ={results[(grp, tier)]['lam']:.4f} × "
                  f"E[T]=£{results[(grp, tier)]['et_corr']:>8,.0f} = "
                  f"£{contrib:>8,.0f}/firm  ({share:.1%})")


if __name__ == "__main__":
    main()
