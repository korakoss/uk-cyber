"""Unified tiered estimate: condition both λ and T on the same type-tier space,
with max-bias correction applied to T.

Type tiers:
  expensive: Ransomware (1), DoS (3), Hacking (4) — rare, high E[T]
  mid:       Malware (2), Impersonation (5) + sparse (7, 8, 16)
  cheap:     Phishing (6) — universal, low cost

Conditioning space for both λ and T: (size_group, has_expensive[, has_mid]).
Max-bias correction: invert the order-statistic relationship using
  F(t) = log(1 + G(t)·(exp(λ) - 1)) / λ
to recover the per-draw T from the observed-max T.

Run: ``python src/estimation/unified_tiered_estimate.py``
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BAND_BOUNDS, SIZE_LABELS, load

PEAK_CUT = 3
N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}

EXPENSIVE = ["type1", "type3", "type4"]
MID = ["type2", "type5", "type7", "type8", "type16"]
CHEAP = ["type6"]


def band_to_gbp(band, tail_multiple=2.0):
    lo, hi = BAND_BOUNDS.get(band, (0, 0))
    if band == 1: return 0.0
    if hi == np.inf: return lo * tail_multiple
    return (lo + hi) / 2


def max_bias_correct_ET(peaks_in_cell, peak_bands, lam):
    """Given observed peak bands and the cell's λ, return corrected E[T].

    Uses F(t) = log(1 + G(t)·(exp(λ) - 1)) / λ to recover the per-draw
    CDF from the observed-max CDF, then computes E[T] under the per-draw PMF.
    """
    if lam < 0.01 or len(peaks_in_cell) < 3:
        # λ ≈ 0 means almost no peaks; correction is negligible
        return peaks_in_cell.map(band_to_gbp).mean()

    band_counts = peaks_in_cell.value_counts().sort_index()
    n = len(peaks_in_cell)

    # Build observed-max CDF G(t)
    cum = 0
    G = {}
    for b in peak_bands:
        cum += band_counts.get(b, 0)
        G[b] = cum / n

    # Invert to per-draw CDF F(t)
    exp_lam_m1 = np.exp(lam) - 1
    F = {}
    for b in peak_bands:
        F[b] = np.log(1 + G[b] * exp_lam_m1) / lam

    # Per-draw PMF → E[T]
    et = 0.0
    prev_F = 0.0
    for b in peak_bands:
        f = F[b] - prev_F
        et += f * band_to_gbp(b)
        prev_F = F[b]

    return et


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)

    # Prep
    att = df[df["attacked"]].copy()
    v = att[att["band"].notna() & att["freq"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["peak"] = v["band"] >= PEAK_CUT
    v["size_group"] = np.where(v["sizeb"] == 1, "Micro", "Rest")

    # Type tier flags
    for name, cols in [("expensive", EXPENSIVE), ("mid", MID), ("cheap", CHEAP)]:
        present = [c for c in cols if c in v.columns]
        v[f"has_{name}"] = (sum(v[c] == 1 for c in present) > 0).astype(int)

    # Also on full df for prevalence (not strictly needed but consistent)
    peak_bands = sorted(v[v["peak"]]["band"].unique())

    # Prevalence (always from full df, weighted)
    prev = {}
    for sz in SIZE_LABELS:
        sub = df[df["sizeb"] == sz]
        prev[sz] = (sub["attacked"].astype(int) * sub["weight"]).sum() / sub["weight"].sum()

    # ── Sparsity check ──
    print("=" * 74)
    print("SPARSITY CHECK — cell sizes for (size, has_expensive, has_mid)")
    print("=" * 74)

    for sz, label in SIZE_LABELS.items():
        for he in [0, 1]:
            for hm in [0, 1]:
                cell = v[(v["sizeb"] == sz) & (v["has_expensive"] == he) & (v["has_mid"] == hm)]
                peaks = cell[cell["peak"]]
                tag = f"{label}, exp={he}, mid={hm}"
                print(f"  {tag:<35s} n={len(cell):>4d}  peaks={len(peaks):>3d}")

    # ── Define conditioning spaces to compare ──
    SPACES = {
        "A: size_group only (baseline)": lambda row: (row["size_group"],),
        "B: size_group × has_expensive": lambda row: (row["size_group"], row["has_expensive"]),
        "C: size_group × has_expensive × has_mid": lambda row: (row["size_group"], row["has_expensive"], row["has_mid"]),
    }

    for space_label, key_fn in SPACES.items():
        print(f"\n{'=' * 74}")
        print(f"ESTIMATE — {space_label}")
        print("=" * 74)

        # Step 1: compute λ per (size, n_types, tier_key) — but actually,
        # we want λ conditioned on (size, n_types) as before for the Poisson
        # inversion, but T conditioned on (size_group, tier).
        # Wait — the user wants λ AND T on the same conditioning space.
        # So λ should be conditioned on (size, tier_key) not (size, n_types).

        # Build T per conditioning cell, with max-bias correction
        # For the correction, we need a representative λ per T cell.
        # Use the cell-average λ (from P(peak) in that cell).

        # Step 1: E[T] per conditioning cell (with max-bias correction)
        ET = {}
        cell_lambda = {}

        # Get unique keys
        keys = set()
        for _, row in v.iterrows():
            keys.add(key_fn(row))

        for key in sorted(keys):
            mask = v.apply(lambda row: key_fn(row) == key, axis=1)
            cell = v[mask]
            peaks = cell[cell["peak"]]

            # λ for this conditioning cell (for max-bias correction)
            p_peak = cell["peak"].mean()
            lam = -np.log(1 - p_peak) if p_peak < 1 else 5.0
            cell_lambda[key] = lam

            if len(peaks) >= 3:
                et_obs = peaks["band"].map(band_to_gbp).mean()
                et_corr = max_bias_correct_ET(peaks["band"], peak_bands, lam)
                ET[key] = {"observed": et_obs, "corrected": et_corr,
                           "n_peaks": len(peaks), "lam": lam}
            else:
                # Fall back to parent (size_group only)
                grp = key[0]
                fallback_peaks = v[(v["peak"]) & (v["size_group"] == grp)]
                p_fb = v[v["size_group"] == grp]["peak"].mean()
                lam_fb = -np.log(1 - p_fb) if p_fb < 1 else 5.0
                et_obs = fallback_peaks["band"].map(band_to_gbp).mean()
                et_corr = max_bias_correct_ET(fallback_peaks["band"], peak_bands, lam_fb)
                ET[key] = {"observed": et_obs, "corrected": et_corr,
                           "n_peaks": len(fallback_peaks), "lam": lam_fb,
                           "fallback": True}

        # Print E[T] table
        print(f"\n  E[T] by conditioning cell:")
        print(f"    {'key':<40s} {'n_peaks':>7s} {'λ':>6s} {'E[T] obs':>10s} {'E[T] corr':>10s} {'bias':>7s}")
        for key in sorted(ET.keys()):
            info = ET[key]
            fb = " (fb)" if info.get("fallback") else ""
            print(f"    {str(key):<40s} {info['n_peaks']:>7d} {info['lam']:>6.3f} "
                  f"£{info['observed']:>9,.0f} £{info['corrected']:>9,.0f} "
                  f"{info['observed']/info['corrected'] - 1:>+6.1%}{fb}")

        # Step 2: national total under both observed and corrected T
        for t_variant in ["observed", "corrected"]:
            total = 0
            for sz, sz_label in SIZE_LABELS.items():
                grp = "Micro" if sz == 1 else "Rest"

                # λ cells: same conditioning space, within this size band
                cell_cost = 0
                cell_wt = 0

                # Iterate over unique tier keys within this size
                sz_data = v[v["sizeb"] == sz]
                tier_keys_in_sz = set()
                for _, row in sz_data.iterrows():
                    tier_keys_in_sz.add(key_fn(row))

                for tk in sorted(tier_keys_in_sz):
                    mask = sz_data.apply(lambda row, k=tk: key_fn(row) == k, axis=1)
                    cell = sz_data[mask]
                    if len(cell) < 5:
                        continue

                    p_peak = cell["peak"].mean()
                    lam = -np.log(1 - p_peak) if p_peak < 1 else 5.0

                    # E[T] for this cell's conditioning key
                    et = ET[tk][t_variant]

                    cell_cost += len(cell) * lam * et
                    cell_wt += len(cell)

                e_cost = cell_cost / cell_wt if cell_wt > 0 else 0
                nat = N_BY_SIZE[sz] * prev[sz] * e_cost
                total += nat

            label_suffix = "(max-bias corrected)" if t_variant == "corrected" else "(uncorrected)"
            print(f"\n  National total {label_suffix}:")
            # Recompute for printing per-size
            total = 0
            for sz, sz_label in SIZE_LABELS.items():
                grp = "Micro" if sz == 1 else "Rest"
                sz_data = v[v["sizeb"] == sz]
                tier_keys_in_sz = set()
                for _, row in sz_data.iterrows():
                    tier_keys_in_sz.add(key_fn(row))

                cell_cost = 0
                cell_wt = 0
                for tk in sorted(tier_keys_in_sz):
                    mask = sz_data.apply(lambda row, k=tk: key_fn(row) == k, axis=1)
                    cell = sz_data[mask]
                    if len(cell) < 5:
                        continue
                    p_peak = cell["peak"].mean()
                    lam = -np.log(1 - p_peak) if p_peak < 1 else 5.0
                    et = ET[tk][t_variant]
                    cell_cost += len(cell) * lam * et
                    cell_wt += len(cell)

                e_cost = cell_cost / cell_wt if cell_wt > 0 else 0
                nat = N_BY_SIZE[sz] * prev[sz] * e_cost
                total += nat
                print(f"    {sz_label:<10s} £{nat/1e9:.3f}bn")
            print(f"    {'TOTAL':<10s} £{total/1e9:.3f}bn")

    # ── Hybrid: n_types for λ, tiers for T ──
    print(f"\n{'=' * 74}")
    print("HYBRID — n_types for λ, tier space for T (with max-bias correction)")
    print("=" * 74)

    # Build T per (size_group, has_expensive) with correction — reuse from space B
    tier_key = lambda row: (row["size_group"], row["has_expensive"])
    ET_tier = {}
    for grp in ["Micro", "Rest"]:
        for he in [0, 1]:
            cell = v[(v["size_group"] == grp) & (v["has_expensive"] == he)]
            peaks = cell[cell["peak"]]
            p_peak = cell["peak"].mean()
            lam = -np.log(1 - p_peak) if p_peak < 1 else 5.0
            if len(peaks) >= 3:
                et_obs = peaks["band"].map(band_to_gbp).mean()
                et_corr = max_bias_correct_ET(peaks["band"], peak_bands, lam)
            else:
                fb_peaks = v[(v["peak"]) & (v["size_group"] == grp)]
                p_fb = v[v["size_group"] == grp]["peak"].mean()
                lam_fb = -np.log(1 - p_fb) if p_fb < 1 else 5.0
                et_obs = fb_peaks["band"].map(band_to_gbp).mean()
                et_corr = max_bias_correct_ET(fb_peaks["band"], peak_bands, lam_fb)
            ET_tier[(grp, he)] = {"observed": et_obs, "corrected": et_corr}

    print("\n  E[T] by (size_group, has_expensive):")
    for key in sorted(ET_tier.keys()):
        info = ET_tier[key]
        print(f"    {str(key):<30s} obs £{info['observed']:>9,.0f}  corr £{info['corrected']:>9,.0f}")

    for t_variant in ["observed", "corrected"]:
        label_suffix = "(max-bias corrected)" if t_variant == "corrected" else "(uncorrected)"
        print(f"\n  National total {label_suffix}:")
        total = 0
        for sz, sz_label in SIZE_LABELS.items():
            grp = "Micro" if sz == 1 else "Rest"
            sz_data = v[v["sizeb"] == sz]

            cell_cost = 0
            cell_wt = 0
            for nt in sorted(v["n_types"].unique()):
                for he in [0, 1]:
                    cell = sz_data[(sz_data["n_types"] == nt) & (sz_data["has_expensive"] == he)]
                    if len(cell) < 5:
                        continue
                    p_peak = cell["peak"].mean()
                    lam = -np.log(1 - p_peak) if p_peak < 1 else 5.0
                    et = ET_tier[(grp, he)][t_variant]
                    cell_cost += len(cell) * lam * et
                    cell_wt += len(cell)

            e_cost = cell_cost / cell_wt if cell_wt > 0 else 0
            nat = N_BY_SIZE[sz] * prev[sz] * e_cost
            total += nat
            print(f"    {sz_label:<10s} £{nat/1e9:.3f}bn")
        print(f"    {'TOTAL':<10s} £{total/1e9:.3f}bn")

    # Also run the original n_types-only λ with pooled T for reference
    print(f"\n  Reference: n_types λ + pooled T (original methodology):")
    for t_variant in ["observed", "corrected"]:
        ET_pooled = {}
        for grp in ["Micro", "Rest"]:
            peaks = v[(v["peak"]) & (v["size_group"] == grp)]
            p_all = v[v["size_group"] == grp]["peak"].mean()
            lam_all = -np.log(1 - p_all) if p_all < 1 else 5.0
            ET_pooled[grp] = {
                "observed": peaks["band"].map(band_to_gbp).mean(),
                "corrected": max_bias_correct_ET(peaks["band"], peak_bands, lam_all),
            }

        total = 0
        for sz, sz_label in SIZE_LABELS.items():
            grp = "Micro" if sz == 1 else "Rest"
            sz_data = v[v["sizeb"] == sz]
            cell_cost = 0
            cell_wt = 0
            for nt in sorted(v["n_types"].unique()):
                cell = sz_data[sz_data["n_types"] == nt]
                if len(cell) < 5:
                    continue
                p_peak = cell["peak"].mean()
                lam = -np.log(1 - p_peak) if p_peak < 1 else 5.0
                et = ET_pooled[grp][t_variant]
                cell_cost += len(cell) * lam * et
                cell_wt += len(cell)
            e_cost = cell_cost / cell_wt if cell_wt > 0 else 0
            nat = N_BY_SIZE[sz] * prev[sz] * e_cost
            total += nat
        label_suffix = "corrected" if t_variant == "corrected" else "uncorrected"
        print(f"    {label_suffix}: £{total/1e9:.3f}bn")

    # ── Summary ──
    print(f"\n{'=' * 74}")
    print("SUMMARY")
    print("=" * 74)
    print("  Conditioning for λ / T → total (uncorr) → total (corrected)")
    print("  A: tier-only λ, pooled T                   see above")
    print("  B: tier-only λ, tier T                     see above")
    print("  C: tier-only λ, tier×mid T                 see above")
    print("  Hybrid: n_types λ, tier T                  see above")
    print("  Reference: n_types λ, pooled T             see above")


if __name__ == "__main__":
    main()
