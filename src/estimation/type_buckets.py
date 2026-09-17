"""Cluster types into cost buckets and test as conditioning variables.

Buckets:
  - expensive: Ransomware (1), DoS (3), Hacking (4)  — rare, high E[T]
  - mid: Impersonation (5), Malware (2) — moderate prevalence/cost
  - cheap: Phishing (6) — universal, low cost

For the remaining sparse types (7, 8, 15, 16), assign to mid bucket
(they're rare, moderate-enrichment, no reliable E[T]).

Run: ``python src/estimation/type_buckets.py``
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BAND_BOUNDS, GENUINE_TYPE_COLS, SIZE_LABELS, load

N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}

EXPENSIVE = ["type1", "type3", "type4"]  # Ransomware, DoS, Hacking
MID = ["type2", "type5", "type7", "type8", "type16"]  # Malware, Impersonation, + sparse
CHEAP = ["type6"]  # Phishing

BUCKETS = {"expensive": EXPENSIVE, "mid": MID, "cheap": CHEAP}


def mid_gbp(b):
    lo, hi = BAND_BOUNDS.get(b, (0, 0))
    if b == 1: return 0
    if hi == np.inf: return lo * 2
    return (lo + hi) / 2


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)
    att = df[df["attacked"]].copy()
    v = att[att["band"].notna() & att["freq"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["peak"] = v["band"] >= 3
    v["size_group"] = np.where(v["sizeb"] == 1, "Micro", "Rest")
    v["cost"] = v["band"].map(mid_gbp)

    type_cols = [c for c in GENUINE_TYPE_COLS if c in v.columns]

    # Build bucket features
    for bname, cols in BUCKETS.items():
        present = [c for c in cols if c in v.columns]
        v[f"has_{bname}"] = (sum(v[c] == 1 for c in present) > 0).astype(int)
        v[f"n_{bname}"] = sum(v[c] == 1 for c in present).astype(int)

    peaks = v[v["peak"]]

    # Prevalence
    prev = {}
    for sz in SIZE_LABELS:
        sub = df[df["sizeb"] == sz]
        prev[sz] = (sub["attacked"].astype(int) * sub["weight"]).sum() / sub["weight"].sum()

    # --- 1. E[T] by bucket features ---
    print("=" * 74)
    print("1. E[T] BY BUCKET FEATURES")
    print("=" * 74)

    for grp in ["Micro", "Rest"]:
        print(f"\n  {grp}:")
        grp_peaks = peaks[peaks["size_group"] == grp]

        # By has_expensive
        for he in [0, 1]:
            sub = grp_peaks[grp_peaks["has_expensive"] == he]
            if len(sub) >= 3:
                print(f"    has_expensive={he}: n={len(sub):>4d}  E[T]=£{sub['cost'].mean():>10,.0f}")

        # By (has_expensive, has_mid)
        print()
        for he in [0, 1]:
            for hm in [0, 1]:
                sub = grp_peaks[(grp_peaks["has_expensive"] == he) & (grp_peaks["has_mid"] == hm)]
                if len(sub) >= 5:
                    print(f"    expensive={he} mid={hm}: n={len(sub):>4d}  E[T]=£{sub['cost'].mean():>10,.0f}")

    # --- 2. P(peak) by bucket features ---
    print(f"\n{'='*74}")
    print("2. P(peak) BY BUCKET FEATURES")
    print("=" * 74)

    for grp_label, sz_list in [("Micro", [1]), ("Rest", [2, 3, 4])]:
        print(f"\n  {grp_label}:")
        sub = v[v["sizeb"].isin(sz_list)]
        for he in [0, 1]:
            for hm in [0, 1]:
                cell = sub[(sub["has_expensive"] == he) & (sub["has_mid"] == hm)]
                if len(cell) >= 10:
                    pp = cell["peak"].mean()
                    print(f"    expensive={he} mid={hm}: n={len(cell):>4d}  P(peak)={pp:.3f}")

    # --- 3. Compute national total under different T-conditioning ---
    print(f"\n{'='*74}")
    print("3. NATIONAL TOTAL UNDER DIFFERENT T-CONDITIONING")
    print("=" * 74)

    # Approach A: current (T by size_group only)
    ET_pooled = {}
    for grp in ["Micro", "Rest"]:
        ET_pooled[grp] = peaks[peaks["size_group"] == grp]["cost"].mean()

    # Approach B: T by (size_group, has_expensive)
    ET_he = {}
    for grp in ["Micro", "Rest"]:
        for he in [0, 1]:
            sub = peaks[(peaks["size_group"] == grp) & (peaks["has_expensive"] == he)]
            ET_he[(grp, he)] = sub["cost"].mean() if len(sub) >= 3 else ET_pooled[grp]

    # Approach C: T by (size_group, has_expensive, has_mid)
    ET_hem = {}
    for grp in ["Micro", "Rest"]:
        for he in [0, 1]:
            for hm in [0, 1]:
                sub = peaks[(peaks["size_group"] == grp) &
                            (peaks["has_expensive"] == he) &
                            (peaks["has_mid"] == hm)]
                ET_hem[(grp, he, hm)] = sub["cost"].mean() if len(sub) >= 5 else ET_he.get((grp, he), ET_pooled[grp])

    approaches = [
        ("A: T by size_group only",
         lambda row: ET_pooled[row["size_group"]]),
        ("B: T by (size_group, has_expensive)",
         lambda row: ET_he[(row["size_group"], row["has_expensive"])]),
        ("C: T by (size_group, has_expensive, has_mid)",
         lambda row: ET_hem.get((row["size_group"], row["has_expensive"], row["has_mid"]),
                                ET_he.get((row["size_group"], row["has_expensive"]), ET_pooled[row["size_group"]]))),
    ]

    for label, et_fn in approaches:
        print(f"\n  --- {label} ---")
        total = 0
        for sz, sz_label in SIZE_LABELS.items():
            grp = "Micro" if sz == 1 else "Rest"

            # Get unique conditioning cells
            cell_cost = 0
            cell_n = 0

            for nt in sorted(v["n_types"].unique()):
                for he in [0, 1]:
                    for hm in [0, 1]:
                        cell = v[(v["sizeb"] == sz) & (v["n_types"] == nt) &
                                 (v["has_expensive"] == he) & (v["has_mid"] == hm)]
                        if len(cell) < 5:
                            continue
                        p_peak = cell["peak"].mean()
                        lam = -np.log(1 - p_peak) if p_peak < 1 else 5.0
                        # Get E[T] for this cell's bucket profile
                        row = {"size_group": grp, "has_expensive": he, "has_mid": hm}
                        et = et_fn(row)
                        cell_cost += len(cell) * lam * et
                        cell_n += len(cell)

            e_cost = cell_cost / cell_n if cell_n > 0 else 0
            nat = N_BY_SIZE[sz] * prev[sz] * e_cost
            total += nat
            print(f"    {sz_label:<10s} £{nat/1e9:.3f}bn")
        print(f"    {'TOTAL':<10s} £{total/1e9:.3f}bn")

    # --- 4. Show the E[T] tables used ---
    print(f"\n{'='*74}")
    print("4. E[T] LOOKUP TABLES")
    print("=" * 74)
    print("\n  Approach A:")
    for grp in ["Micro", "Rest"]:
        print(f"    {grp}: £{ET_pooled[grp]:,.0f}")
    print("\n  Approach B:")
    for grp in ["Micro", "Rest"]:
        for he in [0, 1]:
            print(f"    {grp}, expensive={he}: £{ET_he[(grp, he)]:,.0f}")
    print("\n  Approach C:")
    for grp in ["Micro", "Rest"]:
        for he in [0, 1]:
            for hm in [0, 1]:
                k = (grp, he, hm)
                if k in ET_hem:
                    print(f"    {grp}, expensive={he}, mid={hm}: £{ET_hem[k]:,.0f}")


if __name__ == "__main__":
    main()
