"""Compare impersonation's cost/freq profile against phishing F_T/F_M benchmarks.

Where does impersonation sit in the three-channel picture?
- F_M (mass phishing): ~60% no-cost, mean band ~2, high freq
- F_T (targeted phishing): ~0% no-cost, mean band ~4, low freq
- Serious: high success rate, heavy tail

Run: PYTHONPATH=/home/user/md-clean/src python3 src/estimation/impersonation_profile.py
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BAND_BOUNDS, SIZE_LABELS, load, FREQ_LABELS, GENUINE_TYPE_COLS

COST_BANDS = list(range(1, 11))


def band_to_gbp(band):
    lo, hi = BAND_BOUNDS.get(band, (0, 0))
    if band == 1: return 0.0
    if hi == np.inf: return lo * 2.0
    return (lo + hi) / 2


def profile(sub, label):
    """Print cost/freq profile for a subsample."""
    n = len(sub)
    if n == 0:
        print(f"  {label}: n=0")
        return

    print(f"\n  {label} (n={n})")

    # Freq distribution
    fd = sub["freq"].value_counts(normalize=True).sort_index()
    parts = " ".join(f"f{f}={fd.get(f,0):.2f}" for f in range(1, 7))
    print(f"    Freq: {parts}")
    print(f"    Mean freq code: {sub['freq'].mean():.2f}")

    # Band distribution
    bd = sub["band"].value_counts(normalize=True).sort_index()
    p_success = (sub["band"] >= 2).mean()
    mean_band = sub["band"].mean()
    print(f"    P(success) = P(band>=2) = {p_success:.3f}")
    print(f"    Mean band = {mean_band:.2f}")

    # Band PMF
    band_costs = np.array([band_to_gbp(b) for b in COST_BANDS])
    pmf = np.array([bd.get(b, 0) for b in COST_BANDS])
    et = (pmf * band_costs).sum()
    print(f"    E[T] = £{et:,.0f}")

    parts = " ".join(f"b{b}={bd.get(b,0):.2f}" for b in COST_BANDS)
    print(f"    Band PMF: {parts}")

    # Conditional on success
    successful = sub[sub["band"] >= 2]
    if len(successful) > 0:
        bd_s = successful["band"].value_counts(normalize=True).sort_index()
        pmf_s = np.array([bd_s.get(b, 0) for b in range(2, 11)])
        costs_s = np.array([band_to_gbp(b) for b in range(2, 11)])
        et_s = (pmf_s * costs_s).sum()
        print(f"    E[T|success] = £{et_s:,.0f} (n={len(successful)})")
        parts_s = " ".join(f"b{b}={bd_s.get(b,0):.2f}" for b in range(2, 11))
        print(f"    Band PMF|success: {parts_s}")


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)
    att = df[df["attacked"]].copy()
    v = att[att["band"].notna() & att["freq"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["freq"] = v["freq"].astype(int)

    # Type flags
    v["has_phishing"] = (v["type6"] == 1) if "type6" in v.columns else False
    v["has_impersonation"] = (v["type5"] == 1) if "type5" in v.columns else False

    serious_cols = [f"type{i}" for i in (1, 2, 3, 4, 7, 8, 13, 15, 16)
                    if f"type{i}" in v.columns]
    v["has_serious"] = (v[serious_cols] == 1).any(axis=1)

    # Subsamples
    # Impersonation-only (no phishing, no serious)
    imp_only = v[v["has_impersonation"] & ~v["has_phishing"] & ~v["has_serious"]]
    # Phishing-only (no impersonation, no serious)
    phish_only = v[v["has_phishing"] & ~v["has_impersonation"] & ~v["has_serious"]]
    # Serious-only
    ser_only = v[v["has_serious"] & ~v["has_phishing"] & ~v["has_impersonation"]]
    # Impersonation + phishing only (no serious)
    imp_phish = v[v["has_impersonation"] & v["has_phishing"] & ~v["has_serious"]]

    print("=" * 74)
    print("IMPERSONATION vs PHISHING vs SERIOUS: profile comparison")
    print("=" * 74)

    profile(phish_only, "Phishing-only (no imp, no serious)")
    profile(imp_only, "Impersonation-only (no phish, no serious)")
    profile(ser_only, "Serious-only (no phish, no imp)")
    profile(imp_phish, "Impersonation + Phishing (no serious)")

    # Freq=1 subsamples for clean single draws
    print(f"\n{'='*74}")
    print("FREQ=1 SUBSAMPLES (clean single draws)")
    print(f"{'='*74}")

    profile(phish_only[phish_only["freq"] == 1], "Phishing-only, freq=1")
    profile(imp_only[imp_only["freq"] == 1], "Impersonation-only, freq=1")
    profile(ser_only[ser_only["freq"] == 1], "Serious-only, freq=1")

    # Disrupta breakdown within impersonation-only firms
    print(f"\n{'='*74}")
    print("DISRUPTA within impersonation-only firms")
    print(f"{'='*74}")

    if "disrupta" in v.columns:
        imp_d = imp_only[imp_only["disrupta"].notna()].copy()
        imp_d["disrupta"] = imp_d["disrupta"].astype(int)
        from data import DISRUPTA_LABELS
        print(f"\n  n={len(imp_d)} with valid disrupta:")
        for d, cnt in imp_d["disrupta"].value_counts().sort_index().items():
            label = DISRUPTA_LABELS.get(d, f"code {d}")
            print(f"    disrupta={d} ({label}): {cnt}")

    # Compare impersonation to the phishing F_T/F_M benchmarks
    print(f"\n{'='*74}")
    print("COMPARISON TO PHISHING MIXTURE BENCHMARKS")
    print(f"{'='*74}")
    print(f"\n  F_M (mass phishing) benchmark:")
    print(f"    ~58-66% no-cost, mean band ~1.7-2.0, concentrated in bands 1-2")
    print(f"  F_T (targeted phishing) benchmark:")
    print(f"    ~0% no-cost, mean band ~3.3-4.0, extends to bands 7-8")
    print(f"\n  Impersonation-only:")
    if len(imp_only) > 0:
        p_nc = (imp_only["band"] == 1).mean()
        mb = imp_only["band"].mean()
        print(f"    No-cost fraction: {p_nc:.1%}")
        print(f"    Mean band: {mb:.2f}")
        if p_nc > 0.5:
            print(f"    → Closer to F_M (mass)")
        elif p_nc < 0.1:
            print(f"    → Closer to F_T (targeted)")
        else:
            print(f"    → Intermediate — possibly a mix, or its own thing")

    # By freq within impersonation-only
    print(f"\n{'='*74}")
    print("IMPERSONATION-ONLY by freq band")
    print(f"{'='*74}")
    for f in sorted(imp_only["freq"].unique()):
        sf = imp_only[imp_only["freq"] == f]
        n = len(sf)
        p_s = (sf["band"] >= 2).mean()
        mb = sf["band"].mean()
        bd = sf["band"].value_counts(normalize=True).sort_index()
        parts = " ".join(f"b{b}={bd.get(b,0):.2f}" for b in range(1, 11))
        print(f"  freq={f} (n={n:3d}): P(success)={p_s:.2f}, mean_band={mb:.2f}")
        print(f"    {parts}")


if __name__ == "__main__":
    main()
