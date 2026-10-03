"""What's the best conditioning variable for lambda?

Compare n_types against alternatives: which-types matters more than
how-many-types for predicting peak status.

Run: ``python src/estimation/lambda_conditioning.py``
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import GENUINE_TYPE_COLS, SIZE_LABELS, load


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)
    attacked = df[df["attacked"]].copy()
    v = attacked[attacked["band"].notna() & attacked["freq"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["peak"] = v["band"] >= 3

    type_cols = [c for c in GENUINE_TYPE_COLS if c in v.columns]

    print("=" * 74)
    print("WHAT PREDICTS PEAK STATUS?")
    print("=" * 74)

    # 1. Per-type peak enrichment
    print("\n1. PER-TYPE PEAK ENRICHMENT")
    print("-" * 50)
    print(f"  {'type':<20s} {'prev':>6s} {'P(peak|t=1)':>12s} {'P(peak|t=0)':>12s} {'enrichment':>10s}")

    enrichments = {}
    for tc in type_cols:
        has = v[v[tc] == 1]
        hasnt = v[v[tc] == 0]
        if len(has) < 10:
            continue
        p1 = has["peak"].mean()
        p0 = hasnt["peak"].mean()
        enrichments[tc] = p1 / p0 if p0 > 0 else np.inf
        print(f"  {tc:<20s} {has.shape[0]/len(v):>6.1%} {p1:>12.3f} {p0:>12.3f} {enrichments[tc]:>10.2f}x")

    # 2. Which types are "peak-predictive" vs "flood-predictive"?
    print("\n2. CLASSIFICATION")
    print("-" * 50)
    peak_types = [tc for tc, e in enrichments.items() if e > 1.5]
    flood_types = [tc for tc, e in enrichments.items() if e < 0.8]
    neutral = [tc for tc in enrichments if tc not in peak_types and tc not in flood_types]
    print(f"  Peak-predictive (enrich > 1.5x): {peak_types}")
    print(f"  Flood-predictive (enrich < 0.8x): {flood_types}")
    print(f"  Neutral: {neutral}")

    # 3. Construct alternative conditioning variables
    v["n_peak_types"] = sum(v[tc] == 1 for tc in peak_types).astype(int)
    v["n_flood_types"] = sum(v[tc] == 1 for tc in flood_types).astype(int) if flood_types else 0
    v["has_any_peak_type"] = (v["n_peak_types"] > 0).astype(int)

    # Severity-weighted score
    weights = {tc: enrichments.get(tc, 1.0) for tc in type_cols}
    v["severity_score"] = sum(v[tc] * weights[tc] for tc in type_cols if tc in weights)
    # Bin it for cell estimation
    v["severity_bin"] = pd.qcut(v["severity_score"], q=4, labels=False, duplicates="drop")

    print("\n3. COMPARE CONDITIONING VARIABLES — P(peak) by cell")
    print("-" * 74)

    # For each candidate, show P(peak) spread and within-cell homogeneity
    candidates = {
        "n_types": "n_types",
        "n_peak_types": "n_peak_types",
        "has_any_peak_type": "has_any_peak_type",
        "severity_bin": "severity_bin",
    }

    for name, col in candidates.items():
        print(f"\n  --- {name} ---")
        vals = sorted(v[col].dropna().unique())
        cells = []
        print(f"    {'val':>5s} {'n':>6s} {'P(peak)':>8s}")
        for val in vals:
            cell = v[v[col] == val]
            pp = cell["peak"].mean()
            cells.append({"val": val, "n": len(cell), "p_peak": pp})
            print(f"    {val:>5.0f} {len(cell):>6d} {pp:>8.3f}")

        # Measure: how much variance in peak status does this variable explain?
        # Use eta-squared (between-group variance / total variance)
        groups = [v[v[col] == val]["peak"].astype(float).values for val in vals]
        f_stat, p_val = stats.f_oneway(*[g for g in groups if len(g) > 1])
        # Eta-squared
        grand_mean = v["peak"].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = sum((v["peak"].astype(float) - grand_mean)**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        print(f"    F={f_stat:.1f}, p={p_val:.2e}, η²={eta_sq:.4f}")

    # 4. Does "which types" add info beyond n_types?
    print(f"\n\n4. DOES 'WHICH TYPES' ADD INFO BEYOND n_types?")
    print("-" * 50)
    print("  P(peak) by (n_types, has_any_peak_type):")
    print(f"    {'n_types':>7s} {'no_peak_type':>15s} {'has_peak_type':>15s} {'diff':>8s}")
    for nt in sorted(v["n_types"].unique()):
        for hpt in [0, 1]:
            cell = v[(v["n_types"] == nt) & (v["has_any_peak_type"] == hpt)]
            if len(cell) < 5:
                continue
        c0 = v[(v["n_types"] == nt) & (v["has_any_peak_type"] == 0)]
        c1 = v[(v["n_types"] == nt) & (v["has_any_peak_type"] == 1)]
        if len(c0) >= 5 and len(c1) >= 5:
            p0 = c0["peak"].mean()
            p1 = c1["peak"].mean()
            print(f"    {nt:>7d} {p0:>12.3f} (n={len(c0):<4d}) {p1:>12.3f} (n={len(c1):<4d}) {p1-p0:>+8.3f}")

    # 5. Logistic regression comparison
    print(f"\n\n5. LOGISTIC REGRESSION — McFadden pseudo-R² comparison")
    print("-" * 50)
    import statsmodels.api as sm

    y = v["peak"].astype(int).values

    for name, X_cols in [
        ("size only", ["sizeb"]),
        ("size + n_types", ["sizeb", "n_types"]),
        ("size + n_peak_types", ["sizeb", "n_peak_types"]),
        ("size + has_any_peak_type", ["sizeb", "has_any_peak_type"]),
        ("size + all type flags", ["sizeb"] + type_cols),
        ("size + n_types + has_any_peak_type", ["sizeb", "n_types", "has_any_peak_type"]),
        ("size + severity_score", ["sizeb", "severity_score"]),
    ]:
        X = sm.add_constant(v[X_cols].values.astype(float))
        try:
            model = sm.Logit(y, X).fit(disp=0)
            print(f"  {name:<40s} pseudo-R² = {model.prsquared:.4f}")
        except Exception as e:
            print(f"  {name:<40s} FAILED: {e}")


if __name__ == "__main__":
    main()
