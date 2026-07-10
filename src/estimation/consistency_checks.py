"""
Consistency checks for the i.i.d. order-statistics model of attack costs.

Model: within each size stratum, each firm has an underlying attack cost
distribution F(x). For a firm with k attacks, the observed cost (of the
largest attack) is max(X_1,...,X_k) where X_i ~i.i.d. F. The CDF of this
maximum is F(x)^k.

Two checks per size stratum:
1. Monotonicity: stochastic dominance of cost_band distribution as freq
   increases (higher k should yield stochastically larger maxima).
2. Chi-square goodness-of-fit: use freq=1 group as the reference estimate
   of F; predict band proportions for freq=k groups as F(b)^k - F(b-1)^k;
   compare against observed counts.

Both checks run in weighted and unweighted variants.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
sys.path.insert(0, '.')
from proc import get_business_data

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

df = get_business_data()

# Keep only attacked firms with valid freq and valid damage_bands.
# damage_bands special codes (don't know / refused) include 997, 999 and
# non-integer variants like 999.62004 from SPSS encoding; filter with > 100.
# freq special codes 997/999 are exact integers so isin() suffices there.
INVALID_FREQ = {997, 999, 997.0, 999.0}
attacked = df[
    df['freq'].notna() &
    ~df['freq'].isin(INVALID_FREQ) &
    (df['freq'] > 0) &
    df['damage_bands'].notna() &
    (df['damage_bands'] < 100)
].copy()

# Map freq to groups: 1, 2, 3, 4, 5+ (pool high-freq due to small cells)
def freq_group(f):
    f = int(f)
    return f if f <= 4 else 5

attacked['freq_group'] = attacked['freq'].apply(freq_group)

FREQ_LABELS = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5+'}
SIZE_LABELS = {1: 'Micro (1-9)', 2: 'Small (10-49)', 3: 'Medium (50-249)', 4: 'Large (250+)'}

bands = sorted(attacked['damage_bands'].unique())
band_arr = np.array(bands)

# ---------------------------------------------------------------------------
# Helper: weighted empirical CDF at each band value
# ---------------------------------------------------------------------------

def empirical_cdf(subset, weights=None):
    """Return dict: band -> P(damage_bands <= band), using weights if given."""
    total = weights.sum() if weights is not None else len(subset)
    cdf = {}
    for b in bands:
        mask = subset['damage_bands'] <= b
        w = weights[mask].sum() if weights is not None else mask.sum()
        cdf[b] = w / total
    return cdf


def empirical_pmf(subset, weights=None):
    """Return dict: band -> P(damage_bands == band)."""
    total = weights.sum() if weights is not None else len(subset)
    pmf = {}
    for b in bands:
        mask = subset['damage_bands'] == b
        w = weights[mask].sum() if weights is not None else mask.sum()
        pmf[b] = w / total
    return pmf

# ---------------------------------------------------------------------------
# Cell size summary
# ---------------------------------------------------------------------------

print("=" * 80)
print("CELL SIZE SUMMARY: sizeb x freq_group (n respondents, attacked only)")
print("=" * 80)
print(pd.crosstab(attacked['sizeb'], attacked['freq_group'],
                  rownames=['sizeb'], colnames=['freq_group'], margins=True))
print()

# ---------------------------------------------------------------------------
# Run checks per size band and per weighting variant
# ---------------------------------------------------------------------------

for weighted in [False, True]:
    label = "WEIGHTED" if weighted else "UNWEIGHTED"
    print("=" * 80)
    print(f"CHECKS — {label}")
    print("=" * 80)

    for size in sorted(attacked['sizeb'].unique()):
        size_data = attacked[attacked['sizeb'] == size]
        size_name = SIZE_LABELS.get(size, str(size))
        print(f"\n--- Size band: {size_name} ---")

        # ----------------------------------------------------------------
        # 1. Monotonicity check
        # ----------------------------------------------------------------
        print("\n  [1] Monotonicity (mean damage_band rank by freq group)")
        print(f"  {'freq_group':<12} {'n':>6} {'mean_band':>12} {'monotone?':>12}")

        freq_groups = sorted(size_data['freq_group'].unique())
        prev_mean = -np.inf
        all_monotone = True
        for fg in freq_groups:
            sub = size_data[size_data['freq_group'] == fg]
            w = sub['weight'] if weighted else None
            if weighted:
                mean_band = np.average(sub['damage_bands'], weights=sub['weight'])
            else:
                mean_band = sub['damage_bands'].mean()
            n = len(sub)
            mono = "YES" if mean_band >= prev_mean else "NO *"
            if mean_band < prev_mean:
                all_monotone = False
            print(f"  {FREQ_LABELS.get(fg, str(fg)):<12} {n:>6} {mean_band:>12.3f} {mono:>12}")
            prev_mean = mean_band

        print(f"  Overall monotone: {'YES' if all_monotone else 'NO — violation(s) flagged above'}")

        # ----------------------------------------------------------------
        # 2. Chi-square goodness-of-fit against freq=1 baseline
        # ----------------------------------------------------------------
        print("\n  [2] Chi-square GOF (observed vs predicted from freq=1 F)")

        freq1 = size_data[size_data['freq_group'] == 1]
        if len(freq1) < 5:
            print("  freq=1 group too small to estimate F reliably — skipping")
            continue

        w1 = freq1['weight'] if weighted else None
        f1_cdf = empirical_cdf(freq1, weights=w1 if weighted else None)

        # Predicted PMF for max of k draws: P(band=b) = F(b)^k - F(b_prev)^k
        def predicted_pmf_max_k(k, f_cdf):
            pmf = {}
            prev_cdf = 0.0
            for b in bands:
                cdf_b = f_cdf[b] ** k
                pmf[b] = cdf_b - prev_cdf
                prev_cdf = cdf_b
            return pmf

        for fg in freq_groups:
            if fg == 1:
                continue  # skip baseline
            sub = size_data[size_data['freq_group'] == fg]
            n_sub = len(sub)
            if n_sub < 5:
                print(f"  freq_group={FREQ_LABELS.get(fg, str(fg))}: n={n_sub}, too small — skipping")
                continue

            # k for the group (use group value; for 5+ use 5 as approximation)
            k = fg
            pred = predicted_pmf_max_k(k, f1_cdf)

            # Observed counts (unweighted for chi-square)
            obs_counts = sub['damage_bands'].value_counts().reindex(bands, fill_value=0)
            pred_counts = np.array([pred[b] * n_sub for b in bands])

            # Pool cells with expected count < 1
            obs_arr = obs_counts.values.astype(float)
            valid_mask = (pred_counts >= 1) | (obs_arr > 0)
            obs_use = obs_arr[valid_mask]
            pred_use = pred_counts[valid_mask]

            # Normalise predicted to match observed total
            pred_use = pred_use / pred_use.sum() * obs_use.sum()

            if len(obs_use) < 2:
                print(f"  freq_group={FREQ_LABELS.get(fg, str(fg))}: too few non-empty cells — skipping")
                continue

            from scipy.stats import chisquare
            chi2_stat, p_val = chisquare(obs_use, f_exp=pred_use)
            df_chi = len(obs_use) - 1

            # Flag small expected cells
            small_cells = (pred_use < 5).sum()
            flag = f" [warning: {small_cells} cell(s) with expected<5]" if small_cells else ""

            print(f"  freq_group={FREQ_LABELS.get(fg, str(fg))}: "
                  f"n={n_sub}, χ²={chi2_stat:.2f}, df={df_chi}, p={p_val:.4f}{flag}")

print()
print("Done.")

# ---------------------------------------------------------------------------
# Anomaly investigation: Small band (sizeb=2), freq_group=1 has mean_band=19.56
# far outside the range of all other groups. Investigate damage_bands values.
# ---------------------------------------------------------------------------

print()
print("=" * 80)
print("ANOMALY INVESTIGATION: damage_bands values in Small (sizeb=2), freq=1")
print("=" * 80)

small_f1 = attacked[(attacked['sizeb'] == 2) & (attacked['freq_group'] == 1)]
print(f"\nAll damage_bands values (value_counts):")
print(small_f1['damage_bands'].value_counts().sort_index())

print(f"\nSummary stats: min={small_f1['damage_bands'].min()}, "
      f"max={small_f1['damage_bands'].max()}, "
      f"mean={small_f1['damage_bands'].mean():.2f}, "
      f"median={small_f1['damage_bands'].median()}")

print(f"\nFor reference — damage_bands value_counts across ALL attacked businesses:")
print(attacked['damage_bands'].value_counts().sort_index())

print(f"\nAll unique damage_bands values in full attacked dataset:")
print(sorted(attacked['damage_bands'].unique()))

print(f"\nRows with damage_bands > 10 in Small freq=1 group:")
outliers = small_f1[small_f1['damage_bands'] > 10]
print(f"  Count: {len(outliers)}")
if len(outliers) > 0:
    print(outliers[['sizeb', 'freq', 'freq_group', 'damage_bands', 'weight']])

print(f"\nFor comparison — damage_bands value_counts for Small (sizeb=2) ALL freq groups:")
small_all = attacked[attacked['sizeb'] == 2]
print(pd.crosstab(small_all['damage_bands'], small_all['freq_group'],
                  rownames=['damage_bands'], colnames=['freq_group'], margins=True))

# ---------------------------------------------------------------------------
# Decomposition: is the "mean damage_band declines with freq" trend driven
# entirely by the rising "no cost" (band=1) fraction, or does severity ALSO
# decline conditional on nonzero cost? The earlier Analysis 2 in
# attack_type_breakdown.py computed mean_db and P(damage>=b) unconditionally
# (denominator = all firms in the freq group, including no-cost ones) - this
# does not distinguish "no-cost share rising" from "conditional-on-nonzero
# severity falling". Check both components separately, pooled across sizes
# and per major attack type.
# ---------------------------------------------------------------------------

print()
print("=" * 80)
print("DECOMPOSITION: no-cost share vs. conditional-on-nonzero severity, by freq")
print("Unconditional mean can fall purely because more firms report 'no cost',")
print("even if the cost distribution among firms WITH a cost is unchanged.")
print("=" * 80)


def decompose_by_freq(subset, label):
    print(f"\n--- {label} (n={len(subset)}) ---")
    print(f"  {'freq_group':<12} {'n':>5} {'no_cost_%':>10} "
          f"{'uncond_mean':>13} {'cond_mean(>1)':>14} {'n_nonzero':>10}")
    for fg in sorted(subset['freq_group'].unique()):
        sub = subset[subset['freq_group'] == fg]
        n = len(sub)
        if n < 5:
            continue
        no_cost_pct = 100 * (sub['damage_bands'] == 1).mean()
        uncond_mean = sub['damage_bands'].mean()
        nonzero = sub[sub['damage_bands'] > 1]
        cond_mean = nonzero['damage_bands'].mean() if len(nonzero) >= 5 else float('nan')
        print(f"  {FREQ_LABELS.get(fg, str(fg)):<12} {n:>5} {no_cost_pct:>9.1f}% "
              f"{uncond_mean:>13.3f} {cond_mean:>14.3f} {len(nonzero):>10}")


decompose_by_freq(attacked, "ALL attacked firms, pooled across size and type")

# Per major attack type (disrupta = most disruptive type), pooled across sizes
DISRUPTA_LABELS_LOCAL = {
    1: 'Ransomware', 2: 'Other malware', 3: 'Denial of service',
    4: 'Hacking', 5: 'Impersonation', 6: 'Phishing',
}
if 'disrupta' in attacked.columns:
    for d, name in DISRUPTA_LABELS_LOCAL.items():
        sub_type = attacked[attacked['disrupta'] == d]
        if len(sub_type) < 20:
            print(f"\n--- {name} (disrupta={d}): n={len(sub_type)}, too small — skipping ---")
            continue
        decompose_by_freq(sub_type, f"{name} (disrupta={d})")

print()
print("Done.")
