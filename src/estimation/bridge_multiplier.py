"""
Bridge multiplier: empirical estimate of E[total annual cost] / E[single largest attack cost].

Approach: for each (size stratum, freq group) cell, treat the observed damage_bands
distribution as the per-attack cost distribution, draw k attacks (k = implied count
for the freq band), compute sum and max, and report the ratio.

Important caveat: the observed damage_bands distribution IS the distribution of the
maximum over k attacks, not the per-attack distribution. Using it as the per-attack
distribution overestimates each additional attack's cost, so the multiplier produced
here is an UPPER BOUND on the true multiplier. The true per-attack distribution is
cheaper than the max distribution (especially for high-freq groups).

For freq=1: multiplier = 1 exactly (one attack, sum = max).

Outputs:
- Multiplier table by (size, freq): mean and 90% CI from bootstrap
- Implied E[total cost per firm] under conservative (m=1), upper-bound (empirical m),
  and a sensitivity sweep.
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

df = get_business_data()
INVALID_FREQ = {997, 999, 997.0, 999.0}

attacked = df[
    df['freq'].notna() &
    ~df['freq'].isin(INVALID_FREQ) &
    (df['freq'] > 0) &
    df['damage_bands'].notna() &
    (df['damage_bands'] < SPECIAL_CODE_THRESHOLD)
].copy()

# Geometric midpoints for damage_bands (£)
DAMAGE_MIDPOINTS = {
    1: 0,
    2: 50,
    3: 224,
    4: 707,
    5: 2236,
    6: 7071,
    7: 14142,
    8: 31623,
    9: 70711,
    10: 223607,
}

# Implied attack count k per freq band
# freq=2 is most uncertain (2-11/year); using midpoint ~6
K_IMPLIED = {1: 1, 2: 6, 3: 12, 4: 52, 5: 365, 6: 1000}
K_LOWER   = {1: 1, 2: 2, 3: 10, 4: 40, 5: 300, 6: 700}
K_UPPER   = {1: 1, 2: 11, 3: 15, 4: 70, 5: 500, 6: 1500}

FREQ_LABELS = {
    1: 'Once only', 2: '>once <monthly', 3: '~monthly',
    4: '~weekly', 5: '~daily', 6: 'Several/day'
}
SIZE_LABELS = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}

N_MC = 5000  # Monte Carlo iterations per cell

# ---------------------------------------------------------------------------
# Monte Carlo multiplier for a given empirical distribution and k
# ---------------------------------------------------------------------------

def compute_multiplier(probs, midpoints, k, n_mc=N_MC):
    """
    Draw k samples from the cost distribution, compute mean(sum)/mean(max).
    Returns (mean_multiplier, p5, p95).
    probs: array of probabilities for each midpoint value.
    midpoints: array of £ midpoint values.
    k: number of attacks to draw.
    """
    if k == 1:
        return 1.0, 1.0, 1.0

    ratios = []
    for _ in range(n_mc):
        draws = rng.choice(midpoints, size=k, p=probs)
        total = draws.sum()
        maximum = draws.max()
        if maximum > 0:
            ratios.append(total / maximum)
        # If maximum == 0 (all attacks free), ratio is trivially 1 (total=max=0)
        else:
            ratios.append(1.0)

    ratios = np.array(ratios)
    return ratios.mean(), np.percentile(ratios, 5), np.percentile(ratios, 95)

# ---------------------------------------------------------------------------
# Main: compute multipliers per (size, freq) cell, weighted and unweighted
# ---------------------------------------------------------------------------

bands = sorted(DAMAGE_MIDPOINTS.keys())
midpoint_arr = np.array([DAMAGE_MIDPOINTS[b] for b in bands], dtype=float)

print("=" * 80)
print("BRIDGE MULTIPLIER: E[total annual cost] / E[single largest attack cost]")
print("Using within-cell damage_bands distribution as per-attack distribution.")
print("NOTE: this is an UPPER BOUND — see module docstring.")
print("=" * 80)

results = []

for weighted in [False, True]:
    label = "WEIGHTED" if weighted else "UNWEIGHTED"
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  {'Size':<10} {'Freq':<18} {'n':>5} {'k':>5} {'E[max] £':>10} "
          f"{'E[tot] £':>10} {'mult mean':>11} {'90% CI':>16} {'note'}")

    for size in sorted(attacked['sizeb'].dropna().unique()):
        for freq in sorted(attacked['freq'].dropna().unique()):
            cell = attacked[(attacked['sizeb'] == size) & (attacked['freq'] == freq)]
            n = len(cell)
            if n < 5:
                continue

            # Empirical distribution of damage_bands
            if weighted:
                w = cell['weight'].values
                counts = np.array([
                    (cell['damage_bands'] == b).values @ w for b in bands
                ], dtype=float)
            else:
                counts = np.array([(cell['damage_bands'] == b).sum() for b in bands], dtype=float)

            if counts.sum() == 0:
                continue
            probs = counts / counts.sum()

            k = K_IMPLIED.get(int(freq), 1)
            mult_mean, p5, p95 = compute_multiplier(probs, midpoint_arr, k)

            e_max = float(np.dot(probs, midpoint_arr))
            e_total = e_max * mult_mean

            note = "exact" if freq == 1 else ("upper bound" if k > 1 else "")

            print(f"  {SIZE_LABELS.get(size, size):<10} {FREQ_LABELS.get(int(freq), str(freq)):<18} "
                  f"{n:>5} {k:>5} {e_max:>10,.0f} {e_total:>10,.0f} "
                  f"{mult_mean:>11.2f} [{p5:.2f}, {p95:.2f}] {note}")

            results.append(dict(
                weighted=weighted, size=size, freq=freq, n=n, k=k,
                e_max=e_max, e_total=e_total,
                mult_mean=mult_mean, mult_p5=p5, mult_p95=p95
            ))

# ---------------------------------------------------------------------------
# Sensitivity table: how much does total cost estimate change by multiplier?
# Aggregated across size bands (for a rough sense of total UK impact direction)
# ---------------------------------------------------------------------------

print()
print("=" * 80)
print("SENSITIVITY: relative impact of multiplier choice on total cost estimate")
print("Shows E[total cost per attacked firm] under different multiplier assumptions,")
print("collapsed across size bands (unweighted).")
print("=" * 80)

# For each freq group, get mean E[max] across size bands
unweighted_results = [r for r in results if not r['weighted']]

print(f"\n  {'Freq':<18} {'n firms':>8} {'E[max] £':>10} | "
      f"{'m=1 (cons)':>12} {'m=emp (UB)':>12} {'m=2':>8} {'m=5':>8}")

for freq in sorted(set(r['freq'] for r in unweighted_results)):
    freq_rows = [r for r in unweighted_results if r['freq'] == freq]
    if not freq_rows:
        continue
    total_n = sum(r['n'] for r in freq_rows)
    # Weighted average E[max] across size bands by n
    avg_emax = sum(r['e_max'] * r['n'] for r in freq_rows) / total_n
    avg_mult = sum(r['mult_mean'] * r['n'] for r in freq_rows) / total_n

    label = FREQ_LABELS.get(int(freq), str(freq))
    print(f"  {label:<18} {total_n:>8} {avg_emax:>10,.0f} | "
          f"{avg_emax:>12,.0f} {avg_emax*avg_mult:>12,.0f} "
          f"{avg_emax*2:>8,.0f} {avg_emax*5:>8,.0f}")

print()
print("Columns: m=1 (conservative: total=max), m=emp (empirical upper bound),")
print("         m=2 and m=5 (fixed sensitivity values).")
print("All values are E[cost per attacked firm in that freq group], in £.")
print()
print("Done.")
