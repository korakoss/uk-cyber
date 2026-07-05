"""
Mixture-model bridge: from observed max attack cost to total annual cost.

Builds on the rank-2 SVD result (mixture_model.py) for phishing (disrupta=6).

Steps:
  1. Re-extract F_T, F_M, pi_f from SVD (phishing, pooled across sizes)
  2. Recover P(freq | type T) and P(freq | type M) via Bayes
  3. Fit Poisson(lambda) to the ordinal freq data within each type —
     tests whether a unimodal Poisson describes the within-type freq profile
  4. Monte Carlo bridge (using empirical P(freq | type) + uniform k within band):
     type -> freq band -> k -> k cost draws from F_type -> (total, max)
  5. Report E[total]/E[max] by type and as a mixture-weighted overall multiplier

Key question: how much does total annual cost exceed the observed max?
If both types give a bridge close to 1, the conservative lower bound (total=max)
is quantitatively validated.

Limitation: phishing only, pooled across size bands.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist
from scipy.optimize import minimize_scalar
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

df = get_business_data()
INVALID_FREQ = {997, 999, 997.0, 999.0}

attacked = df[
    df['freq'].notna() & ~df['freq'].isin(INVALID_FREQ) & (df['freq'] > 0) &
    df['damage_bands'].notna() & (df['damage_bands'] < SPECIAL_CODE_THRESHOLD) &
    df['disrupta'].notna() & (df['disrupta'] > 0) & (df['disrupta'] != 997)
].copy()

phishing = attacked[attacked['disrupta'] == 6].copy()

FREQ_LABELS = {1: 'Once only', 2: '>once <monthly', 3: '~monthly',
               4: '~weekly', 5: '~daily', 6: 'Several/day'}
bands = list(range(1, 11))
BAND_MIDPOINTS = {1: 0, 2: 50, 3: 224, 4: 707, 5: 2236,
                  6: 7071, 7: 14142, 8: 31623, 9: 70711, 10: 223607}
midpoint_arr = np.array([BAND_MIDPOINTS[b] for b in bands], dtype=float)

# Count ranges implied by each freq band
FREQ_K_BOUNDS = {
    1: (1,    1),
    2: (2,    11),
    3: (12,   51),
    4: (52,   364),
    5: (365,  999),
    6: (1000, 5000),
}

# ---------------------------------------------------------------------------
# 1. SVD extraction: F_T, F_M, pi_f
# ---------------------------------------------------------------------------

MIN_PER_FREQ = 10
freq_groups = sorted(phishing['freq'].unique())
rows, valid_freqs, ns = [], [], []
for fg in freq_groups:
    sub = phishing[phishing['freq'] == fg]
    if len(sub) < MIN_PER_FREQ:
        continue
    counts = np.array([(sub['damage_bands'] == b).sum() for b in bands], dtype=float)
    if counts.sum() == 0:
        continue
    rows.append(counts / counts.sum())
    valid_freqs.append(int(fg))
    ns.append(len(sub))

P = np.array(rows)
U, S, Vt = np.linalg.svd(P, full_matrices=False)
direction = Vt[0]
mean_p = P.mean(axis=0)

def max_valid_extent(base, d):
    t_max, t_min = np.inf, -np.inf
    for i in range(len(base)):
        if d[i] > 1e-9:
            t_max = min(t_max, (1.0 - base[i]) / d[i])
            t_min = max(t_min, -base[i] / d[i])
        elif d[i] < -1e-9:
            t_min = max(t_min, (1.0 - base[i]) / d[i])
            t_max = min(t_max, -base[i] / d[i])
    return t_min, t_max

t_min, t_max = max_valid_extent(mean_p, direction)
F1 = np.clip(mean_p + t_max * direction, 0, None); F1 /= F1.sum()
F2 = np.clip(mean_p + t_min * direction, 0, None); F2 /= F2.sum()
F_T, F_M = (F1, F2) if np.dot(F1, bands) >= np.dot(F2, bands) else (F2, F1)

denom = np.dot(F_T - F_M, F_T - F_M)
pi_f = {}
for i, fg in enumerate(valid_freqs):
    pi = np.dot(P[i] - F_M, F_T - F_M) / denom if denom > 1e-10 else float('nan')
    pi_f[fg] = float(np.clip(pi, 0, 1))

print("=" * 65)
print("STEP 1: Mixture extraction (phishing, all sizes pooled)")
print("=" * 65)
var2 = (S[0]**2 + S[1]**2) / (S**2).sum()
print(f"\n  SVD rank-2 variance: {var2:.1%}")
print(f"  F_T: mean band={np.dot(F_T, bands):.2f}, no-cost={F_T[0]:.3f}")
print(f"  F_M: mean band={np.dot(F_M, bands):.2f}, no-cost={F_M[0]:.3f}")
print(f"\n  {'Freq group':<20} {'n':>5}  {'pi_f':>6}")
for fg, n in zip(valid_freqs, ns):
    print(f"  {FREQ_LABELS[fg]:<20} {n:>5}  {pi_f[fg]:>6.3f}")

# ---------------------------------------------------------------------------
# 2. Bayes: P(freq | T) and P(freq | M)
# ---------------------------------------------------------------------------

n_phishing_valid = sum((phishing['freq'] == fg).sum() for fg in valid_freqs)
p_freq = {fg: int((phishing['freq'] == fg).sum()) / n_phishing_valid for fg in valid_freqs}
pi_overall = sum(pi_f[fg] * p_freq[fg] for fg in valid_freqs)

p_freq_T = {fg: pi_f[fg] * p_freq[fg] / pi_overall for fg in valid_freqs}
p_freq_M = {fg: (1 - pi_f[fg]) * p_freq[fg] / (1 - pi_overall) for fg in valid_freqs}

print(f"\n{'=' * 65}")
print("STEP 2: P(freq | type) via Bayes")
print(f"{'=' * 65}")
print(f"\n  pi_overall = {pi_overall:.3f}  ({pi_overall:.1%} of phishing firms are type T)")
print(f"\n  {'Freq group':<20} {'P(freq)':>9} {'P(freq|T)':>10} {'P(freq|M)':>10}")
for fg in valid_freqs:
    print(f"  {FREQ_LABELS[fg]:<20} {p_freq[fg]:>9.3f} {p_freq_T[fg]:>10.3f} {p_freq_M[fg]:>10.3f}")

# ---------------------------------------------------------------------------
# 3. Poisson fit for k | type
# ---------------------------------------------------------------------------

def band_prob_poisson(fg, lam):
    lo, hi = FREQ_K_BOUNDS[fg]
    if fg == 6:
        return 1.0 - poisson_dist.cdf(lo - 1, lam)
    return float(poisson_dist.cdf(hi, lam) - poisson_dist.cdf(lo - 1, lam))

def fit_poisson(p_freq_type, label):
    def nll(log_lam):
        lam = np.exp(log_lam)
        ll = sum(p_freq_type[fg] * np.log(max(band_prob_poisson(fg, lam), 1e-12))
                 for fg in valid_freqs)
        return -ll

    res = minimize_scalar(nll, bounds=(-3, 12), method='bounded')
    lam = np.exp(res.x)

    print(f"\n  Poisson fit — {label}: lambda = {lam:.1f}")
    print(f"  {'Freq group':<20} {'P(freq|type)':>13} {'Poisson pred':>13} {'Residual':>10}")
    for fg in valid_freqs:
        obs = p_freq_type[fg]
        fit = band_prob_poisson(fg, lam)
        print(f"  {FREQ_LABELS[fg]:<20} {obs:>13.3f} {fit:>13.3f} {obs-fit:>10.3f}")
    return lam

print(f"\n{'=' * 65}")
print("STEP 3: Poisson fit for k | type")
print("(Tests whether a single Poisson describes within-type freq profile)")
print(f"{'=' * 65}")
lambda_T = fit_poisson(p_freq_T, "type T (targeted)")
lambda_M = fit_poisson(p_freq_M, "type M (mass-market)")

# ---------------------------------------------------------------------------
# 4. Monte Carlo bridge
# ---------------------------------------------------------------------------
# Use empirical P(freq | type) + uniform k within band — more honest than
# Poisson given the freq=6 anomaly for type T.

print(f"\n{'=' * 65}")
print("STEP 4: Monte Carlo bridge")
print("Sampling: type -> freq band (empirical) -> k uniform in band -> costs")
print(f"{'=' * 65}")

N_MC = 4000

def mc_bridge(F, p_freq_type, label, n_mc=N_MC):
    freq_vals = np.array(valid_freqs)
    freq_probs = np.array([p_freq_type[fg] for fg in valid_freqs])

    totals = np.zeros(n_mc)
    maxima = np.zeros(n_mc)
    k_draws = np.zeros(n_mc)

    for i in range(n_mc):
        # Draw freq band, then k uniformly within band
        fg = int(rng.choice(freq_vals, p=freq_probs))
        lo, hi = FREQ_K_BOUNDS[fg]
        k = int(rng.integers(lo, hi + 1))
        k_draws[i] = k

        draws = rng.choice(midpoint_arr, size=k, p=F)
        totals[i] = draws.sum()
        maxima[i] = draws.max()

    # Bridge = total / max; when max=0 (all attacks free), total=0 too, ratio=1
    nonzero = maxima > 0
    ratio = np.where(nonzero, totals / maxima, 1.0)

    print(f"\n  {label}")
    print(f"    E[k]:                {k_draws.mean():.1f}  (median {np.median(k_draws):.0f})")
    print(f"    E[total annual cost]: £{totals.mean():>10,.0f}")
    print(f"    E[max attack cost]:   £{maxima.mean():>10,.0f}")
    print(f"    E[total]/E[max]:      {totals.mean()/maxima.mean():.3f}  (ratio of means)")
    print(f"    E[total/max]:         {ratio.mean():.3f}  (mean of per-firm ratios)")
    print(f"    P(max=0, all free):   {(~nonzero).mean():.3f}")
    pct = np.percentile(ratio, [10, 50, 90])
    print(f"    10/50/90th pct of total/max: {pct[0]:.2f} / {pct[1]:.2f} / {pct[2]:.2f}")

    return totals.mean(), maxima.mean(), ratio.mean()

# NOTE: mc_bridge below is SUPERSEDED by the analytical bridge in Step 4b.
# It was wrong because it draws k costs from F_type (the max distribution)
# rather than from G_type (the per-attack distribution). See Step 4b below.
e_tot_T, e_max_T, mult_T = mc_bridge(F_T, p_freq_T, "Type T (targeted)")
e_tot_M, e_max_M, mult_M = mc_bridge(F_M, p_freq_M, "Type M (mass-market)")

# ---------------------------------------------------------------------------
# 5. Mixture-weighted summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 65}")
print("STEP 5: Mixture-weighted overall bridge")
print(f"{'=' * 65}")

e_tot_mix = pi_overall * e_tot_T + (1 - pi_overall) * e_tot_M
e_max_mix = pi_overall * e_max_T + (1 - pi_overall) * e_max_M
mult_mix = e_tot_mix / e_max_mix

print(f"\n  pi_overall = {pi_overall:.3f}")
print(f"  Type T: E[total]=£{e_tot_T:,.0f},  E[max]=£{e_max_T:,.0f},  mult={mult_T:.3f}")
print(f"  Type M: E[total]=£{e_tot_M:,.0f},  E[max]=£{e_max_M:,.0f},  mult={mult_M:.3f}")
print(f"\n  Mixture E[total] = £{e_tot_mix:,.0f}")
print(f"  Mixture E[max]   = £{e_max_mix:,.0f}")
print(f"  Overall bridge multiplier: {mult_mix:.3f}")

if mult_mix < 1.20:
    verdict = "Conservative lower bound (total=max) is well-supported quantitatively."
elif mult_mix < 1.50:
    verdict = "Modest correction needed — conservative lower bound is a reasonable approximation."
else:
    verdict = "Meaningful correction — bridge multiplier should be applied."
print(f"\n  Verdict: {verdict}")

# ---------------------------------------------------------------------------
# 4b. CORRECTED analytical bridge (supersedes Steps 4 and 5 above)
#
# The MC above was wrong: it drew k costs from F_type (the observed max
# distribution), but F_type is the distribution of the MAXIMUM of k attacks,
# not of each individual attack.
#
# Correct model (i.i.d. within type):
#   Per-attack distribution G satisfies  G_CDF(x)^k = F_CDF(x)
#   => G_CDF(b) = F_CDF(b)^(1/k)
#
# This gives:
#   E[max]   = sum_b F_prob[b] * midpoint[b]          (=E[X|F], the observed mean)
#   E[total] = k * sum_b G_prob[b] * midpoint[b]
#            = k * sum_b (F_CDF[b]^(1/k) - F_CDF[b-1]^(1/k)) * midpoint[b]
#   Bridge   = E[total] / E[max]
#
# For k=1: G=F exactly, bridge=1.  For large k: G concentrates near 0,
# E[total] grows slowly, bridge stays close to 1.
# ---------------------------------------------------------------------------

print(f"\n{'=' * 65}")
print("STEP 4b: CORRECTED analytical bridge")
print("Per-attack distribution G = F^(1/k) (order-statistics inversion)")
print(f"{'=' * 65}")

def analytical_bridge(F_prob, k):
    """
    Bridge = E[total k attacks] / E[max of k attacks], for k i.i.d. draws.
    F_prob: observed (max) distribution over bands. G_CDF = F_CDF^(1/k).
    """
    F_CDF = np.cumsum(F_prob)
    F_CDF_prev = np.concatenate([[0.0], F_CDF[:-1]])
    G_prob = F_CDF ** (1.0 / k) - F_CDF_prev ** (1.0 / k)
    e_max   = float(np.dot(F_prob, midpoint_arr))
    e_total = k * float(np.dot(G_prob, midpoint_arr))
    return e_total, e_max, (e_total / e_max if e_max > 0 else 1.0)

# Show bridge as a function of k for both F_T and F_M
k_values = [1, 2, 5, 10, 25, 52, 100, 365, 1000, 3000]
print(f"\n  Bridge(k) = E[total] / E[max] under i.i.d.-within-type assumption")
print(f"  {'k':>6}  {'Bridge(F_T)':>12}  {'Bridge(F_M)':>12}")
for k in k_values:
    _, _, b_T = analytical_bridge(F_T, k)
    _, _, b_M = analytical_bridge(F_M, k)
    print(f"  {k:>6}  {b_T:>12.4f}  {b_M:>12.4f}")

# Freq band midpoint counts (central estimate for each band)
FREQ_K_MID = {1: 1, 2: 6, 3: 25, 4: 100, 5: 365, 6: 1500}

# Weighted bridge per type: average over freq bands weighted by P(freq | type)
def weighted_bridge(F_prob, p_freq_type, label):
    e_tots, e_maxs, weights = [], [], []
    print(f"\n  {label}:")
    print(f"  {'Freq group':<20} {'k_mid':>6} {'P(freq|type)':>13} {'Bridge(k)':>10} {'E[max] £':>10}")
    for fg in valid_freqs:
        k = FREQ_K_MID[fg]
        w = p_freq_type[fg]
        e_tot, e_max, bridge = analytical_bridge(F_prob, k)
        print(f"  {FREQ_LABELS[fg]:<20} {k:>6} {w:>13.3f} {bridge:>10.4f} {e_max:>10,.0f}")
        e_tots.append(e_tot * w)
        e_maxs.append(e_max * w)
        weights.append(w)
    overall_bridge = sum(e_tots) / sum(e_maxs)
    print(f"  {'Weighted overall':<20} {'':>6} {sum(weights):>13.3f} {overall_bridge:>10.4f}")
    return overall_bridge

print(f"\n  {'=' * 60}")
print("  Per-type weighted bridge (empirical P(freq | type) weights)")
bridge_T = weighted_bridge(F_T, p_freq_T, "Type T (targeted)")
bridge_M = weighted_bridge(F_M, p_freq_M, "Type M (mass-market)")

# Overall mixture bridge
e_tot_T_w = sum(analytical_bridge(F_T, FREQ_K_MID[fg])[0] * p_freq_T[fg] for fg in valid_freqs)
e_max_T_w = sum(analytical_bridge(F_T, FREQ_K_MID[fg])[1] * p_freq_T[fg] for fg in valid_freqs)
e_tot_M_w = sum(analytical_bridge(F_M, FREQ_K_MID[fg])[0] * p_freq_M[fg] for fg in valid_freqs)
e_max_M_w = sum(analytical_bridge(F_M, FREQ_K_MID[fg])[1] * p_freq_M[fg] for fg in valid_freqs)

e_tot_mix = pi_overall * e_tot_T_w + (1 - pi_overall) * e_tot_M_w
e_max_mix = pi_overall * e_max_T_w + (1 - pi_overall) * e_max_M_w
bridge_mix = e_tot_mix / e_max_mix

print(f"\n{'=' * 65}")
print("STEP 5b: Mixture-weighted overall bridge (corrected)")
print(f"{'=' * 65}")
print(f"\n  pi_overall = {pi_overall:.3f}")
print(f"  Type T weighted bridge: {bridge_T:.4f}")
print(f"  Type M weighted bridge: {bridge_M:.4f}")
print(f"\n  Mixture E[total] = £{e_tot_mix:,.0f}")
print(f"  Mixture E[max]   = £{e_max_mix:,.0f}")
print(f"  Overall bridge multiplier: {bridge_mix:.4f}")

if bridge_mix < 1.10:
    verdict = "Conservative lower bound (total=max) is well-supported — correction < 10%."
elif bridge_mix < 1.30:
    verdict = "Modest correction — conservative lower bound understates by 10-30%."
else:
    verdict = "Meaningful correction — bridge multiplier materially exceeds 1."
print(f"\n  Verdict: {verdict}")
print(f"\n  Note: the freq=6 (several/day) anomaly for type T inflates bridge_T.")
print(f"  Excluding freq=6 from type T (treating it as noise), bridge_T would be lower.")

print("\nDone.")
