"""
Mixture model feasibility: 2-component (targeted vs. mass) test.

Hypothesis: within a given (disrupta type, size basket), the cost distribution
at each freq group is a convex mixture of two latent component distributions:
    p_f = π_f * F_T + (1 - π_f) * F_M

Structural implication: the matrix P whose rows are p_f (one per freq group)
should have rank at most 2. All row-vectors lie on a 1D affine subspace (a
line segment between F_T and F_M in the band-probability simplex).

Test: compute SVD of P, examine singular values. If 2 components dominate,
the hypothesis is consistent with the data. Then extract F_T and F_M as the
endpoints of the fitted line segment.

Initial run: phishing (disrupta=6) pooled across all size bands.
Noted limitation: pooling sizes assumes the component distributions F_T, F_M
are the same across sizes. Results here are indicative only; not conclusive.
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

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
    (df['damage_bands'] < SPECIAL_CODE_THRESHOLD) &
    df['disrupta'].notna() &
    (df['disrupta'] > 0) &
    (df['disrupta'] != 997)
].copy()

FREQ_LABELS = {
    1: 'Once only', 2: '>once <monthly', 3: '~monthly',
    4: '~weekly', 5: '~daily', 6: 'Several/day'
}
DISRUPTA_LABELS = {
    1: 'Ransomware', 2: 'Other malware', 3: 'Denial of service',
    4: 'Hacking', 5: 'Impersonation', 6: 'Phishing',
    11: 'Website/social takeover', 12: 'Other',
}
DAMAGE_BAND_LABELS = {
    1: 'No cost', 2: '<£100', 3: '£100-£500', 4: '£500-£1k',
    5: '£1k-£5k', 6: '£5k-£10k', 7: '£10k-£20k', 8: '£20k-£50k',
    9: '£50k-£100k', 10: '£100k-£500k'
}
bands = sorted(DAMAGE_BAND_LABELS.keys())

# ---------------------------------------------------------------------------
# Core function: rank-2 test for a given subset
# ---------------------------------------------------------------------------

def run_mixture_test(subset, label, group_col='freq', group_labels=None, min_per_freq=10):
    """
    For a given subset of firms (a type/size basket), build the group x band
    matrix P of conditional cost distributions (grouping by `group_col`, e.g.
    freq band or phisheng_bands engagement level), run SVD, report rank
    structure, and attempt to extract F_T and F_M.
    """
    if group_labels is None:
        group_labels = FREQ_LABELS
    groups = sorted(subset[group_col].unique())

    # Build matrix P: rows = groups, cols = damage bands
    rows = []
    row_labels = []
    ns = []
    for g in groups:
        sub = subset[subset[group_col] == g]
        if len(sub) < min_per_freq:
            continue
        counts = np.array([(sub['damage_bands'] == b).sum() for b in bands], dtype=float)
        if counts.sum() == 0:
            continue
        probs = counts / counts.sum()
        rows.append(probs)
        row_labels.append(group_labels.get(int(g), str(g)))
        ns.append(len(sub))

    if len(rows) < 3:
        print(f"  {label}: only {len(rows)} freq groups with n≥{min_per_freq} — insufficient for rank test")
        return

    P = np.array(rows)  # shape: (n_freq_groups, n_bands)
    n_rows, n_cols = P.shape

    print(f"\n{'='*70}")
    print(f"  {label}  (n_freq_groups={n_rows}, n_bands={n_cols})")
    print(f"{'='*70}")

    # Show the raw P matrix
    print(f"\n  P matrix (row=freq group, col=damage band, values=proportions):")
    header = f"  {'Group':<20} {'n':>5}  " + "  ".join(f"{DAMAGE_BAND_LABELS[b][:8]:>8}" for b in bands)
    print(header)
    for i, (lbl, n) in enumerate(zip(row_labels, ns)):
        row_str = f"  {lbl:<20} {n:>5}  " + "  ".join(f"{P[i,j]:>8.3f}" for j in range(n_cols))
        print(row_str)

    # SVD
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    total_var = (S**2).sum()
    cumvar = np.cumsum(S**2) / total_var

    print(f"\n  Singular values and cumulative variance explained:")
    print(f"  {'Component':>12} {'Singular value':>16} {'Var explained':>14} {'Cumulative':>12}")
    for i, (s, cv) in enumerate(zip(S, cumvar)):
        print(f"  {i+1:>12} {s:>16.4f} {s**2/total_var:>13.1%} {cv:>11.1%}")

    # Rank-2 assessment
    var_2 = cumvar[1] if len(cumvar) > 1 else cumvar[0]
    var_3 = cumvar[2] if len(cumvar) > 2 else 1.0
    gap = var_3 - var_2  # additional variance from 3rd component
    print(f"\n  Variance explained by 2 components: {var_2:.1%}")
    print(f"  Marginal gain from 3rd component:   {gap:.1%}")
    if var_2 > 0.95 and gap < 0.03:
        rank_verdict = "CONSISTENT with rank-2 (2-component mixture plausible)"
    elif var_2 > 0.90:
        rank_verdict = "MARGINAL — rank-2 is approximate but not clean"
    else:
        rank_verdict = "NOT consistent with rank-2 (>2 components needed)"
    print(f"  Verdict: {rank_verdict}")

    # Extract F_T and F_M from the rank-2 approximation
    # Project rows onto PC1 (first right singular vector)
    # The two endpoints of the line segment in 2D are the extremes of PC1 scores
    print(f"\n  Extracting F_T and F_M from rank-2 approximation:")

    # Scores on first two PCs
    scores = U[:, :2] * S[:2]  # shape: (n_rows, 2)
    pc1_scores = scores[:, 0]

    # Reconstruct using rank-2 approximation
    P_approx = U[:, :2] @ np.diag(S[:2]) @ Vt[:2, :]

    # The extreme rows in PC1 space estimate the two components
    idx_max = np.argmax(pc1_scores)
    idx_min = np.argmin(pc1_scores)

    F_candidate_1 = P_approx[idx_max]
    F_candidate_2 = P_approx[idx_min]

    # Check validity (should be non-negative and sum to ~1)
    valid1 = (F_candidate_1 >= -0.01).all() and abs(F_candidate_1.sum() - 1) < 0.05
    valid2 = (F_candidate_2 >= -0.01).all() and abs(F_candidate_2.sum() - 1) < 0.05

    # Try to extrapolate to purer endpoints using the line structure
    # The "true" F_T and F_M are where π_f = 1 and π_f = 0
    # Under rank-2: p_f = F_M + π_f*(F_T - F_M)
    # p_f = mean(P) + (π_f - mean_π)*(F_T - F_M)
    # Direction: first right singular vector v1
    direction = Vt[0]  # 10-dim direction of main variation
    mean_p = P.mean(axis=0)

    # Try to find F_T, F_M as valid distributions at the ends of the segment
    # by finding the max/min extent along the direction that keeps values non-negative
    # p + t*direction >= 0 for all bands → t >= -p[b]/direction[b] for direction[b]>0 etc.
    def max_valid_extent(base, dir_vec):
        t_max = np.inf
        t_min = -np.inf
        for i in range(len(base)):
            if dir_vec[i] > 1e-9:
                t_max = min(t_max, (1.0 - base[i]) / dir_vec[i])
                t_min = max(t_min, -base[i] / dir_vec[i])
            elif dir_vec[i] < -1e-9:
                t_min = max(t_min, (1.0 - base[i]) / dir_vec[i])
                t_max = min(t_max, -base[i] / dir_vec[i])
        return t_min, t_max

    t_min, t_max = max_valid_extent(mean_p, direction)

    F_T_ext = mean_p + t_max * direction
    F_M_ext = mean_p + t_min * direction

    # Normalise
    F_T_ext = np.clip(F_T_ext, 0, None)
    F_M_ext = np.clip(F_M_ext, 0, None)
    if F_T_ext.sum() > 0: F_T_ext /= F_T_ext.sum()
    if F_M_ext.sum() > 0: F_M_ext /= F_M_ext.sum()

    # Which is targeted (higher cost) vs mass (lower cost)?
    # Use mean band index as proxy
    mean_band_1 = np.dot(F_T_ext, bands)
    mean_band_2 = np.dot(F_M_ext, bands)
    if mean_band_1 < mean_band_2:
        F_T_ext, F_M_ext = F_M_ext, F_T_ext
        mean_band_1, mean_band_2 = mean_band_2, mean_band_1

    print(f"\n  Extracted component distributions (F_T = higher cost, F_M = lower cost):")
    print(f"  {'Band':<14} {'F_T (targeted)':>16} {'F_M (mass)':>12}")
    for i, b in enumerate(bands):
        print(f"  {DAMAGE_BAND_LABELS[b]:<14} {F_T_ext[i]:>15.3f}  {F_M_ext[i]:>11.3f}")
    print(f"  {'Mean band':<14} {mean_band_1:>15.2f}  {mean_band_2:>11.2f}")
    print(f"  {'No-cost frac':<14} {F_T_ext[0]:>15.3f}  {F_M_ext[0]:>11.3f}")

    # Show implied π_f for each freq group
    print(f"\n  Implied mixing weights π_f (fraction from targeted component):")
    # Under rank-2: p_f ≈ F_M + π_f*(F_T - F_M)
    # π_f = dot(p_f - F_M, F_T - F_M) / dot(F_T - F_M, F_T - F_M)
    denom = np.dot(F_T_ext - F_M_ext, F_T_ext - F_M_ext)
    print(f"  {'Group':<20} {'n':>5} {'π_f (targeted)':>16}")
    for i, (lbl, n) in enumerate(zip(row_labels, ns)):
        if denom > 1e-10:
            pi = np.dot(P[i] - F_M_ext, F_T_ext - F_M_ext) / denom
            pi = np.clip(pi, 0, 1)
        else:
            pi = float('nan')
        print(f"  {lbl:<20} {n:>5} {pi:>15.3f}")

# ---------------------------------------------------------------------------
# Run: phishing, pooled across all sizes
# ---------------------------------------------------------------------------

print("Mixture model feasibility test")
print("Structural implication of 2-component model: rank(P) ≤ 2")
print("NOTE: pooling across size bands — results indicative, not conclusive\n")

phishing = attacked[attacked['disrupta'] == 6].copy()
run_mixture_test(phishing, "Phishing (disrupta=6), all sizes pooled")

# Also run for impersonation as a secondary check
impersonation = attacked[attacked['disrupta'] == 5].copy()
run_mixture_test(impersonation, "Impersonation (disrupta=5), all sizes pooled", min_per_freq=8)

# ---------------------------------------------------------------------------
# Run: phishing per size band
# ---------------------------------------------------------------------------

SIZE_LABELS = {1: 'Micro (1-9)', 2: 'Small (10-49)', 3: 'Medium (50-249)', 4: 'Large (250+)'}

print("\n\n" + "=" * 70)
print("PER SIZE BAND: Phishing (disrupta=6)")
print("Min 8 firms per freq group required; some bands may be sparse")
print("=" * 70)

for sz in sorted(attacked['sizeb'].dropna().unique()):
    sz_label = SIZE_LABELS.get(int(sz), str(sz))
    phishing_sz = attacked[(attacked['disrupta'] == 6) & (attacked['sizeb'] == sz)].copy()
    run_mixture_test(phishing_sz, f"Phishing — {sz_label}", min_per_freq=8)

# ---------------------------------------------------------------------------
# Run: phishing grouped by phisheng_bands (engagement level) instead of freq
#
# Motivation (src/estimation/phishing_count_validation.py): the freq-based
# F_M component (58.5% no-cost, mean band 2.0) matches almost exactly onto
# firms with phisheng_bands=='None' (57.1% no-cost, mean band 2.02) — a real,
# independent variable, not a freq artefact. Rerunning the rank-2 test with
# engagement level as the grouping variable checks whether engagement produces
# an even cleaner separation than freq did.
# ---------------------------------------------------------------------------

PHISHENG_LABELS = {
    1: 'None', 2: '1', 3: '2-3', 4: '4-5', 5: '6-10',
    6: '11-20', 7: '21-50', 8: '51-100', 9: '100+',
}
INVALID_COUNT = {-9, -1, 997, 999, -9.0, -1.0, 997.0, 999.0}

print("\n\n" + "=" * 70)
print("GROUPED BY ENGAGEMENT (phisheng_bands) INSTEAD OF FREQ")
print("Phishing (disrupta=6), all sizes pooled")
print("=" * 70)

phishing_eng = attacked[
    (attacked['disrupta'] == 6) &
    attacked['phisheng_bands'].notna() &
    ~attacked['phisheng_bands'].isin(INVALID_COUNT)
].copy()

run_mixture_test(
    phishing_eng, "Phishing — grouped by phisheng_bands (engagement level)",
    group_col='phisheng_bands', group_labels=PHISHENG_LABELS, min_per_freq=8
)

print("\nDone.")
