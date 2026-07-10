"""
Does cyber preparedness explain the freq -> no-cost pattern?

Motivating question: the no-cost share among attacked firms climbs with attack
frequency (38.5% at freq=1 to 58.7% at freq=4, pooled - see consistency_checks.py
DECOMPOSITION section), while cost conditional on having any cost stays flat.
Why would more-frequently-attacked firms be more likely to fend off every single
incident for free? One candidate explanation: better-prepared firms deflect more
low-effort/automated attacks at zero cost, and also happen to face (or generate)
more recorded incidents (e.g. more monitoring = more attacks noticed/logged).

Preparedness variables (Q31 rules1-20, technical controls; Q33B trained, staff
training) have full coverage among attacked firms - no routing-related
missingness, unlike policy/review/strategy/corprisk/update which are only
asked to a subset and are noted but not used as primary variables here.

Structure: simple bivariate looks first (preparedness x freq, x no-cost,
x size, run separately), then a multivariate logistic regression of no-cost
on freq + size + preparedness jointly, since size is known to correlate with
both freq (Step 3 finding) and (plausibly) preparedness (bigger firms have
more resources) - any bivariate preparedness-freq or preparedness-no-cost
relationship could just be riding on size.
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD, RULES_COLNAMES

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 15)

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

def freq_group(f):
    f = int(f)
    return f if f <= 4 else 5

attacked['freq_group'] = attacked['freq'].apply(freq_group)
attacked['no_cost'] = (attacked['damage_bands'] == 1).astype(int)

FREQ_LABELS = {1: '1 (once)', 2: '2', 3: '3', 4: '4', 5: '5+'}
SIZE_LABELS = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}

# Real technical controls only - exclude rules10 ("don't know") and
# rules11 ("none of these"), which are meta-responses, not controls.
REAL_CONTROLS = [c for c in RULES_COLNAMES if c not in ('rules10', 'rules11')]
attacked['num_controls'] = attacked[REAL_CONTROLS].sum(axis=1)
attacked['trained_yes'] = (attacked['trained'] == 1).astype(float)
attacked.loc[attacked['trained'].isin([997, 997.0]), 'trained_yes'] = np.nan

print("=" * 80)
print(f"Coverage check: n={len(attacked)} attacked firms")
print(f"  num_controls (0-{len(REAL_CONTROLS)}): missing = {attacked['num_controls'].isna().sum()}")
print(f"  trained_yes: missing = {attacked['trained_yes'].isna().sum()} "
      f"(don't-know responses excluded, not imputed)")
print("=" * 80)

# ---------------------------------------------------------------------------
# STEP 1: simple bivariate looks
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 1a: preparedness (num_controls, trained%) BY FREQUENCY GROUP")
print("=" * 80)
g = attacked.groupby('freq_group').agg(
    n=('num_controls', 'size'),
    mean_controls=('num_controls', 'mean'),
    trained_pct=('trained_yes', lambda x: 100 * x.mean()),
)
g.index = g.index.map(FREQ_LABELS)
print(g.round(2))

print("\n" + "=" * 80)
print("STEP 1b: preparedness BY no-cost vs nonzero-cost outcome")
print("=" * 80)
g = attacked.groupby('no_cost').agg(
    n=('num_controls', 'size'),
    mean_controls=('num_controls', 'mean'),
    trained_pct=('trained_yes', lambda x: 100 * x.mean()),
)
g.index = g.index.map({0: 'Nonzero cost', 1: 'No cost'})
print(g.round(2))

print("\n" + "=" * 80)
print("STEP 1c: preparedness BY size band (sanity check - expect scaling with size)")
print("=" * 80)
g = attacked.groupby('sizeb').agg(
    n=('num_controls', 'size'),
    mean_controls=('num_controls', 'mean'),
    trained_pct=('trained_yes', lambda x: 100 * x.mean()),
)
g.index = g.index.map(SIZE_LABELS)
print(g.round(2))

print("\n" + "=" * 80)
print("STEP 1d: freq_group BY size band (does the freq distribution vary with size,")
print("as needed to explain a confound between preparedness and freq via size)")
print("=" * 80)
print(pd.crosstab(attacked['sizeb'].map(SIZE_LABELS), attacked['freq_group'].map(FREQ_LABELS),
                   normalize='index').round(3) * 100)

print("\n" + "=" * 80)
print("STEP 1e: no-cost share BY size band (does size alone predict no-cost)")
print("=" * 80)
g = attacked.groupby('sizeb').agg(n=('no_cost', 'size'), no_cost_pct=('no_cost', lambda x: 100 * x.mean()))
g.index = g.index.map(SIZE_LABELS)
print(g.round(1))

# ---------------------------------------------------------------------------
# STEP 2: individual control breakdown (which specific controls matter most,
# if any, for the no-cost outcome) - unweighted % Yes among no-cost vs
# nonzero-cost firms, per control
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 2: individual controls - % with control=Yes, no-cost vs nonzero-cost firms")
print("=" * 80)
rows = []
for c in REAL_CONTROLS:
    pct_nocost = 100 * attacked.loc[attacked['no_cost'] == 1, c].mean()
    pct_nonzero = 100 * attacked.loc[attacked['no_cost'] == 0, c].mean()
    rows.append({'control': c, 'pct_yes_no_cost_firms': pct_nocost,
                 'pct_yes_nonzero_cost_firms': pct_nonzero,
                 'diff': pct_nocost - pct_nonzero})
control_df = pd.DataFrame(rows).sort_values('diff', ascending=False)
print(control_df.round(1).to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 3: multivariate - logistic regression of no_cost on freq + size +
# preparedness jointly, to see whether freq's association with no-cost
# survives controlling for size and preparedness, and vice versa.
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 3: multivariate logistic regression, P(no_cost=1 | attacked)")
print("Predictors: freq_group (ordinal 1-5), sizeb (ordinal 1-4), num_controls,")
print("trained_yes. Standardized coefficients not used - report raw + odds ratios.")
print("=" * 80)

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False

reg_data = attacked.dropna(subset=['num_controls', 'trained_yes', 'freq_group', 'sizeb', 'no_cost']).copy()
print(f"\nRegression sample: n={len(reg_data)} (complete cases on all predictors)")

if HAVE_STATSMODELS:
    model = smf.logit(
        'no_cost ~ freq_group + sizeb + num_controls + trained_yes',
        data=reg_data
    ).fit(disp=0)
    print(model.summary())
    print("\nOdds ratios:")
    print(np.exp(model.params).round(3))
else:
    print("statsmodels not installed - falling back to numpy-based logistic regression (no SEs/p-values).")
    from numpy.linalg import lstsq

    X = reg_data[['freq_group', 'sizeb', 'num_controls', 'trained_yes']].values.astype(float)
    X = np.column_stack([np.ones(len(X)), X])
    y = reg_data['no_cost'].values.astype(float)

    beta = np.zeros(X.shape[1])
    for _ in range(100):
        z = X @ beta
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p)
        W = np.clip(W, 1e-6, None)
        grad = X.T @ (y - p)
        H = -(X.T * W) @ X
        beta -= np.linalg.solve(H, grad)

    names = ['intercept', 'freq_group', 'sizeb', 'num_controls', 'trained_yes']
    print("\nCoefficients (Newton-Raphson logistic regression, no SEs):")
    for n, b in zip(names, beta):
        print(f"  {n:<14} {b:>8.4f}   odds ratio = {np.exp(b):.3f}")

# ---------------------------------------------------------------------------
# STEP 4: preparedness x cost, CONTROLLING FOR SIZE explicitly.
# Two things "cost" can mean here, both checked:
#   (a) probability of any cost at all (no_cost outcome) - within each size
#       band separately, not pooled with an assumed-common slope.
#   (b) severity conditional on having any cost (mean damage_band among
#       nonzero-cost firms only) - tests whether preparedness relates to how
#       BAD the incident is, not just whether one was logged as costing
#       anything, mirroring the no-cost/conditional-mean split done earlier
#       for the freq analysis.
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 4a: no-cost % by preparedness level (median split), WITHIN each size band")
print("=" * 80)
median_controls = attacked['num_controls'].median()
attacked['prep_level'] = np.where(attacked['num_controls'] > median_controls, 'High prep', 'Low prep')
print(f"(Median split at num_controls > {median_controls} = High prep; overall n Low={sum(attacked['prep_level']=='Low prep')}, High={sum(attacked['prep_level']=='High prep')})\n")

tab = attacked.groupby(['sizeb', 'prep_level']).agg(
    n=('no_cost', 'size'), no_cost_pct=('no_cost', lambda x: 100 * x.mean())
).reset_index()
tab['sizeb'] = tab['sizeb'].map(SIZE_LABELS)
print(tab.pivot(index='sizeb', columns='prep_level', values=['n', 'no_cost_pct']).round(1))

print("\n" + "=" * 80)
print("STEP 4b: severity CONDITIONAL on nonzero cost, by preparedness level, WITHIN size band")
print("(mean damage_band among firms with damage_bands > 1 only)")
print("=" * 80)
nonzero = attacked[attacked['no_cost'] == 0]
tab2 = nonzero.groupby(['sizeb', 'prep_level']).agg(
    n=('damage_bands', 'size'), mean_severity=('damage_bands', 'mean')
).reset_index()
tab2['sizeb'] = tab2['sizeb'].map(SIZE_LABELS)
print(tab2.pivot(index='sizeb', columns='prep_level', values=['n', 'mean_severity']).round(2))

print("\n" + "=" * 80)
print("STEP 4c: per-size-band logistic regression, P(no_cost=1) ~ num_controls")
print("(no pooling across sizes - lets the slope vary freely by size band)")
print("=" * 80)
if HAVE_STATSMODELS:
    for size, name in SIZE_LABELS.items():
        sub = attacked[attacked['sizeb'] == size].dropna(subset=['num_controls', 'no_cost'])
        if len(sub) < 30 or sub['no_cost'].nunique() < 2:
            print(f"\n{name}: n={len(sub)}, skipping (too small or no variation)")
            continue
        m = smf.logit('no_cost ~ num_controls', data=sub).fit(disp=0)
        coef = m.params['num_controls']
        p = m.pvalues['num_controls']
        print(f"\n{name} (n={len(sub)}): num_controls coef={coef:.4f}, "
              f"OR={np.exp(coef):.3f}, p={p:.3f}")

print("\n" + "=" * 80)
print("STEP 4d: OLS, severity (damage_bands | nonzero cost) ~ num_controls + sizeb")
print("Does preparedness relate to HOW BAD the incident is, given one happened?")
print("=" * 80)
if HAVE_STATSMODELS:
    sev_data = nonzero.dropna(subset=['num_controls', 'sizeb', 'damage_bands'])
    m_sev = smf.ols('damage_bands ~ num_controls + sizeb', data=sev_data).fit()
    print(m_sev.summary())

# ---------------------------------------------------------------------------
# STEP 5: full shape of the nonzero-cost damage_bands distribution, not just
# the mean, broken out separately by freq_group, sizeb, and prep_level.
# Means can hide shape differences (e.g. same mean via different tails).
# ---------------------------------------------------------------------------

DAMAGE_BAND_LABELS = {
    2: '<£100', 3: '£100-£500', 4: '£500-£1k', 5: '£1k-£5k',
    6: '£5k-£10k', 7: '£10k-£20k', 8: '£20k-£50k', 9: '£50k-£100k', 10: '£100k-£500k'
}
nz_bands = sorted(DAMAGE_BAND_LABELS.keys())

def print_distribution(subset, group_col, group_labels, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    groups = sorted(subset[group_col].dropna().unique())
    ns = {g: len(subset[subset[group_col] == g]) for g in groups}
    header = f"  {'Band':<12}" + "".join(
        f"  {str(group_labels.get(g, g)) + ' (n=' + str(ns[g]) + ')':>20}" for g in groups)
    print(header)
    for b in nz_bands:
        row = f"  {DAMAGE_BAND_LABELS[b]:<12}"
        for g in groups:
            grp = subset[subset[group_col] == g]
            pct = 100 * (grp['damage_bands'] == b).mean() if len(grp) > 0 else 0
            row += f"  {pct:>19.1f}%"
        print(row)
    print(f"  {'Mean band':<12}" + "".join(
        f"  {subset[subset[group_col]==g]['damage_bands'].mean():>19.2f} " for g in groups))
    print(f"\n  Cumulative from top: P(damage_bands >= b)")
    print(header)
    for b in nz_bands:
        row = f"  {DAMAGE_BAND_LABELS[b]:<12}"
        for g in groups:
            grp = subset[subset[group_col] == g]
            pct = 100 * (grp['damage_bands'] >= b).mean() if len(grp) > 0 else 0
            row += f"  {pct:>19.1f}%"
        print(row)

print_distribution(nonzero, 'freq_group', FREQ_LABELS,
                    "STEP 5a: nonzero-cost damage_bands distribution BY FREQUENCY GROUP")
print_distribution(nonzero, 'sizeb', SIZE_LABELS,
                    "STEP 5b: nonzero-cost damage_bands distribution BY SIZE BAND")
print_distribution(nonzero, 'prep_level', {'Low prep': 'Low prep', 'High prep': 'High prep'},
                    "STEP 5c: nonzero-cost damage_bands distribution BY PREPAREDNESS LEVEL")

# Joint size x prep_level view - only report where cells are large enough
print("\n" + "=" * 80)
print("STEP 5d: mean severity and P(>=£20k) by size x prep_level jointly (cell n shown)")
print("=" * 80)
joint = nonzero.groupby(['sizeb', 'prep_level']).agg(
    n=('damage_bands', 'size'),
    mean_band=('damage_bands', 'mean'),
    p_ge_20k=('damage_bands', lambda x: 100 * (x >= 8).mean()),
).reset_index()
joint['sizeb'] = joint['sizeb'].map(SIZE_LABELS)
print(joint.round(2).to_string(index=False))
print("\n  (Cells with n<15 are noisy - treat as indicative only, esp. Large.)")

# ---------------------------------------------------------------------------
# STEP 6: direct correlation, freq x preparedness - quantify Step 1a properly
# rather than eyeballing group means. Also check whether raw freq (not just
# the collapsed freq_group) or controlling for size changes the answer.
# ---------------------------------------------------------------------------

from scipy.stats import spearmanr, pearsonr

print("\n" + "=" * 80)
print("STEP 6: freq x preparedness correlation (direct)")
print("=" * 80)

corr_data = attacked.dropna(subset=['num_controls', 'trained_yes', 'freq', 'sizeb'])

for freq_col in ['freq', 'freq_group']:
    for prep_col in ['num_controls', 'trained_yes']:
        rho, p_s = spearmanr(corr_data[freq_col], corr_data[prep_col])
        r, p_p = pearsonr(corr_data[freq_col], corr_data[prep_col])
        print(f"  {freq_col:<12} x {prep_col:<14} Spearman rho={rho:+.4f} (p={p_s:.3f})   "
              f"Pearson r={r:+.4f} (p={p_p:.3f})")

print("\n  Partial correlation, controlling for size (residualize both freq and")
print("  num_controls on sizeb via OLS, correlate the residuals):")
if HAVE_STATSMODELS:
    resid_freq = smf.ols('freq ~ sizeb', data=corr_data).fit().resid
    resid_prep = smf.ols('num_controls ~ sizeb', data=corr_data).fit().resid
    rho_partial, p_partial = spearmanr(resid_freq, resid_prep)
    print(f"  freq x num_controls | sizeb: Spearman rho={rho_partial:+.4f} (p={p_partial:.3f})")

# ---------------------------------------------------------------------------
# STEP 7: mirror of Step 4a/4c, but stratified by FREQUENCY GROUP instead of
# size - within each freq group, does preparedness still relate to no-cost?
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 7a: no-cost % by preparedness level (median split), WITHIN each freq group")
print("=" * 80)
tab = attacked.groupby(['freq_group', 'prep_level']).agg(
    n=('no_cost', 'size'), no_cost_pct=('no_cost', lambda x: 100 * x.mean())
).reset_index()
tab['freq_group'] = tab['freq_group'].map(FREQ_LABELS)
print(tab.pivot(index='freq_group', columns='prep_level', values=['n', 'no_cost_pct']).round(1))

print("\n" + "=" * 80)
print("STEP 7b: per-freq-group logistic regression, P(no_cost=1) ~ num_controls")
print("(no pooling across freq groups - lets the slope vary freely)")
print("=" * 80)
if HAVE_STATSMODELS:
    for fg, name in FREQ_LABELS.items():
        sub = attacked[attacked['freq_group'] == fg].dropna(subset=['num_controls', 'no_cost'])
        if len(sub) < 30 or sub['no_cost'].nunique() < 2:
            print(f"\n{name}: n={len(sub)}, skipping (too small or no variation)")
            continue
        m = smf.logit('no_cost ~ num_controls', data=sub).fit(disp=0)
        coef = m.params['num_controls']
        p = m.pvalues['num_controls']
        print(f"\n{name} (n={len(sub)}): num_controls coef={coef:.4f}, "
              f"OR={np.exp(coef):.3f}, p={p:.3f}")

print("\n" + "=" * 80)
print("STEP 7c: full joint model, no_cost ~ num_controls * freq_group (interaction)")
print("Tests whether the preparedness effect size itself varies by freq group,")
print("controlling for size throughout.")
print("=" * 80)
if HAVE_STATSMODELS:
    inter_data = attacked.dropna(subset=['num_controls', 'freq_group', 'sizeb', 'no_cost'])
    m_inter = smf.logit('no_cost ~ num_controls * C(freq_group) + sizeb', data=inter_data).fit(disp=0)
    print(m_inter.summary())

print("\nDone.")
