import pandas as pd
import numpy as np
from proc import get_charity_data
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm

charity_data = get_charity_data()

print("=" * 80)
print("ORDINAL REGRESSION: DAMAGE COSTS ~ INCOME + FREQUENCY + INCOME×FREQUENCY")
print("CHARITIES")
print("=" * 80)

# ============================================================================
# Prepare data for regression
# ============================================================================

# Filter to charities with valid data
reg_data = charity_data[
    (charity_data['income2'].notna()) &
    (charity_data['freq'].notna()) &
    (~charity_data['freq'].isin([997.0, 999.0])) &
    (charity_data['damage_bands'].notna()) &
    (~charity_data['damage_bands'].isin([997.0, 999.0]))
].copy()

print(f"\nSample size for regression: {len(reg_data)}")

# Convert to integers for ordinal model
reg_data['damage_bands_int'] = reg_data['damage_bands'].astype(int)
reg_data['income2_int'] = reg_data['income2'].astype(int)
reg_data['freq_int'] = reg_data['freq'].astype(int)

print(f"\nDamage bands range: {reg_data['damage_bands_int'].min()} to {reg_data['damage_bands_int'].max()}")
print(f"Income range: {reg_data['income2_int'].min()} to {reg_data['income2_int'].max()}")
print(f"Freq range: {reg_data['freq_int'].min()} to {reg_data['freq_int'].max()}")

print(f"\nUnique damage_bands values: {sorted(reg_data['damage_bands_int'].unique())}")

# ============================================================================
# Model 1: Main effects only (damage ~ income + freq)
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 1: Main Effects Only (damage ~ income + freq)")
print("=" * 80)

# Create design matrix WITHOUT constant
X1 = reg_data[['income2_int', 'freq_int']].copy()
y1 = reg_data['damage_bands_int']

try:
    model1 = OrderedModel(y1, X1, distr='logit')
    result1 = model1.fit(method='bfgs', disp=False)
    print(result1.summary())
    
    print("\n" + "=" * 80)
    print("INTERPRETATION OF MODEL 1 COEFFICIENTS")
    print("=" * 80)
    print("\nCoefficients are in log-odds units:")
    print("- Positive coefficient → higher odds of being in a higher damage band")
    print("- income2_int: Effect of moving up one income category")
    print("- freq_int: Effect of moving up one frequency category")
    
    # Calculate odds ratios
    print("\n" + "-" * 80)
    print("ODDS RATIOS (easier to interpret):")
    print("-" * 80)
    for param_name in ['income2_int', 'freq_int']:
        if param_name in result1.params.index:
            coef = result1.params[param_name]
            odds_ratio = np.exp(coef)
            print(f"\n{param_name}:")
            print(f"  Coefficient: {coef:.4f}")
            print(f"  Odds ratio: {odds_ratio:.4f}")
            print(f"  Interpretation: Moving up 1 category multiplies odds of higher damage by {odds_ratio:.2f}")
    
except Exception as e:
    print(f"Model 1 failed: {e}")
    import traceback
    traceback.print_exc()
    result1 = None

# ============================================================================
# Model 2: With interaction (damage ~ income + freq + income*freq)
# ============================================================================

print("\n\n" + "=" * 80)
print("MODEL 2: With Interaction (damage ~ income + freq + income×freq)")
print("=" * 80)

# Create interaction term
reg_data['income_freq_interaction'] = reg_data['income2_int'] * reg_data['freq_int']

# Create design matrix WITHOUT constant
X2 = reg_data[['income2_int', 'freq_int', 'income_freq_interaction']].copy()
y2 = reg_data['damage_bands_int']

try:
    model2 = OrderedModel(y2, X2, distr='logit')
    result2 = model2.fit(method='bfgs', disp=False)
    print(result2.summary())
    
    print("\n" + "=" * 80)
    print("INTERPRETATION OF MODEL 2 COEFFICIENTS")
    print("=" * 80)
    print("\nInteraction term interpretation:")
    print("- income_freq_interaction: How the freq effect changes as income increases")
    print("- Positive coefficient → freq effect is STRONGER for larger charities")
    print("- Negative coefficient → freq effect is WEAKER for larger charities")
    print("\nTotal effect of frequency at different income levels:")
    
    if 'freq_int' in result2.params.index and 'income_freq_interaction' in result2.params.index:
        freq_coef = result2.params['freq_int']
        interaction_coef = result2.params['income_freq_interaction']
        
        print(f"\nBase freq effect (at income=1, <£100k): {freq_coef:.4f}")
        for income in [1, 2, 3]:
            total_freq_effect = freq_coef + (interaction_coef * income)
            print(f"Total freq effect at income={income}: {total_freq_effect:.4f}")
    
except Exception as e:
    print(f"Model 2 failed: {e}")
    import traceback
    traceback.print_exc()
    result2 = None

# ============================================================================
# Model comparison
# ============================================================================

if result1 is not None and result2 is not None:
    print("\n\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    print(f"\nModel 1 (main effects only):")
    print(f"  Log-likelihood: {result1.llf:.2f}")
    print(f"  AIC: {result1.aic:.2f}")
    print(f"  BIC: {result1.bic:.2f}")
    
    print(f"\nModel 2 (with interaction):")
    print(f"  Log-likelihood: {result2.llf:.2f}")
    print(f"  AIC: {result2.aic:.2f}")
    print(f"  BIC: {result2.bic:.2f}")
    
    # Likelihood ratio test
    lr_stat = 2 * (result2.llf - result1.llf)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    
    print(f"\n" + "=" * 80)
    print("LIKELIHOOD RATIO TEST")
    print("=" * 80)
    print(f"Tests: Is the interaction term significant?")
    print(f"\nLR statistic: {lr_stat:.2f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("\n→ Interaction IS significant (p<0.05)")
        print("→ The frequency effect DEPENDS ON income")
        print("→ Model 2 (with interaction) is preferred")
    else:
        print("\n→ Interaction NOT significant (p≥0.05)")
        print("→ The frequency effect is similar across all income levels")
        print("→ Model 1 (main effects) is sufficient")
    
    # AIC comparison
    if result2.aic < result1.aic:
        print(f"\nAIC also favors Model 2 (lower is better)")
    else:
        print(f"\nAIC favors Model 1 (lower is better)")

# ============================================================================
# Comparison with business results
# ============================================================================

print("\n\n" + "=" * 80)
print("COMPARISON: CHARITIES vs BUSINESSES")
print("=" * 80)

print("""
BUSINESSES:
- Size effect: Strong (coef=0.35, p<0.001)
- Freq effect: Depends on size (interaction p=0.038)
- Interaction: Positive (freq matters MORE for large orgs)
- Sample: n=983

CHARITIES:
- Income effect: See above
- Freq effect: See above  
- Interaction: See above
- Sample: n={n}

Key question: Do charities show the same interaction pattern as businesses?
""".format(n=len(reg_data)))

# ============================================================================
# Summary interpretation
# ============================================================================

print("\n" + "=" * 80)
print("WHAT THIS TELLS US ABOUT CHARITIES")
print("=" * 80)

if result1 is not None:
    print("\nFrom Model 1 (main effects):")
    print("- Does income independently predict costs?")
    print("- Does frequency independently predict costs?")

if result2 is not None:
    print("\nFrom Model 2 (with interaction):")
    print("- Does the freq→cost relationship vary by income level?")
    print("- Do larger charities show the same compounding effect as large businesses?")

print("\n" + "=" * 80)
