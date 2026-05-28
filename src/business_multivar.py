import pandas as pd
from proc import get_business_data
from scipy.stats import chi2_contingency
import numpy as np

business_data = get_business_data()

# Create freq_with_zero for completeness
business_data['freq_with_zero'] = business_data['freq'].copy()
business_data.loc[business_data['freq'].isna(), 'freq_with_zero'] = 0.0

print("=" * 80)
print("CONDITIONAL ANALYSIS: FREQ AND COST DISTRIBUTIONS BY SIZE")
print("BUSINESSES")
print("=" * 80)

# Size labels
size_labels = {
    1.0: "Micro (1-9)",
    2.0: "Small (10-49)",
    3.0: "Medium (50-249)",
    4.0: "Large (250+)"
}

# ============================================================================
# 1. ATTACK FREQUENCY DISTRIBUTION | SIZE (among attacked only)
# ============================================================================

print("\n" + "=" * 80)
print("1. ATTACK FREQUENCY DISTRIBUTION WITHIN EACH SIZE CATEGORY")
print("   (Among attacked businesses only)")
print("=" * 80)

attacked = business_data[
    (business_data['freq'].notna()) & 
    (~business_data['freq'].isin([997.0, 999.0]))
]

print(f"\nTotal attacked businesses with valid freq: {len(attacked)}")

# Absolute counts
print("\n--- ABSOLUTE COUNTS ---")
freq_by_size_counts = pd.crosstab(
    attacked['sizeb'],
    attacked['freq'],
    margins=True
)
print(freq_by_size_counts)

# Percentages within each size (row percentages)
print("\n--- PERCENTAGES WITHIN EACH SIZE (row %) ---")
freq_by_size_pct = pd.crosstab(
    attacked['sizeb'],
    attacked['freq'],
    normalize='index'
) * 100

# Add readable labels
freq_by_size_pct.index = freq_by_size_pct.index.map(lambda x: size_labels.get(x, x))
print(freq_by_size_pct.round(1))

# Test if frequency distribution differs by size (among attacked)
print("\n--- CHI-SQUARE TEST: Does attack frequency distribution differ by size? ---")
contingency = pd.crosstab(attacked['sizeb'], attacked['freq'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"χ²={chi2:.2f}, p={p_value:.4f}, df={dof}")

# Calculate high-frequency rate by size
print("\n--- HIGH FREQUENCY (weekly+) RATE BY SIZE ---")
attacked['high_freq'] = attacked['freq'].isin([4.0, 5.0, 6.0])
high_freq_by_size = attacked.groupby('sizeb')['high_freq'].agg(['sum', 'count', 'mean'])
high_freq_by_size['pct'] = high_freq_by_size['mean'] * 100
high_freq_by_size.index = high_freq_by_size.index.map(size_labels)
print(high_freq_by_size[['sum', 'count', 'pct']].round(1))


# ============================================================================
# 2. DAMAGE COST DISTRIBUTION | SIZE
# ============================================================================

print("\n\n" + "=" * 80)
print("2. DAMAGE COST DISTRIBUTION WITHIN EACH SIZE CATEGORY")
print("=" * 80)

has_damage = business_data[
    (business_data['damage_bands'].notna()) & 
    (~business_data['damage_bands'].isin([997.0, 999.0]))
]

print(f"\nTotal businesses with valid damage_bands: {len(has_damage)}")

# Absolute counts
print("\n--- ABSOLUTE COUNTS ---")
cost_by_size_counts = pd.crosstab(
    has_damage['sizeb'],
    has_damage['damage_bands'],
    margins=True
)
print(cost_by_size_counts)

# Percentages within each size (row percentages)
print("\n--- PERCENTAGES WITHIN EACH SIZE (row %) ---")
cost_by_size_pct = pd.crosstab(
    has_damage['sizeb'],
    has_damage['damage_bands'],
    normalize='index'
) * 100
cost_by_size_pct.index = cost_by_size_pct.index.map(lambda x: size_labels.get(x, x))
print(cost_by_size_pct.round(1))

# Calculate high-cost rate by size
print("\n--- HIGH COST (£500+, band 4+) RATE BY SIZE ---")
has_damage['high_cost'] = has_damage['damage_bands'] >= 4.0
high_cost_by_size = has_damage.groupby('sizeb')['high_cost'].agg(['sum', 'count', 'mean'])
high_cost_by_size['pct'] = high_cost_by_size['mean'] * 100
high_cost_by_size.index = high_cost_by_size.index.map(size_labels)
print(high_cost_by_size[['sum', 'count', 'pct']].round(1))


# ============================================================================
# 3. DAMAGE COST DISTRIBUTION | FREQUENCY (within each size)
# ============================================================================

print("\n\n" + "=" * 80)
print("3. DAMAGE COST DISTRIBUTION BY FREQUENCY, WITHIN EACH SIZE")
print("   (Tests if freq→cost relationship holds after controlling for size)")
print("=" * 80)

# Get businesses with both freq and damage data
both_data = business_data[
    (business_data['freq'].notna()) & 
    (~business_data['freq'].isin([997.0, 999.0])) &
    (business_data['damage_bands'].notna()) &
    (~business_data['damage_bands'].isin([997.0, 999.0]))
]

print(f"\nTotal businesses with both freq and damage_bands: {len(both_data)}")

for size_val, size_label in size_labels.items():
    subset = both_data[both_data['sizeb'] == size_val]
    
    if len(subset) < 10:
        print(f"\n--- {size_label}: INSUFFICIENT DATA (n={len(subset)}) ---")
        continue
    
    print(f"\n--- {size_label} (n={len(subset)}) ---")
    
    # Crosstab: freq × damage_bands
    cost_given_freq = pd.crosstab(
        subset['freq'],
        subset['damage_bands']
    )
    
    print("\nAbsolute counts:")
    print(cost_given_freq)
    
    # Row percentages (distribution of costs within each freq level)
    if len(cost_given_freq) > 0:
        cost_given_freq_pct = pd.crosstab(
            subset['freq'],
            subset['damage_bands'],
            normalize='index'
        ) * 100
        
        print("\nRow percentages (cost distribution within each freq):")
        print(cost_given_freq_pct.round(1))
        
        # Chi-square test if sufficient data
        if cost_given_freq.size >= 6 and cost_given_freq.min().min() >= 1:
            try:
                chi2, p_value, dof, expected = chi2_contingency(cost_given_freq)
                print(f"\nChi-square test: χ²={chi2:.2f}, p={p_value:.4f}, df={dof}")
            except:
                print("\nChi-square test: Cannot compute (sparse data)")
        else:
            print("\nChi-square test: Too sparse for reliable test")
    
    # High cost rate by frequency
    print("\nHigh cost (£500+) rate by frequency:")
    subset['high_cost'] = subset['damage_bands'] >= 4.0
    high_cost_by_freq = subset.groupby('freq')['high_cost'].agg(['sum', 'count', 'mean'])
    high_cost_by_freq['pct'] = high_cost_by_freq['mean'] * 100
    print(high_cost_by_freq[['sum', 'count', 'pct']].round(1))


# ============================================================================
# 4. SUMMARY STATISTICS
# ============================================================================

print("\n\n" + "=" * 80)
print("4. SUMMARY: KEY PATTERNS")
print("=" * 80)

print("\nA. Does size predict attack frequency among attacked businesses?")
print("   High-frequency attack rate (weekly+) by size:")
for size_val, size_label in size_labels.items():
    subset = attacked[attacked['sizeb'] == size_val]
    if len(subset) > 0:
        high_freq_rate = (subset['freq'] >= 4.0).mean() * 100
        print(f"   {size_label}: {high_freq_rate:.1f}%")

print("\nB. Does size predict damage costs?")
print("   High-cost rate (£500+) by size:")
for size_val, size_label in size_labels.items():
    subset = has_damage[has_damage['sizeb'] == size_val]
    if len(subset) > 0:
        high_cost_rate = (subset['damage_bands'] >= 4.0).mean() * 100
        print(f"   {size_label}: {high_cost_rate:.1f}%")

print("\nC. Does frequency predict costs WITHIN size categories?")
print("   (Average damage band by frequency, within each size)")
for size_val, size_label in size_labels.items():
    subset = both_data[both_data['sizeb'] == size_val]
    if len(subset) > 10:
        avg_by_freq = subset.groupby('freq')['damage_bands'].mean()
        print(f"\n   {size_label}:")
        for freq_val, avg_cost in avg_by_freq.items():
            print(f"     freq={freq_val}: avg damage_band={avg_cost:.2f}")
