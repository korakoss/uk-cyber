import pandas as pd
from proc import get_business_data
from scipy.stats import chi2_contingency

business_data = get_business_data()

# Add a new column: freq_with_zero
# freq is NaN for non-attacked (Cybercrime_all = 0), so we set those to 0
# Keep 997/999 (don't know/refused) as-is for now
business_data['freq_with_zero'] = business_data['freq'].copy()
business_data.loc[business_data['freq'].isna(), 'freq_with_zero'] = 0.0

print("=" * 80)
print("FREQUENCY vs SIZE MATRIX - BUSINESSES (with freq=0 for not attacked)")
print("=" * 80)
print("\nRows: freq_with_zero (0=not attacked, 1-6=attack frequency)")
print("Columns: sizeb (employee size bands)")
print("\n" + "=" * 80)

freq_size_matrix = pd.crosstab(
    business_data['freq_with_zero'], 
    business_data['sizeb'], 
    margins=True,
    dropna=False
)
print(freq_size_matrix)

# Chi-square test excluding don't know/refused
valid_data = business_data[
    (business_data['freq_with_zero'].notna()) & 
    (business_data['sizeb'].notna()) &
    (~business_data['freq_with_zero'].isin([997.0, 999.0]))
]
contingency = pd.crosstab(valid_data['freq_with_zero'], valid_data['sizeb'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test: χ²={chi2:.2f}, p={p_value:.4f}, df={dof}")


print("\n" + "=" * 80)
print("FREQUENCY vs DAMAGE COST MATRIX - BUSINESSES (with freq=0)")
print("=" * 80)
print("\nRows: freq_with_zero (0=not attacked, 1-6=attack frequency)")
print("Columns: damage_bands (total damage cost bands)")
print("\n" + "=" * 80)

freq_cost_matrix = pd.crosstab(
    business_data['freq_with_zero'], 
    business_data['damage_bands'], 
    margins=True,
    dropna=False
)
print(freq_cost_matrix)

# Chi-square test
valid_data = business_data[
    (business_data['freq_with_zero'].notna()) & 
    (business_data['damage_bands'].notna()) &
    (~business_data['freq_with_zero'].isin([997.0, 999.0])) &
    (~business_data['damage_bands'].isin([997.0, 999.0]))
]
contingency = pd.crosstab(valid_data['freq_with_zero'], valid_data['damage_bands'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test: χ²={chi2:.2f}, p={p_value:.4f}, df={dof}")


print("\n" + "=" * 80)
print("SIZE vs DAMAGE COST MATRIX - BUSINESSES")
print("=" * 80)
print("\nRows: sizeb (employee size bands)")
print("Columns: damage_bands (total damage cost bands)")
print("\n" + "=" * 80)

size_cost_matrix = pd.crosstab(
    business_data['sizeb'], 
    business_data['damage_bands'], 
    margins=True,
    dropna=False
)
print(size_cost_matrix)

# Chi-square test
valid_data = business_data[
    (business_data['sizeb'].notna()) & 
    (business_data['damage_bands'].notna()) &
    (~business_data['damage_bands'].isin([997.0, 999.0]))
]
contingency = pd.crosstab(valid_data['sizeb'], valid_data['damage_bands'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test: χ²={chi2:.2f}, p={p_value:.4f}, df={dof}")


print("\n" + "=" * 80)
print("SUMMARY: What freq=0 represents")
print("=" * 80)
print("freq_with_zero = 0.0: Not attacked (original freq was NaN)")
print("freq_with_zero = 1.0-6.0: Attack frequency categories (original freq values)")
print("freq_with_zero = 997.0: Don't know (excluded from chi-square)")
print("freq_with_zero = 999.0: Refused (excluded from chi-square)")
