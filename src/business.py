import pandas as pd
from proc import get_business_data
from scipy.stats import chi2_contingency

business_data = get_business_data()

print("=" * 80)
print("FREQUENCY vs SIZE MATRIX - BUSINESSES")
print("=" * 80)
print("\nRows: freq (attack frequency)")
print("Columns: sizeb (employee size bands)")
print("\n" + "=" * 80)

freq_size_matrix = pd.crosstab(
    business_data['freq'], 
    business_data['sizeb'], 
    margins=True,
    dropna=False
)
print(freq_size_matrix)

valid_data = business_data[
    (business_data['freq'].notna()) & 
    (business_data['sizeb'].notna()) &
    (~business_data['freq'].isin([997.0, 999.0]))
]
contingency = pd.crosstab(valid_data['freq'], valid_data['sizeb'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test: χ²={chi2:.2f}, p={p_value:.4f}, df={dof}")


print("\n" + "=" * 80)
print("FREQUENCY vs DAMAGE COST MATRIX - BUSINESSES")
print("=" * 80)
print("\nRows: freq (attack frequency)")
print("Columns: damage_bands (total damage cost bands)")
print("\n" + "=" * 80)

freq_cost_matrix = pd.crosstab(
    business_data['freq'], 
    business_data['damage_bands'], 
    margins=True,
    dropna=False
)
print(freq_cost_matrix)

valid_data = business_data[
    (business_data['freq'].notna()) & 
    (business_data['damage_bands'].notna()) &
    (~business_data['freq'].isin([997.0, 999.0])) &
    (~business_data['damage_bands'].isin([997.0, 999.0]))
]
contingency = pd.crosstab(valid_data['freq'], valid_data['damage_bands'])
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

valid_data = business_data[
    (business_data['sizeb'].notna()) & 
    (business_data['damage_bands'].notna()) &
    (~business_data['damage_bands'].isin([997.0, 999.0]))
]
contingency = pd.crosstab(valid_data['sizeb'], valid_data['damage_bands'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test: χ²={chi2:.2f}, p={p_value:.4f}, df={dof}")
