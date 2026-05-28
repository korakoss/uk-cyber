import pandas as pd
from proc import get_charity_data
from scipy.stats import chi2_contingency

charity_data = get_charity_data()

# Add a new column: freq_with_zero
# freq is NaN for non-attacked, so we set those to 0
charity_data['freq_with_zero'] = charity_data['freq'].copy()
charity_data.loc[charity_data['freq'].isna(), 'freq_with_zero'] = 0.0

print("=" * 80)
print("FREQUENCY vs INCOME MATRIX - CHARITIES (with freq=0 for not attacked)")
print("=" * 80)
print("\nRows: freq_with_zero (0=not attacked, 1-6=attack frequency)")
print("Columns: income2 (charity income bands)")
print("\n" + "=" * 80)

freq_income_matrix = pd.crosstab(
    charity_data['freq_with_zero'], 
    charity_data['income2'], 
    margins=True,
    dropna=False
)
print(freq_income_matrix)

# Chi-square test excluding don't know/refused
valid_data = charity_data[
    (charity_data['freq_with_zero'].notna()) & 
    (charity_data['income2'].notna()) &
    (~charity_data['freq_with_zero'].isin([997.0, 999.0]))
]
contingency = pd.crosstab(valid_data['freq_with_zero'], valid_data['income2'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test: χ²={chi2:.2f}, p={p_value:.4f}, df={dof}")


print("\n" + "=" * 80)
print("FREQUENCY vs DAMAGE COST MATRIX - CHARITIES (with freq=0)")
print("=" * 80)
print("\nRows: freq_with_zero (0=not attacked, 1-6=attack frequency)")
print("Columns: damage_bands (total damage cost bands)")
print("\n" + "=" * 80)

freq_cost_matrix = pd.crosstab(
    charity_data['freq_with_zero'], 
    charity_data['damage_bands'], 
    margins=True,
    dropna=False
)
print(freq_cost_matrix)

# Chi-square test
valid_data = charity_data[
    (charity_data['freq_with_zero'].notna()) & 
    (charity_data['damage_bands'].notna()) &
    (~charity_data['freq_with_zero'].isin([997.0, 999.0])) &
    (~charity_data['damage_bands'].isin([997.0, 999.0]))
]
contingency = pd.crosstab(valid_data['freq_with_zero'], valid_data['damage_bands'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test: χ²={chi2:.2f}, p={p_value:.4f}, df={dof}")


print("\n" + "=" * 80)
print("INCOME vs DAMAGE COST MATRIX - CHARITIES")
print("=" * 80)
print("\nRows: income2 (charity income bands)")
print("Columns: damage_bands (total damage cost bands)")
print("\n" + "=" * 80)

income_cost_matrix = pd.crosstab(
    charity_data['income2'], 
    charity_data['damage_bands'], 
    margins=True,
    dropna=False
)
print(income_cost_matrix)

# Chi-square test
valid_data = charity_data[
    (charity_data['income2'].notna()) & 
    (charity_data['damage_bands'].notna()) &
    (~charity_data['damage_bands'].isin([997.0, 999.0]))
]
contingency = pd.crosstab(valid_data['income2'], valid_data['damage_bands'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test: χ²={chi2:.2f}, p={p_value:.4f}, df={dof}")


print("\n" + "=" * 80)
print("SUMMARY: What freq=0 represents")
print("=" * 80)
print("freq_with_zero = 0.0: Not attacked (original freq was NaN)")
print("freq_with_zero = 1.0-6.0: Attack frequency categories (original freq values)")
print("freq_with_zero = 997.0: Don't know (excluded from chi-square)")
print("freq_with_zero = 999.0: Refused (excluded from chi-square)")
print("\nIncome bands:")
print("income2 = 1.0: £0 to <£100,000")
print("income2 = 2.0: £100,000 to <£500,000")
print("income2 = 3.0: £500,000+")
