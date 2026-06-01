import pandas as pd
from proc import get_business_data

b_data = get_business_data()

print("\nBUSINESSES \n")

print("'sizeb' meanings:")
print("1.0 -> 1-9 ppl")
print("2.0 -> 10-49 ppl")
print("3.0 -> 50-249 ppl")
print("4.0 -> 250+ ppl")
print("\n\n")


print("Experiencing attack")
print(pd.crosstab(
    b_data["sizeb"],
    b_data["Cybercrime_all"],
    margins=True,
    normalize="index",
    dropna=False,
))


att_b_data = b_data[b_data["Cybercrime_all"] == 1.0]

print("\nCrosstabs on attacked businesses:")

print("\nFreq vs size")
print(pd.crosstab(
    att_b_data["sizeb"],
    att_b_data["freq"],
    margins=True,
    dropna=False,
))

print("\nFreq vs size (normed for sizes)")
print(pd.crosstab(
    att_b_data["sizeb"],
    att_b_data["freq"],
    margins=False,
    dropna=False,
    normalize="index"
))

print("\nFreq vs size (normed, non-NaN)")
print(pd.crosstab(
    att_b_data["sizeb"],
    att_b_data["freq"],
    margins=False,
    dropna=True,
    normalize="index"
))

print("\n\n")
print("Damage vs size")
print(pd.crosstab(
    att_b_data["sizeb"],
    att_b_data["damage_bands"],
    margins=True,
    dropna=False,
))

print("\nDamage vs size (normed for sizes)")
print(pd.crosstab(
    att_b_data["sizeb"],
    att_b_data["damage_bands"],
    margins=False,
    dropna=False,
    normalize="index"
))

print("\nDamage vs size (normed, non-NaN)")
print(pd.crosstab(
    att_b_data["sizeb"],
    att_b_data["damage_bands"],
    margins=False,
    dropna=True,
    normalize="index"
))

