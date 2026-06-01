import pandas as pd
from pandas.core.reshape.api import crosstab
from proc import get_business_data
from scipy.stats import chi2_contingency, contingency
import numpy as np


b_data = get_business_data()

b_data = b_data[b_data["damage_bands"] < 100] # to filter out the weird 9** categories

print("FREQ V. COST MATRICES WITHIN SIZE BANDS\n\n")


for band in [1.0, 2.0, 3.0, 4.0]:
    print("\n\nMatrix for class ", band, ":\n")

    band_dataset = b_data[b_data["sizeb"] == band]
    crosstab = pd.crosstab(
        band_dataset["freq"],
        band_dataset["damage_bands"],
        margins=False,
        dropna=True
    )
    print(crosstab)

    print("\nCorrelation (Pearson):", band_dataset["freq"].corr(band_dataset["damage_bands"], method="pearson"))
    print("\nCorrelation (Spearman):", band_dataset["freq"].corr(band_dataset["damage_bands"], method="spearman"))
    chi2, pval, dof,_ = chi2_contingency(crosstab)
    print("Chi2 test results:")
    print("\tchi^2=", chi2)
    print("\tp-val=", pval)    
    print("\tdegs=", dof)
    n = crosstab.sum().sum()
    cramerv = np.sqrt(chi2 / (n *( min(crosstab.shape) -1 )))
    print("Cramér's V:", cramerv)

