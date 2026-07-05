"""
Prevalence analysis: fraction of businesses attacked, by size band.

"Attacked" = firm reported freq in {1,2,3,4,5,6} (i.e. at least one incident).
Special codes (997=don't know, 999=refused) and missing are treated as not-attacked
for the base rate; sensitivity to this choice is noted.

Analysis:
1. Empirical attacked/total by size band, weighted and unweighted
2. Logistic regression in log(size_midpoint) — check whether a smooth parametric
   curve fits the 4 data points
3. Note implications for the parametric vs. empirical choice in estimation
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid
sys.path.insert(0, '.')
from proc import get_business_data

df = get_business_data()

VALID_FREQ = {1, 2, 3, 4, 5, 6}
SIZE_LABELS = {1: 'Micro (1-9)', 2: 'Small (10-49)', 3: 'Medium (50-249)', 4: 'Large (250+)'}

# Size band midpoints: geometric mean of band endpoints
# Large: assume 500 as approximate typical large firm (ONS distribution is right-skewed;
# using geometric midpoint of 250-1000 as a reasonable central value)
SIZE_MIDPOINTS = {1: 3.0, 2: 22.1, 3: 111.8, 4: 500.0}

df['attacked'] = df['freq'].isin(VALID_FREQ).astype(int)

# ---------------------------------------------------------------------------
# 1. Empirical prevalence by size band
# ---------------------------------------------------------------------------

print("=" * 65)
print("PREVALENCE: fraction of businesses attacked, by size band")
print("=" * 65)

sizes = sorted(df['sizeb'].dropna().unique())

rows = []
for sz in sizes:
    sub = df[df['sizeb'] == sz]
    n_total = len(sub)
    n_attacked = sub['attacked'].sum()

    # Weighted prevalence
    w = sub['weight'].values
    prev_w = (sub['attacked'].values * w).sum() / w.sum()

    prev_uw = n_attacked / n_total

    rows.append(dict(
        size=sz,
        label=SIZE_LABELS[sz],
        midpoint=SIZE_MIDPOINTS[sz],
        n_total=n_total,
        n_attacked=int(n_attacked),
        prev_unweighted=prev_uw,
        prev_weighted=prev_w,
    ))
    print(f"  {SIZE_LABELS[sz]:<20} n={n_total:>5}  attacked={int(n_attacked):>5}"
          f"  unweighted={prev_uw:.3f}  weighted={prev_w:.3f}")

results = pd.DataFrame(rows)

# Sensitivity: how many firms had freq in {997, 999} or missing?
missing_or_dk = df['freq'].isna() | df['freq'].isin({997, 999, 997.0, 999.0})
n_missing = missing_or_dk.sum()
n_total = len(df)
print(f"\n  Note: {n_missing}/{n_total} firms ({n_missing/n_total:.1%}) have freq missing/DK/refused")
print(f"  These are counted as 'not attacked' in the base rate above.")

# ---------------------------------------------------------------------------
# 2. Logistic model: P(attacked) = sigmoid(a + b * log(midpoint))
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("LOGISTIC FIT: P(attacked) = sigmoid(a + b * log(size_midpoint))")
print("=" * 65)

log_mp = np.log(results['midpoint'].values)

def neg_log_lik(params, log_x, y_prop, n):
    a, b = params
    p = expit(a + b * log_x)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = (y_prop * n * np.log(p) + (1 - y_prop) * n * np.log(1 - p)).sum()
    return -ll

for variant, col in [("Unweighted", "prev_unweighted"), ("Weighted", "prev_weighted")]:
    y = results[col].values
    n = results['n_total'].values

    res = minimize(neg_log_lik, x0=[0.0, 0.5], args=(log_mp, y, n), method='Nelder-Mead')
    a_hat, b_hat = res.x
    fitted = expit(a_hat + b_hat * log_mp)
    residuals = y - fitted
    rmse = np.sqrt((residuals**2).mean())

    print(f"\n  {variant}:")
    print(f"    Fitted: a={a_hat:.3f}, b={b_hat:.3f}")
    print(f"    {'Size band':<20} {'Empirical':>10} {'Fitted':>10} {'Residual':>10}")
    for i, row in results.iterrows():
        print(f"    {row['label']:<20} {y[i]:>10.3f} {fitted[i]:>10.3f} {residuals[i]:>10.3f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    Interpretation: b={b_hat:.3f} → a doubling of firm size is associated")
    print(f"      with a {expit(a_hat + b_hat*(np.log(2)+log_mp[0])) - expit(a_hat + b_hat*log_mp[0]):.1%} increase in attack probability (at Micro baseline)")

# ---------------------------------------------------------------------------
# 3. Quick view of the monotonicity and magnitude
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("SUMMARY: is prevalence monotone in size? How large is the gradient?")
print("=" * 65)

uw = results['prev_unweighted'].values
w  = results['prev_weighted'].values
print(f"\n  Unweighted: " + " → ".join(f"{v:.3f}" for v in uw))
print(f"  Weighted:   " + " → ".join(f"{v:.3f}" for v in w))
print(f"  Monotone (unweighted): {all(uw[i] <= uw[i+1] for i in range(len(uw)-1))}")
print(f"  Monotone (weighted):   {all(w[i] <= w[i+1] for i in range(len(w)-1))}")
print(f"  Range (unweighted): {uw.min():.3f} – {uw.max():.3f}")
print(f"  Range (weighted):   {w.min():.3f} – {w.max():.3f}")

print("\nDone.")
