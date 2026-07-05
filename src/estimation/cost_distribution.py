"""
Cost distribution: lognormal fit to damage_bands.

damage_bands is a discretised version of a continuous cost variable. Each band
corresponds to an interval [L_b, U_b). Band 1 is a point mass at 0 (no cost).
Bands 2-10 have known lower/upper bounds. Band 10 is open-ended (£100k+).

Fitting approach:
- Model cost X as a zero-inflated lognormal: with probability p0, X=0 (no cost);
  with probability 1-p0, X ~ Lognormal(mu, sigma).
- p0 is estimated as the observed fraction in band 1 (no cost).
- For bands 2-10, we fit mu and sigma by maximum likelihood on the conditional
  lognormal using band-interval probabilities.
- Band 10 (£100k-£500k) is treated as right-censored at £500k for fitting;
  the fitted tail beyond £500k is reported separately as a key output.

Runs:
1. Per size band (pooled across freq) — main check
2. Per (size, freq) cell — to see how parameters vary
3. Within freq=1 only per size — cleanest subgroup (no bridge ambiguity)

Key outputs:
- Fitted (p0, mu, sigma) per stratum
- Goodness of fit: predicted vs observed band proportions (chi-square and visual table)
- Fitted tail probability P(cost > £500k) — the key unknown for the top band
- Comparison: how much does mu vary by size? by freq?
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

df = get_business_data()
INVALID_FREQ = {997, 999, 997.0, 999.0}

attacked = df[
    df['freq'].notna() &
    ~df['freq'].isin(INVALID_FREQ) &
    (df['freq'] > 0) &
    df['damage_bands'].notna() &
    (df['damage_bands'] < SPECIAL_CODE_THRESHOLD)
].copy()

SIZE_LABELS = {1: 'Micro (1-9)', 2: 'Small (10-49)', 3: 'Medium (50-249)', 4: 'Large (250+)'}
FREQ_LABELS = {1: 'Once only', 2: '>once <monthly', 3: '~monthly',
               4: '~weekly', 5: '~daily', 6: 'Several/day'}

# Band boundaries (lower, upper) in £
# Band 1: point mass at 0
# Band 10: lower=100000, upper=500000 (right-censored for fitting; tail reported separately)
BAND_BOUNDS = {
    2:  (1,      100),
    3:  (100,    500),
    4:  (500,    1000),
    5:  (1000,   5000),
    6:  (5000,   10000),
    7:  (10000,  20000),
    8:  (20000,  50000),
    9:  (50000,  100000),
    10: (100000, 500000),
}
BAND_LABELS = {
    1: 'No cost', 2: '<£100', 3: '£100-£500', 4: '£500-£1k',
    5: '£1k-£5k', 6: '£5k-£10k', 7: '£10k-£20k', 8: '£20k-£50k',
    9: '£50k-£100k', 10: '£100k-£500k'
}
bands = list(range(1, 11))

# ---------------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------------

def lognormal_band_probs(mu, sigma, p0):
    """
    Compute predicted probability for each band under zero-inflated lognormal.
    Band 1 (no cost) = p0.
    Bands 2-10: (1-p0) * P(L_b <= X < U_b | X in [1, 500k]) where X ~ Lognormal(mu, sigma).
    Band 10 is treated as [100k, 500k] for fitting (right-censored).
    """
    probs = np.zeros(10)
    probs[0] = p0
    raw = np.zeros(9)  # raw lognormal interval probabilities for bands 2-10
    for i, b in enumerate(range(2, 11)):
        L, U = BAND_BOUNDS[b]
        raw[i] = norm.cdf(np.log(U), mu, sigma) - norm.cdf(np.log(L), mu, sigma)
    # Renormalise within [1, 500k] then scale by (1-p0)
    s = raw.sum()
    if s > 1e-9:
        probs[1:] = (1 - p0) * raw / s
    return probs

def fit_lognormal(counts, verbose=False):
    """
    Fit zero-inflated lognormal to band counts.
    p0 is fixed at the no-cost fraction.
    mu, sigma are fitted by MLE on bands 2-10.
    Returns (p0, mu, sigma, fitted_probs) or None if fitting fails.
    """
    total = counts.sum()
    if total == 0:
        return None

    p0 = counts[0] / total  # fraction in band 1 (no cost)

    # If all observations are in band 1, lognormal not identifiable
    if counts[1:].sum() == 0:
        return None

    # MLE: maximise log-likelihood over bands 2-10 conditional on being in band 2+
    # (equivalently, minimise negative log-likelihood of the lognormal on the conditional counts)
    cond_counts = counts[1:]  # bands 2-10, shape (9,)

    def nll(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        if sigma < 1e-6:
            return 1e10
        # Predicted probabilities for bands 2-10 conditional on X > 0
        pred = np.zeros(9)
        for i, b in enumerate(range(2, 11)):
            L, U = BAND_BOUNDS[b]
            pred[i] = norm.cdf(np.log(U), mu, sigma) - norm.cdf(np.log(L), mu, sigma)
        # Normalise within [1, 500k]
        s = pred.sum()
        if s < 1e-9:
            return 1e10
        pred = pred / s
        pred = np.clip(pred, 1e-12, 1)
        ll = (cond_counts * np.log(pred)).sum()
        return -ll

    # Multiple starting points to avoid local minima
    best_res = None
    for mu0 in [5, 7, 9, 11]:
        for lsig0 in [0.5, 1.0, 1.5]:
            res = minimize(nll, x0=[mu0, lsig0], method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
            if best_res is None or res.fun < best_res.fun:
                best_res = res

    mu_hat = best_res.x[0]
    sigma_hat = np.exp(best_res.x[1])
    fitted_probs = lognormal_band_probs(mu_hat, sigma_hat, p0)

    return p0, mu_hat, sigma_hat, fitted_probs

def report_fit(label, counts, n_min=15):
    """Fit and print results for a given set of band counts."""
    total = counts.sum()
    if total < n_min:
        print(f"  {label}: n={int(total)} < {n_min}, skipping")
        return None

    obs_probs = counts / total
    result = fit_lognormal(counts)
    if result is None:
        print(f"  {label}: fit failed (all no-cost?)")
        return None

    p0, mu, sigma, fitted = result

    # Tail: P(cost > £500k) = (1-p0) * P(X > 500k | X > 0)
    tail_prob = (1 - p0) * (1 - norm.cdf(np.log(500000), mu, sigma))
    # Expected cost under fitted model (conditional on X > 0): E[X | X > 0] for lognormal
    e_cost_given_nonzero = np.exp(mu + sigma**2 / 2)  # true lognormal mean, no truncation

    # GOF: chi-square on bands 2-10 (skip band 1 since p0 is fixed)
    obs_c = counts[1:]
    exp_c = fitted[1:] * total
    exp_c = np.clip(exp_c, 1e-9, None)
    chisq = ((obs_c - exp_c)**2 / exp_c).sum()
    df_chisq = 8 - 2  # 9 cells - 1 (normalisation) - 2 params (mu, sigma) — rough

    print(f"\n  {label}  (n={int(total)})")
    print(f"    p0={p0:.3f}  mu={mu:.3f}  sigma={sigma:.3f}")
    print(f"    E[cost|nonzero] = £{e_cost_given_nonzero:,.0f}  (lognormal mean, no truncation)")
    print(f"    P(cost > £500k) = {tail_prob:.4f}  ({tail_prob:.2%})")
    print(f"    Chi-sq (bands 2-10): {chisq:.1f}  (df≈{df_chisq}, rough)")
    print(f"    {'Band':<14} {'Observed':>10} {'Fitted':>10} {'Ratio':>8}")
    for i, b in enumerate(bands):
        obs = obs_probs[i]
        fit = fitted[i]
        ratio = fit / obs if obs > 1e-6 else float('nan')
        flag = " *" if abs(obs - fit) > 0.05 else ""
        print(f"    {BAND_LABELS[b]:<14} {obs:>10.3f} {fit:>10.3f} {ratio:>8.2f}{flag}")

    return dict(label=label, n=int(total), p0=p0, mu=mu, sigma=sigma,
                tail_prob=tail_prob, e_cost_nonzero=e_cost_given_nonzero, chisq=chisq)

# ---------------------------------------------------------------------------
# 1. Per size band (pooled across freq)
# ---------------------------------------------------------------------------

print("=" * 70)
print("LOGNORMAL FIT: per size band (pooled across freq groups)")
print("=" * 70)

size_results = []
for sz in sorted(attacked['sizeb'].dropna().unique()):
    sub = attacked[attacked['sizeb'] == sz]
    counts = np.array([(sub['damage_bands'] == b).sum() for b in bands], dtype=float)
    r = report_fit(SIZE_LABELS.get(int(sz), str(sz)), counts)
    if r:
        size_results.append(r)

# ---------------------------------------------------------------------------
# 2. freq=1 only, per size band (cleanest subgroup: total cost = single attack cost)
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("LOGNORMAL FIT: freq=1 (once only) per size band")
print("No bridge needed here — reported cost IS the total annual cost")
print("=" * 70)

freq1_results = []
for sz in sorted(attacked['sizeb'].dropna().unique()):
    sub = attacked[(attacked['sizeb'] == sz) & (attacked['freq'] == 1)]
    counts = np.array([(sub['damage_bands'] == b).sum() for b in bands], dtype=float)
    r = report_fit(f"{SIZE_LABELS.get(int(sz), str(sz))} | freq=1", counts, n_min=10)
    if r:
        freq1_results.append(r)

# ---------------------------------------------------------------------------
# 3. Per (size, freq) cell — check how mu varies
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("LOGNORMAL FIT: per (size, freq) cell — parameter variation")
print("Only cells with n >= 20 shown")
print("=" * 70)

print(f"\n  {'Stratum':<30} {'n':>5} {'p0':>6} {'mu':>7} {'sigma':>7} "
      f"{'E[cost|nz] £':>14} {'P(>500k)':>10} {'chi-sq':>8}")
cell_results = []
for sz in sorted(attacked['sizeb'].dropna().unique()):
    for freq in sorted(attacked['freq'].dropna().unique()):
        sub = attacked[(attacked['sizeb'] == sz) & (attacked['freq'] == freq)]
        counts = np.array([(sub['damage_bands'] == b).sum() for b in bands], dtype=float)
        n = int(counts.sum())
        if n < 20:
            continue
        result = fit_lognormal(counts)
        if result is None:
            continue
        p0, mu, sigma, fitted = result
        tail_prob = (1 - p0) * (1 - norm.cdf(np.log(500000), mu, sigma))
        e_nz = np.exp(mu + sigma**2 / 2)
        obs_probs = counts / n
        exp_c = fitted[1:] * n
        exp_c = np.clip(exp_c, 1e-9, None)
        chisq = ((counts[1:] - exp_c)**2 / exp_c).sum()
        lbl = f"{SIZE_LABELS.get(int(sz),'?')[:12]} | {FREQ_LABELS.get(int(freq),'?')[:14]}"
        print(f"  {lbl:<30} {n:>5} {p0:>6.3f} {mu:>7.2f} {sigma:>7.2f} "
              f"{e_nz:>14,.0f} {tail_prob:>10.4f} {chisq:>8.1f}")
        cell_results.append(dict(size=sz, freq=freq, n=n, p0=p0, mu=mu, sigma=sigma,
                                 tail_prob=tail_prob, e_nz=e_nz, chisq=chisq))

# ---------------------------------------------------------------------------
# 4. Summary: how does mu vary? Is sigma stable?
# ---------------------------------------------------------------------------

if cell_results:
    df_cells = pd.DataFrame(cell_results)
    print(f"\n  mu range across cells:    {df_cells['mu'].min():.2f} – {df_cells['mu'].max():.2f}")
    print(f"  sigma range across cells: {df_cells['sigma'].min():.2f} – {df_cells['sigma'].max():.2f}")
    print(f"  median sigma: {df_cells['sigma'].median():.2f}")
    print(f"  Note: sigma is the lognormal shape parameter (log-scale SD).")
    print(f"  A stable sigma suggests a common shape with shifting location (mu).")

print("\nDone.")
