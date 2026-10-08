"""Two-process latent mixture MLE: commodity (H) vs serious (L) incidents.

Generative model per attacked firm:
  K_H ~ Poisson(λ_H)   commodity/nuisance incidents
  K_L ~ Poisson(λ_L)   serious/targeted incidents
  Each H-incident: cost ~ Lognormal(μ_H, σ_H), discretized to bands 1-10
  Each L-incident: cost ~ Lognormal(μ_L, σ_L), discretized to bands 1-10
  K = K_H + K_L         total incident count
  Observed: (freq_band, max_cost_band) where freq_band bins K

The two processes overlap in cost support — H can produce band-3+ costs,
L can produce band-1 costs — but they differ in rate and shape.

Freq band -> K ranges:
  1: K=1
  2: K in [2, 11]    (more than once but <monthly)
  3: K in [12, 12]   (~monthly, use point)
  4: K in [52, 52]   (~weekly, use point)
  5: K in [365, 365] (~daily, use point)
  6: K in [730, 730]  (several times a day, use point)

For bands 2-6, we marginalize over K values in the range.

Run: ``python src/estimation/two_process_mle.py``
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import lognorm, poisson

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BAND_BOUNDS, SIZE_LABELS, load

COST_BANDS = list(range(1, 11))
N_COST_BANDS = len(COST_BANDS)

# Freq band -> range of annual K values
FREQ_K_RANGES = {
    1: (1, 1),
    2: (2, 11),
    3: (12, 12),
    4: (52, 52),
    5: (365, 365),
    6: (730, 730),
}

# For high-K freq bands, we can't enumerate all K. Use a representative point.
# For freq=2 we marginalize over K=2..11.
# For freq>=3, K is large enough that max-cost is essentially determined by
# the process CDFs raised to high powers, so a point K is fine.

BAND_EDGES = []
for b in COST_BANDS:
    lo, hi = BAND_BOUNDS[b]
    BAND_EDGES.append((lo, hi))


def lognormal_band_cdf(mu, sigma):
    """CDF of the discretized lognormal: P(band <= b) for b=1..10."""
    cdf = np.zeros(N_COST_BANDS)
    for j, (lo, hi) in enumerate(BAND_EDGES):
        if hi == np.inf or hi == 0:
            cdf[j] = 1.0 if hi == np.inf else 0.0
        else:
            cdf[j] = lognorm.cdf(hi, s=sigma, scale=np.exp(mu))
    # band 1 = (0,0): P(cost <= 0) — interpret as P(cost is negligible)
    # Use the lognormal's CDF at a small threshold, say £1
    cdf[0] = lognorm.cdf(1.0, s=sigma, scale=np.exp(mu))
    # Ensure monotone
    for j in range(1, N_COST_BANDS):
        cdf[j] = max(cdf[j], cdf[j-1])
    cdf[-1] = 1.0  # band 10 upper = 500k, treat as ceiling
    return cdf


def band_pmf_from_cdf(cdf):
    """P(band = b) from CDF."""
    pmf = np.zeros(N_COST_BANDS)
    pmf[0] = cdf[0]
    for j in range(1, N_COST_BANDS):
        pmf[j] = cdf[j] - cdf[j-1]
    return np.maximum(pmf, 1e-300)


def max_cdf(cdf_h, cdf_l, k_h, k_l):
    """CDF of max(H_1,...,H_{k_h}, L_1,...,L_{k_l}) at each band."""
    # P(max <= b) = P(all H <= b) * P(all L <= b) = CDF_H(b)^k_h * CDF_L(b)^k_l
    result = np.ones(N_COST_BANDS)
    if k_h > 0:
        result *= np.power(np.maximum(cdf_h, 1e-300), k_h)
    if k_l > 0:
        result *= np.power(np.maximum(cdf_l, 1e-300), k_l)
    return result


def max_pmf(cdf_h, cdf_l, k_h, k_l):
    """PMF of max cost band given k_h H-draws and k_l L-draws."""
    if k_h == 0 and k_l == 0:
        # No incidents — shouldn't happen for attacked firms, but handle
        pmf = np.zeros(N_COST_BANDS)
        pmf[0] = 1.0
        return pmf
    mc = max_cdf(cdf_h, cdf_l, k_h, k_l)
    pmf = np.zeros(N_COST_BANDS)
    pmf[0] = mc[0]
    for j in range(1, N_COST_BANDS):
        pmf[j] = mc[j] - mc[j-1]
    return np.maximum(pmf, 1e-300)


def neg_log_likelihood(theta, obs_counts, max_k_enumerate=50):
    """
    theta: [log_lam_H, log_lam_L, mu_H, log_sigma_H, mu_L, log_sigma_L]
    obs_counts: dict (freq_band, cost_band) -> count
    """
    lam_H = np.exp(np.clip(theta[0], -10, 8))
    lam_L = np.exp(np.clip(theta[1], -10, 4))
    mu_H = theta[2]
    sigma_H = np.exp(np.clip(theta[3], -3, 3))
    mu_L = theta[4]
    sigma_L = np.exp(np.clip(theta[5], -3, 4))

    cdf_h = lognormal_band_cdf(mu_H, sigma_H)
    cdf_l = lognormal_band_cdf(mu_L, sigma_L)

    lam_total = lam_H + lam_L

    ll = 0.0
    for (f, b), count in obs_counts.items():
        if count == 0:
            continue
        b_idx = b - 1  # 0-indexed

        k_lo, k_hi = FREQ_K_RANGES[f]

        # For large K ranges or large K, we need to be smart about enumeration.
        # P(freq=f, band=b) = sum_{K in range} P(K|lam_total) * P(band=b|K, lam_H, lam_L)
        # P(band=b|K) = sum_{k_h=0}^{K} P(k_h|K, lam_H, lam_L) * P(max=b|k_h, K-k_h)
        # where k_h|K ~ Binomial(K, lam_H/(lam_H+lam_L))

        p_fb = 0.0
        p_h = lam_H / lam_total if lam_total > 1e-20 else 0.5

        for K in range(k_lo, min(k_hi, max_k_enumerate) + 1):
            # P(K | lam_total): for freq=1, condition on K>=1
            if f == 1:
                p_K = poisson.pmf(K, lam_total) / max(1 - poisson.pmf(0, lam_total), 1e-300)
            else:
                p_K = poisson.pmf(K, lam_total)

            if p_K < 1e-300:
                continue

            # Marginalize over k_h splits: k_h ~ Binom(K, p_h)
            p_band_given_K = 0.0
            from scipy.stats import binom
            for k_h in range(K + 1):
                p_split = binom.pmf(k_h, K, p_h)
                if p_split < 1e-300:
                    continue
                k_l = K - k_h
                mp = max_pmf(cdf_h, cdf_l, k_h, k_l)
                p_band_given_K += p_split * mp[b_idx]

            p_fb += p_K * p_band_given_K

        # For freq=2 (K in 2..11), also need P(K in [2,11])
        # Normalize: P(freq=f) = sum P(K in range)
        # But we're computing the joint P(freq=f, band=b) directly

        # For freq bands 3+ with large K, the Poisson PMF at the point K
        # stands in for P(freq=f|K) — this is approximate
        # (really P(freq=f) = P(K in range) but we use a point)

        ll += count * np.log(max(p_fb, 1e-300))

    return -ll


def fit(obs_counts, n_restarts=5):
    """Fit the two-process model with multiple restarts."""
    best_nll = np.inf
    best_result = None

    for restart in range(n_restarts):
        if restart == 0:
            # Informed start: H = high-rate low-cost, L = low-rate high-cost
            theta0 = np.array([
                np.log(2.0),    # lam_H ~ 2/yr
                np.log(0.1),    # lam_L ~ 0.1/yr
                np.log(10),     # mu_H: median cost ~£10
                np.log(1.0),    # sigma_H ~ 1
                np.log(5000),   # mu_L: median cost ~£5000
                np.log(2.0),    # sigma_L ~ 2
            ])
        else:
            # Random restarts
            theta0 = np.array([
                np.log(np.random.uniform(0.5, 10)),
                np.log(np.random.uniform(0.01, 1)),
                np.log(np.random.uniform(1, 100)),
                np.log(np.random.uniform(0.5, 3)),
                np.log(np.random.uniform(100, 50000)),
                np.log(np.random.uniform(1, 4)),
            ])

        try:
            result = minimize(neg_log_likelihood, theta0,
                              args=(obs_counts,),
                              method="L-BFGS-B",
                              options={"maxiter": 5000, "ftol": 1e-10})
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception as e:
            print(f"  restart {restart} failed: {e}")

    return best_result


def band_to_gbp(band, tail_multiple=2.0):
    lo, hi = BAND_BOUNDS.get(band, (0, 0))
    if band == 1: return 0.0
    if hi == np.inf: return lo * tail_multiple
    return (lo + hi) / 2


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)

    att = df[df["attacked"]].copy()
    v = att[att["band"].notna() & att["freq"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["freq"] = v["freq"].astype(int)
    v["size_group"] = np.where(v["sizeb"] == 1, "Micro", "Rest")

    N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}
    prev = {}
    for sz in SIZE_LABELS:
        sub = df[df["sizeb"] == sz]
        prev[sz] = (sub["attacked"].astype(int) * sub["weight"]).sum() / sub["weight"].sum()

    for grp in ["Micro", "Rest"]:
        g = v[v["size_group"] == grp]
        print(f"\n{'='*74}")
        print(f"  {grp} (n={len(g)})")
        print(f"{'='*74}")

        # Build observed (freq, band) counts
        obs_counts = {}
        for (f, b), cnt in g.groupby(["freq", "band"]).size().items():
            obs_counts[(int(f), int(b))] = int(cnt)

        print(f"\n  Observed cells: {len(obs_counts)}")
        print(f"  Fitting two-process model (6 params)...")

        result = fit(obs_counts)
        theta = result.x

        lam_H = np.exp(np.clip(theta[0], -10, 8))
        lam_L = np.exp(np.clip(theta[1], -10, 4))
        mu_H = theta[2]
        sigma_H = np.exp(np.clip(theta[3], -3, 3))
        mu_L = theta[4]
        sigma_L = np.exp(np.clip(theta[5], -3, 4))

        print(f"\n  Fitted parameters:")
        print(f"    H (commodity): λ={lam_H:.3f}, μ={mu_H:.2f}, σ={sigma_H:.2f}")
        print(f"      median cost = £{np.exp(mu_H):,.0f}, "
              f"mean cost = £{np.exp(mu_H + sigma_H**2/2):,.0f}")
        print(f"    L (serious):   λ={lam_L:.4f}, μ={mu_L:.2f}, σ={sigma_L:.2f}")
        print(f"      median cost = £{np.exp(mu_L):,.0f}, "
              f"mean cost = £{np.exp(mu_L + sigma_L**2/2):,.0f}")
        print(f"    λ_total = {lam_H + lam_L:.3f}")
        print(f"    P(at least one H) = {1 - np.exp(-lam_H):.3f}")
        print(f"    P(at least one L) = {1 - np.exp(-lam_L):.4f}")

        # E[T] using band midpoints under fitted distributions
        cdf_h = lognormal_band_cdf(mu_H, sigma_H)
        cdf_l = lognormal_band_cdf(mu_L, sigma_L)
        pmf_h = band_pmf_from_cdf(cdf_h)
        pmf_l = band_pmf_from_cdf(cdf_l)
        band_costs = np.array([band_to_gbp(b) for b in COST_BANDS])

        et_h = (pmf_h * band_costs).sum()
        et_l = (pmf_l * band_costs).sum()

        print(f"\n    E[T_H] (band midpoints) = £{et_h:,.0f}")
        print(f"    E[T_L] (band midpoints) = £{et_l:,.0f}")
        print(f"    E[cost|att] = λ_H·E[T_H] + λ_L·E[T_L]"
              f" = £{lam_H*et_h + lam_L*et_l:,.0f}")

        print(f"\n    H PMF across bands: {[f'{p:.3f}' for p in pmf_h]}")
        print(f"    L PMF across bands: {[f'{p:.3f}' for p in pmf_l]}")

        print(f"\n    NLL = {result.fun:.2f}, converged = {result.success}")

    # National total
    print(f"\n{'='*74}")
    print("NATIONAL TOTAL")
    print(f"{'='*74}")

    total = 0
    for grp in ["Micro", "Rest"]:
        g = v[v["size_group"] == grp]
        obs_counts = {}
        for (f, b), cnt in g.groupby(["freq", "band"]).size().items():
            obs_counts[(int(f), int(b))] = int(cnt)
        result = fit(obs_counts, n_restarts=3)
        theta = result.x
        lam_H = np.exp(np.clip(theta[0], -10, 8))
        lam_L = np.exp(np.clip(theta[1], -10, 4))
        mu_H = theta[2]
        sigma_H = np.exp(np.clip(theta[3], -3, 3))
        mu_L = theta[4]
        sigma_L = np.exp(np.clip(theta[5], -3, 4))

        cdf_h = lognormal_band_cdf(mu_H, sigma_H)
        cdf_l = lognormal_band_cdf(mu_L, sigma_L)
        pmf_h = band_pmf_from_cdf(cdf_h)
        pmf_l = band_pmf_from_cdf(cdf_l)
        band_costs = np.array([band_to_gbp(b) for b in COST_BANDS])
        et_h = (pmf_h * band_costs).sum()
        et_l = (pmf_l * band_costs).sum()
        e_cost = lam_H * et_h + lam_L * et_l

        for sz in SIZE_LABELS:
            sz_grp = "Micro" if sz == 1 else "Rest"
            if sz_grp == grp:
                nat = N_BY_SIZE[sz] * prev[sz] * e_cost
                total += nat
                print(f"  {SIZE_LABELS[sz]}: N={N_BY_SIZE[sz]:,}, prev={prev[sz]:.3f}, "
                      f"E[cost|att]=£{e_cost:,.0f}, contrib=£{nat/1e9:.3f}bn")

    print(f"\n  NATIONAL TOTAL: £{total/1e9:.3f}bn")


if __name__ == "__main__":
    main()
