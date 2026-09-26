"""Two-process mixture with free PMFs over cost bands.

Generative model per attacked firm:
  K_H ~ Poisson(λ_H)   commodity/nuisance incidents
  K_L ~ Poisson(λ_L)   serious/targeted incidents
  Each H-incident: cost band ~ Categorical(π_H) over bands 1-10
  Each L-incident: cost band ~ Categorical(π_L) over bands 1-10
  Observed: (freq_band, max_cost_band)

Both processes have full support over all bands — the data determines
where each puts its mass.

Parameters: log(λ_H), log(λ_L), π_H (9 free, softmax), π_L (9 free, softmax)
Total: 2 + 9 + 9 = 20 params per size group.

Observable cells: 6 freq bands × 10 cost bands = 60 (minus empties).

Freq band -> K mapping:
  1: K=1 (conditioned on K>=1)
  2: K in [2, 11]
  3: K=12
  4: K=52
  5: K=365
  6: K=730

For large K, P(max=b|K) is dominated by the upper tail of the per-draw
CDF, so we can compute it efficiently: P(max<=b) = CDF(b)^K where
CDF is the mixture CDF.

Run: ``python src/estimation/two_process_free_pmf.py``
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import softmax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BAND_BOUNDS, SIZE_LABELS, FREQ_LABELS, load

COST_BANDS = list(range(1, 11))
N_BANDS = len(COST_BANDS)

FREQ_K_RANGES = {
    1: (1, 1),
    2: (2, 11),
    3: (12, 12),
    4: (52, 52),
    5: (365, 365),
    6: (730, 730),
}

N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}


def unpack(theta):
    lam_H = np.exp(np.clip(theta[0], -10, 8))
    lam_L = np.exp(np.clip(theta[1], -10, 4))
    pi_H = softmax(theta[2:2+N_BANDS])
    pi_L = softmax(theta[2+N_BANDS:2+2*N_BANDS])
    return lam_H, lam_L, pi_H, pi_L


def max_band_pmf(cdf_H, cdf_L, k_h, k_l):
    """PMF of max(H draws, L draws) over bands."""
    mc = np.ones(N_BANDS)
    if k_h > 0:
        mc *= np.power(np.maximum(cdf_H, 1e-300), k_h)
    if k_l > 0:
        mc *= np.power(np.maximum(cdf_L, 1e-300), k_l)
    pmf = np.zeros(N_BANDS)
    pmf[0] = mc[0]
    for j in range(1, N_BANDS):
        pmf[j] = mc[j] - mc[j-1]
    return np.maximum(pmf, 1e-300)


def neg_log_likelihood(theta, obs_counts):
    lam_H, lam_L, pi_H, pi_L = unpack(theta)
    lam_tot = lam_H + lam_L
    p_h = lam_H / lam_tot if lam_tot > 1e-20 else 0.5

    cdf_H = np.cumsum(pi_H)
    cdf_L = np.cumsum(pi_L)

    # Precompute max PMFs for each (k_h, k_l) we'll need
    # For efficiency, cache by K and enumerate k_h splits
    # For large K, use the mixture CDF directly: each draw is H with prob p_h
    # so the per-draw CDF is p_h*CDF_H + (1-p_h)*CDF_L, and max of K iid draws
    # has CDF = (mixture_CDF)^K

    # For K <= 20: enumerate k_h splits exactly
    # For K > 20: use mixture-CDF approximation (exact in the limit,
    #   and the binomial over k_h concentrates tightly for large K)
    EXACT_THRESHOLD = 20

    mix_cdf = p_h * cdf_H + (1 - p_h) * cdf_L

    ll = 0.0
    for (f, b), count in obs_counts.items():
        if count == 0:
            continue
        b_idx = b - 1
        k_lo, k_hi = FREQ_K_RANGES[f]

        p_fb = 0.0
        for K in range(k_lo, k_hi + 1):
            if f == 1:
                p_K = poisson.pmf(K, lam_tot) / max(1 - np.exp(-lam_tot), 1e-300)
            else:
                p_K = poisson.pmf(K, lam_tot)

            if p_K < 1e-300:
                continue

            if K <= EXACT_THRESHOLD:
                p_band = 0.0
                for k_h in range(K + 1):
                    p_split = binom.pmf(k_h, K, p_h)
                    if p_split < 1e-300:
                        continue
                    mp = max_band_pmf(cdf_H, cdf_L, k_h, K - k_h)
                    p_band += p_split * mp[b_idx]
            else:
                # Mixture approximation: each draw ~ mix_CDF
                mc_K = np.power(np.maximum(mix_cdf, 1e-300), K)
                if b_idx == 0:
                    p_band = mc_K[0]
                else:
                    p_band = mc_K[b_idx] - mc_K[b_idx - 1]
                p_band = max(p_band, 1e-300)

            p_fb += p_K * p_band

        ll += count * np.log(max(p_fb, 1e-300))

    return -ll


def fit(obs_counts, n_restarts=8):
    best_nll = np.inf
    best_result = None

    for restart in range(n_restarts):
        if restart == 0:
            # Informed: H concentrates on band 1, L spreads across 3+
            logits_H = np.array([5, 1, -1, -2, -3, -4, -5, -6, -7, -8], dtype=float)
            logits_L = np.array([-2, 0, 2, 1, 1, 0, -1, -2, -3, -3], dtype=float)
            theta0 = np.concatenate([
                [np.log(3.0), np.log(0.2)],
                logits_H, logits_L
            ])
        elif restart == 1:
            # H = band 1-2, L = band 2-6
            logits_H = np.array([3, 2, 0, -2, -4, -5, -6, -7, -8, -9], dtype=float)
            logits_L = np.array([-3, 1, 2, 2, 2, 1, 0, -1, -2, -2], dtype=float)
            theta0 = np.concatenate([
                [np.log(5.0), np.log(0.3)],
                logits_H, logits_L
            ])
        elif restart == 2:
            # Swapped labels
            logits_L = np.array([5, 1, -1, -2, -3, -4, -5, -6, -7, -8], dtype=float)
            logits_H = np.array([-2, 0, 2, 1, 1, 0, -1, -2, -3, -3], dtype=float)
            theta0 = np.concatenate([
                [np.log(0.2), np.log(3.0)],
                logits_H, logits_L
            ])
        else:
            # Random
            theta0 = np.concatenate([
                [np.log(np.random.uniform(0.1, 10)),
                 np.log(np.random.uniform(0.01, 2))],
                np.random.randn(N_BANDS) * 2,
                np.random.randn(N_BANDS) * 2,
            ])

        try:
            result = minimize(neg_log_likelihood, theta0,
                              args=(obs_counts,),
                              method="L-BFGS-B",
                              options={"maxiter": 5000, "ftol": 1e-12})
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception as e:
            print(f"  restart {restart} failed: {e}")

    return best_result


def band_to_gbp(band):
    lo, hi = BAND_BOUNDS.get(band, (0, 0))
    if band == 1: return 0.0
    if hi == np.inf: return lo * 2.0
    return (lo + hi) / 2

BAND_COSTS = np.array([band_to_gbp(b) for b in COST_BANDS])


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)

    att = df[df["attacked"]].copy()
    v = att[att["band"].notna() & att["freq"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["freq"] = v["freq"].astype(int)
    v["size_group"] = np.where(v["sizeb"] == 1, "Micro", "Rest")

    prev = {}
    for sz in SIZE_LABELS:
        sub = df[df["sizeb"] == sz]
        prev[sz] = (sub["attacked"].astype(int) * sub["weight"]).sum() / sub["weight"].sum()

    results = {}
    for grp in ["Micro", "Rest"]:
        g = v[v["size_group"] == grp]
        print(f"\n{'='*74}")
        print(f"  {grp} (n={len(g)})")
        print(f"{'='*74}")

        obs_counts = {}
        for (f, b), cnt in g.groupby(["freq", "band"]).size().items():
            obs_counts[(int(f), int(b))] = int(cnt)

        n_cells = len(obs_counts)
        print(f"  Observable cells: {n_cells}, Parameters: 20")
        print(f"  Fitting...")

        result = fit(obs_counts)
        lam_H, lam_L, pi_H, pi_L = unpack(result.x)

        # Ensure H is the high-rate process
        if lam_H < lam_L:
            lam_H, lam_L = lam_L, lam_H
            pi_H, pi_L = pi_L, pi_H

        et_H = (pi_H * BAND_COSTS).sum()
        et_L = (pi_L * BAND_COSTS).sum()

        print(f"\n  H (commodity): λ={lam_H:.3f}")
        print(f"    PMF: {['%.3f' % p for p in pi_H]}")
        print(f"    E[T_H] = £{et_H:,.0f}")
        print(f"    P(at least one H) = {1-np.exp(-lam_H):.3f}")

        print(f"\n  L (serious): λ={lam_L:.4f}")
        print(f"    PMF: {['%.3f' % p for p in pi_L]}")
        print(f"    E[T_L] = £{et_L:,.0f}")
        print(f"    P(at least one L) = {1-np.exp(-lam_L):.4f}")

        e_cost = lam_H * et_H + lam_L * et_L
        print(f"\n  E[cost|att] = λ_H·E[T_H] + λ_L·E[T_L]")
        print(f"    = {lam_H:.3f}×£{et_H:,.0f} + {lam_L:.4f}×£{et_L:,.0f}")
        print(f"    = £{lam_H*et_H:,.0f} + £{lam_L*et_L:,.0f}")
        print(f"    = £{e_cost:,.0f}")

        print(f"\n  NLL = {result.fun:.2f}, converged = {result.success}")

        results[grp] = {"lam_H": lam_H, "lam_L": lam_L,
                         "pi_H": pi_H, "pi_L": pi_L,
                         "et_H": et_H, "et_L": et_L, "e_cost": e_cost}

        # GOF: predicted vs observed freq×band table
        print(f"\n  Predicted vs observed (freq × band):")
        lam_tot = lam_H + lam_L
        p_h = lam_H / lam_tot
        cdf_H = np.cumsum(pi_H)
        cdf_L = np.cumsum(pi_L)
        mix_cdf = p_h * cdf_H + (1 - p_h) * cdf_L

        chi2_sum = 0.0
        n_gof_cells = 0
        for f in range(1, 7):
            row_obs = []
            row_pred = []
            k_lo, k_hi = FREQ_K_RANGES[f]
            # P(freq=f)
            p_freq = sum(poisson.pmf(K, lam_tot) for K in range(k_lo, k_hi+1))
            if f == 1:
                p_freq = poisson.pmf(1, lam_tot) / max(1 - np.exp(-lam_tot), 1e-300)
                # But we also need P(K>=1), since all firms are attacked
                # Actually p_freq for freq=1 is P(K=1|K>=1)
            for b in COST_BANDS:
                obs = obs_counts.get((f, b), 0)
                # Compute predicted
                b_idx = b - 1
                p_fb = 0.0
                for K in range(k_lo, min(k_hi, 20) + 1):
                    if f == 1:
                        p_K = poisson.pmf(K, lam_tot) / max(1 - np.exp(-lam_tot), 1e-300)
                    else:
                        p_K = poisson.pmf(K, lam_tot)
                    if p_K < 1e-300:
                        continue
                    if K <= 20:
                        p_band = 0.0
                        for k_h in range(K+1):
                            p_split = binom.pmf(k_h, K, p_h)
                            if p_split < 1e-300:
                                continue
                            mp = max_band_pmf(cdf_H, cdf_L, k_h, K-k_h)
                            p_band += p_split * mp[b_idx]
                    else:
                        mc_K = np.power(np.maximum(mix_cdf, 1e-300), K)
                        p_band = mc_K[b_idx] - (mc_K[b_idx-1] if b_idx > 0 else 0)
                        p_band = max(p_band, 1e-300)
                    p_fb += p_K * p_band

                # For K > 20 in freq=2 range
                for K in range(21, k_hi+1):
                    p_K = poisson.pmf(K, lam_tot)
                    if p_K < 1e-300:
                        continue
                    mc_K = np.power(np.maximum(mix_cdf, 1e-300), K)
                    p_band = mc_K[b_idx] - (mc_K[b_idx-1] if b_idx > 0 else 0)
                    p_fb += p_K * max(p_band, 1e-300)

                pred = p_fb * len(g)
                row_obs.append(obs)
                row_pred.append(pred)
                if pred > 1:
                    chi2_sum += (obs - pred)**2 / pred
                    n_gof_cells += 1

            # Print compact row
            obs_str = ' '.join(f'{o:>4d}' for o in row_obs)
            pred_str = ' '.join(f'{p:>4.0f}' for p in row_pred)
            if any(o > 0 for o in row_obs):
                print(f'    freq={f} obs:  {obs_str}')
                print(f'           pred: {pred_str}')

        from scipy.stats import chi2
        dof = n_gof_cells - 20
        p_val = 1 - chi2.cdf(chi2_sum, max(dof, 1)) if dof > 0 else float('nan')
        print(f"\n  GOF: chi2={chi2_sum:.1f}, cells={n_gof_cells}, "
              f"params=20, df={dof}, p={p_val:.3f}")

    # National total
    print(f"\n{'='*74}")
    print("NATIONAL TOTAL")
    print(f"{'='*74}")
    total = 0
    for sz in SIZE_LABELS:
        grp = "Micro" if sz == 1 else "Rest"
        e_cost = results[grp]["e_cost"]
        nat = N_BY_SIZE[sz] * prev[sz] * e_cost
        total += nat
        print(f"  {SIZE_LABELS[sz]}: E[cost|att]=£{e_cost:,.0f}, "
              f"national=£{nat/1e9:.3f}bn")
    print(f"\n  TOTAL: £{total/1e9:.3f}bn")

    # Decomposition
    print(f"\n{'='*74}")
    print("COST DECOMPOSITION")
    print(f"{'='*74}")
    for grp in ["Micro", "Rest"]:
        r = results[grp]
        h_cost = r["lam_H"] * r["et_H"]
        l_cost = r["lam_L"] * r["et_L"]
        print(f"  {grp}:")
        print(f"    Commodity (H): £{h_cost:,.0f}/firm ({h_cost/r['e_cost']*100:.1f}%)")
        print(f"    Serious  (L):  £{l_cost:,.0f}/firm ({l_cost/r['e_cost']*100:.1f}%)")


if __name__ == "__main__":
    main()
