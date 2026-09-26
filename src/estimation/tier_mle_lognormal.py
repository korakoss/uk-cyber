"""MLE for independent Poisson peak processes with lognormal T per tier.

Same model as tier_mle.py but T_X ~ Lognormal(μ_X, σ_X) truncated to
bands 3-10, giving 3 λ + 3×(μ,σ) = 9 parameters instead of 24.

Run: ``python src/estimation/tier_mle_lognormal.py``
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import lognorm, chi2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BAND_BOUNDS, SIZE_LABELS, load

PEAK_CUT = 3
N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}
PEAK_BANDS = [3, 4, 5, 6, 7, 8, 9, 10]
N_BANDS = len(PEAK_BANDS)
TIER_NAMES = ["expensive", "mid", "cheap"]
N_TIERS = 3

TIER_MAP = {
    1: "expensive", 3: "expensive", 4: "expensive",
    2: "mid", 5: "mid", 7: "mid", 8: "mid", 11: "mid", 12: "mid", 9: "mid",
    6: "cheap",
}


def band_to_gbp(band, tail_multiple=2.0):
    lo, hi = BAND_BOUNDS.get(band, (0, 0))
    if band == 1: return 0.0
    if hi == np.inf: return lo * tail_multiple
    return (lo + hi) / 2

BAND_COSTS = np.array([band_to_gbp(b) for b in PEAK_BANDS])

# Band edges in £ for the lognormal CDF
BAND_EDGES = []
for b in PEAK_BANDS:
    lo, hi = BAND_BOUNDS[b]
    BAND_EDGES.append((lo, hi))


def lognormal_band_pmf(mu, sigma):
    """Compute P(cost falls in band j | cost is in bands 3-10) under Lognormal(mu, sigma)."""
    raw = np.zeros(N_BANDS)
    for j, (lo, hi) in enumerate(BAND_EDGES):
        cdf_lo = lognorm.cdf(lo, s=sigma, scale=np.exp(mu)) if lo > 0 else 0.0
        if hi == np.inf:
            cdf_hi = 1.0
        else:
            cdf_hi = lognorm.cdf(hi, s=sigma, scale=np.exp(mu))
        raw[j] = max(cdf_hi - cdf_lo, 1e-300)

    total = raw.sum()
    if total < 1e-200:
        return np.ones(N_BANDS) / N_BANDS
    return raw / total


def unpack_params(theta):
    """theta = [log_λ_e, log_λ_m, log_λ_c, μ_e, log_σ_e, μ_m, log_σ_m, μ_c, log_σ_c]"""
    log_lams = theta[:3]
    lams = np.exp(np.clip(log_lams, -10, 5))

    pmfs = []
    for i in range(3):
        mu = theta[3 + 2*i]
        sigma = np.exp(np.clip(theta[4 + 2*i], -3, 4))
        pmf = lognormal_band_pmf(mu, sigma)
        pmfs.append(pmf)

    return lams, pmfs


def neg_log_likelihood(theta, n_no_peak, tier_band_counts):
    lams, pmfs = unpack_params(theta)

    cdfs = [np.cumsum(pmf) for pmf in pmfs]

    pm_leq, pm_lt, pm_eq = [], [], []
    for x in range(N_TIERS):
        leq = np.exp(-lams[x] * (1 - cdfs[x]))
        lt = np.empty(N_BANDS)
        lt[0] = np.exp(-lams[x])
        lt[1:] = leq[:-1]
        eq = leq - lt
        pm_leq.append(leq)
        pm_lt.append(lt)
        pm_eq.append(np.maximum(eq, 1e-300))

    ll = 0.0
    ll += n_no_peak * (-lams.sum())

    for x in range(N_TIERS):
        for j in range(N_BANDS):
            count = tier_band_counts.get((x, j), 0)
            if count == 0:
                continue
            log_p = np.log(pm_eq[x][j])
            for y in range(N_TIERS):
                if y != x:
                    log_p += np.log(np.maximum(pm_lt[y][j], 1e-300))
            ll += count * log_p

    return -ll


def fit_group(v_grp, label):
    peaks = v_grp[v_grp["peak"]].copy()
    n_no_peak = len(v_grp) - len(peaks)
    n_total = len(v_grp)

    tier_band_counts = {}
    for _, row in peaks.iterrows():
        tier = row["tier"]
        band = int(row["band"])
        if tier not in TIER_NAMES or band not in PEAK_BANDS:
            continue
        x = TIER_NAMES.index(tier)
        j = PEAK_BANDS.index(band)
        tier_band_counts[(x, j)] = tier_band_counts.get((x, j), 0) + 1

    n_assigned = sum(tier_band_counts.values())
    print(f"\n  {label}: n={n_total}, no_peak={n_no_peak}, peaks={len(peaks)}, "
          f"assigned={n_assigned}")

    # Initial guess
    lam_init = []
    mu_sigma_init = []
    for x, tier in enumerate(TIER_NAMES):
        n_tier_peaks = sum(v for (xx, j), v in tier_band_counts.items() if xx == x)
        p = n_tier_peaks / n_total
        lam = -np.log(1 - p) if 0 < p < 1 else 0.01
        lam_init.append(np.log(lam))
        # Rough mu/sigma from tier peak band midpoints
        costs = []
        for (xx, j), cnt in tier_band_counts.items():
            if xx == x:
                costs.extend([BAND_COSTS[j]] * cnt)
        if costs:
            costs = np.array(costs)
            costs = costs[costs > 0]
            if len(costs) > 0:
                mu_sigma_init.extend([np.log(np.median(costs)), np.log(1.0)])
            else:
                mu_sigma_init.extend([np.log(1000), np.log(1.0)])
        else:
            mu_sigma_init.extend([np.log(1000), np.log(1.0)])

    theta0 = np.array(lam_init + mu_sigma_init)

    # Try multiple restarts
    best_result = None
    best_nll = np.inf
    for attempt in range(5):
        if attempt > 0:
            theta_try = theta0 + np.random.randn(len(theta0)) * 0.5
        else:
            theta_try = theta0

        result = minimize(neg_log_likelihood, theta_try,
                          args=(n_no_peak, tier_band_counts),
                          method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-12})
        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result

    if not best_result.success:
        print(f"  WARNING: optimization did not converge: {best_result.message}")

    lams, pmfs = unpack_params(best_result.x)
    cdfs = [np.cumsum(pmf) for pmf in pmfs]

    # Extract mu, sigma for reporting
    mu_sigmas = []
    for i in range(3):
        mu = best_result.x[3 + 2*i]
        sigma = np.exp(np.clip(best_result.x[4 + 2*i], -3, 4))
        mu_sigmas.append((mu, sigma))

    return lams, pmfs, cdfs, best_result, mu_sigmas


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)

    att = df[df["attacked"]].copy()
    v = att[att["band"].notna() & att["freq"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["peak"] = v["band"] >= PEAK_CUT
    v["size_group"] = np.where(v["sizeb"] == 1, "Micro", "Rest")
    v["tier"] = v["disrupta"].map(TIER_MAP)

    prev = {}
    for sz in SIZE_LABELS:
        sub = df[df["sizeb"] == sz]
        prev[sz] = (sub["attacked"].astype(int) * sub["weight"]).sum() / sub["weight"].sum()

    print("=" * 74)
    print("MLE FIT — LOGNORMAL T PER TIER (9 parameters per group)")
    print("=" * 74)

    all_results = {}
    for grp in ["Micro", "Rest"]:
        grp_data = v[v["size_group"] == grp]
        lams, pmfs, cdfs, opt, mu_sigmas = fit_group(grp_data, grp)
        all_results[grp] = {"lams": lams, "pmfs": pmfs, "cdfs": cdfs,
                            "opt": opt, "mu_sigmas": mu_sigmas}

        print(f"\n  Fitted parameters:")
        for x, tier in enumerate(TIER_NAMES):
            mu, sigma = mu_sigmas[x]
            print(f"    {tier:<12s} λ={lams[x]:.4f}, μ={mu:.2f}, σ={sigma:.2f}  "
                  f"(median £{np.exp(mu):,.0f})")
        print(f"    {'TOTAL':<12s} λ={lams.sum():.4f}")

        print(f"\n  Implied band PMFs (per-draw, truncated to bands 3-10):")
        print(f"    {'band':>5s} {'£':>10s}", end="")
        for tier in TIER_NAMES:
            print(f" {tier:>12s}", end="")
        print()
        for j, b in enumerate(PEAK_BANDS):
            print(f"    {b:>5d} {BAND_COSTS[j]:>10,.0f}", end="")
            for x in range(N_TIERS):
                print(f" {pmfs[x][j]:>12.4f}", end="")
            print()

        print(f"\n  E[T] per tier (per-draw, within bands 3-10):")
        for x, tier in enumerate(TIER_NAMES):
            et = (pmfs[x] * BAND_COSTS).sum()
            print(f"    {tier:<12s} E[T]=£{et:>10,.0f}")

    # ── GOF ──
    print(f"\n{'=' * 74}")
    print("GOODNESS OF FIT")
    print("=" * 74)

    for grp in ["Micro", "Rest"]:
        grp_data = v[v["size_group"] == grp]
        n_total = len(grp_data)
        lams = all_results[grp]["lams"]
        pmfs = all_results[grp]["pmfs"]
        cdfs = [np.cumsum(pmf) for pmf in pmfs]

        pm_leq, pm_lt, pm_eq = [], [], []
        for x in range(N_TIERS):
            leq = np.exp(-lams[x] * (1 - cdfs[x]))
            lt = np.empty(N_BANDS)
            lt[0] = np.exp(-lams[x])
            lt[1:] = leq[:-1]
            eq = leq - lt
            pm_leq.append(leq)
            pm_lt.append(lt)
            pm_eq.append(eq)

        print(f"\n  {grp} (n={n_total}):")
        print(f"    {'tier':<12s} {'band':>5s} {'obs':>6s} {'pred':>8s} {'diff':>6s}")

        obs_list = []
        exp_list = []

        obs_no = n_total - len(grp_data[grp_data["peak"]])
        exp_no = n_total * np.exp(-lams.sum())
        obs_list.append(obs_no)
        exp_list.append(exp_no)
        print(f"    {'no peak':<12s} {'':>5s} {obs_no:>6d} {exp_no:>8.1f} {obs_no-exp_no:>+6.1f}")

        for x, tier in enumerate(TIER_NAMES):
            for j, b in enumerate(PEAK_BANDS):
                obs = len(grp_data[(grp_data["peak"]) & (grp_data["tier"] == tier)
                                   & (grp_data["band"] == b)])
                p = pm_eq[x][j]
                for y in range(N_TIERS):
                    if y != x:
                        p *= pm_lt[y][j]
                exp = p * n_total
                obs_list.append(obs)
                exp_list.append(exp)
                if obs > 0 or exp > 0.5:
                    print(f"    {tier:<12s} {b:>5d} {obs:>6d} {exp:>8.1f} {obs-exp:>+6.1f}")

        obs_arr = np.array(obs_list, dtype=float)
        exp_arr = np.array(exp_list, dtype=float)

        # Pool small cells
        pooled_obs, pooled_exp = [], []
        pool_o, pool_e = 0, 0
        for i in range(len(obs_arr)):
            pool_o += obs_arr[i]
            pool_e += exp_arr[i]
            if pool_e >= 5:
                pooled_obs.append(pool_o)
                pooled_exp.append(pool_e)
                pool_o, pool_e = 0, 0
        if pool_e > 0:
            if pooled_obs:
                pooled_obs[-1] += pool_o
                pooled_exp[-1] += pool_e
            else:
                pooled_obs.append(pool_o)
                pooled_exp.append(pool_e)

        pooled_obs = np.array(pooled_obs)
        pooled_exp = np.array(pooled_exp)

        n_params = 9
        dof = len(pooled_obs) - 1 - n_params

        chi_sq = ((pooled_obs - pooled_exp) ** 2 / pooled_exp).sum()
        mask = pooled_obs > 0
        g_stat = 2 * (pooled_obs[mask] * np.log(pooled_obs[mask] / pooled_exp[mask])).sum()

        if dof > 0:
            p_val = 1 - chi2.cdf(chi_sq, dof)
            p_val_g = 1 - chi2.cdf(g_stat, dof)
        else:
            p_val = p_val_g = float('nan')

        print(f"\n    Pooled cells (exp≥5): {len(pooled_obs)}, params: {n_params}, df: {dof}")
        if dof > 0:
            print(f"    χ² = {chi_sq:.2f}, p = {p_val:.4f}")
            print(f"    G  = {g_stat:.2f}, p = {p_val_g:.4f}")
        else:
            print(f"    χ² = {chi_sq:.2f}, df = {dof} (still too few cells)")

        nonzero = obs_arr > 0
        mape = np.abs((obs_arr[nonzero] - exp_arr[nonzero]) / obs_arr[nonzero]).mean()
        print(f"    MAPE (nonzero cells) = {mape:.1%}")
        print(f"    Σ|obs-exp| = {np.abs(obs_arr - exp_arr).sum():.1f}")
        print(f"    max|obs-exp| = {np.abs(obs_arr - exp_arr).max():.1f}")

    # Also compare with free-PMF MLE's NLL
    print(f"\n  Log-likelihood comparison:")
    for grp in ["Micro", "Rest"]:
        nll = all_results[grp]["opt"].fun
        print(f"    {grp}: NLL(lognormal, 9 params) = {nll:.2f}")

    # ── National total ──
    print(f"\n{'=' * 74}")
    print("NATIONAL TOTAL")
    print("=" * 74)

    total = 0
    for sz, sz_label in SIZE_LABELS.items():
        grp = "Micro" if sz == 1 else "Rest"
        lams = all_results[grp]["lams"]
        pmfs = all_results[grp]["pmfs"]

        e_cost = 0
        for x in range(N_TIERS):
            et = (pmfs[x] * BAND_COSTS).sum()
            e_cost += lams[x] * et

        nat = N_BY_SIZE[sz] * prev[sz] * e_cost
        total += nat
        print(f"  {sz_label:<10s} E[cost|att]=£{e_cost:>8,.0f}  national=£{nat/1e9:.3f}bn")

    print(f"\n  TOTAL: £{total/1e9:.3f}bn")

    print(f"\n  Per-tier contribution:")
    for grp in ["Micro", "Rest"]:
        lams = all_results[grp]["lams"]
        pmfs = all_results[grp]["pmfs"]
        total_grp = sum(lams[x] * (pmfs[x] * BAND_COSTS).sum() for x in range(N_TIERS))
        print(f"\n  {grp}:")
        for x, tier in enumerate(TIER_NAMES):
            et = (pmfs[x] * BAND_COSTS).sum()
            contrib = lams[x] * et
            print(f"    {tier:<12s} λ={lams[x]:.4f} × E[T]=£{et:>8,.0f} = "
                  f"£{contrib:>8,.0f}/firm ({contrib/total_grp:.1%})")


if __name__ == "__main__":
    main()
