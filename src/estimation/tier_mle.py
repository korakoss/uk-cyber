"""MLE for independent Poisson peak processes per type tier.

Model: three independent Poisson processes (expensive, mid, cheap), each
with its own rate λ_X and per-draw cost distribution T_X (discrete on
bands 3-10). We observe per firm:
  - no peak (band < 3): all three K_X = 0
  - peak with disrupta ∈ tier X at band b: tier X's max draw = b, and
    all other tiers' max draws < b (strict; ties assigned to no one)

Key identity: P(M_X ≤ b) = exp(-λ_X · (1 - F_X(b))), where F_X is the
per-draw CDF. This makes the likelihood clean.

Parameters (per size group): λ_e, λ_m, λ_c + PMF of T_e, T_m, T_c on
bands 3-10 (7 free each) = 24 total.

Run: ``python src/estimation/tier_mle.py``
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def unpack_params(theta):
    """Unpack flat parameter vector into (lambdas, pmfs).

    theta layout: [log_λ_e, log_λ_m, log_λ_c,
                   raw_pmf_e (7 vals), raw_pmf_m (7), raw_pmf_c (7)]
    = 3 + 3*7 = 24 parameters.

    PMFs use softmax on 8 raw values (first 7 free + implicit 0 for last band).
    """
    log_lams = theta[:3]
    lams = np.exp(np.clip(log_lams, -10, 5))

    pmfs = []
    for i in range(3):
        start = 3 + i * (N_BANDS - 1)
        raw = np.zeros(N_BANDS)
        raw[:N_BANDS - 1] = theta[start:start + N_BANDS - 1]
        # softmax
        raw -= raw.max()
        exp_raw = np.exp(raw)
        pmf = exp_raw / exp_raw.sum()
        pmfs.append(pmf)

    return lams, pmfs


def neg_log_likelihood(theta, n_no_peak, tier_band_counts):
    """Compute -LL for one size group.

    tier_band_counts: dict (tier_idx, band_idx) -> count
    n_no_peak: number of firms with band < 3
    """
    lams, pmfs = unpack_params(theta)

    # CDF for each tier: F_X(b_j) = Σ_{i≤j} pmf_X(i)
    cdfs = [np.cumsum(pmf) for pmf in pmfs]

    # P(M_X ≤ b_j) = exp(-λ_X * (1 - F_X(b_j)))
    # P(M_X < b_j) = P(M_X ≤ b_{j-1}) for j>0, = exp(-λ_X) for j=0
    # P(M_X = b_j) = P(M_X ≤ b_j) - P(M_X < b_j)
    pm_leq = []  # P(M_X ≤ b_j) for each tier, band
    pm_lt = []   # P(M_X < b_j)
    pm_eq = []   # P(M_X = b_j)

    for x in range(N_TIERS):
        leq = np.exp(-lams[x] * (1 - cdfs[x]))  # shape (N_BANDS,)
        lt = np.empty(N_BANDS)
        lt[0] = np.exp(-lams[x])  # P(K_X = 0)
        lt[1:] = leq[:-1]
        eq = leq - lt
        pm_leq.append(leq)
        pm_lt.append(lt)
        pm_eq.append(np.maximum(eq, 1e-300))

    # Log-likelihood
    ll = 0.0

    # No-peak firms: P(no peak) = exp(-(λ_e + λ_m + λ_c))
    ll += n_no_peak * (-lams.sum())

    # Peak firms: P(disrupta=X, band=b_j) = P(M_X = b_j) · Π_{Y≠X} P(M_Y < b_j)
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
    """Fit the three-tier Poisson model for one size group."""
    peaks = v_grp[v_grp["peak"]].copy()
    n_no_peak = len(v_grp) - len(peaks)
    n_total = len(v_grp)

    # Build tier_band_counts
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

    # Initial guess: λ from empirical tier rates, PMF uniform
    lam_init = []
    for x, tier in enumerate(TIER_NAMES):
        n_tier_peaks = sum(v for (xx, j), v in tier_band_counts.items() if xx == x)
        p = n_tier_peaks / n_total
        lam = -np.log(1 - p) if p < 1 and p > 0 else 0.01
        lam_init.append(np.log(lam))
    theta0 = np.array(lam_init + [0.0] * (N_TIERS * (N_BANDS - 1)))

    result = minimize(neg_log_likelihood, theta0,
                      args=(n_no_peak, tier_band_counts),
                      method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-12})

    if not result.success:
        print(f"  WARNING: optimization did not converge: {result.message}")

    lams, pmfs = unpack_params(result.x)
    cdfs = [np.cumsum(pmf) for pmf in pmfs]

    return lams, pmfs, cdfs, result


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)

    att = df[df["attacked"]].copy()
    v = att[att["band"].notna() & att["freq"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["peak"] = v["band"] >= PEAK_CUT
    v["size_group"] = np.where(v["sizeb"] == 1, "Micro", "Rest")
    v["tier"] = v["disrupta"].map(TIER_MAP)

    # Prevalence
    prev = {}
    for sz in SIZE_LABELS:
        sub = df[df["sizeb"] == sz]
        prev[sz] = (sub["attacked"].astype(int) * sub["weight"]).sum() / sub["weight"].sum()

    # ── Fit per size group ──
    print("=" * 74)
    print("MLE FIT — THREE INDEPENDENT POISSON PEAK PROCESSES")
    print("=" * 74)

    all_results = {}
    for grp in ["Micro", "Rest"]:
        grp_data = v[v["size_group"] == grp]
        lams, pmfs, cdfs, opt = fit_group(grp_data, grp)
        all_results[grp] = {"lams": lams, "pmfs": pmfs, "cdfs": cdfs, "opt": opt}

        print(f"\n  Fitted λ values:")
        for x, tier in enumerate(TIER_NAMES):
            print(f"    {tier:<12s} λ={lams[x]:.4f}")
        print(f"    {'TOTAL':<12s} λ={lams.sum():.4f}")

        # Compare with empirical
        p_peak_emp = grp_data["peak"].mean()
        lam_emp = -np.log(1 - p_peak_emp) if p_peak_emp < 1 else 5.0
        print(f"    Empirical λ_total (from binary): {lam_emp:.4f}")

        print(f"\n  Fitted T distributions (per-draw PMF):")
        print(f"    {'band':>5s} {'£':>10s}", end="")
        for tier in TIER_NAMES:
            print(f" {tier:>12s}", end="")
        print()
        for j, b in enumerate(PEAK_BANDS):
            cost = band_to_gbp(b)
            print(f"    {b:>5d} {cost:>10,.0f}", end="")
            for x in range(N_TIERS):
                print(f" {pmfs[x][j]:>12.4f}", end="")
            print()

        # E[T] per tier
        print(f"\n  E[T] per tier (per-draw):")
        for x, tier in enumerate(TIER_NAMES):
            et = (pmfs[x] * BAND_COSTS).sum()
            print(f"    {tier:<12s} E[T]=£{et:>10,.0f}")

        # For comparison: empirical E[T] among disrupta-assigned peaks
        print(f"\n  Empirical E[T] per tier (observed max, no correction):")
        for tier in TIER_NAMES:
            tp = grp_data[(grp_data["peak"]) & (grp_data["tier"] == tier)]
            if len(tp) >= 3:
                et_emp = tp["band"].map(band_to_gbp).mean()
                print(f"    {tier:<12s} E[T]=£{et_emp:>10,.0f} (n={len(tp)})")

    # ── Goodness of fit ──
    print(f"\n{'=' * 74}")
    print("GOODNESS OF FIT — predicted vs observed tier-band counts")
    print("=" * 74)

    for grp in ["Micro", "Rest"]:
        grp_data = v[v["size_group"] == grp]
        n_total = len(grp_data)
        lams = all_results[grp]["lams"]
        pmfs = all_results[grp]["pmfs"]
        cdfs = [np.cumsum(pmf) for pmf in pmfs]

        # Compute predicted probabilities
        pm_leq = []
        pm_lt = []
        pm_eq = []
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
        print(f"    {'tier':<12s} {'band':>5s} {'obs':>6s} {'pred':>8s}")

        for x, tier in enumerate(TIER_NAMES):
            for j, b in enumerate(PEAK_BANDS):
                obs = len(grp_data[(grp_data["peak"]) & (grp_data["tier"] == tier)
                                   & (grp_data["band"] == b)])
                # P(disrupta=X, band=b_j)
                p = pm_eq[x][j]
                for y in range(N_TIERS):
                    if y != x:
                        p *= pm_lt[y][j]
                pred = p * n_total
                if obs > 0 or pred > 0.5:
                    print(f"    {tier:<12s} {b:>5d} {obs:>6d} {pred:>8.1f}")

        # No-peak
        obs_no = n_total - len(grp_data[grp_data["peak"]])
        pred_no = n_total * np.exp(-lams.sum())
        print(f"    {'no peak':<12s} {'':>5s} {obs_no:>6d} {pred_no:>8.1f}")

    # ── Formal GOF ──
    print(f"\n{'=' * 74}")
    print("FORMAL GOODNESS OF FIT")
    print("=" * 74)

    from scipy.stats import chi2

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

        # Collect observed and expected counts (pool cells with expected < 5)
        obs_list = []
        exp_list = []
        labels = []

        # No-peak cell
        obs_no = n_total - len(grp_data[grp_data["peak"]])
        exp_no = n_total * np.exp(-lams.sum())
        obs_list.append(obs_no)
        exp_list.append(exp_no)
        labels.append("no_peak")

        # Tier × band cells
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
                labels.append(f"{tier}_{b}")

        obs_arr = np.array(obs_list, dtype=float)
        exp_arr = np.array(exp_list, dtype=float)

        # Pool small-expected cells for chi-sq validity
        pooled_obs = []
        pooled_exp = []
        pool_obs = 0
        pool_exp = 0
        n_cells_raw = len(obs_arr)
        for i in range(len(obs_arr)):
            pool_obs += obs_arr[i]
            pool_exp += exp_arr[i]
            if pool_exp >= 3:
                pooled_obs.append(pool_obs)
                pooled_exp.append(pool_exp)
                pool_obs = 0
                pool_exp = 0
        if pool_exp > 0:
            if len(pooled_obs) > 0:
                pooled_obs[-1] += pool_obs
                pooled_exp[-1] += pool_exp
            else:
                pooled_obs.append(pool_obs)
                pooled_exp.append(pool_exp)

        pooled_obs = np.array(pooled_obs)
        pooled_exp = np.array(pooled_exp)

        chi_sq = ((pooled_obs - pooled_exp) ** 2 / pooled_exp).sum()
        n_params = 24  # 3 λ + 3×7 PMF
        dof = len(pooled_obs) - 1 - n_params
        if dof > 0:
            p_val = 1 - chi2.cdf(chi_sq, dof)
        else:
            p_val = float('nan')

        # Also compute G-test (log-likelihood ratio)
        mask = pooled_obs > 0
        g_stat = 2 * (pooled_obs[mask] * np.log(pooled_obs[mask] / pooled_exp[mask])).sum()

        print(f"\n  {grp}:")
        print(f"    Raw cells: {n_cells_raw}, pooled cells (exp≥3): {len(pooled_obs)}")
        print(f"    χ² = {chi_sq:.2f}, df = {dof}, p = {p_val:.4f}" if dof > 0
              else f"    χ² = {chi_sq:.2f}, df = {dof} (too few cells for valid test)")
        print(f"    G-test = {g_stat:.2f}")
        print(f"    Σ|obs-exp| = {np.abs(obs_arr - exp_arr).sum():.1f}")
        print(f"    max|obs-exp| = {np.abs(obs_arr - exp_arr).max():.1f}")
        # Mean absolute percentage error (on cells with obs > 0)
        nonzero = obs_arr > 0
        if nonzero.any():
            mape = np.abs((obs_arr[nonzero] - exp_arr[nonzero]) / obs_arr[nonzero]).mean()
            print(f"    MAPE (on nonzero cells) = {mape:.1%}")

    # ── National total ──
    print(f"\n{'=' * 74}")
    print("NATIONAL TOTAL — MLE-BASED")
    print("=" * 74)

    # E[cost | attacked, size_group] = Σ_tier λ_tier · E[T_tier]
    # (using fitted per-draw T, so max-bias is already corrected)
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

    # Per-tier contribution
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

    # ── Comparison summary ──
    print(f"\n{'=' * 74}")
    print("COMPARISON WITH PREVIOUS ESTIMATES")
    print("=" * 74)
    print(f"  Original (n_types λ, pooled T, no correction):    ~£2.17bn")
    print(f"  Original + max-bias correction:                    ~£1.85bn")
    print(f"  Tier-decomposed (disrupta λ, disrupta T, corr):   ~£1.60bn")
    print(f"  MLE (this script):                                 £{total/1e9:.2f}bn")


if __name__ == "__main__":
    main()
