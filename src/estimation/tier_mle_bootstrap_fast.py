"""Bootstrap the lognormal-T tier MLE — fast vectorized version.

Run: ``python src/estimation/tier_mle_bootstrap_fast.py``
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import lognorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BAND_BOUNDS, SIZE_LABELS, load

PEAK_CUT = 3
N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}
PEAK_BANDS = [3, 4, 5, 6, 7, 8, 9, 10]
N_BANDS = len(PEAK_BANDS)
TIER_NAMES = ["expensive", "mid", "cheap"]
N_TIERS = 3
B_REPS = 200

TIER_MAP = {
    1: "expensive", 3: "expensive", 4: "expensive",
    2: "mid", 5: "mid", 7: "mid", 8: "mid", 11: "mid", 12: "mid", 9: "mid",
    6: "cheap",
}

BAND_COSTS = np.array([band_to_gbp(b) for b in PEAK_BANDS] if False else [])

def _init_costs():
    global BAND_COSTS, BAND_EDGES
    costs = []
    edges = []
    for b in PEAK_BANDS:
        lo, hi = BAND_BOUNDS[b]
        if b == 1: costs.append(0.0)
        elif hi == np.inf: costs.append(lo * 2.0)
        else: costs.append((lo + hi) / 2)
        edges.append((lo, hi))
    BAND_COSTS = np.array(costs)
    BAND_EDGES = edges

_init_costs()


def lognormal_band_pmf(mu, sigma):
    raw = np.zeros(N_BANDS)
    for j, (lo, hi) in enumerate(BAND_EDGES):
        cdf_lo = lognorm.cdf(lo, s=sigma, scale=np.exp(mu)) if lo > 0 else 0.0
        cdf_hi = 1.0 if hi == np.inf else lognorm.cdf(hi, s=sigma, scale=np.exp(mu))
        raw[j] = max(cdf_hi - cdf_lo, 1e-300)
    total = raw.sum()
    if total < 1e-200:
        return np.ones(N_BANDS) / N_BANDS
    return raw / total


def unpack_params(theta):
    lams = np.exp(np.clip(theta[:3], -10, 5))
    pmfs = []
    for i in range(3):
        mu = theta[3 + 2*i]
        sigma = np.exp(np.clip(theta[4 + 2*i], -3, 4))
        pmfs.append(lognormal_band_pmf(mu, sigma))
    return lams, pmfs


def neg_log_likelihood(theta, n_no_peak, tier_band_counts):
    lams, pmfs = unpack_params(theta)
    cdfs = [np.cumsum(pmf) for pmf in pmfs]

    pm_eq = []
    pm_lt = []
    for x in range(N_TIERS):
        leq = np.exp(-lams[x] * (1 - cdfs[x]))
        lt = np.empty(N_BANDS)
        lt[0] = np.exp(-lams[x])
        lt[1:] = leq[:-1]
        pm_eq.append(np.maximum(leq - lt, 1e-300))
        pm_lt.append(lt)

    ll = n_no_peak * (-lams.sum())
    for x in range(N_TIERS):
        for j in range(N_BANDS):
            count = tier_band_counts[x, j]
            if count == 0:
                continue
            log_p = np.log(pm_eq[x][j])
            for y in range(N_TIERS):
                if y != x:
                    log_p += np.log(np.maximum(pm_lt[y][j], 1e-300))
            ll += count * log_p
    return -ll


def build_counts_fast(peak_arr, tier_arr, n_total):
    """Build tier_band_counts as a (3, 8) array from pre-encoded arrays."""
    counts = np.zeros((N_TIERS, N_BANDS), dtype=int)
    for i in range(len(peak_arr)):
        if peak_arr[i] >= 0 and tier_arr[i] >= 0:
            counts[tier_arr[i], peak_arr[i]] += 1
    n_no_peak = n_total - counts.sum()
    return n_no_peak, counts


def fit_from_counts(n_no_peak, counts, theta0):
    result = minimize(neg_log_likelihood, theta0,
                      args=(n_no_peak, counts),
                      method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-10})
    lams, pmfs = unpack_params(result.x)
    return lams, pmfs, result


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

    # Pre-encode for fast bootstrap
    # peak_band_idx: -1 if not peak or no tier, else index into PEAK_BANDS
    # tier_idx: -1 if no tier, else index into TIER_NAMES
    group_data = {}
    for grp in ["Micro", "Rest"]:
        gd = v[v["size_group"] == grp].copy()
        n = len(gd)

        peak_band_idx = np.full(n, -1, dtype=int)
        tier_idx = np.full(n, -1, dtype=int)

        for i, (_, row) in enumerate(gd.iterrows()):
            if row["peak"] and row["tier"] in TIER_NAMES:
                band = int(row["band"])
                if band in PEAK_BANDS:
                    peak_band_idx[i] = PEAK_BANDS.index(band)
                    tier_idx[i] = TIER_NAMES.index(row["tier"])

        group_data[grp] = {"n": n, "peak_band_idx": peak_band_idx, "tier_idx": tier_idx}

    # Point estimate
    print("=" * 74)
    print("POINT ESTIMATE")
    print("=" * 74)

    point_results = {}
    for grp in ["Micro", "Rest"]:
        gd = group_data[grp]
        n_no_peak, counts = build_counts_fast(gd["peak_band_idx"], gd["tier_idx"], gd["n"])

        # Initial theta
        lam_init = []
        mu_sigma_init = []
        for x in range(N_TIERS):
            n_tier = counts[x].sum()
            p = n_tier / gd["n"]
            lam = -np.log(1 - p) if 0 < p < 1 else 0.01
            lam_init.append(np.log(lam))
            tier_costs = []
            for j in range(N_BANDS):
                tier_costs.extend([BAND_COSTS[j]] * counts[x, j])
            tier_costs = [c for c in tier_costs if c > 0]
            if tier_costs:
                mu_sigma_init.extend([np.log(np.median(tier_costs)), np.log(1.0)])
            else:
                mu_sigma_init.extend([np.log(1000), np.log(1.0)])

        theta0 = np.array(lam_init + mu_sigma_init)
        lams, pmfs, opt = fit_from_counts(n_no_peak, counts, theta0)
        point_results[grp] = {"lams": lams, "pmfs": pmfs, "theta": opt.x.copy()}

        e_cost = sum(lams[x] * (pmfs[x] * BAND_COSTS).sum() for x in range(N_TIERS))
        print(f"\n  {grp}: E[cost|att] = £{e_cost:,.0f}")
        for x, tier in enumerate(TIER_NAMES):
            et = (pmfs[x] * BAND_COSTS).sum()
            mu = opt.x[3 + 2*x]
            sigma = np.exp(np.clip(opt.x[4 + 2*x], -3, 4))
            print(f"    {tier:<12s} λ={lams[x]:.4f}, E[T]=£{et:>8,.0f} (μ={mu:.2f}, σ={sigma:.2f})")

    total_point = 0
    for sz in SIZE_LABELS:
        grp = "Micro" if sz == 1 else "Rest"
        lams = point_results[grp]["lams"]
        pmfs = point_results[grp]["pmfs"]
        e_cost = sum(lams[x] * (pmfs[x] * BAND_COSTS).sum() for x in range(N_TIERS))
        total_point += N_BY_SIZE[sz] * prev[sz] * e_cost
    print(f"\n  National total: £{total_point/1e9:.3f}bn")

    # Bootstrap
    print(f"\n{'=' * 74}")
    print(f"BOOTSTRAP ({B_REPS} replicates)")
    print("=" * 74)

    np.random.seed(42)

    boot_totals = []
    boot_ecost = {"Micro": [], "Rest": []}
    boot_lams = {g: {t: [] for t in TIER_NAMES} for g in ["Micro", "Rest"]}
    boot_ets = {g: {t: [] for t in TIER_NAMES} for g in ["Micro", "Rest"]}
    boot_sigmas = {g: {t: [] for t in TIER_NAMES} for g in ["Micro", "Rest"]}
    n_failed = 0

    for rep in range(B_REPS):
        if (rep + 1) % 50 == 0:
            print(f"  rep {rep+1}/{B_REPS}...")

        rep_results = {}
        ok = True
        for grp in ["Micro", "Rest"]:
            gd = group_data[grp]
            n = gd["n"]
            idx = np.random.choice(n, size=n, replace=True)
            boot_pbi = gd["peak_band_idx"][idx]
            boot_ti = gd["tier_idx"][idx]

            n_no_peak, counts = build_counts_fast(boot_pbi, boot_ti, n)

            try:
                lams, pmfs, opt = fit_from_counts(n_no_peak, counts, point_results[grp]["theta"])
                e_cost = sum(lams[x] * (pmfs[x] * BAND_COSTS).sum() for x in range(N_TIERS))
                boot_ecost[grp].append(e_cost)

                for x, tier in enumerate(TIER_NAMES):
                    et = (pmfs[x] * BAND_COSTS).sum()
                    boot_lams[grp][tier].append(lams[x])
                    boot_ets[grp][tier].append(et)
                    sigma = np.exp(np.clip(opt.x[4 + 2*x], -3, 4))
                    boot_sigmas[grp][tier].append(sigma)

                rep_results[grp] = {"lams": lams, "pmfs": pmfs}
            except Exception:
                n_failed += 1
                ok = False

        if ok:
            total = 0
            for sz in SIZE_LABELS:
                grp = "Micro" if sz == 1 else "Rest"
                lams = rep_results[grp]["lams"]
                pmfs = rep_results[grp]["pmfs"]
                e_cost = sum(lams[x] * (pmfs[x] * BAND_COSTS).sum() for x in range(N_TIERS))
                total += N_BY_SIZE[sz] * prev[sz] * e_cost
            boot_totals.append(total)

    print(f"\n  Completed: {len(boot_totals)} replicates, {n_failed} failures")

    bt = np.array(boot_totals)
    print(f"\n{'=' * 74}")
    print("NATIONAL TOTAL — BOOTSTRAP DISTRIBUTION")
    print("=" * 74)
    print(f"  Point estimate:   £{total_point/1e9:.3f}bn")
    print(f"  Bootstrap mean:   £{bt.mean()/1e9:.3f}bn")
    print(f"  Bootstrap median: £{np.median(bt)/1e9:.3f}bn")
    print(f"  Bootstrap SD:     £{bt.std()/1e9:.3f}bn")
    print(f"  5th pctile:       £{np.percentile(bt, 5)/1e9:.3f}bn")
    print(f"  25th pctile:      £{np.percentile(bt, 25)/1e9:.3f}bn")
    print(f"  75th pctile:      £{np.percentile(bt, 75)/1e9:.3f}bn")
    print(f"  95th pctile:      £{np.percentile(bt, 95)/1e9:.3f}bn")
    print(f"  90% CI:           £{np.percentile(bt, 5)/1e9:.3f}bn – £{np.percentile(bt, 95)/1e9:.3f}bn")

    print(f"\n{'=' * 74}")
    print("E[cost|attacked] — BOOTSTRAP")
    print("=" * 74)
    for grp in ["Micro", "Rest"]:
        ec = np.array(boot_ecost[grp])
        pt = sum(point_results[grp]["lams"][x] * (point_results[grp]["pmfs"][x] * BAND_COSTS).sum()
                 for x in range(N_TIERS))
        print(f"\n  {grp}:")
        print(f"    Point:  £{pt:>8,.0f}")
        print(f"    Mean:   £{ec.mean():>8,.0f}")
        print(f"    Median: £{np.median(ec):>8,.0f}")
        print(f"    90% CI: £{np.percentile(ec, 5):>8,.0f} – £{np.percentile(ec, 95):>8,.0f}")

    print(f"\n{'=' * 74}")
    print("PER-TIER PARAMETERS — BOOTSTRAP")
    print("=" * 74)
    for grp in ["Micro", "Rest"]:
        print(f"\n  {grp}:")
        print(f"    {'tier':<12s} {'λ [90%CI]':>24s}  {'E[T] [90%CI]':>32s}  {'σ [90%CI]':>20s}")
        for x, tier in enumerate(TIER_NAMES):
            la = np.array(boot_lams[grp][tier])
            ea = np.array(boot_ets[grp][tier])
            sa = np.array(boot_sigmas[grp][tier])
            lam_pt = point_results[grp]["lams"][x]
            et_pt = (point_results[grp]["pmfs"][x] * BAND_COSTS).sum()
            sig_pt = np.exp(np.clip(point_results[grp]["theta"][4 + 2*x], -3, 4))

            print(f"    {tier:<12s} "
                  f"{lam_pt:.4f} [{np.percentile(la,5):.4f}, {np.percentile(la,95):.4f}]  "
                  f"£{et_pt:>8,.0f} [£{np.percentile(ea,5):>7,.0f}, £{np.percentile(ea,95):>7,.0f}]  "
                  f"{sig_pt:.2f} [{np.percentile(sa,5):.2f}, {np.percentile(sa,95):.2f}]")

    print(f"\n{'=' * 74}")
    print("λ×E[T] PER TIER — BOOTSTRAP")
    print("=" * 74)
    for grp in ["Micro", "Rest"]:
        print(f"\n  {grp}:")
        for x, tier in enumerate(TIER_NAMES):
            la = np.array(boot_lams[grp][tier])
            ea = np.array(boot_ets[grp][tier])
            contrib = la * ea
            pt = point_results[grp]["lams"][x] * (point_results[grp]["pmfs"][x] * BAND_COSTS).sum()
            print(f"    {tier:<12s} point £{pt:>8,.0f}  "
                  f"mean £{contrib.mean():>8,.0f}  "
                  f"90% CI [£{np.percentile(contrib,5):>8,.0f}, £{np.percentile(contrib,95):>8,.0f}]")


if __name__ == "__main__":
    main()
