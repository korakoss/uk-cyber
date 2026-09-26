"""Bootstrap the lognormal-T tier MLE to get confidence intervals.

Resample firms (within size group, with replacement), refit the 9-param
lognormal MLE each time, and collect the spread of national totals,
per-tier λ and E[T], and per-group E[cost|att].

Run: ``python src/estimation/tier_mle_bootstrap.py``
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
B_REPS = 500

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

BAND_EDGES = []
for b in PEAK_BANDS:
    lo, hi = BAND_BOUNDS[b]
    BAND_EDGES.append((lo, hi))


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

    ll = n_no_peak * (-lams.sum())
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


def build_counts(v_grp):
    peaks = v_grp[v_grp["peak"]].copy()
    n_no_peak = len(v_grp) - len(peaks)
    tier_band_counts = {}
    for _, row in peaks.iterrows():
        tier = row["tier"]
        band = int(row["band"])
        if tier not in TIER_NAMES or band not in PEAK_BANDS:
            continue
        x = TIER_NAMES.index(tier)
        j = PEAK_BANDS.index(band)
        tier_band_counts[(x, j)] = tier_band_counts.get((x, j), 0) + 1
    return n_no_peak, tier_band_counts


def fit_group(v_grp, theta0=None):
    n_no_peak, tier_band_counts = build_counts(v_grp)
    n_total = len(v_grp)

    if theta0 is None:
        lam_init = []
        mu_sigma_init = []
        for x in range(N_TIERS):
            n_tier = sum(v for (xx, j), v in tier_band_counts.items() if xx == x)
            p = n_tier / n_total
            lam = -np.log(1 - p) if 0 < p < 1 else 0.01
            lam_init.append(np.log(lam))
            costs = []
            for (xx, j), cnt in tier_band_counts.items():
                if xx == x:
                    costs.extend([BAND_COSTS[j]] * cnt)
            if costs:
                costs = np.array([c for c in costs if c > 0])
                if len(costs) > 0:
                    mu_sigma_init.extend([np.log(np.median(costs)), np.log(1.0)])
                else:
                    mu_sigma_init.extend([np.log(1000), np.log(1.0)])
            else:
                mu_sigma_init.extend([np.log(1000), np.log(1.0)])
        theta0 = np.array(lam_init + mu_sigma_init)

    result = minimize(neg_log_likelihood, theta0,
                      args=(n_no_peak, tier_band_counts),
                      method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-12})

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

    # Point estimate first
    print("=" * 74)
    print("POINT ESTIMATE (full sample)")
    print("=" * 74)

    point_results = {}
    for grp in ["Micro", "Rest"]:
        gd = v[v["size_group"] == grp]
        lams, pmfs, opt = fit_group(gd)
        point_results[grp] = {"lams": lams, "pmfs": pmfs, "theta": opt.x.copy()}
        e_cost = sum(lams[x] * (pmfs[x] * BAND_COSTS).sum() for x in range(N_TIERS))
        print(f"\n  {grp}: E[cost|att] = £{e_cost:,.0f}")
        for x, tier in enumerate(TIER_NAMES):
            et = (pmfs[x] * BAND_COSTS).sum()
            mu = opt.x[3 + 2*x]
            sigma = np.exp(np.clip(opt.x[4 + 2*x], -3, 4))
            print(f"    {tier:<12s} λ={lams[x]:.4f}, E[T]=£{et:>8,.0f} (μ={mu:.2f}, σ={sigma:.2f})")

    total_point = 0
    for sz, sz_label in SIZE_LABELS.items():
        grp = "Micro" if sz == 1 else "Rest"
        lams = point_results[grp]["lams"]
        pmfs = point_results[grp]["pmfs"]
        e_cost = sum(lams[x] * (pmfs[x] * BAND_COSTS).sum() for x in range(N_TIERS))
        nat = N_BY_SIZE[sz] * prev[sz] * e_cost
        total_point += nat
    print(f"\n  National total: £{total_point/1e9:.3f}bn")

    # Bootstrap
    print(f"\n{'=' * 74}")
    print(f"BOOTSTRAP ({B_REPS} replicates)")
    print("=" * 74)

    np.random.seed(42)

    boot_totals = []
    boot_ecost = {"Micro": [], "Rest": []}
    boot_lams = {grp: {t: [] for t in TIER_NAMES} for grp in ["Micro", "Rest"]}
    boot_ets = {grp: {t: [] for t in TIER_NAMES} for grp in ["Micro", "Rest"]}
    boot_mus = {grp: {t: [] for t in TIER_NAMES} for grp in ["Micro", "Rest"]}
    boot_sigmas = {grp: {t: [] for t in TIER_NAMES} for grp in ["Micro", "Rest"]}

    n_failed = 0
    for rep in range(B_REPS):
        if (rep + 1) % 50 == 0:
            print(f"  rep {rep+1}/{B_REPS}...")

        rep_results = {}
        for grp in ["Micro", "Rest"]:
            gd = v[v["size_group"] == grp]
            # Resample with replacement
            idx = np.random.choice(len(gd), size=len(gd), replace=True)
            boot_gd = gd.iloc[idx].reset_index(drop=True)

            try:
                lams, pmfs, opt = fit_group(boot_gd, theta0=point_results[grp]["theta"])
                if not opt.success:
                    # Try from scratch
                    lams, pmfs, opt = fit_group(boot_gd)

                rep_results[grp] = {"lams": lams, "pmfs": pmfs, "theta": opt.x}

                e_cost = sum(lams[x] * (pmfs[x] * BAND_COSTS).sum() for x in range(N_TIERS))
                boot_ecost[grp].append(e_cost)

                for x, tier in enumerate(TIER_NAMES):
                    et = (pmfs[x] * BAND_COSTS).sum()
                    boot_lams[grp][tier].append(lams[x])
                    boot_ets[grp][tier].append(et)
                    mu = opt.x[3 + 2*x]
                    sigma = np.exp(np.clip(opt.x[4 + 2*x], -3, 4))
                    boot_mus[grp][tier].append(mu)
                    boot_sigmas[grp][tier].append(sigma)

            except Exception:
                n_failed += 1
                rep_results[grp] = None

        if all(rep_results[g] is not None for g in ["Micro", "Rest"]):
            total = 0
            for sz in SIZE_LABELS:
                grp = "Micro" if sz == 1 else "Rest"
                lams = rep_results[grp]["lams"]
                pmfs = rep_results[grp]["pmfs"]
                e_cost = sum(lams[x] * (pmfs[x] * BAND_COSTS).sum() for x in range(N_TIERS))
                total += N_BY_SIZE[sz] * prev[sz] * e_cost
            boot_totals.append(total)

    print(f"\n  Completed: {len(boot_totals)} replicates, {n_failed} failures")

    # Results
    bt = np.array(boot_totals)
    print(f"\n{'=' * 74}")
    print("BOOTSTRAP RESULTS — NATIONAL TOTAL")
    print("=" * 74)
    print(f"  Point estimate:  £{total_point/1e9:.3f}bn")
    print(f"  Bootstrap mean:  £{bt.mean()/1e9:.3f}bn")
    print(f"  Bootstrap median:£{np.median(bt)/1e9:.3f}bn")
    print(f"  Bootstrap SD:    £{bt.std()/1e9:.3f}bn")
    print(f"  5th percentile:  £{np.percentile(bt, 5)/1e9:.3f}bn")
    print(f"  25th percentile: £{np.percentile(bt, 25)/1e9:.3f}bn")
    print(f"  75th percentile: £{np.percentile(bt, 75)/1e9:.3f}bn")
    print(f"  95th percentile: £{np.percentile(bt, 95)/1e9:.3f}bn")
    print(f"  90% CI:          £{np.percentile(bt, 5)/1e9:.3f}bn – £{np.percentile(bt, 95)/1e9:.3f}bn")

    # Per-group E[cost|att]
    print(f"\n{'=' * 74}")
    print("BOOTSTRAP RESULTS — E[cost|attacked] by group")
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

    # Per-tier parameters
    print(f"\n{'=' * 74}")
    print("BOOTSTRAP RESULTS — per-tier parameters")
    print("=" * 74)
    for grp in ["Micro", "Rest"]:
        print(f"\n  {grp}:")
        print(f"    {'tier':<12s} {'λ point':>8s} {'λ 90%CI':>20s} "
              f"{'E[T] point':>10s} {'E[T] 90%CI':>24s} "
              f"{'σ point':>8s} {'σ 90%CI':>16s}")
        for x, tier in enumerate(TIER_NAMES):
            lam_pt = point_results[grp]["lams"][x]
            et_pt = (point_results[grp]["pmfs"][x] * BAND_COSTS).sum()
            sigma_pt = np.exp(np.clip(point_results[grp]["theta"][4 + 2*x], -3, 4))

            la = np.array(boot_lams[grp][tier])
            ea = np.array(boot_ets[grp][tier])
            sa = np.array(boot_sigmas[grp][tier])

            print(f"    {tier:<12s} {lam_pt:>8.4f} [{np.percentile(la,5):.4f}, {np.percentile(la,95):.4f}] "
                  f"£{et_pt:>9,.0f} [£{np.percentile(ea,5):>8,.0f}, £{np.percentile(ea,95):>8,.0f}] "
                  f"{sigma_pt:>8.2f} [{np.percentile(sa,5):.2f}, {np.percentile(sa,95):.2f}]")

    # λ×E[T] contribution per tier
    print(f"\n{'=' * 74}")
    print("BOOTSTRAP RESULTS — λ×E[T] contribution per tier")
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
