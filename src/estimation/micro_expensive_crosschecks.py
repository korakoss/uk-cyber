"""Cross-checks on the poorly-identified Micro-expensive cell of the tier MLE.

The lognormal tier MLE's weakest cell is (Micro, expensive): E[T] 90% CI
spans £3k-£31k (10x). Four independent-evidence checks:

  1. BRIDGE CROSS-CHECK: the earlier censored-MLE per-type figures
     (ranscost_bands etc. — a different measurement channel entirely) imply
     a per-attacked-business cost for the expensive types. Compare with the
     MLE's population-weighted λ_e·E[T_e].
  2. CROSS-SIZE RATIO SCALING: Micro/Rest E[T] ratios per tier. Does the
     expensive tier's ratio look like the better-identified tiers'?
  3. SIZE GRADIENT: empirical mean peak cost in the expensive tier for
     Small/Medium/Large separately. Does Micro sit below Small (monotone)?
  4. FREQ=1 DIRECT DRAWS: once-only firms need no max-bias correction —
     their band IS a single draw of T. Compare Micro expensive freq=1
     draws against the fitted lognormal.

Run: ``python src/estimation/micro_expensive_crosschecks.py``
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import lognorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BAND_BOUNDS, SIZE_LABELS, load

PEAK_CUT = 3
N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}
PEAK_BANDS = [3, 4, 5, 6, 7, 8, 9, 10]
TIER_NAMES = ["expensive", "mid", "cheap"]

TIER_MAP = {
    1: "expensive", 3: "expensive", 4: "expensive",
    2: "mid", 5: "mid", 7: "mid", 8: "mid", 11: "mid", 12: "mid", 9: "mid",
    6: "cheap",
}

# Fitted lognormal-MLE point estimates (tier_mle_lognormal.py, 2026-09 session)
MLE_FIT = {
    ("Micro", "expensive"): {"lam": 0.0553, "mu": 0.72, "sigma": 4.64, "ET": 15_060},
    ("Micro", "mid"):       {"lam": 0.1017, "mu": 3.69, "sigma": 2.72, "ET": 4_227},
    ("Micro", "cheap"):     {"lam": 0.1425, "mu": 5.58, "sigma": 1.61, "ET": 1_634},
    ("Rest", "expensive"):  {"lam": 0.0619, "mu": 8.50, "sigma": 3.67, "ET": 53_539},
    ("Rest", "mid"):        {"lam": 0.2122, "mu": 5.17, "sigma": 2.56, "ET": 7_074},
    ("Rest", "cheap"):      {"lam": 0.1810, "mu": 3.81, "sigma": 2.99, "ET": 6_870},
}

# Bridge-side per-type figures (NOTES.md 2026-07-14 handoff): weighted
# £/attacked-business from the censored-MLE fits on the TYPE-SPECIFIC cost
# columns (ranscost_bands, doscost_bands, hackcost_bands...) — an
# independent measurement channel from damage_bands+disrupta.
BRIDGE_PER_TYPE = {
    "Ransomware": 605.71,
    "Denial of service": 52.25,
    "Hacking (broad)": 244.09,
}


def band_to_gbp(band, tail_multiple=2.0):
    lo, hi = BAND_BOUNDS.get(band, (0, 0))
    if band == 1: return 0.0
    if hi == np.inf: return lo * tail_multiple
    return (lo + hi) / 2


def lognormal_band_pmf(mu, sigma):
    raw = []
    for b in PEAK_BANDS:
        lo, hi = BAND_BOUNDS[b]
        cdf_lo = lognorm.cdf(lo, s=sigma, scale=np.exp(mu)) if lo > 0 else 0.0
        cdf_hi = 1.0 if hi == np.inf else lognorm.cdf(hi, s=sigma, scale=np.exp(mu))
        raw.append(max(cdf_hi - cdf_lo, 1e-300))
    raw = np.array(raw)
    return raw / raw.sum()


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

    # ================================================================
    print("=" * 74)
    print("1. BRIDGE CROSS-CHECK — independent measurement channel")
    print("=" * 74)

    bridge_total = sum(BRIDGE_PER_TYPE.values())
    print(f"\n  Bridge-side (censored MLE on type-specific cost columns),")
    print(f"  weighted £/attacked-business, population-representative:")
    for t, val in BRIDGE_PER_TYPE.items():
        print(f"    {t:<20s} £{val:>8,.2f}")
    print(f"    {'SUM (expensive tier)':<20s} £{bridge_total:>8,.2f}")

    # MLE-side: population-weighted expensive-tier λ·E[T] per attacked business
    n_attacked = {}
    for sz in SIZE_LABELS:
        n_attacked[sz] = N_BY_SIZE[sz] * prev[sz]
    total_attacked = sum(n_attacked.values())

    mle_weighted = 0.0
    for sz in SIZE_LABELS:
        grp = "Micro" if sz == 1 else "Rest"
        fit = MLE_FIT[(grp, "expensive")]
        contrib = fit["lam"] * fit["ET"]
        mle_weighted += n_attacked[sz] * contrib
    mle_weighted /= total_attacked

    micro_contrib = MLE_FIT[("Micro", "expensive")]["lam"] * MLE_FIT[("Micro", "expensive")]["ET"]
    rest_contrib = MLE_FIT[("Rest", "expensive")]["lam"] * MLE_FIT[("Rest", "expensive")]["ET"]

    print(f"\n  MLE-side (this model), λ_e·E[T_e] per attacked firm:")
    print(f"    Micro: £{micro_contrib:,.0f}   Rest: £{rest_contrib:,.0f}")
    print(f"    Attacked-population weights: Micro {n_attacked[1]/total_attacked:.1%}, "
          f"Rest {1 - n_attacked[1]/total_attacked:.1%}")
    print(f"    Population-weighted: £{mle_weighted:,.0f}/attacked business")

    print(f"\n  RATIO (MLE / bridge): {mle_weighted / bridge_total:.2f}x")
    print(f"  Note: bridge side counts only firms where the type OCCURRED and is")
    print(f"  built on different cost concepts (type-total vs worst-incident),")
    print(f"  so ~2x agreement is meaningful; ~10x would be a red flag.")

    # Implied Micro-expensive E[T] if we forced the MLE to match the bridge total,
    # holding Rest fixed:
    implied_micro_contrib = (bridge_total * total_attacked
                             - rest_contrib * (total_attacked - n_attacked[1])) / n_attacked[1]
    implied_ET = implied_micro_contrib / MLE_FIT[("Micro", "expensive")]["lam"]
    print(f"\n  If the MLE were forced to match the bridge sum (holding Rest fixed):")
    print(f"    implied Micro λ_e·E[T_e] = £{implied_micro_contrib:,.0f}/firm")
    print(f"    implied Micro E[T_e]     = £{implied_ET:,.0f}")
    print(f"    (fitted: £{MLE_FIT[('Micro','expensive')]['ET']:,.0f}; "
          f"bootstrap 90% CI £3k-£31k)")

    # ================================================================
    print(f"\n{'=' * 74}")
    print("2. CROSS-SIZE RATIO SCALING — Micro/Rest E[T] per tier")
    print("=" * 74)

    print(f"\n  {'tier':<12s} {'Micro E[T]':>12s} {'Rest E[T]':>12s} {'ratio':>7s}")
    for tier in TIER_NAMES:
        m = MLE_FIT[("Micro", tier)]["ET"]
        r = MLE_FIT[("Rest", tier)]["ET"]
        print(f"  {tier:<12s} £{m:>10,.0f} £{r:>10,.0f} {m/r:>7.2f}")

    # Same on raw observed maxima (no model) for reference
    print(f"\n  Raw observed-max means (no correction, for reference):")
    print(f"  {'tier':<12s} {'Micro obs':>12s} {'n':>4s} {'Rest obs':>12s} {'n':>4s} {'ratio':>7s}")
    for tier in TIER_NAMES:
        mp = v[(v["size_group"] == "Micro") & v["peak"] & (v["tier"] == tier)]
        rp = v[(v["size_group"] == "Rest") & v["peak"] & (v["tier"] == tier)]
        m_obs = mp["band"].map(band_to_gbp).mean()
        r_obs = rp["band"].map(band_to_gbp).mean()
        print(f"  {tier:<12s} £{m_obs:>10,.0f} {len(mp):>4d} £{r_obs:>10,.0f} {len(rp):>4d} "
              f"{m_obs/r_obs:>7.2f}")

    # ================================================================
    print(f"\n{'=' * 74}")
    print("3. SIZE GRADIENT — expensive-tier observed peak costs by size band")
    print("=" * 74)

    print(f"\n  Empirical mean/median of band-£ among expensive-tier peaks:")
    print(f"  {'size':<8s} {'n':>4s} {'mean £':>10s} {'median £':>10s}  bands")
    for sz, label in SIZE_LABELS.items():
        p = v[(v["sizeb"] == sz) & v["peak"] & (v["tier"] == "expensive")]
        if len(p) == 0:
            print(f"  {label:<8s} {0:>4d}")
            continue
        costs = p["band"].map(band_to_gbp)
        bands = sorted(p["band"].tolist())
        print(f"  {label:<8s} {len(p):>4d} £{costs.mean():>9,.0f} £{costs.median():>9,.0f}  {bands}")
    print(f"\n  (These are observed MAXIMA — biased up vs per-draw T, but the bias")
    print(f"   is similar across sizes at similar λ, so the ORDERING is informative.)")

    # ================================================================
    print(f"\n{'=' * 74}")
    print("4. FREQ=1 DIRECT DRAWS — no max-bias, single draw of T")
    print("=" * 74)

    for grp in ["Micro", "Rest"]:
        p1 = v[(v["size_group"] == grp) & v["peak"] & (v["tier"] == "expensive")
               & (v["freq"] == 1)]
        fit = MLE_FIT[(grp, "expensive")]
        pmf = lognormal_band_pmf(fit["mu"], fit["sigma"])

        print(f"\n  {grp} expensive, freq=1 (once-only): n={len(p1)}")
        if len(p1) == 0:
            continue
        bands = sorted(p1["band"].tolist())
        print(f"    observed bands: {bands}")
        print(f"    observed £:     {[f'£{band_to_gbp(b):,.0f}' for b in bands]}")

        # Log-likelihood of these draws under the fitted per-draw lognormal
        ll = sum(np.log(pmf[PEAK_BANDS.index(b)]) for b in bands)
        print(f"    fitted per-draw PMF on bands 3-10: "
              f"{[f'{x:.3f}' for x in pmf]}")
        print(f"    log-lik of these draws under fitted T: {ll:.2f} "
              f"(mean per draw {ll/len(bands):.2f})")

        # Compare with alternative sigmas at same median... simple sensitivity
        for sig_alt in [1.5, 2.5, 3.5]:
            pmf_alt = lognormal_band_pmf(fit["mu"], sig_alt)
            ll_alt = sum(np.log(pmf_alt[PEAK_BANDS.index(b)]) for b in bands)
            print(f"      vs sigma={sig_alt}: log-lik {ll_alt:.2f}")

    # Also: what does the freq=1 subset's mean say about E[T] directly?
    print(f"\n  Direct E[T] estimates from freq=1 draws (unbiased, tiny n):")
    for grp in ["Micro", "Rest"]:
        p1 = v[(v["size_group"] == grp) & v["peak"] & (v["tier"] == "expensive")
               & (v["freq"] == 1)]
        if len(p1) > 0:
            et1 = p1["band"].map(band_to_gbp).mean()
            print(f"    {grp}: mean £{et1:,.0f} (n={len(p1)}) vs "
                  f"fitted E[T]=£{MLE_FIT[(grp,'expensive')]['ET']:,.0f}")


if __name__ == "__main__":
    main()
