"""Approach A — the semiparametric national baseline (the ~£2.2bn headline).

This is the SPINE of the whole estimate. It converts the survey's single-worst-
incident banded cost into a national annual total via:

    Total = Sum_size N(size) * E[annual cost per firm | size]

For each firm: keep its OWN observed damage band (empirical band mix — the model
does NOT get to move how many firms sit in the top band), place a pound value
WITHIN that band using the fitted per-size lognormal (band_conditional_mean), and
multiply by the per-attack-type bridge. This is the "A" / semiparametric approach,
contrasted with the fully parametric "B" in estimate_size_type_lognormal.py which
discards the firm's band and uses the fitted cell mean instead.

Why bootstrap rather than multiply means: at N ~ 1.4M firms an i.i.d. draw per
firm would collapse the total to its mean by the CLT and hide the real
uncertainty, which lives in the INPUTS — prevalence, the band mix (esp. how many
firms land in the top band, only 8 in the whole sample), and the fragile
Impersonation bridge. So we bootstrap the survey (resample firms within each size
band, weighted by survey weight) B times; each replicate yields a cost-per-
business per size band; Python scales each by the fixed ONS N(size) and sums into
a distribution of the national total. The fitted lognormal shapes (mu,sigma) are
held at their point estimates across replicates (band-10 membership still varies
across resamples, so the dominant tail uncertainty is captured).

Cost model: band 10 is BOUNDED at £100k-£500k (data.BAND_BOUNDS); no unbounded
lognormal tail is used. The catastrophic >£500k tail is a separate, explicit
sensitivity, not manufactured here.

Bridge: per-attack-type multipliers as documented constants (src/bridge.py);
freq==1 firms get 1.0 (their reported cost IS the annual total); Impersonation is
a RANGE (3.11-6.25x) drawn per replicate.

Provenance: distilled from src/estimation/national_simulation.py (the type-based
decomposition = approach A). Python-only engine here (the original's Squiggle
emit matched this Python aggregation to 2 s.f.; distillation scope treats Squiggle
as optional).

Deterministic given the fixed seed; reproduces the recorded headline
(mean ~£2.2bn, median ~£2.1bn, 90% interval ~£0.8-4.4bn) — checked at the end.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import load_business_data, N_BY_SIZE, SIZE_LABELS, VALID_FREQ
from fitting import fit_lognormal, band_conditional_mean
from bridge import bridge, IMPERSONATION_LOW, IMPERSONATION_HIGH

RNG = np.random.default_rng(20260715)  # same seed as national_simulation.py
B = 500                                # bootstrap replicates

OUT = os.path.join(os.path.dirname(__file__), "results", "estimate_national_baseline.txt")

# Recorded headline this run must reproduce (national_simulation.py, type-based).
EXPECTED = {"mean_bn": 2.2, "median_bn": 2.1, "p5_bn": 0.8, "p95_bn": 4.4}
TOL_BN = 0.15  # absolute £bn tolerance on the reproduction check


def band_cost_lut(mu, sigma):
    """band code (1..10) -> analytic within-band conditional-mean £ cost."""
    return {b: band_conditional_mean(mu, sigma, b) for b in range(1, 11)}


def cost_vector(lut, bands):
    """Map an array of integer band codes to £ costs via the lookup table."""
    out = np.zeros(len(bands))
    for b in range(1, 11):
        m = bands == b
        if m.any():
            out[m] = lut[b]
    return out


def static_type_bridge(sub):
    """Per-firm type bridge that does NOT depend on the per-replicate Impersonation
    draw. Returns (bridge_array, is_impersonation_mask); Impersonation firms get a
    placeholder 1.0 here, filled in per replicate."""
    freq = sub["freq"].values
    dis = sub["disrupta"].values
    br = np.ones(len(sub))
    is_imp = np.zeros(len(sub), dtype=bool)
    for i in range(len(sub)):
        if dis[i] == 5 and freq[i] != 1:
            is_imp[i] = True          # filled per replicate
            br[i] = 1.0
        else:
            br[i] = bridge(dis[i], freq[i])
    return br, is_imp


def main():
    df = load_business_data()

    # --- fit per-size lognormal shapes once (held fixed across bootstrap) ------
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("Fitting per-size lognormal cost shapes (held fixed across bootstrap)...")
    size_fit = {}
    size_lut = {}
    for sz in (1, 2, 3, 4):
        bands = df.loc[df["sizeb"] == sz, "band"].values
        fit = fit_lognormal(bands)
        size_fit[sz] = fit
        size_lut[sz] = band_cost_lut(fit.mu, fit.sigma)
        log(f"  {SIZE_LABELS[sz]:<7} p0={fit.p0:.3f} mu={fit.mu:.2f} sigma={fit.sigma:.2f} "
            f"E[cost|>0]=£{np.exp(fit.mu + fit.sigma ** 2 / 2):,.0f}  (n={fit.n})")

    # --- precompute per-size firm arrays + static bridges ---------------------
    size_dfs = {sz: df[df["sizeb"] == sz].reset_index(drop=True) for sz in (1, 2, 3, 4)}
    type_static = {sz: static_type_bridge(size_dfs[sz]) for sz in (1, 2, 3, 4)}

    boot = {sz: np.zeros(B) for sz in (1, 2, 3, 4)}

    log(f"\nBootstrapping ({B} weighted replicates)...")
    for rep in range(B):
        imp_bridge = RNG.uniform(IMPERSONATION_LOW, IMPERSONATION_HIGH)
        for sz in (1, 2, 3, 4):
            sub = size_dfs[sz]
            n = len(sub)
            w = sub["weight"].values
            idx = RNG.choice(n, size=n, replace=True, p=w / w.sum())  # weighted bootstrap
            rs = sub.iloc[idx]
            att = rs["attacked"].values
            bands = rs["band"].values
            safe_bands = np.where(att & ~np.isnan(bands), bands, 1).astype(int)

            cost = cost_vector(size_lut[sz], safe_bands)
            br, is_imp = type_static[sz]
            br_rs = br[idx].copy()
            br_rs[is_imp[idx]] = imp_bridge
            total = np.where(att, cost * br_rs, 0.0)
            boot[sz][rep] = total.mean()

    # --- aggregate to the national total --------------------------------------
    tot = np.zeros(B)
    for sz in (1, 2, 3, 4):
        tot += N_BY_SIZE[sz] * boot[sz]

    mean_bn = tot.mean() / 1e9
    median_bn = np.median(tot) / 1e9
    p5_bn = np.percentile(tot, 5) / 1e9
    p95_bn = np.percentile(tot, 95) / 1e9

    log(f"\n{'=' * 72}")
    log("APPROACH A — national annual UK business cybercrime cost (headline)")
    log(f"{'=' * 72}")
    log(f"  mean   = £{mean_bn:,.2f}bn")
    log(f"  median = £{median_bn:,.2f}bn")
    log(f"  90% interval = £{p5_bn:,.2f}bn  –  £{p95_bn:,.2f}bn")
    log(f"\n  per-size contribution (mean £bn):")
    for sz in (1, 2, 3, 4):
        c = (N_BY_SIZE[sz] * boot[sz]).mean()
        log(f"    {SIZE_LABELS[sz]:<7} £{c / 1e9:,.2f}bn   "
            f"(£{boot[sz].mean():,.0f}/business x {N_BY_SIZE[sz]:,})")

    # --- reproduction contract ------------------------------------------------
    got = {"mean_bn": mean_bn, "median_bn": median_bn, "p5_bn": p5_bn, "p95_bn": p95_bn}
    log(f"\n{'=' * 72}")
    log("Reproduction check vs recorded headline (national_simulation.py):")
    ok = True
    for k, exp in EXPECTED.items():
        d = abs(got[k] - exp)
        flag = "OK " if d <= TOL_BN else "OFF"
        if d > TOL_BN:
            ok = False
        log(f"  {k:<10} got £{got[k]:.2f}bn  expected ~£{exp:.1f}bn  |Δ|={d:.2f}  [{flag}]")
    log(f"\n  {'ALL CHECKS PASS' if ok else 'MISMATCH — investigate'}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nWrote {OUT}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
