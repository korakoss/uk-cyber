"""Posterior frailty estimates + model-free intensity distribution.

1. Under the binary frailty model: compute P(exposed | observables) per firm.
   If bimodal (pile-up near 0 and 1), binary is a good description.
   If spread across the middle, a continuous frailty fits better.

2. Model-free: distribution of phishing count N among attacked firms.
   Bimodal -> binary; smooth/skewed -> continuous.

Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src:src/estimation \
     python3 src/estimation/frailty_posterior_check.py
"""

import json
import os
import numpy as np

import joint_four_channel as j
import joint_four_channel_frailty as fr


def main():
    d = j.load_firms()
    F, _, _ = j.calibrate_freq(d)
    w = d["weight"].fillna(d["weight"].median()).values
    w = w / w.mean()

    here = os.path.dirname(os.path.abspath(__file__))
    q = json.load(open(os.path.join(here, "build", "joint_four_channel_frailty_weighted.json")))
    q.pop("nll", None)

    # --- 1. Posterior P(exposed | data_i) under binary model ---
    print("=" * 70)
    print("1. POSTERIOR P(exposed | observables) UNDER BINARY FRAILTY")
    print("=" * 70)

    lik = fr.FrailtyLikelihood(d, F, w)
    q_quiet = fr.class_params(q, False)
    q_exposed = fr.class_params(q, True)
    v_quiet = lik.firm_values(q_quiet)
    v_exposed = lik.firm_values(q_exposed)

    # Per firm: P(exposed | data) = pi * L_exposed / (pi * L_exposed + (1-pi) * L_quiet)
    posteriors = []
    firm_weights = []
    firm_attacked = []
    idx = 0
    for (key, fi, yi, li, wt), lq, le in zip(lik.groups, v_quiet, v_exposed):
        num = q["pi"] * le
        den = (1 - q["pi"]) * lq + q["pi"] * le
        post = num / np.maximum(den, 1e-300)
        posteriors.extend(post.tolist())
        firm_weights.extend(wt.tolist())
        fP, fI, fS, nb = key
        attacked = bool(fP or fI or fS)
        firm_attacked.extend([attacked] * len(post))

    posteriors = np.array(posteriors)
    firm_weights = np.array(firm_weights)
    firm_attacked = np.array(firm_attacked, dtype=bool)

    # Histogram of posteriors
    bins = np.linspace(0, 1, 21)
    print(f"\n  Distribution of P(exposed | data) across all {len(posteriors)} firms:")
    print(f"  (pi prior = {q['pi']:.3f})")
    print(f"\n  {'bin':>12s}  {'count':>6s}  {'w.share':>8s}")
    for i in range(len(bins) - 1):
        mask = (posteriors >= bins[i]) & (posteriors < bins[i + 1])
        if i == len(bins) - 2:
            mask = (posteriors >= bins[i]) & (posteriors <= bins[i + 1])
        n = mask.sum()
        ws = firm_weights[mask].sum() / firm_weights.sum()
        print(f"  [{bins[i]:.2f},{bins[i+1]:.2f})  {n:6d}  {ws:8.3f}")

    # Summary stats
    print(f"\n  Summary:")
    print(f"    mean = {np.average(posteriors, weights=firm_weights):.3f}")
    print(f"    median = {np.median(posteriors):.3f}")
    print(f"    P(post < 0.1) = {np.average(posteriors < 0.1, weights=firm_weights):.3f}")
    print(f"    P(post > 0.9) = {np.average(posteriors > 0.9, weights=firm_weights):.3f}")
    print(f"    P(0.2 < post < 0.8) = {np.average((posteriors > 0.2) & (posteriors < 0.8), weights=firm_weights):.3f}")

    # Among attacked only
    pa = posteriors[firm_attacked]
    wa = firm_weights[firm_attacked]
    print(f"\n  Among attacked firms (n={firm_attacked.sum()}):")
    print(f"    mean = {np.average(pa, weights=wa):.3f}")
    print(f"    P(post < 0.2) = {np.average(pa < 0.2, weights=wa):.3f}")
    print(f"    P(post > 0.8) = {np.average(pa > 0.8, weights=wa):.3f}")
    print(f"    P(0.2 < post < 0.8) = {np.average((pa > 0.2) & (pa < 0.8), weights=wa):.3f}")

    # Among non-attacked
    pn = posteriors[~firm_attacked]
    wn = firm_weights[~firm_attacked]
    print(f"\n  Among non-attacked firms (n={(~firm_attacked).sum()}):")
    print(f"    mean = {np.average(pn, weights=wn):.3f}")
    print(f"    P(post < 0.1) = {np.average(pn < 0.1, weights=wn):.3f}")
    print(f"    P(post > 0.5) = {np.average(pn > 0.5, weights=wn):.3f}")

    # --- 2. Model-free: phishing count distribution ---
    print(f"\n{'=' * 70}")
    print("2. MODEL-FREE: PHISHING COUNT N DISTRIBUTION (attacked firms)")
    print("=" * 70)
    att = d["att"] == 1
    N = d.loc[att, "N"].values
    valid = (~np.isnan(N)) & (N >= 0)
    Nv = N[valid]
    wv = w[att.values][valid]
    print(f"  n = {valid.sum()} firms with valid phishing count")
    print(f"  unweighted quantiles: Q10={np.quantile(Nv, 0.1):.0f}  Q25={np.quantile(Nv, 0.25):.0f}  "
          f"Q50={np.quantile(Nv, 0.5):.0f}  Q75={np.quantile(Nv, 0.75):.0f}  "
          f"Q90={np.quantile(Nv, 0.9):.0f}  Q99={np.quantile(Nv, 0.99):.0f}")

    # Log-scale histogram
    log_edges = [0, 1, 2, 3, 5, 10, 20, 50, 100, 500, 10000]
    print(f"\n  {'N range':>12s}  {'count':>6s}  {'w.share':>8s}  {'bar'}")
    for i in range(len(log_edges) - 1):
        lo, hi = log_edges[i], log_edges[i + 1]
        if i == 0:
            mask = Nv == 0
        else:
            mask = (Nv >= lo) & (Nv < hi)
            if i == len(log_edges) - 2:
                mask = Nv >= lo
        n = mask.sum()
        ws = wv[mask].sum() / wv.sum()
        bar = "#" * int(ws * 100)
        if i == 0:
            label = "0"
        elif i == len(log_edges) - 2:
            label = f"{lo}+"
        else:
            label = f"{lo}-{hi-1}"
        print(f"  {label:>12s}  {n:6d}  {ws:8.3f}  {bar}")

    # Is it bimodal? Hartigan's dip test approximation: just check if
    # log(N+1) has a clear dip in density
    logN = np.log1p(Nv)
    print(f"\n  log(N+1) distribution:")
    print(f"    mean={logN.mean():.2f}  std={logN.std():.2f}  skew={((logN - logN.mean())/logN.std()**3).mean():.2f}")

    # Kernel density on log scale - crude binned version
    log_bins = np.linspace(0, np.log1p(1000), 25)
    counts_log, _ = np.histogram(logN, bins=log_bins, weights=wv)
    counts_log = counts_log / counts_log.sum()
    print(f"\n  Weighted density of log(N+1) (25 bins, 0 to ln(1001)):")
    max_d = counts_log.max()
    for i in range(len(log_bins) - 1):
        mid = (log_bins[i] + log_bins[i + 1]) / 2
        n_equiv = np.exp(mid) - 1
        bar = "#" * int(counts_log[i] / max_d * 40)
        print(f"    N~{n_equiv:6.0f}  {counts_log[i]:.3f}  {bar}")

    # --- 3. Posterior vs phishing count scatter ---
    print(f"\n{'=' * 70}")
    print("3. POSTERIOR P(exposed) BY PHISHING COUNT BUCKET")
    print("=" * 70)
    # Need to align posteriors with original firm indices
    # Rebuild posteriors indexed by firm
    post_by_firm = np.full(len(d), np.nan)
    idx = 0
    for (key, fi, yi, li, wt), lq, le in zip(lik.groups, v_quiet, v_exposed):
        # groups contain arrays of firm-level values
        # but we need the original indices... the grouping lost them
        pass

    # Simpler: just cross-tab phishing count buckets with flag patterns
    # (which are a good proxy for posterior)
    n_types = d.loc[att, ["fP", "fI", "fS"]].sum(1).values
    N_att = d.loc[att, "N"].values
    w_att = w[att.values]
    valid_att = (~np.isnan(N_att)) & (N_att >= 0)

    buckets = [(0, 0), (1, 1), (2, 3), (4, 10), (11, 50), (51, 100000)]
    print(f"  {'N bucket':>12s}  {'n':>5s}  {'mean #types':>12s}  {'P(3 types)':>10s}  {'mean band':>10s}")
    for lo, hi in buckets:
        mask = valid_att & (N_att >= lo) & (N_att <= hi)
        if mask.sum() < 3:
            continue
        nt = np.average(n_types[mask], weights=w_att[mask])
        p3 = np.average(n_types[mask] == 3, weights=w_att[mask])
        band = d.loc[att, "band"].values[mask]
        bv = ~np.isnan(band)
        mb = np.average(band[bv], weights=w_att[mask][bv]) if bv.any() else np.nan
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        print(f"  {label:>12s}  {mask.sum():5d}  {nt:12.2f}  {p3:10.3f}  {mb:10.2f}")


if __name__ == "__main__":
    main()
