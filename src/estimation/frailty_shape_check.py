"""Empirical look at the shape of firm-level attack heterogeneity.

Analogous to the SVD that revealed two phishing clusters: here we look at
the joint distribution of attack indicators across channels to see whether
the latent heterogeneity is binary, multi-class, or continuous.

Observables per firm (among attacked): fP, fI, fS (binary flags),
phishing count N (where available), freq band, worst cost band.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation \
     python3 src/estimation/frailty_shape_check.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import product

import joint_four_channel as j


def main():
    d = j.load_firms()
    w = d["weight"].fillna(d["weight"].median()).values
    w = w / w.mean()

    att = d["att"] == 1
    da = d[att].copy()
    wa = w[att.values]

    # --- 1. Flag pattern frequencies (weighted) ---
    print("=" * 70)
    print("1. ATTACK FLAG PATTERNS  (fP, fI, fS)  among attacked firms")
    print("=" * 70)
    flags = da[["fP", "fI", "fS"]].values.astype(int)
    patterns = {}
    for pat in product([0, 1], repeat=3):
        if pat == (0, 0, 0):
            continue
        mask = (flags == pat).all(1)
        patterns[pat] = np.sum(wa[mask])
    total_w = sum(patterns.values())
    for pat, wt in sorted(patterns.items()):
        n_raw = int(((flags == pat).all(1)).sum())
        print(f"  {pat}  weighted share {wt/total_w:.3f}  (n={n_raw})")

    # --- 2. Pairwise flag co-occurrence (tetrachoric-style) ---
    print(f"\n{'=' * 70}")
    print("2. PAIRWISE FLAG ASSOCIATIONS  (weighted)")
    print("=" * 70)
    for i, j_idx, a, b in [(0, 1, "P", "I"), (0, 2, "P", "S"), (1, 2, "I", "S")]:
        fa, fb = flags[:, i], flags[:, j_idx]
        # weighted 2x2 table
        tab = np.zeros((2, 2))
        for vi in [0, 1]:
            for vj in [0, 1]:
                tab[vi, vj] = wa[(fa == vi) & (fb == vj)].sum()
        tab /= tab.sum()
        # odds ratio
        oratio = (tab[1, 1] * tab[0, 0]) / max(tab[1, 0] * tab[0, 1], 1e-15)
        # tetrachoric: P(both) under bivariate normal
        # marginals
        p_a = tab[1, :].sum()
        p_b = tab[:, 1].sum()
        print(f"  {a} x {b}:  P({a})={p_a:.3f}  P({b})={p_b:.3f}  "
              f"P({a}&{b})={tab[1,1]:.3f}  P({a}&{b})|indep={p_a*p_b:.3f}  "
              f"OR={oratio:.1f}")

    # --- 3. Intensity gradient by number of channels hit ---
    print(f"\n{'=' * 70}")
    print("3. INTENSITY BY NUMBER OF CHANNELS HIT")
    print("=" * 70)
    n_channels = flags.sum(1)
    # phishing count where available
    N = da["N"].values if "N" in da.columns else None
    for nc in [1, 2, 3]:
        mask = n_channels == nc
        n_firms = mask.sum()
        wt = wa[mask].sum() / wa.sum()
        # freq distribution
        freq_vals = da.loc[mask, "freq"].values
        freq_valid = ~np.isnan(freq_vals)
        if freq_valid.any():
            mean_freq = np.average(freq_vals[freq_valid], weights=wa[mask][freq_valid])
        else:
            mean_freq = np.nan
        # phishing count
        if N is not None:
            n_valid = (~np.isnan(N[mask])) & (N[mask] >= 0)
            if n_valid.any():
                med_N = np.median(N[mask][n_valid])
                mean_N = np.average(N[mask][n_valid], weights=wa[mask][n_valid])
            else:
                med_N = mean_N = np.nan
        else:
            med_N = mean_N = np.nan
        # worst band
        band = da.loc[mask, "band"].values
        band_valid = ~np.isnan(band)
        if band_valid.any():
            mean_band = np.average(band[band_valid], weights=wa[mask][band_valid])
        else:
            mean_band = np.nan
        print(f"  {nc} channel(s): n={n_firms} ({wt:.1%})  "
              f"mean_freq={mean_freq:.2f}  mean_N={mean_N:.1f} (med {med_N:.0f})  "
              f"mean_band={mean_band:.2f}")

    # --- 4. Is the triple rate "too high" for a single continuous factor? ---
    print(f"\n{'=' * 70}")
    print("4. TRIPLE CO-OCCURRENCE: OBSERVED vs 1-FACTOR MODELS")
    print("=" * 70)
    # Given marginal rates P(fP), P(fI), P(fS) among attacked,
    # what triple rate does a 1-factor probit model predict?
    # Under 1-factor: fX = 1{a_X * Z + e_X > 0}, Z ~ N(0,1), e_X ~ N(0,1)
    # P(fX=1) = Phi(a_X * mu_X) where mu_X = threshold
    # The pairwise correlation rho(fX, fY) = a_X * a_Y / sqrt((1+a_X^2)(1+a_Y^2))
    # Fit from marginals and pairwise associations

    # Simpler: given weighted marginal P(fP|att), P(fI|att), P(fS|att),
    # compute P(all 3) under independence vs observed
    p_P = (flags[:, 0] * wa).sum() / wa.sum()
    p_I = (flags[:, 1] * wa).sum() / wa.sum()
    p_S = (flags[:, 2] * wa).sum() / wa.sum()
    p_triple_obs = patterns.get((1, 1, 1), 0) / total_w
    p_triple_indep = p_P * p_I * p_S

    print(f"  Marginals: P(P|att)={p_P:.3f}  P(I|att)={p_I:.3f}  P(S|att)={p_S:.3f}")
    print(f"  P(all 3 | att):  observed = {p_triple_obs:.3f}  "
          f"independent = {p_triple_indep:.3f}  ratio = {p_triple_obs/p_triple_indep:.1f}x")

    # 1-factor probit simulation: find loadings that match the 3 pairwise ORs,
    # then predict the triple rate
    from scipy.optimize import minimize as _min

    def one_factor_loss(params):
        a_P, a_I, a_S = params
        # thresholds from marginals
        t_P = norm.ppf(1 - p_P)
        t_I = norm.ppf(1 - p_I)
        t_S = norm.ppf(1 - p_S)
        # simulate
        rng = np.random.default_rng(0)
        Z = rng.standard_normal(500_000)
        e_P = rng.standard_normal(500_000)
        e_I = rng.standard_normal(500_000)
        e_S = rng.standard_normal(500_000)
        fP_sim = (a_P * Z + e_P > t_P).astype(float)
        fI_sim = (a_I * Z + e_I > t_I).astype(float)
        fS_sim = (a_S * Z + e_S > t_S).astype(float)
        # target pairwise P(both)
        p_PI_obs = patterns.get((1, 1, 0), 0) / total_w + p_triple_obs
        p_PS_obs = patterns.get((1, 0, 1), 0) / total_w + p_triple_obs
        p_IS_obs = patterns.get((0, 1, 1), 0) / total_w + p_triple_obs
        p_PI_sim = (fP_sim * fI_sim).mean()
        p_PS_sim = (fP_sim * fS_sim).mean()
        p_IS_sim = (fI_sim * fS_sim).mean()
        return (p_PI_sim - p_PI_obs)**2 + (p_PS_sim - p_PS_obs)**2 + (p_IS_sim - p_IS_obs)**2

    res = _min(one_factor_loss, [1.0, 1.0, 1.0], method="Nelder-Mead",
               options=dict(maxiter=500, xatol=0.01))
    a_P, a_I, a_S = res.x
    # predict triple
    rng = np.random.default_rng(0)
    Z = rng.standard_normal(1_000_000)
    fP_sim = (a_P * Z + rng.standard_normal(1_000_000) > norm.ppf(1 - p_P))
    fI_sim = (a_I * Z + rng.standard_normal(1_000_000) > norm.ppf(1 - p_I))
    fS_sim = (a_S * Z + rng.standard_normal(1_000_000) > norm.ppf(1 - p_S))
    p_triple_1factor = (fP_sim & fI_sim & fS_sim).mean()

    print(f"  1-factor probit (loadings a_P={a_P:.2f} a_I={a_I:.2f} a_S={a_S:.2f}):")
    print(f"    predicted P(all 3) = {p_triple_1factor:.3f}  (obs {p_triple_obs:.3f})")

    # Full pattern comparison
    print(f"\n  Pattern comparison (among attacked):")
    print(f"  {'pattern':>10s}  {'observed':>8s}  {'indep':>8s}  {'1-factor':>8s}")
    att_sim = (fP_sim | fI_sim | fS_sim)
    for pat in [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]:
        obs = patterns.get(pat, 0) / total_w
        ind = 1.0
        for k, p_k in enumerate([p_P, p_I, p_S]):
            ind *= p_k if pat[k] else (1 - p_k)
        # 1-factor among attacked
        mask_sim = np.array([pat[0], pat[1], pat[2]], dtype=bool)
        sim_flags = np.column_stack([fP_sim, fI_sim, fS_sim])
        match = (sim_flags == mask_sim).all(1)
        f1 = match[att_sim].mean()
        print(f"  {str(pat):>10s}  {obs:8.3f}  {ind:8.3f}  {f1:8.3f}")

    # --- 5. Distribution of phishing count N by flag pattern ---
    print(f"\n{'=' * 70}")
    print("5. PHISHING COUNT DISTRIBUTION BY FLAG PATTERN")
    print("=" * 70)
    if N is not None:
        for pat_label, pat_mask in [("P only", (1,0,0)), ("P+I", (1,1,0)),
                                     ("P+S", (1,0,1)), ("P+I+S", (1,1,1))]:
            mask = (flags == pat_mask).all(1)
            n_vals = N[mask]
            valid = (~np.isnan(n_vals)) & (n_vals >= 0)
            if valid.sum() < 5:
                continue
            nv = n_vals[valid]
            qs = np.quantile(nv, [0.25, 0.5, 0.75, 0.9])
            print(f"  {pat_label:>8s}: n={valid.sum():3d}  "
                  f"Q25={qs[0]:.0f}  Q50={qs[1]:.0f}  Q75={qs[2]:.0f}  Q90={qs[3]:.0f}  "
                  f"mean={nv.mean():.1f}")


if __name__ == "__main__":
    main()
