"""Two-channel model: Stage 1 calibration + Stage 2 independence check.

Stage 1: For each channel's pure subsample (nuisance-only or serious-only
firms, classified by type flags), fit a per-attack cost PMF over bands 1-10
using MLE on (freq_band, max_cost_band). The freq band implies K attacks;
the max cost band is the max of K iid draws from the per-attack PMF.

No Poisson assumption on K — freq-to-K is taken as given (label-implied
midpoints). The per-attack PMF is the sole fitted object.

Stage 2: Test independence of the two channels via a 2x2 contingency
table on type-flag presence, and compare freq/cost profiles between
pure and mixed subsamples.

Run: PYTHONPATH=/home/user/md-clean/src python3 src/estimation/two_channel_mle.py
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# data.py lives in md-clean/src — set PYTHONPATH accordingly
from data import BAND_BOUNDS, SIZE_LABELS, load, FREQ_LABELS, GENUINE_TYPE_COLS

NUISANCE_TYPE_COLS = ["type5", "type6"]  # impersonation, phishing
SERIOUS_TYPE_COLS = [c for c in GENUINE_TYPE_COLS if c not in NUISANCE_TYPE_COLS]

COST_BANDS = list(range(1, 11))
N_BANDS = len(COST_BANDS)

# Freq band -> representative K (label-implied, not range midpoints)
FREQ_TO_K = {1: 1, 2: 6, 3: 12, 4: 52, 5: 365, 6: 730}

N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}


def band_to_gbp(band):
    lo, hi = BAND_BOUNDS.get(band, (0, 0))
    if band == 1: return 0.0
    if hi == np.inf: return lo * 2.0
    return (lo + hi) / 2


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def neg_log_likelihood(z, obs_list):
    """NLL for per-attack PMF given (K, max_band_0indexed) observations."""
    full_z = np.zeros(N_BANDS)
    full_z[1:] = z  # band 1 fixed at z=0
    pmf = softmax(full_z)
    cdf = np.cumsum(pmf)

    nll = 0.0
    for K, b_idx in obs_list:
        cdf_hi = cdf[b_idx]
        cdf_lo = cdf[b_idx - 1] if b_idx > 0 else 0.0
        p_max = cdf_hi**K - cdf_lo**K
        nll -= np.log(max(p_max, 1e-300))

    return nll


def fit_pmf(obs_list, n_restarts=15):
    """Fit per-attack PMF via MLE with multiple restarts."""
    best_nll = np.inf
    best_z = None

    for restart in range(n_restarts):
        if restart == 0:
            z0 = np.zeros(N_BANDS - 1)
        elif restart == 1:
            # Start with mass on band 1 (high failure rate)
            z0 = np.array([2.0] + [-1.0] * (N_BANDS - 2))
            z0 = z0[:N_BANDS - 1]
        elif restart == 2:
            # Start with mass spread across bands
            z0 = np.zeros(N_BANDS - 1)
        else:
            z0 = np.random.randn(N_BANDS - 1) * 2.0

        try:
            result = minimize(neg_log_likelihood, z0, args=(obs_list,),
                              method="L-BFGS-B",
                              options={"maxiter": 5000, "ftol": 1e-14})
            if result.fun < best_nll:
                best_nll = result.fun
                best_z = result.x
                best_converged = result.success
        except Exception:
            pass

    full_z = np.zeros(N_BANDS)
    full_z[1:] = best_z
    pmf = softmax(full_z)
    return pmf, best_nll, best_converged


def predicted_max_pmf(pmf, K):
    """P(max = band b | K draws from pmf)."""
    cdf = np.cumsum(pmf)
    pred = np.zeros(N_BANDS)
    pred[0] = cdf[0]**K
    for j in range(1, N_BANDS):
        pred[j] = cdf[j]**K - cdf[j-1]**K
    return pred


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)
    att = df[df["attacked"]].copy()

    # Classify firms into regimes based on type flags
    nuis_cols = [c for c in NUISANCE_TYPE_COLS if c in att.columns]
    ser_cols = [c for c in SERIOUS_TYPE_COLS if c in att.columns]

    att["has_nuisance"] = (att[nuis_cols] == 1).any(axis=1)
    att["has_serious"] = (att[ser_cols] == 1).any(axis=1)

    att["regime"] = "neither"
    att.loc[att["has_nuisance"] & ~att["has_serious"], "regime"] = "nuisance_only"
    att.loc[~att["has_nuisance"] & att["has_serious"], "regime"] = "serious_only"
    att.loc[att["has_nuisance"] & att["has_serious"], "regime"] = "both"

    print("REGIME CLASSIFICATION (all attacked firms)")
    print("=" * 60)
    for r in ["nuisance_only", "serious_only", "both", "neither"]:
        n = (att["regime"] == r).sum()
        print(f"  {r:>15s}: {n:>5d} ({n/len(att)*100:.1f}%)")
    print(f"  {'total':>15s}: {len(att):>5d}")

    # Filter to firms with valid freq + band
    v = att[att["band"].notna() & att["freq"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["freq"] = v["freq"].astype(int)

    print(f"\nWith valid freq + band: {len(v)}")
    for r in ["nuisance_only", "serious_only", "both", "neither"]:
        n = (v["regime"] == r).sum()
        print(f"  {r:>15s}: {n:>5d}")

    band_costs = np.array([band_to_gbp(b) for b in COST_BANDS])

    # ──────────────────────────────────────────────────────────────
    # STAGE 1: PURE SUBSAMPLE CALIBRATION
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print("STAGE 1: PURE SUBSAMPLE CALIBRATION (pooled across sizes)")
    print(f"{'='*74}")

    fitted = {}  # store fitted PMFs for later use

    for regime in ["nuisance_only", "serious_only"]:
        sub = v[v["regime"] == regime]
        channel = "Nuisance" if "nuis" in regime else "Serious"

        print(f"\n{'─'*74}")
        print(f"  {channel} channel — pure subsample (n={len(sub)})")
        print(f"{'─'*74}")

        # Observed (freq × band) crosstab
        print(f"\n  Observed freq × max_band:")
        ct = pd.crosstab(sub["freq"], sub["band"])
        for f in sorted(sub["freq"].unique()):
            row = sub[sub["freq"] == f]
            counts = [f"{(row['band'] == b).sum():3d}" for b in COST_BANDS]
            print(f"    freq={f} (K={FREQ_TO_K[f]:>4d}): {' '.join(counts)}  (n={len(row)})")

        # Build observation list
        obs_list = [(FREQ_TO_K[row["freq"]], row["band"] - 1)
                    for _, row in sub.iterrows()]

        # Fit
        pmf, nll, converged = fit_pmf(obs_list)
        fitted[regime] = pmf

        print(f"\n  Fitted per-attack PMF (what a SINGLE attack looks like):")
        for b in COST_BANDS:
            lo, hi = BAND_BOUNDS[b]
            if b == 1:
                label = "no cost"
            elif hi == np.inf:
                label = f"£{lo:,}+"
            else:
                label = f"£{lo:,}-£{hi:,}"
            print(f"    band {b:2d} ({label:>20s}): {pmf[b-1]:.4f}")

        p_success = 1 - pmf[0]
        et = (pmf * band_costs).sum()
        et_success = (pmf[1:] * band_costs[1:]).sum() / p_success if p_success > 1e-10 else 0

        print(f"\n  P(success per attack) = {p_success:.4f}")
        print(f"  E[T per attack] = £{et:,.0f} (unconditional, incl. failures)")
        print(f"  E[T | success]  = £{et_success:,.0f}")
        print(f"  NLL = {nll:.2f}, converged = {converged}")

        # E[K] from empirical freq distribution
        freq_dist = sub["freq"].value_counts(normalize=True).sort_index()
        ek = sum(FREQ_TO_K[f] * p for f, p in freq_dist.items())
        print(f"\n  Empirical freq distribution:")
        for f in sorted(freq_dist.index):
            print(f"    freq={f}: {freq_dist[f]:.3f} (n={int(freq_dist[f]*len(sub))})")
        print(f"  E[K] = {ek:.1f}")
        print(f"  E[total cost per firm] = E[K] × E[T] = {ek:.1f} × £{et:,.0f} = £{ek*et:,.0f}")

        # GOF: predicted vs observed
        print(f"\n  GOF: predicted vs observed max-band (per freq):")
        total_chi2 = 0
        total_cells = 0
        for f in sorted(sub["freq"].unique()):
            K = FREQ_TO_K[f]
            sf = sub[sub["freq"] == f]
            n_f = len(sf)

            pred = predicted_max_pmf(pmf, K)

            obs_counts = np.array([(sf["band"] == b).sum() for b in COST_BANDS])
            exp_counts = pred * n_f

            # Merge sparse cells for chi2
            obs_m, exp_m = [], []
            obs_acc, exp_acc = 0, 0
            for j in range(N_BANDS):
                obs_acc += obs_counts[j]
                exp_acc += exp_counts[j]
                if exp_acc >= 3:  # merge until expected >= 3
                    obs_m.append(obs_acc)
                    exp_m.append(exp_acc)
                    obs_acc, exp_acc = 0, 0
            if obs_acc > 0 or exp_acc > 0:
                if len(obs_m) > 0:
                    obs_m[-1] += obs_acc
                    exp_m[-1] += exp_acc
                else:
                    obs_m.append(obs_acc)
                    exp_m.append(exp_acc)

            chi2_f = sum((o - e)**2 / e for o, e in zip(obs_m, exp_m) if e > 0)
            total_chi2 += chi2_f
            total_cells += len(obs_m)

            obs_str = " ".join(f"{c:3d}" for c in obs_counts)
            pred_str = " ".join(f"{e:5.1f}" for e in exp_counts)
            print(f"    freq={f} (K={K:>4d}, n={n_f:>3d}):")
            print(f"      obs:  {obs_str}")
            print(f"      pred: {pred_str}")

        n_params = N_BANDS - 1  # 9 free PMF params
        dof = max(total_cells - n_params, 1)
        p_gof = 1 - chi2_dist.cdf(total_chi2, dof)
        print(f"\n  Overall: χ²={total_chi2:.1f}, cells={total_cells}, "
              f"params={n_params}, df={dof}, p={p_gof:.3f}")

    # ──────────────────────────────────────────────────────────────
    # STAGE 2: INDEPENDENCE CHECK
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print("STAGE 2: INDEPENDENCE CHECK")
    print(f"{'='*74}")

    # 2×2 contingency table
    print("\n  2×2 table: has_nuisance × has_serious (all attacked firms)")
    ct = pd.crosstab(att["has_nuisance"], att["has_serious"], margins=True)
    ct.index = [f"nuisance={x}" for x in ct.index[:-1]] + ["Total"]
    ct.columns = [f"serious={x}" for x in ct.columns[:-1]] + ["Total"]
    print(ct.to_string())

    from scipy.stats import chi2_contingency
    table = pd.crosstab(att["has_nuisance"], att["has_serious"])
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"\n  χ² = {chi2:.1f}, df={dof}, p={p:.4f}")
    if p < 0.05:
        print("  → Channels are DEPENDENT (reject independence at 5%)")
        # Show direction
        obs_both = table.loc[True, True] if True in table.index and True in table.columns else 0
        exp_both = expected[table.index.tolist().index(True), table.columns.tolist().index(True)]
        print(f"    Observed 'both': {obs_both}, Expected under independence: {exp_both:.0f}")
        if obs_both > exp_both:
            print("    → Positive association (firms with one tend to have the other)")
        else:
            print("    → Negative association")
    else:
        print("  → Consistent with independence")

    # Compare freq profiles: nuisance-only vs nuisance-side-of-both
    print(f"\n  Freq distribution comparison:")
    print(f"  (Does the nuisance channel behave the same with/without serious?)")
    for label, mask in [("nuisance_only", v["regime"] == "nuisance_only"),
                        ("both (all)",    v["regime"] == "both")]:
        sub = v[mask]
        fd = sub["freq"].value_counts(normalize=True).sort_index()
        ek = sum(FREQ_TO_K[f] * p for f, p in fd.items())
        parts = " ".join(f"f{f}={fd.get(f,0):.2f}" for f in range(1, 7))
        print(f"    {label:>20s} (n={len(sub):>3d}): {parts}  E[K]={ek:.0f}")

    print(f"\n  (Does the serious channel behave the same with/without nuisance?)")
    for label, mask in [("serious_only", v["regime"] == "serious_only"),
                        ("both (all)",   v["regime"] == "both")]:
        sub = v[mask]
        fd = sub["freq"].value_counts(normalize=True).sort_index()
        ek = sum(FREQ_TO_K[f] * p for f, p in fd.items())
        parts = " ".join(f"f{f}={fd.get(f,0):.2f}" for f in range(1, 7))
        print(f"    {label:>20s} (n={len(sub):>3d}): {parts}  E[K]={ek:.0f}")

    # Compare max-band profiles: nuisance-only vs nuisance-disrupta-in-both
    # NOTE: for "both" firms with nuisance disrupta, the max is selected
    # (max_N > max_S), so direct comparison is biased. Flag this.
    print(f"\n  Max-band comparison (nuisance channel):")
    print(f"  NOTE: 'both' firms with nuisance disrupta have max_N > max_S")
    print(f"        (selection bias — not directly comparable)")

    v_disrupta = v[v["disrupta"].notna()].copy()
    v_disrupta["disrupta"] = v_disrupta["disrupta"].astype(int)
    v_disrupta["disrupta_is_nuisance"] = v_disrupta["disrupta"].isin({5, 6})

    for label, sub in [("nuisance_only", v[v["regime"] == "nuisance_only"]),
                       ("both, nuis disrupta",
                        v_disrupta[(v_disrupta["regime"] == "both") &
                                   v_disrupta["disrupta_is_nuisance"]])]:
        bd = sub["band"].value_counts(normalize=True).sort_index()
        mean_b = sub["band"].mean()
        p_success = (sub["band"] >= 2).mean()
        parts = " ".join(f"b{b}={bd.get(b,0):.2f}" for b in range(1, 11))
        print(f"    {label:>25s} (n={len(sub):>3d}): mean={mean_b:.2f} "
              f"P(success)={p_success:.2f}")

    # ──────────────────────────────────────────────────────────────
    # NATIONAL TOTAL (using Stage 1 fits)
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print("STAGE 1 NATIONAL TOTAL (pure-subsample parameters, no bridge)")
    print(f"{'='*74}")

    # For each channel, E[total cost per attacked firm from this channel]
    # = E[K_channel] × E[T_channel]
    # National total = Σ_size N(size) × P(att|size) × Σ_channel E[K_c] × E[T_c]

    prev = {}
    for sz in SIZE_LABELS:
        sub = df[df["sizeb"] == sz]
        prev[sz] = (sub["attacked"].astype(int) * sub["weight"]).sum() / sub["weight"].sum()

    for regime in ["nuisance_only", "serious_only"]:
        channel = "Nuisance" if "nuis" in regime else "Serious"
        pmf = fitted[regime]
        et = (pmf * band_costs).sum()
        sub = v[v["regime"] == regime]
        freq_dist = sub["freq"].value_counts(normalize=True).sort_index()
        ek = sum(FREQ_TO_K[f] * p for f, p in freq_dist.items())
        print(f"\n  {channel}: E[K]={ek:.1f}, E[T]=£{et:,.0f}, "
              f"E[total from channel]=£{ek*et:,.0f}")

    # Combine: total E[cost|attacked] = E[K_N]·E[T_N] + E[K_S]·E[T_S]
    # But this requires knowing the rates for EACH channel SEPARATELY,
    # which the pure subsamples give us (but only for firms in that regime).
    # For "both" firms, we'd need to decompose — that's Stage 3.
    # For now, use the pure-subsample E[K] as a proxy.
    print(f"\n  Caveat: E[K] estimated from pure subsamples only.")
    print(f"  'Both' firms (n={len(v[v['regime']=='both'])}) have higher total K")
    print(f"  from BOTH channels — Stage 3 will handle the decomposition.")

    nuis_pmf = fitted["nuisance_only"]
    ser_pmf = fitted["serious_only"]
    et_n = (nuis_pmf * band_costs).sum()
    et_s = (ser_pmf * band_costs).sum()

    nuis_sub = v[v["regime"] == "nuisance_only"]
    ser_sub = v[v["regime"] == "serious_only"]
    fd_n = nuis_sub["freq"].value_counts(normalize=True).sort_index()
    fd_s = ser_sub["freq"].value_counts(normalize=True).sort_index()
    ek_n = sum(FREQ_TO_K[f] * p for f, p in fd_n.items())
    ek_s = sum(FREQ_TO_K[f] * p for f, p in fd_s.items())

    # Rough national total:
    # Each attacked firm contributes E[K_N]·E[T_N] + E[K_S]·E[T_S]
    # where the rates depend on which regime the firm is in.
    # Simplification: use regime shares to weight.
    regime_shares = v["regime"].value_counts(normalize=True)
    p_nuis = regime_shares.get("nuisance_only", 0)
    p_ser = regime_shares.get("serious_only", 0)
    p_both = regime_shares.get("both", 0)

    # For nuisance-only: cost = E[K_N]·E[T_N]
    # For serious-only: cost = E[K_S]·E[T_S]
    # For both: cost ≈ E[K_N]·E[T_N] + E[K_S]·E[T_S] (independence)
    e_cost_nonly = ek_n * et_n
    e_cost_sonly = ek_s * et_s
    e_cost_both = ek_n * et_n + ek_s * et_s  # independence assumption

    e_cost_att = (p_nuis * e_cost_nonly +
                  p_ser * e_cost_sonly +
                  p_both * e_cost_both)

    print(f"\n  Per attacked firm (rough, using pure-subsample rates):")
    print(f"    Nuisance-only ({p_nuis:.1%}): £{e_cost_nonly:,.0f}")
    print(f"    Serious-only  ({p_ser:.1%}):  £{e_cost_sonly:,.0f}")
    print(f"    Both          ({p_both:.1%}): £{e_cost_both:,.0f} (independence assumed)")
    print(f"    Weighted E[cost|att] = £{e_cost_att:,.0f}")

    total = 0
    for sz in SIZE_LABELS:
        nat = N_BY_SIZE[sz] * prev[sz] * e_cost_att
        total += nat
        print(f"    {SIZE_LABELS[sz]:>7s}: £{nat/1e9:.3f}bn")
    print(f"\n  NATIONAL TOTAL: £{total/1e9:.3f}bn")
    print(f"  (Rough — uses pure-subsample E[K], independence for 'both' firms)")


if __name__ == "__main__":
    main()
