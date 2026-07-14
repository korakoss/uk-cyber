"""
Type-specific Monte Carlo bridge for Ransomware and Impersonation, built by
actually combining the inputs catalogued across this whole thread instead of
testing each variable in isolation and giving up when any single one looked
thin. This directly reuses bridge_multiplier.py's already-validated
methodology (draw k per-attack cost samples, sum, compare to max) — the
difference is WHICH per-attack distribution and WHICH k get used.

THE KEY MOVE: for both types, there exists a genuinely clean "exactly one
incident of this type, and nothing else confounding it" reference subsample,
where damage_bands IS the type's total annual cost with zero bridging
ambiguity (same logic as freq==1 in the general framework, just verified via
a type-specific count instead of the contaminated cross-type freq). That
reference subsample's damage_bands distribution is then a properly-justified
"per-single-attack cost" distribution — usable in the same Monte Carlo
sum-of-k-draws technique already accepted elsewhere in this project — for
the smaller remaining slice of firms that DID have more than one incident of
that type.

RANSOMWARE: ranssoft_bands (count of ransomware attacks, Q83E — NOT the
contaminated cross-type freq) directly identifies which disrupta==Ransomware
firms had exactly 1 attack (n=19, weight 15.62) vs more than 1 (n=3, weight
3.05, out of ~18.7 total weight in this population). The n=19 clean singles
ALSO retroactively explain why the earlier censored-lognormal fit degenerated
(type_cost_censored_model.py): the single heaviest-weighted "lower-censored"
observation that blew up sigma (idx=1436, weight=1.405, damage_bands=10,
i.e. cost>=£100k) turns out to be exactly-one-attack per ranssoft_bands —
meaning its damage_bands IS its exact total cost, not merely a lower bound.
Treating it as merely "at least £100k" (as the old model did, for lack of
this variable) manufactured tension that was never really there.

IMPERSONATION: no dedicated count variable exists, but restricting to firms
where Impersonation was disrupta AND it was the ONLY attack type flagged all
year (n_types_real==1, using the REAL attack-type columns only — type9/10/
11/12 are Q53A meta-responses "any other"/"don't know"/"none"/"refused", not
attack flags, and are excluded here) removes the freq-contamination problem
identified earlier this session directly: for these firms, freq unambiguously
measures impersonation-specific frequency, because nothing else happened
that year. This "clean" subsample is n=78, weight 72.48 — 46.4% of ALL
Impersonation-disrupta weight, much bigger than expected. Within it, freq==1
firms (n=39, weight 42.77) are the exact-cost reference class (like
Ransomware's ranssoft_bands==1 subset), and freq>1 clean firms (n=39, weight
29.71) are the multi-incident group needing an actual bridge — now properly
isolated from the confounded majority instead of being modeled altogether.

Caveat carried over from bridge_multiplier.py (unavoidable, already accepted
project-wide): damage_bands among the "exactly 1" reference firms is that
firm's single worst incident of the year FOR THAT TYPE — since it's their
only incident of that type, this is clean. But using this distribution as a
stand-in for "cost of one attack" for MULTI-attack firms still implicitly
assumes each of their k attacks is drawn from the same distribution as a
typical single-incident firm's one attack — plausible, not certain.
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

rng = np.random.default_rng(42)
N_MC = 20000

df = get_business_data()
INVALID_FREQ = {997, 999, 997.0, 999.0}

attacked = df[
    df['freq'].notna() & ~df['freq'].isin(INVALID_FREQ) & (df['freq'] > 0) &
    df['disrupta'].notna() & (df['disrupta'] > 0) & (df['disrupta'] != 997)
].copy()

FULL_MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000,
            9: 75000, 10: 300000, 11: 750000, 12: 3000000, 13: 5000000}
FULL_BANDS = sorted(FULL_MID.keys())

REAL_TYPE_COLS = ['type1', 'type2', 'type3', 'type4', 'type5', 'type6', 'type7',
                   'type8', 'type13', 'type15', 'type16']  # excludes type9/10/11/12 meta-responses
for c in REAL_TYPE_COLS:
    attacked[c] = attacked[c].fillna(0)
attacked['n_types_real'] = attacked[REAL_TYPE_COLS].sum(axis=1)


def empirical_band_probs(damage_bands_series, weights):
    counts = np.array([(damage_bands_series == b).values @ weights.values for b in FULL_BANDS], dtype=float)
    if counts.sum() == 0:
        return None
    return counts / counts.sum()


def mc_sum_over_max(probs, k, n_mc=N_MC):
    if k <= 1:
        return None
    midpoints = np.array([FULL_MID[b] for b in FULL_BANDS])
    draws = rng.choice(midpoints, size=(n_mc, k), p=probs)
    sums = draws.sum(axis=1)
    return sums.mean(), np.percentile(sums, 5), np.percentile(sums, 95)


def floored(mc_mean, own_damage_band):
    """
    A firm's true annual total can never be less than its own directly-
    observed single-worst-incident cost (damage_bands), by definition. The MC
    estimate is drawn from a reference population's per-attack distribution,
    which is not conditioned on this specific firm's own outcome, so nothing
    guarantees mc_mean respects that floor — checked directly and violated in
    practice (see script docstring / the idx=792 case). Enforce it explicitly.
    """
    if pd.isna(own_damage_band) or own_damage_band >= SPECIAL_CODE_THRESHOLD:
        return mc_mean
    own_mid = FULL_MID.get(int(own_damage_band), np.nan)
    if pd.isna(own_mid):
        return mc_mean
    return max(mc_mean, own_mid)


print("=" * 100)
print("RANSOMWARE — Monte Carlo bridge using ranssoft_bands (type-specific attack count)")
print("=" * 100)

rans = attacked[attacked['disrupta'] == 1].copy()
print(f"n(disrupta==Ransomware) = {len(rans)}, weight sum = {rans['weight'].sum():.2f}")

clean_single_r = rans[rans['ranssoft_bands'] == 2]  # band 2 = "1" attack
print(f"\nClean single-attack reference (ranssoft_bands=='1'): n={len(clean_single_r)}, "
      f"weight={clean_single_r['weight'].sum():.2f}")
print("For these firms damage_bands IS the exact total ransomware cost (only 1 attack all year) — no bridge needed.")

RANS_COUNT_MID = {3: 2.5, 4: 4.5, 5: 8, 6: 15.5, 7: 35, 8: 75}  # band -> implied k (midpoint of range)
multi_r = rans[rans['ranssoft_bands'].isin(RANS_COUNT_MID.keys())].copy()
print(f"\nMulti-attack firms needing a bridge (ranssoft_bands >= '2-3'): n={len(multi_r)}, "
      f"weight={multi_r['weight'].sum():.2f}")

probs_r = empirical_band_probs(clean_single_r['damage_bands'], clean_single_r['weight'])
print(f"\nPer-attack cost distribution (from the {len(clean_single_r)}-firm clean reference), £ band probs:")
for b, p in zip(FULL_BANDS, probs_r):
    if p > 0:
        print(f"  band {b} (£{FULL_MID[b]:,}): {p:.3f}")

multi_r_results = []
for idx, row in multi_r.iterrows():
    k = RANS_COUNT_MID[int(row['ranssoft_bands'])]
    mean_sum, p5, p95 = mc_sum_over_max(probs_r, round(k))
    est_total = floored(mean_sum, row['damage_bands'])
    multi_r_results.append(dict(idx=idx, ranssoft_band=row['ranssoft_bands'], k=k,
                                 damage_bands=row['damage_bands'], weight=row['weight'],
                                 est_total=est_total, p5=p5, p95=p95))
    flag = "  <<< FLOORED (raw MC was below own damage_bands)" if est_total > mean_sum + 1e-6 else ""
    print(f"  firm idx={idx}: ranssoft_bands={row['ranssoft_bands']:.0f} (k~{k}), own damage_bands="
          f"{row['damage_bands']:.0f} (£{FULL_MID[int(row['damage_bands'])]:,}), weight={row['weight']:.3f} "
          f"-> MC estimated total = £{mean_sum:,.0f} [90% CI £{p5:,.0f}-£{p95:,.0f}], used = £{est_total:,.0f}{flag}")

# Aggregate: population-weighted E[cost] per business from Ransomware, using
# this structure instead of the degenerate censored-lognormal fit.
pop_w = df['weight'].sum()
clean_contribution = (clean_single_r['damage_bands'].map(FULL_MID) * clean_single_r['weight']).sum()
multi_contribution = sum(r['est_total'] * r['weight'] for r in multi_r_results)

# Residual: disrupta==Ransomware firms with unknown/missing ranssoft_bands —
# fall back to their own damage_bands as a conservative (likely-mild-undercount) estimate
residual_r = rans[~rans.index.isin(clean_single_r.index) & ~rans.index.isin(multi_r.index)]
residual_contribution = (residual_r['damage_bands'].map(
    lambda b: FULL_MID.get(int(b), np.nan) if pd.notna(b) and b < SPECIAL_CODE_THRESHOLD else np.nan
) * residual_r['weight']).sum()
print(f"\nResidual (ranssoft_bands missing/'don't know'): n={len(residual_r)}, weight={residual_r['weight'].sum():.2f}, "
      f"using own damage_bands as a conservative floor: £{residual_contribution:,.0f} weighted contribution")

total_r_contribution = clean_contribution + multi_contribution + residual_contribution
e_cost_r = total_r_contribution / pop_w
print(f"\nTOTAL Ransomware E[cost] per business (population-weighted): £{e_cost_r:,.2f}")
print(f"  (clean-singles contribute £{clean_contribution/pop_w:,.2f}, multi-attack MC estimate contributes "
      f"£{multi_contribution/pop_w:,.2f}, residual £{residual_contribution/pop_w:,.2f})")
print("This replaces the earlier degenerate censored-lognormal result — this one is not degenerate, "
      "is built entirely from exact/clean observations plus a small, explicit Monte Carlo extrapolation "
      "for only 3 firms, and is directly comparable to the £3,269/business the broken fit produced.")

# ---------------------------------------------------------------------------
# THE BRIDGE ITSELF: express this as a multiplier, directly comparable to the
# other 4 types' single-number bridges (Malware 1.93x, DoS 0.36x, Hacking
# 1.01x, Takeover 5.50x from type_cost_censored_model.py). There is no single
# flat ratio here — the rule is per-firm (bridge=1 for clean single-attack
# firms, Monte-Carlo-derived for multi-attack firms) — but the same "naive"
# comparison (what if bridge=1 for EVERY disrupta==Ransomware firm, including
# the multi-attack ones?) gives a directly comparable overall number.
# ---------------------------------------------------------------------------

naive_multi_r = (multi_r['damage_bands'].map(FULL_MID) * multi_r['weight']).sum()
naive_total_r = clean_contribution + naive_multi_r + residual_contribution
naive_e_cost_r = naive_total_r / pop_w
print(f"\nTHE BRIDGE: naive (bridge=1 for every Ransomware firm) would give £{naive_e_cost_r:,.2f}/business. "
      f"This method gives £{e_cost_r:,.2f}/business. Implied overall multiplier: {e_cost_r/naive_e_cost_r:.2f}x "
      f"— but this is an AVERAGE over a rule that is exactly 1.00x for ~86% of Ransomware weight (the clean "
      f"single-attack firms) and much higher (driven by the 3 Monte-Carlo-extrapolated firms) for the rest.")

# ---------------------------------------------------------------------------

print("\n" + "=" * 100)
print("IMPERSONATION — Monte Carlo bridge using the clean (impersonation-only) subsample")
print("=" * 100)

imp = attacked[attacked['disrupta'] == 5].copy()
print(f"n(disrupta==Impersonation) = {len(imp)}, weight sum = {imp['weight'].sum():.2f}")

imp_clean = imp[imp['n_types_real'] == 1].copy()
print(f"\nClean (impersonation was the ONLY type all year): n={len(imp_clean)}, "
      f"weight={imp_clean['weight'].sum():.2f} ({100*imp_clean['weight'].sum()/imp['weight'].sum():.1f}% of "
      f"total Impersonation-disrupta weight)")

imp_clean_f1 = imp_clean[imp_clean['freq'] == 1]
print(f"Of which freq==1 (clean single-incident reference, no bridge needed): n={len(imp_clean_f1)}, "
      f"weight={imp_clean_f1['weight'].sum():.2f}")

K_IMPLIED = {1: 1, 2: 6, 3: 12, 4: 52, 5: 365, 6: 1000}
imp_clean_multi = imp_clean[imp_clean['freq'] > 1].copy()
print(f"Of which freq>1, still clean (needs a bridge, no cross-type contamination): n={len(imp_clean_multi)}, "
      f"weight={imp_clean_multi['weight'].sum():.2f}")
print(imp_clean_multi['freq'].value_counts().sort_index())

probs_i = empirical_band_probs(imp_clean_f1['damage_bands'], imp_clean_f1['weight'])
print(f"\nPer-attack cost distribution (from the {len(imp_clean_f1)}-firm clean reference), £ band probs:")
for b, p in zip(FULL_BANDS, probs_i):
    if p > 0:
        print(f"  band {b} (£{FULL_MID[b]:,}): {p:.3f}")

imp_multi_results = []
for idx, row in imp_clean_multi.iterrows():
    k = K_IMPLIED.get(int(row['freq']), 1)
    mean_sum, p5, p95 = mc_sum_over_max(probs_i, k)
    est_total = floored(mean_sum, row['damage_bands'])
    imp_multi_results.append(dict(idx=idx, freq=row['freq'], k=k, weight=row['weight'], est_total=est_total))

print(f"\nMC-estimated totals for the {len(imp_multi_results)} clean multi-incident firms (grouped by freq):")
imp_multi_df = pd.DataFrame(imp_multi_results)
if len(imp_multi_df) > 0:
    print(imp_multi_df.groupby('freq').agg(n=('weight', 'size'), w=('weight', 'sum'),
                                            est_total=('est_total', 'mean')))

clean_f1_contribution = (imp_clean_f1['damage_bands'].map(FULL_MID) * imp_clean_f1['weight']).sum()
clean_multi_contribution = sum(r['est_total'] * r['weight'] for r in imp_multi_results)

# Contaminated (multi-type) subset — this is the genuinely unresolved residual.
# Apply the clean-derived per-attack distribution as a working (not certain)
# estimate rather than silently falling back to conservative=1 for ALL of it —
# explicitly flagged as the weakest part of this estimate, not hidden.
imp_dirty = imp[imp['n_types_real'] > 1].copy()
print(f"\nContaminated (multi-type) residual: n={len(imp_dirty)}, weight={imp_dirty['weight'].sum():.2f} "
      f"({100*imp_dirty['weight'].sum()/imp['weight'].sum():.1f}% of total) — the genuinely unresolved part.")
imp_dirty_results = []
n_dirty_floored = 0
for idx, row in imp_dirty.iterrows():
    k = K_IMPLIED.get(int(row['freq']), 1)
    if k <= 1:
        est = row['damage_bands']
        est = FULL_MID.get(int(est), np.nan) if pd.notna(est) and est < SPECIAL_CODE_THRESHOLD else np.nan
    else:
        mean_sum, _, _ = mc_sum_over_max(probs_i, k)
        est = floored(mean_sum, row['damage_bands'])
        if est > mean_sum + 1e-6:
            n_dirty_floored += 1
    imp_dirty_results.append(dict(idx=idx, weight=row['weight'], est_total=est))
dirty_contribution = sum(r['est_total'] * r['weight'] for r in imp_dirty_results if pd.notna(r['est_total']))
print(f"n contaminated-residual firms where the floor had to kick in (raw MC was below the firm's own "
      f"single-worst-incident cost): {n_dirty_floored} / {len(imp_dirty)}")

total_i_contribution = clean_f1_contribution + clean_multi_contribution + dirty_contribution
e_cost_i = total_i_contribution / pop_w
print(f"\nTOTAL Impersonation E[cost] per business (population-weighted): £{e_cost_i:,.2f}")
print(f"  clean freq==1 contributes £{clean_f1_contribution/pop_w:,.2f}, clean multi-incident MC contributes "
      f"£{clean_multi_contribution/pop_w:,.2f}, contaminated-residual (using the SAME clean per-attack "
      f"distribution, weakest assumption in this script) contributes £{dirty_contribution/pop_w:,.2f}")
naive_multi_i = (imp_clean_multi['damage_bands'].map(FULL_MID) * imp_clean_multi['weight']).sum()
naive_dirty_i = (imp_dirty['damage_bands'].map(
    lambda b: FULL_MID.get(int(b), np.nan) if pd.notna(b) and b < SPECIAL_CODE_THRESHOLD else np.nan
) * imp_dirty['weight']).sum()
naive_total_i = clean_f1_contribution + naive_multi_i + naive_dirty_i
naive_e_cost_i = naive_total_i / pop_w
print(f"\nTHE BRIDGE: naive (bridge=1 for every Impersonation firm) would give £{naive_e_cost_i:,.2f}/business. "
      f"This method gives £{e_cost_i:,.2f}/business. Implied overall multiplier: {e_cost_i/naive_e_cost_i:.2f}x "
      f"— again an AVERAGE: exactly 1.00x for the clean freq==1 firms (27% of weight), Monte-Carlo-derived "
      f"for the clean multi-incident firms (19% of weight), and Monte-Carlo-derived-from-a-different-"
      f"population (the weakest part) for the contaminated 54%.")
print(
    "\nCaveat on the contaminated-residual piece: applying the impersonation-only per-attack distribution "
    "to firms that ALSO had other attack types assumes their impersonation-specific costs look like a "
    "typical impersonation-only firm's — plausible but unverified, and this is where the known dominant "
    "outlier (idx=792, weight=1.54, damage_bands=10) sits, since it is NOT impersonation-only (it has "
    "Hacking+Phishing+Takeover co-occurring) so it necessarily falls in this weakest bucket."
)

# ---------------------------------------------------------------------------
# LEAVE-ONE-OUT SENSITIVITY CHECK (2026-07-14): the body-vs-tail check earlier
# found the n=39 reference sample's single £35k observation (idx=2066, Small
# firm, weight=0.479) was only moderately expected (1.8% chance) under a
# lognormal fit to the other 38 firms — and this reference distribution feeds
# the extreme-frequency (freq=5/6, K_IMPLIED=365/1000) firms that dominate the
# contaminated-residual bucket. Rebuild everything with that ONE observation
# removed from the reference sample, to see how much the total actually moves.
# ---------------------------------------------------------------------------

print("\n" + "=" * 100)
print("LEAVE-ONE-OUT: rebuild the Impersonation reference distribution WITHOUT idx=2066 (the £35k firm)")
print("=" * 100)

imp_clean_f1_loo = imp_clean_f1.drop(index=2066)
print(f"Reference sample without idx=2066: n={len(imp_clean_f1_loo)} (was {len(imp_clean_f1)}), "
      f"weight={imp_clean_f1_loo['weight'].sum():.2f} (was {imp_clean_f1['weight'].sum():.2f})")

probs_i_loo = empirical_band_probs(imp_clean_f1_loo['damage_bands'], imp_clean_f1_loo['weight'])
print("Per-attack cost distribution WITHOUT idx=2066:")
for b, p in zip(FULL_BANDS, probs_i_loo):
    if p > 0:
        print(f"  band {b} (£{FULL_MID[b]:,}): {p:.3f}")
print("(compare to WITH idx=2066: band 8/£35k had probability 0.012 — now check if it's gone entirely)")

# Recompute the clean multi-incident bucket using the LOO reference distribution
imp_multi_loo_results = []
for idx, row in imp_clean_multi.iterrows():
    k = K_IMPLIED.get(int(row['freq']), 1)
    mean_sum, _, _ = mc_sum_over_max(probs_i_loo, k)
    est_total = floored(mean_sum, row['damage_bands'])
    imp_multi_loo_results.append(dict(idx=idx, weight=row['weight'], est_total=est_total))
clean_multi_contribution_loo = sum(r['est_total'] * r['weight'] for r in imp_multi_loo_results)

# Recompute the contaminated-residual bucket using the LOO reference distribution
imp_dirty_loo_results = []
for idx, row in imp_dirty.iterrows():
    k = K_IMPLIED.get(int(row['freq']), 1)
    if k <= 1:
        est = row['damage_bands']
        est = FULL_MID.get(int(est), np.nan) if pd.notna(est) and est < SPECIAL_CODE_THRESHOLD else np.nan
    else:
        mean_sum, _, _ = mc_sum_over_max(probs_i_loo, k)
        est = floored(mean_sum, row['damage_bands'])
    imp_dirty_loo_results.append(dict(idx=idx, weight=row['weight'], est_total=est))
dirty_contribution_loo = sum(r['est_total'] * r['weight'] for r in imp_dirty_loo_results if pd.notna(r['est_total']))

# Reference-class contribution itself also drops (one fewer firm contributing its own cost)
clean_f1_contribution_loo = (imp_clean_f1_loo['damage_bands'].map(FULL_MID) * imp_clean_f1_loo['weight']).sum()

total_i_loo = clean_f1_contribution_loo + clean_multi_contribution_loo + dirty_contribution_loo
e_cost_i_loo = total_i_loo / pop_w

print(f"\nWITH idx=2066:    clean freq==1 £{clean_f1_contribution/pop_w:,.2f} + clean multi £"
      f"{clean_multi_contribution/pop_w:,.2f} + dirty-residual £{dirty_contribution/pop_w:,.2f} "
      f"= TOTAL £{e_cost_i:,.2f}/business")
print(f"WITHOUT idx=2066: clean freq==1 £{clean_f1_contribution_loo/pop_w:,.2f} + clean multi £"
      f"{clean_multi_contribution_loo/pop_w:,.2f} + dirty-residual £{dirty_contribution_loo/pop_w:,.2f} "
      f"= TOTAL £{e_cost_i_loo:,.2f}/business")
print(f"\nChange from removing ONE reference-sample observation (weight=0.479 out of {pop_w:,.0f} population "
      f"weight): £{e_cost_i:,.2f} -> £{e_cost_i_loo:,.2f}  ({100*(e_cost_i_loo-e_cost_i)/e_cost_i:+.1f}%)")

# ---------------------------------------------------------------------------
# UNWEIGHTED VERSION (2026-07-13, appended per request): everything above uses
# 'weight' — the survey-provided design/adjustment weight (corrects for
# sampling by size stratum, non-response, etc.), same convention used
# throughout this project. Redone here with every firm weight=1, to show how
# much the survey weighting itself is doing versus the raw sample.
# ---------------------------------------------------------------------------

print("\n" + "=" * 100)
print("UNWEIGHTED VERSION (weight=1 for every firm — raw sample, no survey design adjustment)")
print("=" * 100)

attacked_uw = attacked.copy()
attacked_uw['weight'] = 1.0
df_uw_n = len(df)  # unweighted population base = raw firm count

rans_uw = attacked_uw[attacked_uw['disrupta'] == 1]
clean_single_r_uw = rans_uw[rans_uw['ranssoft_bands'] == 2]
multi_r_uw = rans_uw[rans_uw['ranssoft_bands'].isin(RANS_COUNT_MID.keys())]
probs_r_uw = empirical_band_probs(clean_single_r_uw['damage_bands'], clean_single_r_uw['weight'])

multi_r_uw_results = []
for idx, row in multi_r_uw.iterrows():
    k = RANS_COUNT_MID[int(row['ranssoft_bands'])]
    mean_sum, _, _ = mc_sum_over_max(probs_r_uw, round(k))
    multi_r_uw_results.append(floored(mean_sum, row['damage_bands']))

clean_contribution_uw = clean_single_r_uw['damage_bands'].map(FULL_MID).sum()
multi_contribution_uw = sum(multi_r_uw_results)
residual_r_uw = rans_uw[~rans_uw.index.isin(clean_single_r_uw.index) & ~rans_uw.index.isin(multi_r_uw.index)]
residual_contribution_uw = residual_r_uw['damage_bands'].map(
    lambda b: FULL_MID.get(int(b), np.nan) if pd.notna(b) and b < SPECIAL_CODE_THRESHOLD else np.nan
).sum()
e_cost_r_uw = (clean_contribution_uw + multi_contribution_uw + residual_contribution_uw) / df_uw_n
naive_r_uw = (clean_contribution_uw +
              multi_r_uw['damage_bands'].map(FULL_MID).sum() +
              residual_contribution_uw) / df_uw_n

print(f"\nRansomware — WEIGHTED: £{e_cost_r:,.2f}/business (naive £{naive_e_cost_r:,.2f}, {e_cost_r/naive_e_cost_r:.2f}x)")
print(f"Ransomware — UNWEIGHTED: £{e_cost_r_uw:,.2f}/business (naive £{naive_r_uw:,.2f}, "
      f"{e_cost_r_uw/naive_r_uw:.2f}x)")

imp_uw = attacked_uw[attacked_uw['disrupta'] == 5]
imp_clean_uw = imp_uw[imp_uw['n_types_real'] == 1]
imp_clean_f1_uw = imp_clean_uw[imp_clean_uw['freq'] == 1]
imp_clean_multi_uw = imp_clean_uw[imp_clean_uw['freq'] > 1]
probs_i_uw = empirical_band_probs(imp_clean_f1_uw['damage_bands'], imp_clean_f1_uw['weight'])

imp_multi_uw_results = []
for idx, row in imp_clean_multi_uw.iterrows():
    k = K_IMPLIED.get(int(row['freq']), 1)
    mean_sum, _, _ = mc_sum_over_max(probs_i_uw, k)
    imp_multi_uw_results.append(floored(mean_sum, row['damage_bands']))

clean_f1_contribution_uw = imp_clean_f1_uw['damage_bands'].map(FULL_MID).sum()
clean_multi_contribution_uw = sum(imp_multi_uw_results)

imp_dirty_uw = imp_uw[imp_uw['n_types_real'] > 1]
imp_dirty_uw_results = []
for idx, row in imp_dirty_uw.iterrows():
    k = K_IMPLIED.get(int(row['freq']), 1)
    if k <= 1:
        est = row['damage_bands']
        est = FULL_MID.get(int(est), np.nan) if pd.notna(est) and est < SPECIAL_CODE_THRESHOLD else np.nan
    else:
        mean_sum, _, _ = mc_sum_over_max(probs_i_uw, k)
        est = floored(mean_sum, row['damage_bands'])
    imp_dirty_uw_results.append(est)
dirty_contribution_uw = sum(e for e in imp_dirty_uw_results if pd.notna(e))

e_cost_i_uw = (clean_f1_contribution_uw + clean_multi_contribution_uw + dirty_contribution_uw) / df_uw_n
naive_i_uw = (clean_f1_contribution_uw +
              imp_clean_multi_uw['damage_bands'].map(FULL_MID).sum() +
              imp_dirty_uw['damage_bands'].map(
                  lambda b: FULL_MID.get(int(b), np.nan) if pd.notna(b) and b < SPECIAL_CODE_THRESHOLD else np.nan
              ).sum()) / df_uw_n

print(f"\nImpersonation — WEIGHTED: £{e_cost_i:,.2f}/business (naive £{naive_e_cost_i:,.2f}, {e_cost_i/naive_e_cost_i:.2f}x)")
print(f"Impersonation — UNWEIGHTED: £{e_cost_i_uw:,.2f}/business (naive £{naive_i_uw:,.2f}, "
      f"{e_cost_i_uw/naive_i_uw:.2f}x)")

print(
    "\nNote: 'weight' throughout this project is the survey's own design/adjustment weight (corrects for "
    "sampling by size stratum and non-response) — the standard convention used in every weighted figure "
    "reported this session. Comparing to the unweighted (raw sample, weight=1) version shows how much the "
    "survey's own size-stratification correction is doing, separately from the modelling choices in this script."
)

print("\nDone.")
