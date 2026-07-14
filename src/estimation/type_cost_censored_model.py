"""
Distributional inference for per-type total costs, using ALL available
bounding evidence per firm — not just the disrupta==type subset used in
type_specific_bridge.py.

MOTIVATING DISCOVERY (checked here first): the type-specific total-cost
variables (ranscost_bands, viruscost_bands, doscost_bands, hackcost_bands,
tkvrcost_bands) are NOT restricted to firms where that type was the single
most disruptive incident (disrupta==type). They are populated whenever the
type occurred at all, per the type flags — e.g. for Ransomware, 33 firms have
a valid ranscost_bands value, but only 30 have disrupta==Ransomware, and only
30 of those 33 overlap (some valid-cost firms have a DIFFERENT type as their
single worst incident). type_specific_bridge.py only used the disrupta==type
subset (n=30 at best), throwing away real total-cost observations for firms
where the type happened but wasn't the worst incident.

ALSO DISCOVERED: hackcost_bands is not tied to type4 (hacking of bank
accounts) alone — 35 firms have a valid hackcost_bands with type4==0, and of
those, type7 (staff unauthorised access) or type8 (outsider unauthorised
access) is set instead. The variable label confirms this: "Total cost of
DELIBERATE HACKING INCIDENTS" — broader than the single Q53A item. Modelled
here as the union type4 | type7 | type8, with disrupta in {4, 7, 9} (7=staff,
9=outsiders — see disrupta code list in type_specific_bridge.py docstring)
as the corresponding "this was the worst incident and it's in this bucket"
condition.

THE IDEA (user's, 2026-07-13): for every attacked firm, per type, construct
a per-firm cost record with the tightest bound the data actually supports,
not just an exact-or-nothing observation:
  - type flag == 0                          -> cost = 0, exact (point mass)
  - direct cost column valid (any flag state) -> cost in [L, U), exact
    interval from the TYPE_COST 12-band scale (trust the more specific
    variable over the flag when both exist)
  - type flag == 1, no direct cost column, and this type IS the firm's
    disrupta (single most disruptive incident) -> cost >= L, where
    [L, U) is damage_bands' band. Lower-censored: we know at least this
    much (it was the worst incident of this exact type), but not the total.
  - type flag == 1, no direct cost column, and this type is NOT disrupta
    -> cost <= U, where [L, U) is damage_bands' band. Upper-censored: this
    ASSUMES the officially-designated "most disruptive" incident (of some
    other type) cost at least as much as this quieter one. That's a
    disruption-as-cost-proxy assumption, not a certainty — flagged clearly,
    not treated as ground truth.
  - type flag == 1, no direct cost column, disrupta missing/invalid ->
    dropped (fully uninformative, rare).

This is exactly the same censored/interval-MLE machinery as
cost_distribution.py's zero-inflated lognormal fit, just with two new
observation types (lower- and upper-censored) added to the likelihood
alongside the usual exact-interval and point-mass-zero terms. Fit is
survey-weighted (weighted pseudo-MLE) to be population-representative;
unweighted n's are reported alongside for transparency.

Output per type: fitted (p0, mu, sigma), E[cost | type occurred], and the
population-weighted expected annual cost contribution per firm from that
type — directly comparable to the single-worst-incident-only figures in
cost_share_by_type.py and the simple ratio point-estimates in
type_specific_bridge.py.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

df = get_business_data()
INVALID_FREQ = {997, 999, 997.0, 999.0}

attacked = df[
    df['freq'].notna() & ~df['freq'].isin(INVALID_FREQ) & (df['freq'] > 0) &
    df['disrupta'].notna() & (df['disrupta'] > 0) & (df['disrupta'] != 997)
].copy()

print(f"n attacked firms (valid freq + disrupta): {len(attacked)}")
print(f"n all business firms (population base for weighting): {len(df)}, "
      f"weight sum={df['weight'].sum():.2f} vs attacked-only weight sum={attacked['weight'].sum():.2f}")

# ---------------------------------------------------------------------------
# IMPORTANT WEIGHTING FIX (2026-07-13): build_records/fit_censored_lognormal
# must run over the FULL business population (all 2180 firms, weight sum
# ~2179.8), not just the freq>0 "attacked" subsample (weight sum ~916). Firms
# with no attack at all (freq<=0/missing) are legitimately flag=0 for every
# real attack type (checked directly: type1-13/15/16 are all 0 for these
# firms in this dataset — the only flag that fires for them is type11, which
# is actually the Q53A "None of these" response option, not a real attack
# type; TYPE_COLNAMES elsewhere in this project incorrectly includes type9/
# 10/11/12 as if they were attack flags, which are actually "Any other" /
# "Don't know" / "None of these" / "Refused" meta-responses — a pre-existing
# wrinkle worth flagging for any script that sums TYPE_COLNAMES as "n_types",
# not touched here since this script only ever references real attack-type
# columns by name). Using only the attacked subsample as the weight
# denominator would report "E[cost] conditional on being attacked at all",
# not "E[cost] per business" — the two differ by a factor of ~0.42 (916/2180)
# and only the latter is the correct quantity to feed into a national total.

# ---------------------------------------------------------------------------
# Band boundaries, £, both confirmed against the codebook directly (csbs.rtf)
# ---------------------------------------------------------------------------

# FULL 13-band scale: damage_bands, crimecost_bands, hacksumcost_bands, etc.
# Band 1 is a true point mass at 0. Band 13 open-ended (upper=None).
FULL_BAND_BOUNDS = {
    1: (0, 0),
    2: (1, 100), 3: (100, 500), 4: (500, 1_000), 5: (1_000, 5_000),
    6: (5_000, 10_000), 7: (10_000, 20_000), 8: (20_000, 50_000),
    9: (50_000, 100_000), 10: (100_000, 500_000), 11: (500_000, 1_000_000),
    12: (1_000_000, 5_000_000), 13: (5_000_000, None),
}

# TYPE_COST 12-band scale: ranscost_bands, hackcost_bands, doscost_bands,
# viruscost_bands, tkvrcost_bands. No zero band (only asked when the type
# occurred) — band 1 = "less than 100". Band 12 open-ended (upper=None).
TYPE_COST_BAND_BOUNDS = {
    1: (0, 100), 2: (100, 250), 3: (250, 500), 4: (500, 1_000),
    5: (1_000, 2_000), 6: (2_000, 5_000), 7: (5_000, 10_000),
    8: (10_000, 20_000), 9: (20_000, 50_000), 10: (50_000, 100_000),
    11: (100_000, 250_000), 12: (250_000, None),
}

# Ransomware bias correction (2026-07-13): ranscost_bands (the "exact"
# evidence source for Ransomware) was independently shown, earlier this
# session, to be a downward-biased holistic-recall measure relative to
# damage_bands (the itemized/comprehensive measure) — the "unpacking
# effect" (see NOTES.md). type_specific_bridge.py measured the specific
# undershoot for Ransomware directly: median ratio ranscost/damage = 0.483
# among the disrupta==Ransomware subset (n=20, robust to the one extreme
# outlier that drags the mean down). Treating raw ranscost_bands bands as
# ground-truth "exact" observations mixes a systematically-low source with
# the damage_bands-anchored lower-censored bounds (which are NOT biased
# this way) — very likely the actual cause of the degenerate fit reported
# earlier. Correcting by the inverse of this measured undershoot (1/0.483)
# before treating ranscost_bands as exact brings both evidence sources onto
# the same scale. NOTE: this same bias plausibly affects the other 4 types'
# "exact" observations too (the hard-ceiling check found ratio>1 near-absent
# across ALL 4), but only Ransomware's fit was broken badly enough to force
# a fix now — the other 4 are left uncorrected here, flagged as a follow-up.
RANSOMWARE_EXACT_CORRECTION = 1 / 0.483

TYPES = {
    'Ransomware': dict(flag_cols=['type1'], cost_col='ranscost_bands', disrupta_codes={1},
                        exact_correction=RANSOMWARE_EXACT_CORRECTION),
    'Other malware': dict(flag_cols=['type2'], cost_col='viruscost_bands', disrupta_codes={2}),
    'Denial of service': dict(flag_cols=['type3'], cost_col='doscost_bands', disrupta_codes={3}),
    'Hacking (broad: bank acct + unauthorised access staff/outsiders)':
        dict(flag_cols=['type4', 'type7', 'type8'], cost_col='hackcost_bands', disrupta_codes={4, 7, 9}),
    'Website/social/email takeover': dict(flag_cols=['type16'], cost_col='tkvrcost_bands', disrupta_codes={11}),
    # Impersonation (2026-07-13): NO direct total-cost column exists at all
    # (unlike the 5 types above) — impersonationhack/impersonationtkvr and
    # fraud4_comb1/2 were checked earlier this session and found unusable
    # (~95-97% "No" / ~6% coverage). cost_col=None means every flag=1 firm
    # falls into the lower- or upper-censored bucket, anchored purely on
    # damage_bands — no exact observations at all to anchor the fit.
    'Impersonation': dict(flag_cols=['type5'], cost_col=None, disrupta_codes={5}),
}


def safe_logcdf(logx):
    """log(Phi(logx)); logx can be -inf (band lower bound 0) -> returns -inf cleanly."""
    with np.errstate(divide='ignore'):
        return norm.logcdf(logx)


def build_records(sub, flag_cols, cost_col, disrupta_codes, exact_correction=1.0):
    """
    Returns lists of (L, U, weight) tuples per observation kind:
      exact:  finite interval [L, U), U may be None (open-ended top band)
      lower:  cost >= L (censored from above)
      upper:  cost <= U (censored from below at 0, i.e. [0, U])
    Plus n_zero (point mass at 0, weighted count) and n_dropped.

    cost_col=None means no direct total-cost variable exists for this type
    at all (e.g. Impersonation) — every flag=1 firm falls into lower/upper
    censoring, anchored purely on damage_bands, with zero exact evidence.

    exact_correction multiplies both L and U of exact observations before
    use — a calibration factor for a known bias in the direct cost column
    (see RANSOMWARE_EXACT_CORRECTION above), not used for cost_col=None.
    """
    exact, lower, upper = [], [], []
    n_zero_w = 0.0
    n_zero_n = 0
    n_dropped = 0

    has_flag = (sub[flag_cols].fillna(0) == 1).any(axis=1)
    if cost_col is not None:
        has_cost = sub[cost_col].notna() & (sub[cost_col] >= 1) & (sub[cost_col] <= 12)
    else:
        has_cost = pd.Series(False, index=sub.index)
    is_worst = sub['disrupta'].isin(disrupta_codes)

    for idx, row in sub.iterrows():
        w = row['weight']
        if has_cost.loc[idx]:
            b = int(row[cost_col])
            L, U = TYPE_COST_BAND_BOUNDS[b]
            L = L * exact_correction
            U = U * exact_correction if U is not None else None
            exact.append((L, U, w))
        elif not has_flag.loc[idx]:
            n_zero_w += w
            n_zero_n += 1
        else:
            # flag==1, no direct cost column
            db = row['damage_bands']
            if pd.isna(db) or db >= SPECIAL_CODE_THRESHOLD or db < 1:
                n_dropped += 1
                continue
            L, U = FULL_BAND_BOUNDS[int(db)]
            if is_worst.loc[idx]:
                lower.append((L, w))
            else:
                if U is None:
                    n_dropped += 1  # upper-censored at "open-ended" is uninformative
                    continue
                if U <= 0:
                    # damage_bands==1 (no cost) for the worst incident forces
                    # this quieter type's cost to exactly 0 too, under the
                    # disruption-as-cost-proxy assumption — route to the zero
                    # mass rather than an upper-censored lognormal term (a
                    # strictly-positive lognormal can't represent X==0).
                    n_zero_w += w
                    n_zero_n += 1
                    continue
                upper.append((U, w))

    return exact, lower, upper, n_zero_w, n_zero_n, n_dropped


def fit_censored_lognormal(exact, lower, upper, n_zero_w):
    """Weighted MLE for zero-inflated lognormal given exact/lower/upper censored obs."""
    w_exact = sum(w for _, _, w in exact)
    w_lower = sum(w for _, w in lower)
    w_upper = sum(w for _, w in upper)
    w_occurred = w_exact + w_lower + w_upper
    total_w = w_occurred + n_zero_w
    if total_w <= 0 or w_occurred <= 0:
        return None
    p0 = n_zero_w / total_w

    def nll(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        # Cap sigma to a range consistent with cost_distribution.py's fitted cells
        # (observed range there: 1.3-3.8, one pathological outlier at 47 flagged
        # as a known failure mode of unconstrained censored lognormal MLE on
        # sparse tail data — the same failure mode shows up here without a cap).
        if sigma < 1e-6 or sigma > 3.8:
            return 1e10
        ll = 0.0
        for L, U, w in exact:
            logL = -np.inf if L <= 0 else np.log(L)
            if U is None:
                # open-ended top band: survival contribution
                ll += w * (0 - safe_logcdf((logL - mu) / sigma)) if False else 0
                # P(X >= L) = 1 - Phi((logL-mu)/sigma)
                sf = norm.sf((logL - mu) / sigma)
                ll += w * np.log(max(sf, 1e-300))
            else:
                logU = np.log(U)
                cU = norm.cdf((logU - mu) / sigma)
                cL = 0.0 if L <= 0 else norm.cdf((logL - mu) / sigma)
                p = max(cU - cL, 1e-300)
                ll += w * np.log(p)
        for L, w in lower:
            logL = -np.inf if L <= 0 else np.log(L)
            sf = norm.sf((logL - mu) / sigma)
            ll += w * np.log(max(sf, 1e-300))
        for U, w in upper:
            logU = np.log(U)
            cU = norm.cdf((logU - mu) / sigma)
            ll += w * np.log(max(cU, 1e-300))
        return -ll

    best_res = None
    for mu0 in [5, 7, 9, 11]:
        for lsig0 in [0.5, 1.0, 1.5]:
            res = minimize(nll, x0=[mu0, lsig0], method='Nelder-Mead',
                            options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
            if best_res is None or res.fun < best_res.fun:
                best_res = res

    mu_hat, sigma_hat = best_res.x[0], np.exp(best_res.x[1])
    e_nonzero = np.exp(mu_hat + sigma_hat ** 2 / 2)
    return dict(p0=p0, mu=mu_hat, sigma=sigma_hat, e_nonzero=e_nonzero,
                w_exact=w_exact, w_lower=w_lower, w_upper=w_upper, total_w=total_w)


print("=" * 100)
print("CENSORED-LIKELIHOOD LOGNORMAL FIT PER TYPE (using ALL bounding evidence, not just disrupta==type)")
print("=" * 100)

results = []
for label, spec in TYPES.items():
    sub = df  # full population — see weighting fix note above
    exact, lower, upper, n_zero_w, n_zero_n, n_dropped = build_records(
        sub, spec['flag_cols'], spec['cost_col'], spec['disrupta_codes'],
        exact_correction=spec.get('exact_correction', 1.0)
    )
    n_exact, n_lower, n_upper = len(exact), len(lower), len(upper)
    print(f"\n--- {label} ---")
    print(f"  n exact={n_exact}  n lower-censored (>=)={n_lower}  n upper-censored (<=)={n_upper}  "
          f"n zero={n_zero_n}  n dropped(uninformative)={n_dropped}")

    fit = fit_censored_lognormal(exact, lower, upper, n_zero_w)
    if fit is None or (n_exact + n_lower + n_upper) < 8:
        print("  Too few informative observations — skipping fit.")
        continue

    SIGMA_CAP = 3.8
    degenerate = fit['sigma'] > SIGMA_CAP - 0.02
    e_cost = (1 - fit['p0']) * fit['e_nonzero']
    print(f"  p0 (P(type did not occur, weighted)) = {fit['p0']:.4f}")
    print(f"  Fitted mu={fit['mu']:.3f}  sigma={fit['sigma']:.3f}"
          f"{'  <<< PINNED AT CAP — DEGENERATE, NOT A REAL OPTIMUM' if degenerate else ''}")
    print(f"  E[cost | type occurred] = £{fit['e_nonzero']:,.0f}")
    print(f"  E[cost] (unconditional, per business in the WHOLE population, attacked or not) = £{e_cost:,.0f}")
    if degenerate:
        print(
            "  DEGENERATE FIT: sigma is pinned at the optimizer cap, meaning the likelihood keeps "
            "improving as sigma grows without bound — there is no real interior maximum. This happens "
            "when the exact observations (mostly small) and the censored bounds (at least one big, "
            "heavily-weighted lower bound) are in tension that a single lognormal can't reconcile. "
            "DO NOT treat the E[cost] figures above as a real estimate for this type — report as "
            "unidentified with this data instead."
        )

    # Comparison: naive single-worst-incident-only mean, among disrupta==type firms
    naive = attacked[attacked['disrupta'].isin(spec['disrupta_codes']) & (attacked['damage_bands'] < SPECIAL_CODE_THRESHOLD)]
    if len(naive) > 0:
        FULL_MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000,
                    9: 75000, 10: 300000, 11: 750000, 12: 3000000, 13: 5000000}
        naive_mean = np.average(naive['damage_bands'].map(FULL_MID), weights=naive['weight'])
        print(f"  [Comparison] mean single-worst-incident cost among disrupta==type firms: £{naive_mean:,.0f} "
              f"(n={len(naive)}) — the censored-model E[cost|occurred] above is £{fit['e_nonzero']:,.0f}, "
              f"a {fit['e_nonzero']/naive_mean:.2f}x multiplier implied purely by using the richer data.")

    results.append(dict(type=label, **fit, e_cost=e_cost, n_exact=n_exact, n_lower=n_lower, n_upper=n_upper,
                         degenerate=degenerate))

# ---------------------------------------------------------------------------
# Sensitivity check: is Impersonation's fit driven by the single dominant
# outlier row already flagged in impersonation_investigation.py (idx=792,
# weight=1.543, damage_bands=10 i.e. cost>=£100k, disrupta=Impersonation —
# previously found to be 69.5% of Impersonation's ENTIRE weighted cost mass
# on its own, and itself a multi-vector incident, not a clean single-type
# one)? Impersonation's fit above came out non-degenerate (sigma=3.756,
# just under the 3.8 cap) but with an implausibly huge E[cost] — close
# enough to the cap, and suspicious enough given zero exact anchors exist
# for this type at all, to warrant checking directly rather than trusting
# the "not flagged degenerate" label at face value.
# ---------------------------------------------------------------------------

print("\n" + "=" * 100)
print("SENSITIVITY CHECK: is the Impersonation fit driven by the single known dominant-outlier row?")
print("=" * 100)

df_excl = df.drop(index=792)
spec = TYPES['Impersonation']
imp_result = next(r for r in results if r['type'] == 'Impersonation')
e_cost_imp = imp_result['e_cost']
exact_x, lower_x, upper_x, n_zero_w_x, n_zero_n_x, n_dropped_x = build_records(
    df_excl, spec['flag_cols'], spec['cost_col'], spec['disrupta_codes']
)
fit_x = fit_censored_lognormal(exact_x, lower_x, upper_x, n_zero_w_x)
unreliable = False
if fit_x is not None:
    degenerate_x = fit_x['sigma'] > 3.8 - 0.02
    e_cost_x = (1 - fit_x['p0']) * fit_x['e_nonzero']
    print(f"  Excluding idx=792: n lower={len(lower_x)} (was {len(lower_x)+1}), n upper={len(upper_x)}")
    print(f"  Fitted mu={fit_x['mu']:.3f}  sigma={fit_x['sigma']:.3f}"
          f"{'  <<< STILL/NOW PINNED AT CAP' if degenerate_x else ''}")
    print(f"  E[cost | type occurred] = £{fit_x['e_nonzero']:,.0f}  (was £{imp_result['e_nonzero']:,.0f} including the outlier)")
    print(f"  E[cost] (per business) = £{e_cost_x:,.0f}  (was £{e_cost_imp:,.0f} including the outlier)")
    unreliable = degenerate_x or abs(e_cost_x - e_cost_imp) / e_cost_imp > 0.3
    verdict = (
        "The fit swings drastically without this single row — Impersonation is NOT genuinely "
        "identified, the previous result was propped up by one heavily-weighted, already-flagged-as-"
        "multi-vector outlier, exactly as suspected. Do not use either figure as a real estimate."
        if unreliable else
        "Fit is stable without the outlier — less suspicious than initially feared, though still "
        "worth treating cautiously given zero exact anchors exist for this type."
    )
    print(f"\n  Verdict: {verdict}")
else:
    print("  Fit failed entirely without this row — confirms the previous result was propped up by it alone.")
    unreliable = True

if unreliable:
    imp_result['degenerate'] = True

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
if results:
    summ = pd.DataFrame(results)[['type', 'n_exact', 'n_lower', 'n_upper', 'p0', 'mu', 'sigma',
                                   'e_nonzero', 'e_cost', 'degenerate']]
    print(summ.to_string(index=False))

    solid = summ[~summ['degenerate']]
    print(f"\nCombined E[cost] per business (population-level, attacked or not), summed over the {len(solid)} "
          f"well-behaved (non-degenerate) type fits: £{solid['e_cost'].sum():,.0f}")
    if summ['degenerate'].any():
        print(f"Excluded as unidentified: {', '.join(summ.loc[summ['degenerate'], 'type'])}")

print(
    "\nCaveat on the upper-censoring assumption: for firms where a type occurred but was NOT the "
    "single most disruptive incident, this model bounds that type's total cost above by the cost of "
    "whatever WAS most disruptive. That equates 'most disruptive' with 'costliest', which is a proxy "
    "assumption, not guaranteed by the survey design (disrupta is elicited as disruption, not cost). "
    "If cost and disruption diverge often, this understates the true upper bound for those firms and "
    "the fitted E[cost] here is a plausible-but-not-certain lower estimate of the true type-level mean."
)

# ---------------------------------------------------------------------------
# UNWEIGHTED VERSION + full 7-type comparison (2026-07-13, appended per
# request "how heavy is which type, weighted and unweighted"). Reruns the
# 4 solid censored fits (Malware, DoS, Hacking, Takeover) with weight=1 for
# every firm, and pulls in Phishing (established ~1.02x mixture bridge) and
# Ransomware/Impersonation (type_specific_montecarlo_bridge.py's numbers,
# not recomputed here) for one single side-by-side table.
# ---------------------------------------------------------------------------

print("\n" + "=" * 100)
print("UNWEIGHTED RERUN: the 4 solid censored fits, weight=1 for every firm")
print("=" * 100)

df_uw = df.copy()
df_uw['weight'] = 1.0
pop_w_uw = len(df_uw)

SOLID_TYPES = {k: v for k, v in TYPES.items() if k not in ('Ransomware', 'Impersonation')}
uw_results = []
for label, spec in SOLID_TYPES.items():
    exact_u, lower_u, upper_u, n_zero_w_u, n_zero_n_u, n_dropped_u = build_records(
        df_uw, spec['flag_cols'], spec['cost_col'], spec['disrupta_codes'],
        exact_correction=spec.get('exact_correction', 1.0)
    )
    fit_u = fit_censored_lognormal(exact_u, lower_u, upper_u, n_zero_w_u)
    if fit_u is None or (len(exact_u) + len(lower_u) + len(upper_u)) < 8:
        print(f"  {label}: too few observations unweighted — skipping")
        continue
    degenerate_u = fit_u['sigma'] > 3.8 - 0.02
    e_cost_u = (1 - fit_u['p0']) * fit_u['e_nonzero']
    print(f"  {label}: sigma={fit_u['sigma']:.3f}{' <<< DEGENERATE' if degenerate_u else ''}, "
          f"E[cost]/business (unweighted) = £{e_cost_u:,.2f}")
    uw_results.append(dict(type=label, e_cost_uw=e_cost_u, degenerate_uw=degenerate_u))

uw_df = pd.DataFrame(uw_results).set_index('type') if uw_results else pd.DataFrame()

# ---------------------------------------------------------------------------
# Phishing: established mixture-model bridge (~1.02x, mixture_bridge.py),
# applied directly to disrupta==Phishing firms' own damage_bands — both
# weighted and unweighted, for the same side-by-side comparison.
# ---------------------------------------------------------------------------

PHISHING_BRIDGE = 1.02
FULL_MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000,
            9: 75000, 10: 300000, 11: 750000, 12: 3000000, 13: 5000000}
phish = attacked[(attacked['disrupta'] == 6) & (attacked['damage_bands'] < SPECIAL_CODE_THRESHOLD)].copy()
phish['damage_mid'] = phish['damage_bands'].map(FULL_MID)
phish_w_total = (phish['damage_mid'] * phish['weight']).sum() * PHISHING_BRIDGE
phish_e_cost_w = phish_w_total / df['weight'].sum()
phish_e_cost_uw = phish['damage_mid'].sum() * PHISHING_BRIDGE / pop_w_uw

print("\n" + "=" * 100)
print("FULL COMPARISON: E[cost] per business, all 7 types, weighted vs unweighted")
print("=" * 100)

rows = []
rows.append(dict(type='Phishing (mixture bridge ~1.02x)', weighted=phish_e_cost_w, unweighted=phish_e_cost_uw,
                  degenerate_w=False, degenerate_uw=False))
for label in SOLID_TYPES:
    w_val = next((r['e_cost'] for r in results if r['type'] == label), np.nan)
    deg_w = next((r['degenerate'] for r in results if r['type'] == label), False)
    uw_val = uw_df.loc[label, 'e_cost_uw'] if label in uw_df.index else np.nan
    deg_uw = uw_df.loc[label, 'degenerate_uw'] if label in uw_df.index else True
    rows.append(dict(type=label, weighted=w_val, unweighted=uw_val, degenerate_w=deg_w, degenerate_uw=deg_uw))
rows.append(dict(type='Ransomware (Monte Carlo bridge, see type_specific_montecarlo_bridge.py)',
                  weighted=605.71, unweighted=831.97, degenerate_w=False, degenerate_uw=False))
rows.append(dict(type='Impersonation (Monte Carlo bridge, see type_specific_montecarlo_bridge.py)',
                  weighted=1908.61, unweighted=6036.13, degenerate_w=False, degenerate_uw=False))

comp_df = pd.DataFrame(rows)
comp_df['ratio_uw_over_w'] = comp_df['unweighted'] / comp_df['weighted']
pd.set_option('display.width', 120)
print(comp_df.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))

reliable = comp_df[~comp_df['degenerate_w'] & ~comp_df['degenerate_uw']]
print(f"\nTotal weighted, all 7 types (including degenerate/unreliable entries): £{comp_df['weighted'].sum():,.2f}/business")
print(f"Total unweighted, all 7 types (including degenerate/unreliable entries): £{comp_df['unweighted'].sum():,.2f}/business")
if (comp_df['degenerate_w'] | comp_df['degenerate_uw']).any():
    flagged = comp_df[comp_df['degenerate_w'] | comp_df['degenerate_uw']]['type'].tolist()
    print(f"\nFLAGGED AS UNRELIABLE (sigma pinned at cap in at least one of weighted/unweighted): {flagged}")
    print(f"Total weighted using ONLY the {len(reliable)} fully-reliable types: £{reliable['weighted'].sum():,.2f}/business")
    print(f"Total unweighted using ONLY the {len(reliable)} fully-reliable types: £{reliable['unweighted'].sum():,.2f}/business")
print(
    "\nNote: Ransomware/Impersonation figures are pulled in from type_specific_montecarlo_bridge.py, not "
    "recomputed here (different method — Monte Carlo bridge, not censored lognormal MLE). All other "
    "figures are computed directly in this script. 'Weighted' uses the survey's own design/adjustment "
    "weight (population-representative); 'unweighted' treats every sampled firm equally (raw sample). "
    "IMPORTANT: DoS and Takeover's UNWEIGHTED refits came out degenerate (sigma pinned at the optimizer "
    "cap) even though their WEIGHTED fits were solid — the censored MLE is less stable on the raw, "
    "unweighted sample for these two types. Their unweighted £ figures above should not be trusted."
)

print("\nDone.")
