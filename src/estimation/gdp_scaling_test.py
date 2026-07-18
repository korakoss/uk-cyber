"""
Does cybercrime damage scale in proportion to economic size ("share of GDP")?
(2026-07-15)

Tests a premise common in extrapolation methodology: that damages sustained by
two entities scale in proportion to their share of GDP. If true, an entity's
share of total cyber damage should equal its share of GDP; a systematic
departure is evidence the GDP-proportional bridge is biased.

We test at two cuts:
  1. SECTOR (the clean GDP analog). Each SIC-mapped sector's share of business
     cyber damage (from the survey) vs its share of UK GVA. GVA-by-industry IS
     how the ONS measures sectoral contribution to GDP, so this is a genuine
     GDP test, not a proxy.
  2. SIZE BAND (proxied). Each employment-size band's share of damage vs its
     share of TURNOVER (BPE). Turnover is only a proxy for GDP/value-added
     (it double-counts intermediate inputs and rises faster than value-added
     with firm size), so read the size cut as indicative, not a true GDP test.

Damage per firm uses the same type-based construction as the rest of the
pipeline: per-size lognormal within-band conditional mean × per-type bridge
(Impersonation at range midpoint), weighted by survey weight. Shares are
weight-based, so ONS N(size) scaling cancels out.

DATA PROVENANCE for the external denominators:
- GVA by SIC section: ONS 2023 current-price GVA. The ONS publishes a 10-group
  aggregation directly (total £2,497.4bn); values for sections that share a
  group (G/H/I distribution; M/N professional+admin; O/P/Q government;
  R/S other) are APPORTIONED using standard ONS section proportions and are
  marked approximate below. Public administration (O, ~£118bn) has no
  private-business counterpart in the survey and is excluded.
- Turnover by size band: ONS Business Population Estimates 2025, Table C
  (employer frame). Excludes financial & insurance (data unavailable).

MAJOR CAVEAT: Education (P) and Health (Q) GVA include large PUBLIC-sector
provision (state schools, NHS) that is not in the private-business survey
frame, so their GDP share overstates the economic size of the slice we
measured damage on. Flagged per-sector below. Also: per-sector damage inherits
the tail fragility (a few top-band firms drive it), so thin sectors are noisy —
n and top-band counts are printed alongside.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

sys.path.insert(0, '.')
from proc import SPECIAL_CODE_THRESHOLD

BAND_BOUNDS = {
    1: (0, 0), 2: (1, 100), 3: (100, 500), 4: (500, 1000), 5: (1000, 5000),
    6: (5000, 10000), 7: (10000, 20000), 8: (20000, 50000),
    9: (50000, 100000), 10: (100000, 500000),  # bounded per codebook
}
BEST_ESTIMATE_BRIDGE = {1: 2.21, 2: 1.93, 3: 0.36, 4: 1.01, 7: 1.01, 9: 1.01, 6: 1.02, 11: 5.50}
IMP_BRIDGE_MID = (3.11 + 6.25) / 2
VALID_FREQ = {1, 2, 3, 4, 5, 6}

SECTOR_LABELS = {
    1: 'Admin/real estate (L,N)', 2: 'Construction (F)', 3: 'Education (P)',
    4: 'Entertainment/membership (R,S)', 5: 'Finance/insurance (K)',
    6: 'Food/hospitality (I)', 7: 'Health/social care (Q)',
    8: 'Info/comms (J)', 9: 'Professional/sci/tech (M)',
    10: 'Retail/wholesale (G)', 11: 'Transport/storage (H)',
    12: 'Utilities/production (B-E)', 13: 'Agriculture (A)',
}
# GVA £bn by sector_comb2 group (2023, current basic prices). D=direct ONS
# 10-group value; A=apportioned within a shared group (approximate).
GVA_BN = {
    1: 346.85 + 143,   # L real estate (D) + N admin/support (A)  ≈ 489.85
    2: 148.46,         # F construction (D)
    3: 140,            # P education (A, within O/P/Q; PUBLIC-heavy)
    4: 74.65,          # R+S other/arts (D as 'other services' group)
    5: 204.05,         # K finance/insurance (D)
    6: 58,             # I accommodation/food (A, within distribution)
    7: 215,            # Q health/social (A, within O/P/Q; PUBLIC-heavy)
    8: 152.30,         # J information/communication (D)
    9: 197,            # M professional/sci/tech (A, within M/N)
    10: 247,           # G wholesale/retail (A, within distribution)
    11: 97,            # H transport/storage (A, within distribution)
    12: 337.89,        # B-E production (D)
    13: 17.84,         # A agriculture (D)
}
PUBLIC_HEAVY_SECTORS = {3, 7}  # GVA overstates private-business economic size

# Turnover £bn by size band (BPE 2025, Table C, employer frame; excl. finance)
TURNOVER_BN = {1: 701.2, 2: 776.2, 3: 948.6, 4: 2696.0}
SIZE_LABELS = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}


def fit_lognormal(bands):
    bands = bands[~np.isnan(bands)].astype(int)
    counts = np.array([np.sum(bands == b) for b in range(1, 11)], float)
    cond = counts[1:]

    def nll(p):
        mu, ls = p
        sigma = np.exp(ls)
        pred = np.array([norm.cdf(np.log(BAND_BOUNDS[b][1]), mu, sigma)
                         - norm.cdf(np.log(max(BAND_BOUNDS[b][0], 1)), mu, sigma) for b in range(2, 11)])
        s = pred.sum()
        return 1e10 if s < 1e-9 else -(cond * np.log(np.clip(pred / s, 1e-12, 1))).sum()

    best = None
    for mu0 in (5, 7, 9, 11):
        for ls0 in (0.5, 1.0, 1.5):
            r = minimize(nll, [mu0, ls0], method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
            if best is None or r.fun < best.fun:
                best = r
    return best.x[0], float(np.exp(best.x[1]))


def band_conditional_mean(mu, sigma, b):
    if b == 1:
        return 0.0
    L, U = BAND_BOUNDS[b]
    a = (np.log(max(L, 1)) - mu) / sigma
    bU = (np.log(U) - mu) / sigma
    denom = norm.cdf(bU) - norm.cdf(a)
    if denom < 1e-12:
        return np.exp(mu + sigma ** 2 / 2)
    return np.exp(mu + sigma ** 2 / 2) * (norm.cdf(bU - sigma) - norm.cdf(a - sigma)) / denom


def type_bridge(disrupta, freq):
    if freq == 1:
        return 1.0
    if disrupta == 5:
        return IMP_BRIDGE_MID
    return BEST_ESTIMATE_BRIDGE.get(int(disrupta), 1.0) if not pd.isna(disrupta) else 1.0


# ---- load full business data (need sector_comb2, not in get_business_data) ----
d = pd.read_csv('data/proc/data.csv')
d = d[d['typex'] == 1].copy()
d['sizeb'] = d['sizeb'].astype('Int64')
d['attacked'] = d['freq'].isin(VALID_FREQ)
db = d['damage_bands']
d['band'] = np.where(d['attacked'] & db.notna() & (db < SPECIAL_CODE_THRESHOLD) & (db >= 1), db, np.nan)

# per-size lognormal shapes
size_fit = {sz: fit_lognormal(d.loc[d['sizeb'] == sz, 'band'].values) for sz in (1, 2, 3, 4)}

# per-firm expected damage (type-based)
def firm_cost(r):
    if not r['attacked'] or np.isnan(r['band']):
        return 0.0
    mu, sigma = size_fit[int(r['sizeb'])]
    return band_conditional_mean(mu, sigma, int(r['band'])) * type_bridge(r['disrupta'], r['freq'])

d['exp_cost'] = d.apply(firm_cost, axis=1)
d['wcost'] = d['exp_cost'] * d['weight']
total_wcost = d['wcost'].sum()


def proportionality_report(group_col, labels, external_bn, title, external_name, flag=set()):
    print("=" * 92)
    print(title)
    print("=" * 92)
    ext_total = sum(external_bn.values())
    rows = []
    for g in sorted(labels):
        sub = d[d[group_col] == g]
        dmg_share = sub['wcost'].sum() / total_wcost
        ext_share = external_bn[g] / ext_total
        ratio = dmg_share / ext_share if ext_share > 0 else np.nan
        n = len(sub)
        n_top = int((sub['band'] == 10).sum())
        rows.append((g, dmg_share, ext_share, ratio, n, n_top))
    print(f"{'sector/size':<32} {'dmg%':>7} {external_name:>8} {'ratio':>7} {'n':>5} {'ntop':>5}")
    for g, ds, es, ratio, n, n_top in rows:
        star = ' *PUB' if g in flag else ''
        print(f"{labels[g]:<32} {100*ds:>6.1f}% {100*es:>7.1f}% {ratio:>7.2f} {n:>5} {n_top:>5}{star}")
    # correlation between damage share and external share (proportionality => high, slope~1)
    ds_arr = np.array([r[1] for r in rows])
    es_arr = np.array([r[2] for r in rows])
    corr = np.corrcoef(ds_arr, es_arr)[0, 1]
    # OLS slope through the shares (no intercept, since shares) and with intercept
    slope_ols = np.polyfit(es_arr, ds_arr, 1)
    print(f"\n  corr(damage share, {external_name} share) = {corr:.2f}   "
          f"(proportional => ~1.0)")
    print(f"  OLS damage% = {slope_ols[0]:.2f}·{external_name}% + {100*slope_ols[1]:.2f}pp   "
          f"(proportional => slope 1, intercept 0)")
    print(f"  ratio spread: min {min(r[3] for r in rows):.2f} ({labels[min(rows,key=lambda r:r[3])[0]]}), "
          f"max {max(r[3] for r in rows):.2f} ({labels[max(rows,key=lambda r:r[3])[0]]})")
    return rows


print(f"Total weighted expected business cyber damage (relative units): {total_wcost:,.0f}\n")

proportionality_report('sector_comb2', SECTOR_LABELS, GVA_BN,
                       "TEST 1 — SECTOR: cyber-damage share vs share of UK GVA (≈ GDP)",
                       "GVA", flag=PUBLIC_HEAVY_SECTORS)
print("  (*PUB = GVA includes large public-sector provision not in the business survey —")
print("   these sectors' GDP share overstates the private slice we measured.)\n")

proportionality_report('sizeb', SIZE_LABELS, TURNOVER_BN,
                       "TEST 2 — SIZE BAND: cyber-damage share vs share of TURNOVER (GDP proxy)",
                       "turnover")
print("  (turnover is a GDP PROXY, not GDP; excludes finance; read as indicative.)")

print("\nInterpretation: ratio = damage share ÷ economic-size share. ratio≈1 for all")
print("groups would support the GDP-proportional premise; large spread refutes it")
print("(some sectors/sizes take far more or less cyber damage than their economic")
print("size predicts). Watch n/ntop: small-n or 1-2-top-band sectors are unreliable.")


# ==========================================================================
# APPENDED 2026-07-17 — rerun TEST 1 with REAL section-level aGVA
# ==========================================================================
# Replaces the apportioned 10-group GVA above with direct section-level
# figures from the ONS Annual Business Survey (ABS), "Sections A to S" table,
# 2023 aGVA at basic prices (£ million), downloaded from:
#   ons.gov.uk/.../uknonfinancialbusinesseconomyannualbusinesssurveysectionsas
#
# Why this is a genuine improvement over the apportioned GVA above:
#  1. DIRECT, not apportioned — each SIC section has its own published value
#     (no splitting of ONS 10-group aggregates by assumed proportions).
#  2. PRIVATE-SECTOR / BUSINESS-ECONOMY frame — ABS aGVA is the non-financial
#     *business* economy, i.e. the same frame the CSBS measures. This removes
#     the public-sector confound that inflated Education/Health GVA in the
#     total-GVA version: private Education aGVA is £38.3bn (not the ~£140bn
#     total GVA incl. state schools) and private Health £53.3bn (not ~£215bn
#     incl. NHS). So the earlier *PUB flag is no longer needed.
#
# Caveats specific to this source:
#  - Finance & insurance (K, sector_comb2=5) is OUT OF FRAME (ABS = NON-
#    financial business economy). We drop sector 5 from this rerun rather
#    than mix denominators. (The turnover size-proxy also excludes finance,
#    so this is internally consistent.)
#  - Agriculture (A) has partial ABS coverage (small farms below the survey
#    threshold excluded): aGVA only £3.1bn vs ~£17.8bn total GVA. Its ratio
#    will therefore be overstated — flagged.
AGVA_BN = {  # ONS ABS 2023 aGVA, £bn, by sector_comb2 group
    1: 52.291 + 156.089,  # L real estate + N admin/support
    2: 143.519,           # F construction
    3: 38.260,            # P education (PRIVATE only)
    4: 35.015 + 21.971,   # R arts + S other services
    # 5: finance/insurance (K) — out of ABS frame, dropped
    6: 67.002,            # I accommodation/food
    7: 53.339,            # Q health/social (PRIVATE only)
    8: 178.371,           # J information/communication
    9: 250.315,           # M professional/sci/tech
    10: 250.406,          # G wholesale/retail
    11: 111.211,          # H transport/storage
    12: 301.963,          # B-E production
    13: 3.066,            # A agriculture (PARTIAL ABS coverage — understated)
}
AGVA_LABELS = {k: v for k, v in SECTOR_LABELS.items() if k in AGVA_BN}
# damage shares must be recomputed on the SAME universe we have aGVA for
# (i.e. excluding finance), so shares of both sides sum over the same sectors.
print("\n\n")
proportionality_report('sector_comb2', AGVA_LABELS, AGVA_BN,
                       "TEST 1b — SECTOR: damage share vs REAL private-sector aGVA (ONS ABS 2023)",
                       "aGVA", flag={13})
print("  (*flag on Agriculture = partial ABS coverage, aGVA understated ⇒ ratio overstated.)")
print("  (Finance/insurance K excluded — outside the ABS non-financial frame.)")
print("  NB: damage shares here are over the finance-excluded universe, so they")
print("      differ slightly from TEST 1; the comparison is like-for-like within it.")

print("\nDone.")
