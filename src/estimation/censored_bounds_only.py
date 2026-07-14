"""
"Just look at the censored data" — no lognormal fit, no Monte Carlo simulation.

Everything so far has taken the interval-censored records (exact / lower-
censored / upper-censored / zero per firm per type — same construction as
type_cost_censored_model.py) and then run them through EITHER a fitted
lognormal (which degenerated for Ransomware, was fragile for Impersonation)
OR a Monte Carlo simulation anchored on a clean reference subsample (which
worked, but leans on an assumption: that firms needing extrapolation draw
from the same per-attack distribution as the clean reference firms).

This script asks: what can we say using ONLY the raw censored bounds
themselves, with NO distributional assumption of any kind layered on top?

Three numbers per type, all assumption-minimal:
1. FLOOR: the tightest lower bound the data logically supports. Use the
   BAND'S OWN LOWER EDGE for every exact and lower-censored record (not the
   midpoint — the true floor), 0 for upper-censored and zero-flag records.
   This is a hard floor: the true value cannot be below this, full stop,
   given the survey bands are read correctly.
2. NAIVE (bridge=1, i.e. "no bridge"): each firm's own single-worst-incident
   band midpoint, or its exact direct-cost band midpoint where available.
   This is the project's existing conservative default — already computed
   elsewhere for these two types (£273.56 Ransomware, £305.49 Impersonation)
   but rebuilt here directly from the censored records for a self-contained
   comparison.
3. UPPER BOUND: does NOT exist in general, from the censored data alone. Any
   lower-censored record (cost >= L) has no logical ceiling without extra
   information — that's exactly why the Monte Carlo approach needed
   ranssoft_bands (Ransomware) / the clean impersonation-only subsample
   (Impersonation) to say anything past the floor. Demonstrated directly
   below by showing the lower-censored group's weight and what fraction of
   the type's total it represents.
"""

import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

df = get_business_data()
INVALID_FREQ = {997, 999, 997.0, 999.0}

FULL_BAND_BOUNDS = {
    1: (0, 0), 2: (1, 100), 3: (100, 500), 4: (500, 1_000), 5: (1_000, 5_000),
    6: (5_000, 10_000), 7: (10_000, 20_000), 8: (20_000, 50_000),
    9: (50_000, 100_000), 10: (100_000, 500_000), 11: (500_000, 1_000_000),
    12: (1_000_000, 5_000_000), 13: (5_000_000, None),
}
FULL_MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000,
            9: 75000, 10: 300000, 11: 750000, 12: 3000000, 13: 5000000}
TYPE_COST_BAND_BOUNDS = {
    1: (0, 100), 2: (100, 250), 3: (250, 500), 4: (500, 1_000),
    5: (1_000, 2_000), 6: (2_000, 5_000), 7: (5_000, 10_000),
    8: (10_000, 20_000), 9: (20_000, 50_000), 10: (50_000, 100_000),
    11: (100_000, 250_000), 12: (250_000, None),
}
TYPE_COST_MID = {1: 50, 2: 175, 3: 375, 4: 750, 5: 1500, 6: 3500, 7: 7500,
                  8: 15000, 9: 35000, 10: 75000, 11: 175000, 12: 500000}

TYPES = {
    'Ransomware': dict(flag_cols=['type1'], cost_col='ranscost_bands', disrupta_codes={1}),
    'Impersonation': dict(flag_cols=['type5'], cost_col=None, disrupta_codes={5}),
}

pop_w = df['weight'].sum()


def build_records(sub, flag_cols, cost_col, disrupta_codes):
    exact, lower, upper = [], [], []
    n_zero_w = 0.0
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
            exact.append((L, U, w, TYPE_COST_MID[b]))
        elif not has_flag.loc[idx]:
            n_zero_w += w
        else:
            db = row['damage_bands']
            if pd.isna(db) or db >= SPECIAL_CODE_THRESHOLD or db < 1:
                continue
            L, U = FULL_BAND_BOUNDS[int(db)]
            mid = FULL_MID[int(db)]
            if is_worst.loc[idx]:
                lower.append((L, w, mid))
            else:
                if U is None:
                    continue
                upper.append((U, w, mid))
    return exact, lower, upper, n_zero_w


for label, spec in TYPES.items():
    print("=" * 100)
    print(f"{label} — RAW CENSORED BOUNDS ONLY (no distributional fit of any kind)")
    print("=" * 100)

    exact, lower, upper, n_zero_w = build_records(df, spec['flag_cols'], spec['cost_col'], spec['disrupta_codes'])
    w_exact = sum(w for _, _, w, _ in exact)
    w_lower = sum(w for _, w, _ in lower)
    w_upper = sum(w for _, w, _ in upper)
    w_occurred = w_exact + w_lower + w_upper
    print(f"n exact={len(exact)} (w={w_exact:.2f}), n lower-censored={len(lower)} (w={w_lower:.2f}), "
          f"n upper-censored={len(upper)} (w={w_upper:.2f}), zero weight={n_zero_w:.2f}")

    # 1. FLOOR: band's own lower edge for exact + lower-censored, 0 for upper-censored/zero
    floor_contribution = sum(L * w for L, U, w, mid in exact) + sum(L * w for L, w, mid in lower)
    floor_e_cost = floor_contribution / pop_w

    # 2. NAIVE (bridge=1): band midpoint for exact and lower-censored (own worst incident),
    #    band midpoint for upper-censored too (their own worst incident, of some OTHER type,
    #    contributes 0 to THIS type's naive total since this type wasn't their worst — but for
    #    the "own-incident" convention used elsewhere, upper-censored firms' contribution to
    #    THIS type is bounded by knowing they're <= that other incident; naive treats unknown
    #    type-specific cost as effectively unresolved, so it's excluded from naive on purpose —
    #    naive here mirrors exactly what "bridge=1 applied to disrupta==type firms" means)
    naive_contribution = sum(mid * w for L, U, w, mid in exact) + sum(mid * w for L, w, mid in lower)
    naive_e_cost = naive_contribution / pop_w

    print(f"\n1. FLOOR (hard lower bound, zero distributional assumption): £{floor_e_cost:,.2f}/business")
    print(f"2. NAIVE (bridge=1, each firm's own worst-incident band midpoint): £{naive_e_cost:,.2f}/business")
    print(f"   Gap between floor and naive: £{naive_e_cost - floor_e_cost:,.2f} — this gap is just band-width "
          f"rounding (midpoint vs lower edge of the same known band), NOT model uncertainty.")

    print(f"\n3. UPPER BOUND: does not exist from censored data alone.")
    print(f"   The {len(lower)} lower-censored records (weight={w_lower:.2f}, "
          f"{100*w_lower/w_occurred:.1f}% of this type's 'occurred' weight) only tell us cost >= L — "
          f"nothing in the censored data itself caps how much higher the true total could be.")
    print(f"   This is exactly why a model (lognormal fit or Monte Carlo from a clean reference class) is "
          f"NECESSARY to say anything beyond the floor — the censored bounds alone are silent on the upper side.")

print("\nDone.")
