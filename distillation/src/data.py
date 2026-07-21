"""Data-access foundation for the distillation.

Single authoritative source for (a) loading the CSBS business survey and
(b) the codebook constants every downstream unit needs. Nothing here fits a
model, samples, or writes results — it is pure load + constants + the one
piece of data cleaning that must be done identically everywhere.

The raw survey lives in the (read-only) working tree at data/proc/data.csv.
This module reads it in place; the distillation never copies the data.

Provenance: distilled from the original proc.py (get_business_data) and the
constants that were re-declared ad hoc across src/estimation/*.py (notably
national_simulation.py's BAND_BOUNDS / N_BY_SIZE). Consolidated here so there
is exactly one copy of each.
"""

import os
import pandas as pd

# --------------------------------------------------------------------------
# Locating the raw data. The working tree is the read-only source of truth;
# resolve its path relative to this file so runs work from any cwd.
# --------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_CSV = os.path.join(_REPO_ROOT, "data", "proc", "data.csv")

# --------------------------------------------------------------------------
# Codebook constants
# --------------------------------------------------------------------------

# damage_bands special codes (997=Don't know, 999=Refused, and an SPSS
# non-integer variant 999.62004) all sit at >= 100. Real cost band codes are
# the integers 1..10. Filter with `< SPECIAL_CODE_THRESHOLD`, NOT isin([997,999]).
SPECIAL_CODE_THRESHOLD = 100

# Business size bands (sizeb). Employer frame: Micro is 1-9 employees, so
# zero-employee sole traders are OUT of frame (a deliberate scope choice).
SIZE_LABELS = {1: "Micro", 2: "Small", 3: "Medium", 4: "Large"}

# damage_bands cost band code -> (lower, upper) bound in whole pounds.
# Band 1 is "no cost". Band 10 is BOUNDED at £100k-£500k per the CSBS codebook:
# the scale explicitly continues (11=£500k-1M, 12=£1M-5M, 13=£5M+) and NO firm
# ever selected a band above 10, so there is no survey evidence of any >£500k
# most-disruptive incident. Treating band 10 as open-ended manufactures tail
# cost the data does not contain.
BAND_BOUNDS = {
    1: (0, 0),
    2: (1, 100),
    3: (100, 500),
    4: (500, 1000),
    5: (1000, 5000),
    6: (5000, 10000),
    7: (10000, 20000),
    8: (20000, 50000),
    9: (50000, 100000),
    10: (100000, 500000),
}

# freq (Q54) encodes PERIODICITY bands, not literal attack counts. freq==1
# ("once only") is the clean anchor: one incident occurred, so its reported
# cost IS the total annual cost and needs no bridge.
#   1 = Once only
#   2 = More than once but less than monthly (~2-11/yr)
#   3 = Roughly monthly (~12/yr)
#   4 = Roughly weekly (~52/yr)
#   5 = Roughly daily (~365/yr)
#   6 = Several times a day (~1000+/yr)
# A firm is "attacked" iff freq is one of these six valid bands.
VALID_FREQ = {1, 2, 3, 4, 5, 6}

# ONS BPE 2025 employer-frame business counts by size band (see the original
# src/estimation/ons_business_counts.py). Excludes the 4.27M zero-employee
# businesses, which are outside the survey frame.
N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}

# Empirical prevalence P(attacked | size) — used directly (the logistic-in-log-
# size fit was validated but adds nothing within the four known strata).
PREVALENCE = {1: 0.40, 2: 0.51, 3: 0.66, 4: 0.69}

# disrupta (Q64A): type of the single MOST DISRUPTIVE attack, for every
# attacked firm. This is the axis the type-based bridge is indexed on.
DISRUPTA_LABELS = {
    1: "Ransomware",
    2: "Other malware",
    3: "Denial of service",
    4: "Hacking",
    5: "Impersonation",
    6: "Phishing",
    7: "Hacking (unauth. access)",
    9: "Hacking (other)",
    11: "Website/social/email takeover",
}

# Columns the distilled analyses actually consume. Loading a curated set (vs.
# proc.py's full column dump) keeps the frame legible; extend here if a new
# unit needs more.
_CORE_COLS = ["sizeb", "weight", "freq", "damage_bands", "disrupta", "disrupt",
              "sector_comb2"]


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------
def load_business_data(path: str = DATA_CSV, clean: bool = True) -> pd.DataFrame:
    """Load the CSBS business-survey frame (typex==1).

    Returns a DataFrame with the curated analytical columns. When clean=True
    (default) it also adds two derived, centrally-defined columns every
    downstream unit would otherwise recompute:

      attacked : bool  — freq is a valid attack-frequency band
      band     : float — damage_bands as a clean 1..10 code for attacked firms,
                         NaN otherwise (special codes 997/999/999.62 stripped).

    Set clean=False to get the raw curated columns with no derivations.
    """
    all_data = pd.read_csv(path)
    biz = all_data[all_data["typex"] == 1].copy()
    cols = [c for c in _CORE_COLS if c in all_data.columns]
    biz = biz[cols]
    if not clean:
        return biz.reset_index(drop=True)

    biz = biz[biz["sizeb"].notna()].copy()
    biz["sizeb"] = biz["sizeb"].astype(int)
    biz["attacked"] = biz["freq"].isin(VALID_FREQ)

    db = biz["damage_bands"]
    is_real = biz["attacked"] & db.notna() & (db < SPECIAL_CODE_THRESHOLD) & (db >= 1)
    biz["band"] = db.where(is_real)
    return biz.reset_index(drop=True)


if __name__ == "__main__":
    df = load_business_data()
    print(f"loaded {len(df)} business firms from {DATA_CSV}")
    print("\nby size band:")
    for sz, lab in SIZE_LABELS.items():
        sub = df[df["sizeb"] == sz]
        att = sub["attacked"].sum()
        print(f"  {lab:<7} n={len(sub):>5}  attacked={att:>5} "
              f"({att / len(sub):.0%})  with valid cost band={sub['band'].notna().sum():>4}")
