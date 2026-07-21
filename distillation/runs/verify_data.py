"""Verification: the distilled loader reproduces the original proc.py on every
shared column, and the centralized cleaning matches national_simulation.py's
inline logic. Run once to trust src/data.py as the base for everything else.
"""

import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))  # distilled src/
sys.path.insert(0, _REPO)                               # original proc.py

import data as ddata          # distilled
from proc import get_business_data, SPECIAL_CODE_THRESHOLD  # original

# --- original loader, cleaned exactly as national_simulation.py does it -----
orig = get_business_data()
orig = orig[orig["sizeb"].notna()].copy()
orig["sizeb"] = orig["sizeb"].astype(int)
orig["attacked"] = orig["freq"].isin({1, 2, 3, 4, 5, 6})
db = orig["damage_bands"]
orig["band"] = np.where(orig["attacked"] & db.notna()
                        & (db < SPECIAL_CODE_THRESHOLD) & (db >= 1), db, np.nan)

# --- distilled loader -------------------------------------------------------
new = ddata.load_business_data()

print(f"row count   original={len(orig)}  distilled={len(new)}  "
      f"-> {'MATCH' if len(orig) == len(new) else 'MISMATCH'}")

checks = []
for col in ["sizeb", "weight", "freq", "damage_bands", "disrupta"]:
    a = orig[col].reset_index(drop=True)
    b = new[col].reset_index(drop=True)
    same = a.equals(b) or np.allclose(a.astype(float), b.astype(float), equal_nan=True)
    checks.append(same)
    print(f"  col {col:<14} -> {'MATCH' if same else 'MISMATCH'}")

# derived columns vs the original inline construction
att_match = orig["attacked"].reset_index(drop=True).equals(new["attacked"].reset_index(drop=True))
band_match = np.allclose(orig["band"].reset_index(drop=True).astype(float),
                         new["band"].reset_index(drop=True).astype(float), equal_nan=True)
checks += [att_match, band_match]
print(f"  derived attacked -> {'MATCH' if att_match else 'MISMATCH'}")
print(f"  derived band     -> {'MATCH' if band_match else 'MISMATCH'}")

# constants sanity: bounds, N(size) present for all four bands
const_ok = (set(ddata.BAND_BOUNDS) == set(range(1, 11))
            and set(ddata.N_BY_SIZE) == {1, 2, 3, 4}
            and ddata.BAND_BOUNDS[10] == (100000, 500000))
checks.append(const_ok)
print(f"  constants (bands 1-10, N by size, band-10 bounded) -> "
      f"{'OK' if const_ok else 'FAIL'}")

print("\nRESULT:", "ALL CHECKS PASS" if (len(orig) == len(new) and all(checks))
      else "*** DISCREPANCY — do not build on this loader yet ***")
