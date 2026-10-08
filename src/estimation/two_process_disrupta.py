"""Two-process parameter inference using disrupta attribution.

Instead of latent-variable MLE, we use the survey's own `disrupta` variable
(which attack type was the MOST DISRUPTIVE) to attribute each firm's max cost
to either the nuisance or serious process. This gives direct, model-free
reads on each process's cost distribution.

Three layers:
  1. Success gate: P(band >= 2 | process)  — did the attack cause any cost?
  2. Cost distribution: P(band | success, process) — cost conditional on success
  3. Rates: how many attacks per year from each process?

We use:
  - Nuisance types: phishing (disrupta=6) + impersonation (disrupta=5)
  - Serious types:  everything else (disrupta in {1,2,3,4,7,8,9,10,11,12})

For firms with freq=1, the observed cost IS the single draw from the
process identified by disrupta.  For freq>1, the max cost is still
attributed to the disrupta process — it's the process that generated
the most disruptive incident.

Run: ``python src/estimation/two_process_disrupta.py``
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BAND_BOUNDS, SIZE_LABELS, load, FREQ_LABELS

NUISANCE_TYPES = {5, 6}  # impersonation, phishing
COST_BANDS = list(range(1, 11))

N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}


def band_to_gbp(band, tail_multiple=2.0):
    lo, hi = BAND_BOUNDS.get(band, (0, 0))
    if band == 1: return 0.0
    if hi == np.inf: return lo * tail_multiple
    return (lo + hi) / 2


def main():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "proc", "data.csv")
    df = load(path)

    att = df[df["attacked"]].copy()
    v = att[att["band"].notna() & att["freq"].notna() & att["disrupta"].notna()].copy()
    v["band"] = v["band"].astype(int)
    v["freq"] = v["freq"].astype(int)
    v["disrupta"] = v["disrupta"].astype(int)
    v["is_nuisance"] = v["disrupta"].isin(NUISANCE_TYPES)
    v["size_group"] = np.where(v["sizeb"] == 1, "Micro", "Rest")

    print(f"Total firms with valid band + freq + disrupta: {len(v)}")
    print(f"  Nuisance disrupta (phishing/impersonation): {v['is_nuisance'].sum()}")
    print(f"  Serious disrupta (everything else): {(~v['is_nuisance']).sum()}")

    # Compute prevalence for national scaling
    prev = {}
    for sz in SIZE_LABELS:
        sub = df[df["sizeb"] == sz]
        prev[sz] = (sub["attacked"].astype(int) * sub["weight"]).sum() / sub["weight"].sum()

    band_costs = np.array([band_to_gbp(b) for b in COST_BANDS])

    for grp in ["Micro", "Rest"]:
        g = v[v["size_group"] == grp]
        print(f"\n{'='*74}")
        print(f"  {grp} (n={len(g)})")
        print(f"{'='*74}")

        for process_name, mask in [("Nuisance", g["is_nuisance"]),
                                    ("Serious", ~g["is_nuisance"])]:
            sub = g[mask]
            print(f"\n  --- {process_name} (disrupta-attributed, n={len(sub)}) ---")

            # Layer 1: success gate (band >= 2)
            n_total = len(sub)
            n_success = (sub["band"] >= 2).sum()
            p_success = n_success / n_total if n_total > 0 else 0
            print(f"    P(success) = P(band>=2) = {n_success}/{n_total} = {p_success:.3f}")

            # Layer 2: cost distribution conditional on success
            successful = sub[sub["band"] >= 2]
            print(f"    Cost distribution | success (n={len(successful)}):")
            band_dist = successful["band"].value_counts().sort_index()
            for b in COST_BANDS:
                if b == 1:
                    continue
                cnt = band_dist.get(b, 0)
                pct = cnt / len(successful) * 100 if len(successful) > 0 else 0
                lo, hi = BAND_BOUNDS[b]
                hi_str = f"£{hi:,}" if hi != np.inf else "£500k+"
                print(f"      band {b:2d} (£{lo:,}-{hi_str}): {cnt:3d} ({pct:5.1f}%)")

            # E[T|success] using band midpoints
            if len(successful) > 0:
                pmf_success = np.zeros(len(COST_BANDS))
                for b in COST_BANDS:
                    if b == 1:
                        continue
                    pmf_success[b-1] = band_dist.get(b, 0) / len(successful)
                et_success = (pmf_success * band_costs).sum()
                print(f"    E[T|success] = £{et_success:,.0f}")
            else:
                et_success = 0

            # E[T] (unconditional, including band=1 as £0)
            if n_total > 0:
                pmf_all = np.zeros(len(COST_BANDS))
                all_dist = sub["band"].value_counts().sort_index()
                for b in COST_BANDS:
                    pmf_all[b-1] = all_dist.get(b, 0) / n_total
                et_all = (pmf_all * band_costs).sum()
                print(f"    E[T] (unconditional) = £{et_all:,.0f}")

            # By freq — show how cost varies with freq within each process
            print(f"\n    By freq band:")
            for f in sorted(sub["freq"].unique()):
                sf = sub[sub["freq"] == f]
                n_f = len(sf)
                p_s = (sf["band"] >= 2).sum() / n_f if n_f > 0 else 0
                mean_band = sf["band"].mean()
                sf_success = sf[sf["band"] >= 2]
                if len(sf_success) > 0:
                    pmf_f = np.zeros(len(COST_BANDS))
                    fd = sf_success["band"].value_counts().sort_index()
                    for b in COST_BANDS:
                        if b == 1: continue
                        pmf_f[b-1] = fd.get(b, 0) / len(sf_success)
                    et_f = (pmf_f * band_costs).sum()
                    et_str = f"£{et_f:,.0f}"
                else:
                    et_str = "n/a"
                print(f"      freq={f} ({FREQ_LABELS.get(f,'?'):>25s}): n={n_f:3d}, "
                      f"P(success)={p_s:.2f}, mean_band={mean_band:.2f}, "
                      f"E[T|success]={et_str}")

        # Now: the two-process national cost estimate
        # For each process, E[cost per attacked firm from this process]
        # = λ_process · E[T_process]
        # We don't observe λ directly, but we can estimate the SHARE
        # of disrupta attributed to each process, which reflects relative rates
        print(f"\n  --- Process shares and national cost ---")
        n_nuis = g["is_nuisance"].sum()
        n_ser = (~g["is_nuisance"]).sum()
        share_nuis = n_nuis / len(g)
        share_ser = n_ser / len(g)
        print(f"    Share nuisance (disrupta): {share_nuis:.3f} ({n_nuis}/{len(g)})")
        print(f"    Share serious (disrupta):  {share_ser:.3f} ({n_ser}/{len(g)})")

        # Approach: E[cost|attacked] = E[cost|disrupta=nuisance]*P(disrupta=nuisance)
        #                             + E[cost|disrupta=serious]*P(disrupta=serious)
        # where E[cost|disrupta=X] uses the unconditional (including band=1) distribution
        nuis_sub = g[g["is_nuisance"]]
        ser_sub = g[~g["is_nuisance"]]

        pmf_n = np.zeros(len(COST_BANDS))
        nd = nuis_sub["band"].value_counts().sort_index()
        for b in COST_BANDS:
            pmf_n[b-1] = nd.get(b, 0) / len(nuis_sub) if len(nuis_sub) > 0 else 0
        et_n = (pmf_n * band_costs).sum()

        pmf_s = np.zeros(len(COST_BANDS))
        sd = ser_sub["band"].value_counts().sort_index()
        for b in COST_BANDS:
            pmf_s[b-1] = sd.get(b, 0) / len(ser_sub) if len(ser_sub) > 0 else 0
        et_s = (pmf_s * band_costs).sum()

        # This gives E[max cost | attacked], not total annual cost
        # The bridge from max to total is separate
        e_max = share_nuis * et_n + share_ser * et_s
        print(f"    E[max cost|attacked, nuisance disrupta] = £{et_n:,.0f}")
        print(f"    E[max cost|attacked, serious disrupta]  = £{et_s:,.0f}")
        print(f"    E[max cost|attacked] = {share_nuis:.3f}×£{et_n:,.0f} + "
              f"{share_ser:.3f}×£{et_s:,.0f} = £{e_max:,.0f}")

    # Freq=1 subsample: clean single draws
    print(f"\n{'='*74}")
    print("FREQ=1 SUBSAMPLE: direct single-draw cost distributions")
    print(f"{'='*74}")

    f1 = v[v["freq"] == 1]
    for grp in ["Micro", "Rest"]:
        g = f1[f1["size_group"] == grp]
        print(f"\n  {grp} (freq=1, n={len(g)}):")

        for process_name, mask in [("Nuisance", g["is_nuisance"]),
                                    ("Serious", ~g["is_nuisance"])]:
            sub = g[mask]
            n = len(sub)
            if n == 0:
                print(f"    {process_name}: n=0")
                continue

            n_success = (sub["band"] >= 2).sum()
            p_success = n_success / n
            successful = sub[sub["band"] >= 2]

            band_dist = sub["band"].value_counts().sort_index()
            pmf = np.zeros(len(COST_BANDS))
            for b in COST_BANDS:
                pmf[b-1] = band_dist.get(b, 0) / n
            et = (pmf * band_costs).sum()

            print(f"    {process_name}: n={n}, P(success)={p_success:.3f}, E[T]=£{et:,.0f}")
            print(f"      PMF: {[f'{p:.3f}' for p in pmf]}")
            if len(successful) > 0:
                pmf_s = np.zeros(len(COST_BANDS))
                sd = successful["band"].value_counts().sort_index()
                for b in COST_BANDS:
                    if b == 1: continue
                    pmf_s[b-1] = sd.get(b, 0) / len(successful)
                et_s = (pmf_s * band_costs).sum()
                print(f"      E[T|success]=£{et_s:,.0f}, "
                      f"PMF|success: {[f'{p:.3f}' for p in pmf_s[1:]]}")

    # National total using the simple disrupta-weighted approach
    print(f"\n{'='*74}")
    print("NATIONAL TOTAL (disrupta-weighted max cost, no bridge)")
    print(f"{'='*74}")

    total = 0
    for grp in ["Micro", "Rest"]:
        g = v[v["size_group"] == grp]
        pmf_all = np.zeros(len(COST_BANDS))
        bd = g["band"].value_counts().sort_index()
        for b in COST_BANDS:
            pmf_all[b-1] = bd.get(b, 0) / len(g)
        e_max = (pmf_all * band_costs).sum()

        for sz in SIZE_LABELS:
            sz_grp = "Micro" if sz == 1 else "Rest"
            if sz_grp == grp:
                nat = N_BY_SIZE[sz] * prev[sz] * e_max
                total += nat
                print(f"  {SIZE_LABELS[sz]:>7s}: N={N_BY_SIZE[sz]:>9,}, prev={prev[sz]:.3f}, "
                      f"E[max cost|att]=£{e_max:,.0f}, contrib=£{nat/1e9:.3f}bn")

    print(f"\n  NATIONAL TOTAL (max cost only, no bridge): £{total/1e9:.3f}bn")


if __name__ == "__main__":
    main()
