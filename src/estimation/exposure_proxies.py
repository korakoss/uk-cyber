"""Do observable firm characteristics predict attack frequency within a size group?

Motivation: attack counts are far more spread out than a shared-rate Poisson allows
(once-only and daily firms coexist in the same size group and channel). An
attacker-side reading attributes this to firm-level exposure h_i. If exposure is
real, freq should correlate with observable exposure proxies (sector, turnover,
preparedness) *within* a size group.

Run: PYTHONPATH=/home/user/md-clean/src python3 src/estimation/exposure_proxies.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

from data import load_raw, SIZE_LABELS, GENUINE_TYPE_COLS

CANDIDATES = [
    "sector_comb2", "income2", "sizeb_comb1", "sizeb_comb2",
    "priority", "insurex", "trained", "audit", "strategy",
    "rules_comb1", "rules_comb2", "policy_comb", "AllEssentials", "Sum10Steps",
    "Cybercrime_phishsum", "Cybercrime_allsum",
]
SERIOUS = [f"type{i}" for i in (1, 2, 3, 4, 7, 8, 13, 15, 16)]


def clean(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s < 100))


def main():
    raw = load_raw()
    df = pd.DataFrame({"sizeb": clean(raw["sizeb"]), "freq": clean(raw["freq"])})
    for c in CANDIDATES:
        df[c] = pd.to_numeric(raw[c], errors="coerce") if c.startswith("Cybercrime") \
            else clean(raw[c])
    types = {c: clean(raw[c]) == 1 for c in GENUINE_TYPE_COLS}
    ser = pd.concat([types[c] for c in SERIOUS], axis=1).any(axis=1)
    df["phish_only"] = types["type6"] & ~types["type5"] & ~ser
    att = df[(raw["type_comb1"] == 1) & df["freq"].between(1, 6)]

    print("=" * 74)
    print("1. CANDIDATE COLUMNS: coverage among attacked firms + value range")
    print("=" * 74)
    for c in CANDIDATES:
        s = att[c]
        vc = s.value_counts().sort_index()
        head = ", ".join(f"{k:g}:{v}" for k, v in list(vc.items())[:10])
        more = " ..." if len(vc) > 10 else ""
        print(f"  {c:>20s}: cov={s.notna().mean():5.1%}, levels={len(vc):3d} | {head}{more}")

    print("\n" + "=" * 74)
    print("2. SPEARMAN(proxy, freq) within size group")
    print("   (a) phishing-only firms   (b) all attacked firms")
    print("=" * 74)
    for label, pool in [("phishing-only", att[att["phish_only"]]), ("all attacked", att)]:
        print(f"\n  --- {label} ---")
        print(f"  {'proxy':>20s} " + " ".join(f"{SIZE_LABELS[s]:>17s}" for s in (1, 2, 3, 4)))
        for c in CANDIDATES:
            if c == "sector_comb2":
                continue
            cells = []
            for sz in (1, 2, 3, 4):
                g = pool[(pool["sizeb"] == sz) & pool[c].notna()]
                if len(g) < 15 or g[c].nunique() < 2:
                    cells.append(f"{'n=' + str(len(g)):>17s}")
                    continue
                r, p = spearmanr(g[c], g["freq"])
                star = "*" if p < 0.05 else " "
                cells.append(f"{r:+.2f} p={p:.3f}{star}(n={len(g)})"[:17].rjust(17))
            print(f"  {c:>20s} " + " ".join(cells))

    print("\n" + "=" * 74)
    print("3. SECTOR (categorical): freq profile by sector, phishing-only, Micro+Small")
    print("   Kruskal-Wallis test of freq across sectors")
    print("=" * 74)
    ph = att[att["phish_only"] & att["sizeb"].isin([1, 2]) & att["sector_comb2"].notna()]
    rows = []
    for sec, g in ph.groupby("sector_comb2"):
        if len(g) < 8:
            continue
        rows.append((sec, len(g), (g["freq"] >= 4).mean(), (g["freq"] == 1).mean(),
                     g["freq"].mean()))
    for sec, n, wk, once, mf in sorted(rows, key=lambda r: -r[2]):
        print(f"  sector={sec:>4g}: n={n:3d}, P(weekly+)={wk:.2f}, P(once)={once:.2f}, "
              f"mean freq={mf:.2f}")
    groups = [g["freq"].values for _, g in ph.groupby("sector_comb2") if len(g) >= 8]
    if len(groups) >= 2:
        h, p = kruskal(*groups)
        print(f"  Kruskal-Wallis H={h:.2f}, p={p:.4f} ({len(groups)} sectors with n>=8)")

    print("\n" + "=" * 74)
    print("4. Cybercrime_*sum: are these literal counts? cross-tab vs freq (attacked)")
    print("=" * 74)
    for c in ["Cybercrime_phishsum", "Cybercrime_allsum"]:
        g = att[att[c].notna()]
        print(f"\n  {c}: n={len(g)}, describe: {g[c].describe().round(2).to_dict()}")
        for f in range(1, 7):
            s = g.loc[g["freq"] == f, c]
            if len(s):
                print(f"    freq={f}: n={len(s):3d}, median={s.median():g}, mean={s.mean():.2f}")


if __name__ == "__main__":
    main()
