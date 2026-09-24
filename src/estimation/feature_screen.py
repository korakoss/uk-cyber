"""Agnostic screen: do any pre-incident firm features predict the latent-related targets?

Targets (each on its own population):
  T1 exposure  : log Cybercrime_phishsum        phishing firms with a count
  T2 exposure  : freq band (ordinal, as numeric) attacked firms with valid freq
  T3 vulnerab. : any cost (band>=2) | log N      phishing firms with count + band
  T4 severity  : cost band | band>=2             attacked firms with a costly worst incident

Method: HistGradientBoosting, repeated 5-fold CV, out-of-sample gain of
(baseline + all features) over baseline (size, sector [, log N for T3]).
Permutation importance only reported if the gain is positive.
Univariate Spearman screen with Benjamini-Hochberg FDR as a second lens.

Leakage guard: a predictor is dropped per population if its coverage < 80%
or if its missingness is associated with the target (p < 0.05) — routed
questions whose asked/not-asked status depends on incidents would otherwise
predict the target by construction.

Run: PYTHONPATH=/home/user/md-clean/src python3 src/estimation/feature_screen.py
"""

import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from statsmodels.stats.multitest import multipletests

from data import load_raw, GENUINE_TYPE_COLS

warnings.filterwarnings("ignore")

PREDICTOR_PATTERNS = [
    r"priority(_comb)?", r"update(_comb\d)?", r"insurex(_comb)?", r"info\d+", r"info_comb\d",
    r"scheme\d+", r"scheme_any", r"govtact\d+", r"govtact_comb\d", r"manage\d+", r"manage_comb",
    r"comply\d+", r"software", r"ident\d+", r"ident_comb", r"audit", r"rules\d+",
    r"rules_comb\d", r"policy\d+", r"policy_comb", r"ransom", r"review(_comb\d)?", r"trained",
    r"strategy", r"stratint", r"corporate", r"corprisk", r"supplyrisk\d", r"supplyrisk_any",
    r"supplycert\d", r"supplycert_any", r"incidcontent\d+", r"incidaction[a-h]",
    r"AllEssentials", r"Step\d+", r"Any10Steps", r"Sum10Steps", r"title",
]
BASE = ["sizeb", "sector_comb2"]


def banded(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s < 100))


def predictor_cols(raw):
    pat = re.compile("^(" + "|".join(PREDICTOR_PATTERNS) + ")$")
    return [c for c in raw.columns if pat.match(c)]


def build_targets(raw):
    att = raw["type_comb1"] == 1
    t6 = banded(raw["type6"]) == 1
    N = pd.to_numeric(raw["Cybercrime_phishsum"], errors="coerce")
    band = banded(raw["damage_bands"])
    freq = banded(raw["freq"])
    logN = np.log(N.where(N >= 1))
    out = {}
    m = att & t6 & (N >= 1)
    out["T1 exposure: log phishing count"] = (m, logN, "reg", [])
    m = att & freq.between(1, 6)
    out["T2 exposure: freq band"] = (m, freq, "reg", [])
    m = att & t6 & (N >= 1) & band.between(1, 10)
    out["T3 vulnerability: any cost | log N"] = (m, (band >= 2).astype(float), "clf", ["logN"])
    m = att & band.between(2, 10)
    out["T4 severity: band | band>=2"] = (m, band, "reg", [])
    return out, logN


def leakage_filter(X, y, cols, kind):
    keep, dropped_cov, dropped_leak = [], 0, 0
    for c in cols:
        miss = X[c].isna()
        if 1 - miss.mean() < 0.8:
            dropped_cov += 1
            continue
        if miss.any() and (~miss).any():
            p = mannwhitneyu(y[miss], y[~miss]).pvalue
            if p < 0.05:
                dropped_leak += 1
                continue
        if X[c].nunique() < 2:
            continue
        keep.append(c)
    return keep, dropped_cov, dropped_leak


def cv_score(X, y, kind, cat_mask, seed=0):
    if kind == "reg":
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed)
        model = lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                      max_iter=200, min_samples_leaf=20,
                                                      categorical_features=cat_mask,
                                                      random_state=seed)
        split = cv.split(X)
    else:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
        model = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                       max_iter=200, min_samples_leaf=20,
                                                       categorical_features=cat_mask,
                                                       random_state=seed)
        split = cv.split(X, y)
    scores = []
    for tr, te in split:
        mdl = model().fit(X.iloc[tr], y.iloc[tr])
        if kind == "reg":
            scores.append(r2_score(y.iloc[te], mdl.predict(X.iloc[te])))
        else:
            scores.append(roc_auc_score(y.iloc[te], mdl.predict_proba(X.iloc[te])[:, 1]))
    return np.mean(scores), np.std(scores) / np.sqrt(len(scores))


def importance(X, y, kind, cat_mask, top=10):
    mdl = (HistGradientBoostingRegressor if kind == "reg" else HistGradientBoostingClassifier)(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
        categorical_features=cat_mask, random_state=0)
    n = len(X)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n):]
    mdl.fit(X.iloc[tr], y.iloc[tr])
    pi = permutation_importance(mdl, X.iloc[te], y.iloc[te], n_repeats=20, random_state=0,
                                scoring="r2" if kind == "reg" else "roc_auc")
    order = np.argsort(-pi.importances_mean)[:top]
    return [(X.columns[i], pi.importances_mean[i], pi.importances_std[i]) for i in order]


def main():
    raw = load_raw()
    pcols = predictor_cols(raw)
    feats = pd.DataFrame({c: banded(raw[c]) for c in pcols + BASE})
    targets, logN = build_targets(raw)
    feats["logN"] = logN
    print(f"Candidate predictor columns: {len(pcols)} (+ size, sector)")

    for name, (mask, y_all, kind, extra) in targets.items():
        y = y_all[mask].reset_index(drop=True)
        X = feats[mask].reset_index(drop=True)
        ok = y.notna() & X["sizeb"].notna() & X["sector_comb2"].notna()
        y, X = y[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
        keep, dcov, dleak = leakage_filter(X, y, pcols, kind)
        base_cols = BASE + extra
        full_cols = base_cols + keep
        print("\n" + "=" * 78)
        print(f"{name}   n={len(y)}   predictors kept={len(keep)} "
              f"(dropped: {dcov} low coverage, {dleak} missingness-leaky)")
        print("=" * 78)

        cat_base = [c == "sector_comb2" for c in base_cols]
        cat_full = [c == "sector_comb2" for c in full_cols]
        b, bse = cv_score(X[base_cols], y, kind, cat_base)
        f, fse = cv_score(X[full_cols], y, kind, cat_full)
        metric = "R2" if kind == "reg" else "AUC"
        print(f"  CV {metric}: baseline {b:+.3f} (±{bse:.3f})  |  full {f:+.3f} (±{fse:.3f})  "
              f"|  gain {f - b:+.3f}")

        if f - b > 2 * max(bse, fse):
            print("  Permutation importance (full model, 30% holdout):")
            for c, m, s in importance(X[full_cols], y, kind, cat_full):
                print(f"    {c:>18s}: {m:+.4f} ± {s:.4f}")
        else:
            print("  No gain beyond noise -> importance not reported.")

        rows = []
        for c in keep:
            d = pd.concat([X[c], y], axis=1).dropna()
            if d.iloc[:, 0].nunique() < 2:
                continue
            r, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
            rows.append((c, r, p, len(d)))
        if rows:
            ps = np.array([r[2] for r in rows])
            rej, q, _, _ = multipletests(ps, alpha=0.10, method="fdr_bh")
            hits = sorted([(r, qq) for r, qq, k in zip(rows, q, rej) if k], key=lambda t: t[1])
            print(f"  Univariate Spearman, BH q<0.10: {len(hits)} of {len(rows)}")
            for (c, r, p, n), qq in hits[:12]:
                print(f"    {c:>18s}: rho={r:+.3f}, q={qq:.4f}, n={n}")


if __name__ == "__main__":
    main()
