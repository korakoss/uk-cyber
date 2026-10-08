"""Fit the two-channel phishing model to real (count N, worst-incident band M) pairs.

Model (per firm, phishing only):
  K_T ~ Poisson(lam_T)                 targeted attacks, same rate for all firms
  K_M ~ NegBin(mean m, size r)          mass attacks, rate varies across firms (gamma exposure)
  N   = K_T + K_M                       observed as Cybercrime_phishsum (gross count)
  each attack: cost band 1 w.p. 1-p_c; else banded lognormal(mu_c, sigma_c) on bands 2..10
  M   = max band over the N attacks     observed as damage_bands

Baseline A: all N attacks iid from one success-gated lognormal (3 params) —
the one-channel model md-clean's iid_test rejected.

Population: phishing-only firms (only type6 flagged), N >= 1, valid band.
Sensitivity: disrupta == 6 firms.  Sizes pooled (sparsity).

Run: PYTHONPATH=/home/user/md-clean/src python3 src/estimation/phishing_count_mixture.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, poisson, nbinom

from data import load_raw, BAND_BOUNDS, GENUINE_TYPE_COLS

BANDS = np.arange(1, 11)
LO = np.array([BAND_BOUNDS[b][0] for b in range(2, 11)], float)
HI = np.array([BAND_BOUNDS[b][1] for b in range(2, 11)], float)
JMAX = 40
rng = np.random.default_rng(0)


def banded(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s < 100))


def build(raw, mode):
    t = {c: banded(raw[c]) == 1 for c in GENUINE_TYPE_COLS}
    others = pd.concat([t[c] for c in GENUINE_TYPE_COLS if c != "type6"], axis=1).any(axis=1)
    N = pd.to_numeric(raw["Cybercrime_phishsum"], errors="coerce")
    band = banded(raw["damage_bands"])
    base = (raw["type_comb1"] == 1) & (N >= 1) & band.between(1, 10)
    if mode == "phishing-only":
        sel = base & t["type6"] & ~others
    else:
        sel = base & (banded(raw["disrupta"]) == 6)
    return N[sel].astype(int).values, band[sel].astype(int).values


def cdf_table(p, mu, sigma):
    """G[b] = P(single-attack band <= b), b = 0..10."""
    upper = norm.cdf((np.log(HI) - mu) / sigma)
    lower = norm.cdf((np.log(np.maximum(LO, 1e-9)) - mu) / sigma)
    lower[0] = 0.0
    mass = np.maximum(upper - lower, 0) + 1e-15
    mass = mass / mass.sum()
    G = np.zeros(11)
    G[1] = 1 - p
    G[2:] = (1 - p) + p * np.cumsum(mass)
    G[-1] = 1.0
    return G


def sig(v):
    return 1 / (1 + np.exp(-np.clip(v, -30, 30)))


def unpack_C(x):
    return dict(lamT=np.exp(np.clip(x[0], -6, 3)), m=np.exp(np.clip(x[1], -3, 8)),
                r=np.exp(np.clip(x[2], -5, 5)), pT=sig(x[3]), pM=sig(x[4]),
                muT=np.clip(x[5], -2, 14), sT=np.exp(np.clip(x[6], np.log(0.3), np.log(5))),
                muM=np.clip(x[7], -2, 14), sM=np.exp(np.clip(x[8], np.log(0.3), np.log(5))))


def log_band_prob(logGhi, logGlo):
    """log(exp(a) - exp(b)) for a >= b, elementwise."""
    d = np.minimum(logGlo - logGhi, 0.0)
    return logGhi + np.log(-np.expm1(np.minimum(d, -1e-300)))


def firm_terms_C(q, N, M):
    """Per-firm log P(N) and log P(N, M), not yet conditioned on N >= 1."""
    with np.errstate(divide="ignore"):
        lGT = np.log(cdf_table(q["pT"], q["muT"], q["sT"]))
        lGM = np.log(cdf_table(q["pM"], q["muM"], q["sM"]))
    nb_p = q["r"] / (q["r"] + q["m"])
    lw, lwm = [], []
    for j in range(JMAX + 1):
        k = N - j
        ok = k >= 0
        kk = np.maximum(k, 0)
        w = poisson.logpmf(j, q["lamT"]) + np.where(ok, nbinom.logpmf(kk, q["r"], nb_p), -np.inf)
        a = j * lGT[M] + kk * lGM[M]
        b = np.where(M > 1, j * lGT[M - 1] + kk * lGM[M - 1], -np.inf)
        lw.append(w)
        lwm.append(w + log_band_prob(a, b))
    return logsumexp(np.array(lw), axis=0), logsumexp(np.array(lwm), axis=0)


def nll_C(x, N, M, conditional, wts):
    q = unpack_C(x)
    lpN, lpNM = firm_terms_C(q, N, M)
    if conditional:
        ll = lpNM - lpN
    else:
        lp0 = poisson.logpmf(0, q["lamT"]) + nbinom.logpmf(0, q["r"], q["r"] / (q["r"] + q["m"]))
        ll = lpNM - np.log(-np.expm1(lp0))
    v = -(wts * ll).sum()
    return v if np.isfinite(v) else 1e12


def nll_A(x, N, M, wts):
    G = cdf_table(sig(x[0]), np.clip(x[1], -2, 14), np.exp(np.clip(x[2], np.log(0.3), np.log(5))))
    pr = G[M] ** N - G[M - 1] ** N
    return -(wts * np.log(np.maximum(pr, 1e-300))).sum()


def fit(fun, dim, starts, args, sampler):
    best = None
    for _ in range(starts):
        res = minimize(fun, sampler(), args=args, method="Nelder-Mead",
                       options=dict(maxiter=20000, maxfev=20000, xatol=1e-4, fatol=1e-4))
        if best is None or res.fun < best.fun:
            best = res
    return best


def predicted_by_group(pmf_fn, N, M, groups):
    rows = []
    for lab, lo, hi in groups:
        idx = (N >= lo) & (N <= hi)
        if idx.sum() == 0:
            continue
        pm = np.array([pmf_fn(n) for n in N[idx]])
        rows.append((lab, idx.sum(), (M[idx] == 1).mean(), pm[:, 0].mean(),
                     M[idx].mean(), (pm * BANDS).sum(1).mean(),
                     (M[idx] >= 5).mean(), pm[:, 4:].sum(1).mean()))
    return rows


def pmf_A_factory(x):
    G = cdf_table(sig(x[0]), np.clip(x[1], -2, 14), np.exp(np.clip(x[2], np.log(0.3), np.log(5))))
    return lambda n: np.array([G[b] ** n - G[b - 1] ** n for b in BANDS])


def pmf_C_factory(q):
    def f(n):
        lpN, lpNM = firm_terms_C(q, np.full(10, n), BANDS)
        return np.exp(lpNM - lpN)
    return f


def trunc_lognormal_mean(mu, s, cap=500_000):
    a = (np.log(cap) - mu) / s
    return np.exp(mu + s ** 2 / 2) * norm.cdf(a - s) / norm.cdf(a)


def describe_C(lab, res):
    q = unpack_C(res.x)
    p0 = nbinom.pmf(0, q["r"], q["r"] / (q["r"] + q["m"]))
    eT = q["pT"] * trunc_lognormal_mean(q["muT"], q["sT"])
    eM = q["pM"] * trunc_lognormal_mean(q["muM"], q["sM"])
    print(f"\n  {lab}: -ll={res.fun:.2f}")
    print(f"    counts  : lam_T={q['lamT']:.3f} | mass NegBin m={q['m']:.2f}, r={q['r']:.3f}, "
          f"P(K_M=0)={p0:.2f}")
    print(f"    targeted: p={q['pT']:.3f}, mu={q['muT']:.2f}, sigma={q['sT']:.2f} "
          f"-> E[cost/attack]=£{eT:,.0f}")
    print(f"    mass    : p={q['pM']:.4f}, mu={q['muM']:.2f}, sigma={q['sM']:.2f} "
          f"-> E[cost/attack]=£{eM:,.2f}")
    print(f"    per-firm Wald (unconditional rates): lam_T*E_T=£{q['lamT']*eT:,.0f}, "
          f"m*E_M=£{q['m']*eM:,.0f}")


def run(raw, mode, starts, cov):
    N, M = build(raw, mode)
    ones = np.ones(len(N))
    ipw = 1 / cov.reindex(M).values
    ipw = ipw / ipw.mean()
    print("\n" + "=" * 78)
    print(f"POPULATION: {mode}  n={len(N)}  N: median={np.median(N):g}, mean={N.mean():.1f}, "
          f"max={N.max()}")
    print("=" * 78)

    def samp_A():
        return np.array([rng.normal(-3, 1.5), rng.normal(5, 1.5), rng.normal(0.7, 0.3)])

    def samp_C():
        return np.array([rng.normal(-1, 1), rng.normal(2.5, 1), rng.normal(-1, 1),
                         rng.normal(0.5, 1), rng.normal(6, 1.5), rng.normal(0.7, 0.3),
                         rng.normal(-4, 1.5), rng.normal(5, 1.5), rng.normal(0.7, 0.3)])

    resA = fit(nll_A, 3, starts, (N, M, ones), samp_A)
    resC = fit(nll_C, 9, starts, (N, M, True, ones), samp_C)
    resJ = fit(nll_C, 9, starts, (N, M, False, ones), samp_C)
    resW = fit(nll_C, 9, starts, (N, M, False, ipw), samp_C)

    print("\nConditional fit of P(M | N):")
    print(f"  A one-channel iid : -ll={resA.fun:8.2f}  k=3  AIC={2*resA.fun+6:8.2f}  "
          f"(p={sig(resA.x[0]):.4f}, mu={resA.x[1]:.2f}, sigma={np.exp(resA.x[2]):.2f})")
    print(f"  C two-channel     : -ll={resC.fun:8.2f}  k=9  AIC={2*resC.fun+18:8.2f}")
    describe_C("C conditional P(M|N)", resC)
    describe_C("C joint P(N,M | N>=1)", resJ)
    describe_C("C joint, IPW for count coverage by band", resW)

    groups = [("N=1", 1, 1), ("N=2-3", 2, 3), ("N=4-10", 4, 10),
              ("N=11-50", 11, 50), ("N=51+", 51, 10**6)]
    print("\nObserved vs predicted by count group (A; C = joint fit):")
    print(f"  {'group':>8s} {'n':>4s} | {'P(no cost)':^20s} | {'mean band':^20s} | {'P(band>=5)':^20s}")
    print(f"  {'':>8s} {'':>4s} |" + " |".join([f" {'obs':>6s}{'A':>7s}{'C':>7s}"] * 3))
    rA = predicted_by_group(pmf_A_factory(resA.x), N, M, groups)
    rC = predicted_by_group(pmf_C_factory(unpack_C(resJ.x)), N, M, groups)
    for a, c in zip(rA, rC):
        print(f"  {a[0]:>8s} {a[1]:4d} | {a[2]:6.2f}{a[3]:7.2f}{c[3]:7.2f} | {a[4]:6.2f}{a[5]:7.2f}"
              f"{c[5]:7.2f} | {a[6]:6.2f}{a[7]:7.2f}{c[7]:7.2f}")


def coverage_check(raw):
    print("=" * 78)
    print("COVERAGE: is Cybercrime_phishsum missing at random among phishing firms?")
    print("=" * 78)
    ph = (raw["type_comb1"] == 1) & (banded(raw["type6"]) == 1)
    has = pd.to_numeric(raw["Cybercrime_phishsum"], errors="coerce") >= 1
    d = pd.DataFrame({"has": has[ph], "freq": banded(raw["freq"])[ph],
                      "sizeb": banded(raw["sizeb"])[ph], "half": raw["half"][ph],
                      "band": banded(raw["damage_bands"])[ph]})
    print(f"  phishing firms: {len(d)}, with count: {d['has'].sum()} ({d['has'].mean():.1%})")
    for c in ["half", "sizeb", "freq", "band"]:
        print(f"  by {c}: " + ", ".join(f"{k:g}:{v:.2f}" for k, v in
                                        d.groupby(c)["has"].mean().items()))
    cov = d.groupby("band")["has"].mean()
    cov.index = cov.index.astype(int)
    return cov


def main():
    raw = load_raw()
    cov = coverage_check(raw)
    run(raw, "phishing-only", 8, cov)
    run(raw, "disrupta==6", 8, cov)


if __name__ == "__main__":
    main()
