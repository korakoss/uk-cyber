"""Joint likelihood fit of the four-channel generative model on the full business sample.

Channels (independent):  T targeted phishing, M mass phishing, I impersonation, S serious.
  K_T ~ Poisson(lam_T)     K_M ~ NegBin(m_M, r_M)
  K_I ~ Poisson(lam_I)     K_S ~ NegBin(m_S, r_S)
  each attack: no cost w.p. 1-p; else banded lognormal(mu, sigma) on bands 2..13 (open tail)

Per firm we observe (any may be missing):
  flags      phishing = 1{K_T+K_M>=1}, impersonation = 1{K_I>=1}, serious = 1{K_S>=1}
  freq       noisy measurement of total K via P(freq | K), calibrated on firms with real counts
  N          Cybercrime_phishsum = K_T + K_M (MAR given the observed worst band)
  M          worst cost band over all attacks
  D          disrupta, read as the channel that produced the worst incident (ties split evenly)

Exact computation: each channel is a table over (count bucket, channel max band);
channels are combined pairwise (bucketed count sum, max band, argmax label).
Non-attacked firms contribute P(all K = 0).

Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src \
     python3 src/estimation/joint_four_channel.py [--weighted]
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, poisson, nbinom

from data import load_raw, BAND_BOUNDS, GENUINE_TYPE_COLS

KMAX = 1100
EDGES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 21, 31, 51, 76, 101, 151, 251,
                  401, 701, KMAX + 1])
NB = len(EDGES) - 1
REP = (EDGES[:-1] + EDGES[1:] - 1) / 2.0
WIDTH = EDGES[1:] - EDGES[:-1]
NY, NL = 14, 4            # max band 0..13 (0 = no attacks); label none/P/I/S
LO = np.array([BAND_BOUNDS[b][0] for b in range(2, 14)], float)
HI = np.array([BAND_BOUNDS[b][1] for b in range(2, 14)], float)
HI[-1] = 5e8              # finite cap for band 13 (£5M+ open interval) — wide enough not to bind
SERIOUS = [c for c in GENUINE_TYPE_COLS if c not in ("type5", "type6")]
CAL_EDGES = np.array([1, 2, 3, 5, 9, 17, 33, 65, 129, 10**6])

K = np.arange(KMAX + 1)
BUCKET_OF_K = np.searchsorted(EDGES, K, side="right") - 1
SUM_IDX = np.clip(np.searchsorted(EDGES, REP[:, None] + REP[None, :], side="right") - 1, 0, NB - 1)
SUM_ONEHOT = np.zeros((NB * NB, NB))
SUM_ONEHOT[np.arange(NB * NB), SUM_IDX.ravel()] = 1.0

NAMES = ["lamT", "pT", "muT", "sT", "mM", "rM", "pM", "muM", "sM",
         "lamI", "pI", "muI", "sI", "mS", "rS", "pS", "muS", "sS"]


# ----------------------------------------------------------------------------- data
def banded(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s < 100))


def load_firms():
    raw = load_raw()
    t = {c: banded(raw[c]) == 1 for c in GENUINE_TYPE_COLS}
    att = banded(raw["type_comb1"])
    d = pd.DataFrame({
        "att": att,
        "fP": t["type6"].astype(int), "fI": t["type5"].astype(int),
        "fS": pd.concat([t[c] for c in SERIOUS], axis=1).any(axis=1).astype(int),
        "freq": banded(raw["freq"]), "band": banded(raw["damage_bands"]),
        "disrupta": banded(raw["disrupta"]), "sizeb": banded(raw["sizeb"]),
        "weight": pd.to_numeric(raw["weight"], errors="coerce"),
        "N": pd.to_numeric(raw["Cybercrime_phishsum"], errors="coerce"),
    })
    d = d[d["att"].isin([0, 1])].copy()
    flagged = (d["fP"] + d["fI"] + d["fS"]) > 0
    d = d[(d["att"] == 0) | flagged].copy()
    d.loc[d["att"] == 0, ["fP", "fI", "fS"]] = 0
    d.loc[d["att"] == 0, ["freq", "band", "disrupta", "N"]] = np.nan
    d.loc[(d["fP"] == 0) | ~(d["N"] >= 1), "N"] = np.nan
    d.loc[~d["freq"].between(1, 6), "freq"] = np.nan
    d.loc[~d["band"].between(1, 13), "band"] = np.nan
    lab = d["disrupta"].map(lambda v: 1 if v == 6 else 2 if v == 5 else
                            3 if v in (1, 2, 3, 4, 7, 8, 9, 10, 11, 12) else np.nan)
    ok = ((lab == 1) & (d["fP"] == 1)) | ((lab == 2) & (d["fI"] == 1)) | ((lab == 3) & (d["fS"] == 1))
    d["D"] = lab.where(ok)
    return d.reset_index(drop=True)


def calibrate_freq(d):
    """P(freq | total K) from firms with a real phishing count, on coarse count bins."""
    c = d[d["N"].notna() & d["freq"].notna()]
    cb = np.searchsorted(CAL_EDGES, c["N"].values, side="right") - 1
    tab = np.full((len(CAL_EDGES) - 1, 6), 0.5)
    for b, f in zip(cb, c["freq"].astype(int).values):
        tab[b, f - 1] += 1
    tab /= tab.sum(1, keepdims=True)
    F = np.zeros((NB, 6))
    F[1:] = tab[np.searchsorted(CAL_EDGES, REP[1:], side="right") - 1]
    return F, tab, len(c)


# ----------------------------------------------------------------------------- model
def sig(v):
    return 1 / (1 + np.exp(-np.clip(v, -30, 30)))


def unpack(x):
    e = lambda v, lo, hi: np.exp(np.clip(v, lo, hi))
    s = lambda v: e(v, np.log(0.3), np.log(5))
    mu = lambda v: np.clip(v, -2, 14)
    return dict(lamT=e(x[0], -8, 3), pT=sig(x[1]), muT=mu(x[2]), sT=s(x[3]),
                mM=e(x[4], -6, 7), rM=e(x[5], -5, 5), pM=sig(x[6]), muM=mu(x[7]), sM=s(x[8]),
                lamI=e(x[9], -8, 3), pI=sig(x[10]), muI=mu(x[11]), sI=s(x[12]),
                mS=e(x[13], -8, 6), rS=e(x[14], -5, 5), pS=sig(x[15]), muS=mu(x[16]), sS=s(x[17]))


def pack(q):
    lg = lambda p: np.log(p / (1 - p))
    return np.array([np.log(q["lamT"]), lg(q["pT"]), q["muT"], np.log(q["sT"]),
                     np.log(q["mM"]), np.log(q["rM"]), lg(q["pM"]), q["muM"], np.log(q["sM"]),
                     np.log(q["lamI"]), lg(q["pI"]), q["muI"], np.log(q["sI"]),
                     np.log(q["mS"]), np.log(q["rS"]), lg(q["pS"]), q["muS"], np.log(q["sS"])])


def band_cdf(p, mu, s):
    """G[y] = P(single-attack band <= y), y = 0..13 (G[0] = 0). Open tail, no renorm."""
    up = norm.cdf((np.log(HI) - mu) / s)
    lo = norm.cdf((np.log(LO) - mu) / s)
    lo[0] = 0.0
    mass = np.maximum(up - lo, 0) + 1e-15
    G = np.zeros(NY)
    G[1] = 1 - p
    G[2:] = (1 - p) + p * np.cumsum(mass)
    G[-1] = 1.0
    return G


def count_pmf(kind, a, b=None):
    if kind == "pois":
        pmf = poisson.pmf(K, a)
        tail = poisson.sf(KMAX, a)
    else:
        pr = b / (b + a)
        pmf = nbinom.pmf(K, b, pr)
        tail = nbinom.sf(KMAX, b, pr)
    pmf = pmf.copy()
    pmf[-1] += tail
    return pmf


def channel_table(pmf, G, label):
    """S[bucket, y, label] = P(K in bucket, channel max = y)."""
    Gk = G[None, :] ** K[:, None]                     # (K, 11); Gk[k,0]=0 for k>=1, 1 for k=0
    py = np.diff(np.concatenate([np.zeros((len(K), 1)), Gk], axis=1), axis=1)
    py[0] = 0.0
    py[0, 0] = 1.0                                    # k=0 -> y=0
    w = pmf[:, None] * py
    tab = np.add.reduceat(w, EDGES[:-1], axis=0)      # (NB, 11)
    S = np.zeros((NB, NY, NL))
    S[:, 0, 0] = tab[:, 0]
    S[:, 1:, label] = tab[:, 1:]
    return S


def combine(A, B):
    As, Bs = A.sum(2), B.sum(2)
    Alt, Blt = np.cumsum(As, 1) - As, np.cumsum(Bs, 1) - Bs
    T = (A[:, None] * (Blt[None, :, :, None] + 0.5 * Bs[None, :, :, None])
         + B[None, :] * (Alt[:, None, :, None] + 0.5 * As[:, None, :, None]))
    return (SUM_ONEHOT.T @ T.reshape(NB * NB, NY * NL)).reshape(NB, NY, NL)


def restrict(S, flag):
    R = S.copy()
    if flag:
        R[0] = 0.0
    else:
        R[1:] = 0.0
    return R


def tables(q):
    ST = channel_table(count_pmf("pois", q["lamT"]), band_cdf(q["pT"], q["muT"], q["sT"]), 1)
    SM = channel_table(count_pmf("nb", q["mM"], q["rM"]), band_cdf(q["pM"], q["muM"], q["sM"]), 1)
    SI = channel_table(count_pmf("pois", q["lamI"]), band_cdf(q["pI"], q["muI"], q["sI"]), 2)
    SS = channel_table(count_pmf("nb", q["mS"], q["rS"]), band_cdf(q["pS"], q["muS"], q["sS"]), 3)
    return combine(ST, SM), SI, SS


class Likelihood:
    def __init__(self, d, F, weights):
        self.F = F
        d = d.copy()
        d["nb"] = np.where(d["N"].notna(),
                           np.searchsorted(EDGES, d["N"].fillna(1).clip(upper=KMAX), side="right") - 1,
                           -1)
        d["fi"] = d["freq"].fillna(7).astype(int) - 1          # 6 = missing
        d["yi"] = d["band"].fillna(-1).astype(int)             # -1 -> missing
        d.loc[d["att"] == 0, "yi"] = 0
        d.loc[d["yi"] < 0, "yi"] = NY                          # 11 = missing
        d["li"] = d["D"].fillna(-1).astype(int)
        d.loc[d["att"] == 0, "li"] = 0
        d.loc[d["li"] < 0, "li"] = NL                          # 4 = missing
        d["w"] = weights
        self.groups = []
        for key, g in d.groupby(["fP", "fI", "fS", "nb"]):
            self.groups.append((key, g["fi"].values, g["yi"].values, g["li"].values, g["w"].values))

    def nll(self, x):
        q = unpack(x)
        P, SI, SS = tables(q)
        X = {(fi, fs): combine(restrict(SI, fi), restrict(SS, fs)) for fi in (0, 1) for fs in (0, 1)}
        Fext = np.vstack([self.F.T, np.ones(NB)])             # (7, NB): 6 = freq missing
        total = 0.0
        for (fP, fI, fS, nb), fi, yi, li, w in self.groups:
            if nb >= 0:
                Pr = np.zeros_like(P)
                Pr[nb] = P[nb] / WIDTH[nb]
            else:
                Pr = restrict(P, fP)
            C = combine(X[(fI, fS)], Pr)                       # (NB, 11, 4)
            Q = np.einsum("fc,cyl->fyl", Fext, C)
            Q = np.concatenate([Q, Q.sum(1, keepdims=True)], 1)
            Q = np.concatenate([Q, Q.sum(2, keepdims=True)], 2)
            v = Q[fi, yi, li]
            total += (w * np.log(np.maximum(v, 1e-300))).sum()
        return -total if np.isfinite(total) else 1e15


# ----------------------------------------------------------------------------- reporting
def trunc_mean(mu, s, cap=None):
    """E[C | success]. No cap: full lognormal mean. With cap: truncated."""
    if cap is None:
        return np.exp(mu + s ** 2 / 2)
    a = (np.log(cap) - mu) / s
    return np.exp(mu + s ** 2 / 2) * norm.cdf(a - s) / norm.cdf(a)


def simulate(q, F, n, rng):
    Kt = rng.poisson(q["lamT"], n)
    Km = rng.negative_binomial(q["rM"], q["rM"] / (q["rM"] + q["mM"]), n)
    Ki = rng.poisson(q["lamI"], n)
    Ks = rng.negative_binomial(q["rS"], q["rS"] / (q["rS"] + q["mS"]), n)

    def chmax(k, p, mu, s):
        G = band_cdf(p, mu, s)
        u = rng.random(n)
        Gk = G[None, :] ** k[:, None]
        y = (Gk < u[:, None]).sum(1)
        return np.where(k == 0, 0, y)

    Y = np.stack([chmax(Kt, q["pT"], q["muT"], q["sT"]), chmax(Km, q["pM"], q["muM"], q["sM"]),
                  chmax(Ki, q["pI"], q["muI"], q["sI"]), chmax(Ks, q["pS"], q["muS"], q["sS"])], 1)
    M = Y.max(1)
    lab_of = np.array([1, 1, 2, 3])
    tie = (Y == M[:, None]) & (M[:, None] > 0)
    pick = np.argmax(tie * rng.random((n, 4)), axis=1)
    D = np.where(M > 0, lab_of[pick], 0)
    Ktot = Kt + Km + Ki + Ks
    b = BUCKET_OF_K[np.minimum(Ktot, KMAX)]
    cf = F[b].cumsum(1)
    freq = np.where(Ktot > 0, (cf < rng.random(n)[:, None]).sum(1) + 1, 0)
    return pd.DataFrame({"fP": (Kt + Km > 0).astype(int), "fI": (Ki > 0).astype(int),
                         "fS": (Ks > 0).astype(int), "N": Kt + Km, "M": M, "D": D,
                         "freq": np.minimum(freq, 6)})


def moment_checks(d, sim):
    print("\nMOMENT CHECKS (observed vs simulated from fit)")
    att_o, att_s = d[d["att"] == 1], sim[(sim[["fP", "fI", "fS"]].sum(1) > 0)]
    print(f"  prevalence: obs {(d['att'] == 1).mean():.3f}  sim {len(att_s) / len(sim):.3f}")
    print("  flag pattern among attacked (P,I,S):")
    for pat in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]:
        o = ((att_o[["fP", "fI", "fS"]].values == pat).all(1)).mean()
        s = ((att_s[["fP", "fI", "fS"]].values == pat).all(1)).mean()
        print(f"    {pat}: obs {o:.3f}  sim {s:.3f}")
    for lab, fo, fs in [("phishing-only", (att_o.fP == 1) & (att_o.fI == 0) & (att_o.fS == 0),
                         (att_s.fP == 1) & (att_s.fI == 0) & (att_s.fS == 0)),
                        ("imp-only", (att_o.fP == 0) & (att_o.fI == 1) & (att_o.fS == 0),
                         (att_s.fP == 0) & (att_s.fI == 1) & (att_s.fS == 0)),
                        ("serious-only", (att_o.fP == 0) & (att_o.fI == 0) & (att_o.fS == 1),
                         (att_s.fP == 0) & (att_s.fI == 0) & (att_s.fS == 1)),
                        ("all attacked", att_o.fP >= 0, att_s.fP >= 0)]:
        o = att_o[fo]["freq"].dropna().astype(int).value_counts(normalize=True).reindex(range(1, 7), fill_value=0)
        s = att_s[fs]["freq"].value_counts(normalize=True).reindex(range(1, 7), fill_value=0)
        print(f"  freq | {lab:>13s}: obs " + " ".join(f"{v:.2f}" for v in o) +
              "  |  sim " + " ".join(f"{v:.2f}" for v in s))
    o = att_o["D"].dropna().value_counts(normalize=True).reindex([1, 2, 3], fill_value=0)
    s = att_s["D"].value_counts(normalize=True).reindex([1, 2, 3], fill_value=0)
    print("  most-disruptive share P/I/S: obs " + " ".join(f"{v:.2f}" for v in o) +
          "  |  sim " + " ".join(f"{v:.2f}" for v in s))
    o = att_o["band"].dropna().astype(int).value_counts(normalize=True).reindex(range(1, 14), fill_value=0)
    s = att_s["M"].value_counts(normalize=True).reindex(range(1, 14), fill_value=0)
    print("  worst band | attacked: obs " + " ".join(f"{v:.3f}" for v in o))
    print("                         sim " + " ".join(f"{v:.3f}" for v in s))
    for lab in (3,):
        oo = att_o[att_o["D"] == lab]["band"].dropna().astype(int)
        ss = att_s[att_s["D"] == lab]["M"]
        print(f"  worst band | serious most disruptive: obs mean {oo.mean():.2f}, P(>=5) {(oo >= 5).mean():.2f}"
              f"  |  sim mean {ss.mean():.2f}, P(>=5) {(ss >= 5).mean():.2f}")
    no = att_o[att_o["N"].notna()]["N"]
    ns = att_s[att_s["fP"] == 1]["N"]
    print("  phishing count quantiles (25/50/75/90): obs " +
          " ".join(f"{no.quantile(p):g}" for p in (.25, .5, .75, .9)) +
          "  |  sim " + " ".join(f"{ns.quantile(p):g}" for p in (.25, .5, .75, .9)))
    po = att_o[(att_o.fI == 0) & (att_o.fS == 0) & att_o["N"].notna() & att_o["band"].notna()]
    ps = att_s[(att_s.fP == 1) & (att_s.fI == 0) & (att_s.fS == 0)]
    print("  P(no cost | N) phishing-only:")
    for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 50), (51, 10**6)]:
        o = po[po["N"].between(lo, hi)]
        s = ps[ps["N"].between(lo, hi)]
        print(f"    N {lo}-{hi if hi < 10**6 else '+'}: obs {(o['band'] == 1).mean():.2f} (n={len(o)})"
              f"  sim {(s['M'] == 1).mean():.2f}")


def report(q, d, F, label, nll):
    print("\n" + "=" * 78)
    print(f"FIT: {label}   -loglik = {nll:.2f}   firms = {len(d)}")
    print("=" * 78)
    rows = [("targeted T", q["lamT"], "Poisson", q["pT"], q["muT"], q["sT"]),
            ("mass M", q["mM"], f"NegBin r={q['rM']:.3f}", q["pM"], q["muM"], q["sM"]),
            ("imperson. I", q["lamI"], "Poisson", q["pI"], q["muI"], q["sI"]),
            ("serious S", q["mS"], f"NegBin r={q['rS']:.3f}", q["pS"], q["muS"], q["sS"])]
    total = 0
    print(f"  {'channel':>12s} {'E[K]/firm':>10s} {'count law':>16s} {'p':>7s} {'mu':>6s} {'sigma':>6s}"
          f" {'E[C|succ]':>10s} {'E[cost/attack]':>14s} {'E[cost/firm]':>13s}")
    for name, ek, law, p, mu, s in rows:
        ec = trunc_mean(mu, s)
        total += ek * p * ec
        print(f"  {name:>12s} {ek:10.3f} {law:>16s} {p:7.4f} {mu:6.2f} {s:6.2f} £{ec:9,.0f}"
              f" £{p * ec:13,.1f} £{ek * p * ec:12,.1f}")
    print(f"  {'TOTAL':>12s}{'':>77s} £{total:12,.1f}  per firm (sample population)")
    print(f"  crude national (x 1,417,730 employer businesses, pooled sizes): £{total * 1_417_730 / 1e9:.2f}bn")
    moment_checks(d, simulate(q, F, 300_000, np.random.default_rng(1)))


def main():
    weighted = "--weighted" in sys.argv
    d = load_firms()
    F, tab, ncal = calibrate_freq(d)
    print(f"Firms: {len(d)} (attacked {int((d['att'] == 1).sum())}), weighted={weighted}")
    print(f"freq measurement model from {ncal} firms with counts; P(freq | count bin):")
    for i in range(len(CAL_EDGES) - 1):
        lo, hi = CAL_EDGES[i], CAL_EDGES[i + 1] - 1
        print(f"  N {lo:>3d}-{hi if hi < 10**5 else '+':>4}: " + " ".join(f"{v:.2f}" for v in tab[i]))

    w = d["weight"].fillna(d["weight"].median()).values if weighted else np.ones(len(d))
    w = w / w.mean()
    lik = Likelihood(d, F, w)
    q0 = dict(lamT=0.2, pT=0.55, muT=4.9, sT=2.2, mM=8.0, rM=0.1, pM=0.008, muM=4.5, sM=1.6,
              lamI=0.2, pI=0.5, muI=4.4, sI=2.4, mS=0.5, rS=0.3, pS=0.75, muS=6.8, sS=2.6)
    x0 = pack(q0)
    t = time.time()
    print(f"initial -loglik {lik.nll(x0):.2f} ({time.time() - t:.2f}s per eval)")
    best = None
    rng = np.random.default_rng(0)
    for start in range(3):
        xs = x0 if start == 0 else x0 + rng.normal(0, 0.5, len(x0))
        res = minimize(lik.nll, xs, method="L-BFGS-B", options=dict(maxiter=400))
        res = minimize(lik.nll, res.x, method="Nelder-Mead",
                       options=dict(maxiter=6000, maxfev=6000, xatol=1e-4, fatol=1e-3))
        print(f"  start {start}: -loglik {res.fun:.2f}")
        if best is None or res.fun < best.fun:
            best = res
    q = unpack(best.x)
    label = "survey-weighted pseudo-likelihood" if weighted else "unweighted"
    report(q, d, F, label, best.fun)
    out = os.path.join(os.path.dirname(__file__), "build",
                       f"joint_four_channel_{'weighted' if weighted else 'unweighted'}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({k: float(v) for k, v in q.items()} | {"nll": float(best.fun)}, fh, indent=1)
    print(f"\nparams saved to {out}")


if __name__ == "__main__":
    main()
