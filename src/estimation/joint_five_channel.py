"""Five-channel frailty model: ransomware split out of 'serious'.

Channels: T targeted phishing (Poisson), M mass phishing (NegBin), I impersonation (Poisson),
R ransomware (NegBin), S other serious (NegBin). Binary frailty on all five rates.
Observables as in joint_four_channel.py, plus the ransomware flag (type1) separately from the
other serious flags, and disrupta resolved to 5 labels (none/P/I/R/S).

Two fits on identical data:
  free  - R has its own success gate and cost lognormal       (29 params)
  tied  - R shares (p, mu, sigma) with S; only counts split    (26 params)
LR(free vs tied) asks whether ransomware's per-attack cost differs from other serious.

Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src:src/estimation \
     python3 src/estimation/joint_five_channel.py {free|tied} [rcount]
rcount: also use Cybercrime_ranssum (ransomware count) where reported, like the phishing count.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import joint_four_channel as j
from data import load_raw

HERE = os.path.dirname(os.path.abspath(__file__))
NL = 6                                   # labels none/P/I/R/S + 1 for "missing"
NLAB = 5
CH = "TMIRS"
RATE = {"T": "lamT", "M": "mM", "I": "lamI", "R": "mR", "S": "mS"}
LAB = {"T": 1, "M": 1, "I": 2, "R": 3, "S": 4}
OTHER_SERIOUS = [c for c in j.SERIOUS if c != "type1"]


def load_firms():
    d = j.load_firms()
    raw = load_raw()
    t = {c: j.banded(raw[c]) == 1 for c in j.GENUINE_TYPE_COLS}
    att = j.banded(raw["type_comb1"])
    keep = att.isin([0, 1])
    fR = t["type1"].astype(int)[keep]
    fS2 = pd.concat([t[c] for c in OTHER_SERIOUS], axis=1).any(axis=1).astype(int)[keep]
    fP = t["type6"].astype(int)[keep]
    fI = t["type5"].astype(int)[keep]
    sel = (att[keep] == 0) | ((fP + fI + fR + fS2) > 0)
    fR, fS2 = fR[sel].reset_index(drop=True), fS2[sel].reset_index(drop=True)
    assert len(fR) == len(d)
    d["fR"] = np.where(d["att"] == 1, fR, 0)
    d["fS"] = np.where(d["att"] == 1, fS2, 0)
    lab = d["disrupta"].map(lambda v: 1 if v == 6 else 2 if v == 5 else 3 if v == 1 else
                            4 if v in (2, 3, 4, 7, 8, 9, 10, 11, 12) else np.nan)
    ok = (((lab == 1) & (d["fP"] == 1)) | ((lab == 2) & (d["fI"] == 1)) |
          ((lab == 3) & (d["fR"] == 1)) | ((lab == 4) & (d["fS"] == 1)))
    d["D"] = lab.where(ok)
    nr = pd.to_numeric(raw["Cybercrime_ranssum"], errors="coerce")[keep][sel].reset_index(drop=True)
    d["NR"] = nr.where((d["fR"] == 1) & (nr >= 1))
    return d


def channel_table(pmf, G, label):
    Gk = G[None, :] ** j.K[:, None]
    py = np.diff(np.concatenate([np.zeros((len(j.K), 1)), Gk], axis=1), axis=1)
    py[0] = 0.0
    py[0, 0] = 1.0
    tab = np.add.reduceat(pmf[:, None] * py, j.EDGES[:-1], axis=0)
    S = np.zeros((j.NB, j.NY, NLAB))
    S[:, 0, 0] = tab[:, 0]
    S[:, 1:, label] = tab[:, 1:]
    return S


def combine(A, B):
    As, Bs = A.sum(2), B.sum(2)
    Alt, Blt = np.cumsum(As, 1) - As, np.cumsum(Bs, 1) - Bs
    T = (A[:, None] * (Blt[None, :, :, None] + 0.5 * Bs[None, :, :, None])
         + B[None, :] * (Alt[:, None, :, None] + 0.5 * As[:, None, :, None]))
    return (j.SUM_ONEHOT.T @ T.reshape(j.NB * j.NB, j.NY * NLAB)).reshape(j.NB, j.NY, NLAB)


def unpack(x, tied):
    e = lambda v, lo, hi: float(np.exp(np.clip(v, lo, hi)))
    s = lambda v: e(v, np.log(0.3), np.log(5))
    mu = lambda v: float(np.clip(v, -2, 14))
    sg = lambda v: float(j.sig(v))
    q = j.unpack(x[:18])
    q["mR"], q["rR"] = e(x[18], -8, 6), e(x[19], -5, 5)
    i = 20
    if tied:
        q["pR"], q["muR"], q["sR"] = q["pS"], q["muS"], q["sS"]
    else:
        q["pR"], q["muR"], q["sR"] = sg(x[20]), mu(x[21]), s(x[22])
        i = 23
    q["pi"] = sg(x[i])
    for k, ch in enumerate(CH):
        q["d" + ch] = float(np.clip(x[i + 1 + k], -6, 8))
    return q


def pack(q, tied):
    lg = lambda p: np.log(p / (1 - p))
    x = list(j.pack(q)) + [np.log(q["mR"]), np.log(q["rR"])]
    if not tied:
        x += [lg(q["pR"]), q["muR"], np.log(q["sR"])]
    return np.array(x + [lg(q["pi"])] + [q["d" + ch] for ch in CH])


def class_q(q, exposed):
    qc = dict(q)
    if exposed:
        for ch, key in RATE.items():
            qc[key] = q[key] * np.exp(q["d" + ch])
    return qc


def restrict(S, flag):
    R = S.copy()
    if flag:
        R[0] = 0.0
    else:
        R[1:] = 0.0
    return R


class Lik:
    def __init__(self, d, F, w, tied, use_rcount=False):
        self.F, self.tied = F, tied
        d = d.copy()
        d["nb"] = np.where(d["N"].notna(), np.searchsorted(
            j.EDGES, d["N"].fillna(1).clip(upper=j.KMAX), side="right") - 1, -1)
        d["fi"] = d["freq"].fillna(7).astype(int) - 1
        d["yi"] = d["band"].fillna(-1).astype(int)
        d.loc[d["att"] == 0, "yi"] = 0
        d.loc[d["yi"] < 0, "yi"] = j.NY
        d["li"] = d["D"].fillna(-1).astype(int)
        d.loc[d["att"] == 0, "li"] = 0
        d.loc[d["li"] < 0, "li"] = NLAB
        d["w"] = w
        nr = d["NR"] if use_rcount else pd.Series(np.nan, index=d.index)
        d["nbR"] = np.where(nr.notna(), np.searchsorted(
            j.EDGES, nr.fillna(1).clip(upper=j.KMAX), side="right") - 1, -1)
        self.groups = [(key, g["fi"].values, g["yi"].values, g["li"].values, g["w"].values)
                       for key, g in d.groupby(["fP", "fI", "fR", "fS", "nb", "nbR"])]

    def values(self, q):
        tab = lambda ch, kind: channel_table(
            j.count_pmf(kind, q[RATE[ch]], q.get("r" + ch)),
            j.band_cdf(q["p" + ch], q["mu" + ch], q["s" + ch]), LAB[ch])
        P = combine(tab("T", "pois"), tab("M", "nb"))
        SI, SR, SS = tab("I", "pois"), tab("R", "nb"), tab("S", "nb")
        X = {}

        def xtab(a, b, c, nbR):
            if (a, b, c, nbR) not in X:
                if nbR >= 0:
                    R = np.zeros_like(SR)
                    R[nbR] = SR[nbR] / j.WIDTH[nbR]
                else:
                    R = restrict(SR, b)
                X[(a, b, c, nbR)] = combine(combine(restrict(SI, a), R), restrict(SS, c))
            return X[(a, b, c, nbR)]

        Fext = np.vstack([self.F.T, np.ones(j.NB)])
        out = []
        for (fP, fI, fR, fS, nb, nbR), fi, yi, li, w in self.groups:
            if nb >= 0:
                Pr = np.zeros_like(P)
                Pr[nb] = P[nb] / j.WIDTH[nb]
            else:
                Pr = restrict(P, fP)
            C = combine(xtab(fI, fR, fS, nbR), Pr)
            Q = np.einsum("fc,cyl->fyl", Fext, C)
            Q = np.concatenate([Q, Q.sum(1, keepdims=True)], 1)
            Q = np.concatenate([Q, Q.sum(2, keepdims=True)], 2)
            out.append(Q[fi, yi, li])
        return out

    def nll(self, x):
        q = unpack(x, self.tied)
        a, b = self.values(class_q(q, False)), self.values(class_q(q, True))
        total = 0.0
        for g, va, vb in zip(self.groups, a, b):
            v = (1 - q["pi"]) * va + q["pi"] * vb
            total += (g[4] * np.log(np.maximum(v, 1e-300))).sum()
        return -total if np.isfinite(total) else 1e15


def main():
    tied = sys.argv[1] == "tied"
    rcount = "rcount" in sys.argv
    tag = sys.argv[1] + ("_rcount" if rcount else "")
    d = load_firms()
    F, _, _ = j.calibrate_freq(d)
    w = d["weight"].fillna(d["weight"].median()).values
    w = w / w.mean()
    lik = Lik(d, F, w, tied, use_rcount=rcount)

    q4 = json.load(open(os.path.join(HERE, "build", "joint_four_channel_frailty_weighted.json")))
    q0 = dict(q4)
    q0.update(mR=0.25 * q4["mS"], rR=q4["rS"], pR=min(0.9, q4["pS"] * 2), muR=q4["muS"] + 1,
              sR=q4["sS"], mS=0.8 * q4["mS"], dR=q4["dS"])
    rng = np.random.default_rng(5)
    best = None
    for start in range(2):
        x0 = pack(q0, tied) + (rng.normal(0, 0.3, len(pack(q0, tied))) if start else 0)
        res = minimize(lik.nll, x0, method="L-BFGS-B", options=dict(maxiter=400))
        res = minimize(lik.nll, res.x, method="Nelder-Mead",
                       options=dict(maxiter=6000, maxfev=6000, xatol=1e-4, fatol=1e-3))
        print(f"  {tag} start {start}: -loglik {res.fun:.2f}", flush=True)
        if best is None or res.fun < best.fun:
            best = res
    q = unpack(best.x, tied)
    print(f"\n{tag.upper()}: -loglik {best.fun:.2f}  params {len(best.x)}  pi {q['pi']:.3f}")
    print(f"  {'ch':>2s} {'rate q':>7s} {'rate e':>7s} {'E[K]':>7s} {'r':>6s} {'p':>6s} {'mu':>6s}"
          f" {'sigma':>6s} {'E[C|succ]':>11s} {'P(>£500k|s)':>11s} {'£/firm':>9s}")
    from scipy.stats import norm
    total = 0.0
    out = {}
    for ch in CH:
        rq = q[RATE[ch]]
        re_ = rq * np.exp(q["d" + ch])
        ek = (1 - q["pi"]) * rq + q["pi"] * re_
        ec = j.trunc_mean(q["mu" + ch], q["s" + ch])
        c = ek * q["p" + ch] * ec
        total += c
        out[ch] = c
        r = q.get("r" + ch)
        print(f"  {ch:>2s} {rq:7.3f} {re_:7.3f} {ek:7.3f} {r if r else float('nan'):6.3f} {q['p' + ch]:6.3f}"
              f" {q['mu' + ch]:6.2f} {q['s' + ch]:6.2f} £{ec:10,.0f} "
              f"{norm.sf((np.log(5e5) - q['mu' + ch]) / q['s' + ch]):11.4f} £{c:8,.0f}")
    print(f"  total £{total:,.0f}/firm -> crude pooled national £{total * 1_417_730 / 1e9:.2f}bn")
    with open(os.path.join(HERE, "build", f"joint_five_channel_{tag}_weighted.json"), "w") as fh:
        json.dump({k: float(v) for k, v in q.items()} | {"nll": float(best.fun), "npar": len(best.x),
                   "cost_per_firm": float(total)} | {f"cost_{k}": float(v) for k, v in out.items()},
                  fh, indent=1)


if __name__ == "__main__":
    main()
