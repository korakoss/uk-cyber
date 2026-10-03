"""Free K-class frailty mixture (NPMLE-style): no assumed frailty shape.

Class 0 is the reference (base rates); classes 1..K-1 each get a free log-multiplier
per channel (T, M, I, S) and a free weight (softmax). K=2 is the binary model.
Increase K until the fit stops improving; the fitted (weight, multiplier) points are
the agnostic "frailty coefficients". Also reports per-firm posterior class
probabilities and their spread.

Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src:src/estimation \
     python3 src/estimation/frailty_free_mixture.py K
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import minimize

import joint_four_channel as j
import joint_four_channel_frailty as fr

HERE = os.path.dirname(os.path.abspath(__file__))


def unpack(x, K):
    q = j.unpack(x[:18])
    logits = np.concatenate([[0.0], np.clip(x[18:18 + K - 1], -12, 12)])
    wts = np.exp(logits - logits.max())
    q["w"] = wts / wts.sum()
    q["D"] = np.zeros((K, 4))
    q["D"][1:] = np.clip(x[18 + K - 1:].reshape(K - 1, 4), -8, 10)
    return q


def pack(q, K):
    return np.concatenate([j.pack(q), np.log(q["w"][1:] / q["w"][0]), q["D"][1:].ravel()])


def class_q(q, k):
    qc = dict(q)
    for c, (ch, key) in enumerate(fr.CHANNEL_RATE.items()):
        qc[key] = q[key] * np.exp(q["D"][k, c])
    return qc


class FreeMixLik(fr.FrailtyLikelihood):
    def __init__(self, d, F, w, K):
        super().__init__(d, F, w)
        self.K = K

    def class_values(self, q):
        return [self.firm_values(class_q(q, k)) for k in range(self.K)]

    def nll(self, x):
        q = unpack(x, self.K)
        vals = self.class_values(q)
        total = 0.0
        for g, (key, fi, yi, li, w) in enumerate(self.groups):
            v = sum(q["w"][k] * vals[k][g] for k in range(self.K))
            total += (w * np.log(np.maximum(v, 1e-300))).sum()
        return -total if np.isfinite(total) else 1e15


def init(qb, K, rng, jitter):
    """Start from the binary fit: quiet class = base, others spread along the binary delta."""
    q = {k: v for k, v in qb.items() if k not in ("pi", "dT", "dM", "dI", "dS")}
    dbin = np.array([qb["d" + ch] for ch in "TMIS"])
    fracs = np.linspace(0, 1, K)[1:] if K > 2 else np.array([1.0])
    if K > 2:
        fracs = np.linspace(0.4, 1.3, K - 1)
    q["D"] = np.vstack([np.zeros(4)] + [f * dbin for f in fracs])
    w = np.array([1 - qb["pi"]] + [qb["pi"] / (K - 1)] * (K - 1))
    q["w"] = w / w.sum()
    x = pack(q, K)
    if jitter:
        x = x + rng.normal(0, 0.3, len(x))
    return x


def main():
    K = int(sys.argv[1])
    d = j.load_firms()
    F, _, _ = j.calibrate_freq(d)
    w = d["weight"].fillna(d["weight"].median()).values
    w = w / w.mean()
    lik = FreeMixLik(d, F, w, K)
    qb = json.load(open(os.path.join(HERE, "build", "joint_four_channel_frailty_weighted.json")))
    qb.pop("nll", None)

    rng = np.random.default_rng(K)
    best = None
    for start in range(3):
        x0 = init(qb, K, rng, jitter=start > 0)
        res = minimize(lik.nll, x0, method="L-BFGS-B", options=dict(maxiter=400))
        res = minimize(lik.nll, res.x, method="Nelder-Mead",
                       options=dict(maxiter=6000, maxfev=6000, xatol=1e-4, fatol=1e-3))
        print(f"  K={K} start {start}: -loglik {res.fun:.2f}", flush=True)
        if best is None or res.fun < best.fun:
            best = res

    q = unpack(best.x, K)
    npar = 18 + (K - 1) * 5
    n_eff = w.sum()
    print(f"\nK={K}: -loglik {best.fun:.2f}  params {npar}  BIC {2 * best.fun + npar * np.log(n_eff):.1f}")

    # classes sorted by overall intensity (mean log-multiplier)
    order = np.argsort(q["D"].mean(1))
    print("\n  class  weight   rate multiplier vs quietest class      absolute E[K]/firm-year")
    print(f"  {'':5s} {'':7s}  {'T':>7s} {'M':>8s} {'I':>7s} {'S':>7s}     "
          f"{'T':>6s} {'M':>7s} {'I':>6s} {'S':>6s}")
    base = q["D"][order[0]]
    rates = np.array([q[fr.CHANNEL_RATE[ch]] for ch in "TMIS"])
    for k in order:
        m = np.exp(q["D"][k] - base)
        a = rates * np.exp(q["D"][k])
        print(f"  {k:5d} {q['w'][k]:7.3f}  {m[0]:7.1f} {m[1]:8.1f} {m[2]:7.1f} {m[3]:7.1f}     "
              f"{a[0]:6.3f} {a[1]:7.2f} {a[2]:6.3f} {a[3]:6.3f}")

    total = 0.0
    for c, ch in enumerate("TMIS"):
        ek = sum(q["w"][k] * rates[c] * np.exp(q["D"][k, c]) for k in range(K))
        total += ek * q["p" + ch] * j.trunc_mean(q["mu" + ch], q["s" + ch])
    print(f"\n  cost/firm £{total:,.0f}  -> crude pooled national £{total * 1_417_730 / 1e9:.2f}bn")

    # per-firm posterior class membership
    vals = lik.class_values(q)
    post_max, post_att, post_w = [], [], []
    for g, (key, fi, yi, li, wt) in enumerate(lik.groups):
        L = np.array([q["w"][k] * vals[k][g] for k in range(K)])
        P = L / np.maximum(L.sum(0), 1e-300)
        post_max.append(P.max(0))
        post_att.append(np.full(len(wt), bool(key[0] or key[1] or key[2])))
        post_w.append(wt)
    pm, pa, pw = map(np.concatenate, (post_max, post_att, post_w))
    for lab, m in (("all", np.ones_like(pa)), ("attacked", pa), ("not attacked", ~pa)):
        print(f"  posterior max-class prob, {lab:12s} (n={m.sum()}): mean {np.average(pm[m], weights=pw[m]):.2f}"
              f"  share >0.9 {np.average(pm[m] > 0.9, weights=pw[m]):.2f}")

    with open(os.path.join(HERE, "build", f"frailty_free_K{K}_weighted.json"), "w") as fh:
        json.dump({"K": K, "nll": float(best.fun), "npar": npar, "w": q["w"].tolist(),
                   "D": q["D"].tolist(), "cost_per_firm": float(total),
                   **{k: float(v) for k, v in q.items() if k not in ("w", "D")}}, fh, indent=1)


if __name__ == "__main__":
    main()
