"""Frailty tiers and frailty-on-costs in the size model.

Base: five-channel model with ransomware count (joint_five_channel.py), four size groups where
attack rates and exposure vary by size and success gates / cost lognormals are shared
(joint_size_model.py 'rates', the chosen structure). Generalised along two axes:
  K  : number of frailty classes (class 0 = quiet; classes 1..K-1 get per-channel log-rate
       multipliers D[k] and weight logits l_k). Size shifts all non-quiet logits by e_g.
  cf : cost frailty - classes 1..K-1 also shift each channel's cost location mu (C[k]).
K=2 without cf reproduces joint_size_model 'rates' exactly.

Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src:src/estimation \
     python3 src/estimation/joint_size_frailty.py K [cf]
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import minimize

import joint_four_channel as j
import joint_five_channel as f
import joint_size_model as sm

HERE = os.path.dirname(os.path.abspath(__file__))
NB0 = 23          # non-frailty base params (f.pack layout before pi / deltas)


def unpack(x, K, cf):
    q = f.unpack(np.r_[x[:NB0], 0.0, np.zeros(5)], False)
    for k in ("pi", *("d" + ch for ch in f.CH)):
        q.pop(k)
    i = NB0
    lg = np.r_[0.0, np.clip(x[i:i + K - 1], -12, 12)]
    i += K - 1
    D = np.vstack([np.zeros(5), np.clip(x[i:i + (K - 1) * 5].reshape(K - 1, 5), -8, 10)])
    i += (K - 1) * 5
    S = np.clip(x[i:i + 18].reshape(3, 6), -8, 8)
    i += 18
    C = np.zeros((K, 5))
    if cf:
        C[1:] = np.clip(x[i:i + (K - 1) * 5].reshape(K - 1, 5), -6, 6)
    return q, lg, D, S, C


def class_params(q, lg, D, S, C, g):
    """List of (weight, params) for size index g."""
    e = S[g - 1, 5] if g else 0.0
    logits = lg + np.r_[0.0, np.full(len(lg) - 1, e)]
    w = np.exp(logits - logits.max())
    w /= w.sum()
    out = []
    for k in range(len(lg)):
        qc = dict(q)
        for c, ch in enumerate(f.CH):
            qc[f.RATE[ch]] = q[f.RATE[ch]] * np.exp(D[k, c] + (S[g - 1, c] if g else 0.0))
            qc["mu" + ch] = q["mu" + ch] + C[k, c]
        out.append((w[k], qc))
    return out


class Lik:
    def __init__(self, d, F, K, cf):
        self.K, self.cf = K, cf
        self.liks = []
        for s in sm.SIZES:
            ds = d[d["sizeb"] == s].reset_index(drop=True)
            w = ds["weight"].fillna(ds["weight"].median()).values
            self.liks.append(f.Lik(ds, F, w / w.mean(), tied=False, use_rcount=True))

    def nll(self, x):
        q, lg, D, S, C = unpack(x, self.K, self.cf)
        tot = 0.0
        for g, lik in enumerate(self.liks):
            cls = class_params(q, lg, D, S, C, g)
            vals = [lik.values(qc) for _, qc in cls]
            for gi, grp in enumerate(lik.groups):
                v = sum(wk * vals[k][gi] for k, (wk, _) in enumerate(cls))
                tot -= (grp[4] * np.log(np.maximum(v, 1e-300))).sum()
        return tot if np.isfinite(tot) else 1e15


def x_from_rates(K, cf, spread=(0.6, 1.5)):
    """Initial vector from the fitted size model 'rates' (K=2 frailty)."""
    r = json.load(open(os.path.join(HERE, "build", "size_model_rates.json")))
    x = np.array(r["x"])
    base, pi_logit, d = x[:NB0], x[NB0], x[NB0 + 1:NB0 + 6]
    S = x[29:47]
    if K == 2:
        lg, D = [pi_logit], [d]
    else:
        lg = [pi_logit - np.log(1.3), pi_logit - np.log(1.3) - 1.5]
        D = [spread[0] * d, spread[1] * d]
    parts = [base, lg, np.ravel(D), S]
    if cf:
        parts.append(np.zeros((K - 1) * 5))
    return np.concatenate([np.atleast_1d(np.asarray(p, float)) for p in parts])


def cost_by_size(q, lg, D, S, C):
    res = []
    for g in range(4):
        tot = 0.0
        per = {ch: 0.0 for ch in f.CH}
        for wk, qc in class_params(q, lg, D, S, C, g):
            for ch in f.CH:
                c = wk * qc[f.RATE[ch]] * qc["p" + ch] * j.trunc_mean(qc["mu" + ch], qc["s" + ch])
                per[ch] += c
                tot += c
        res.append((tot, per))
    return res


def main():
    K = int(sys.argv[1])
    cf = "cf" in sys.argv
    tag = f"K{K}{'_cf' if cf else ''}"
    d = f.load_firms()
    F, _, _ = j.calibrate_freq(d)
    lik = Lik(d, F, K, cf)
    x0 = x_from_rates(K, cf)
    print(f"  {tag}: {len(x0)} params, nll at init {lik.nll(x0):.2f}", flush=True)

    best = None
    rng = np.random.default_rng(K + 10 * cf)
    for start in range(2):
        xs = x0 + (rng.normal(0, 0.2, len(x0)) if start else 0)
        res = minimize(lik.nll, xs, method="L-BFGS-B", options=dict(maxiter=1500, maxfun=300_000))
        print(f"  {tag} start {start}: -loglik {res.fun:.2f}  ({res.nit} it)", flush=True)
        if best is None or res.fun < best.fun:
            best = res

    q, lg, D, S, C = unpack(best.x, K, cf)
    print(f"\n{tag}: -loglik {best.fun:.2f}  params {len(best.x)}")
    for g in range(4):
        cls = class_params(q, lg, D, S, C, g)
        print(f"  size {g + 1} class weights: " + " ".join(f"{wk:.3f}" for wk, _ in cls))
    print("  class rate multipliers vs quiet (T M I R S):")
    for k in range(K):
        print(f"    class {k}: " + " ".join(f"{np.exp(v):8.1f}" for v in D[k]))
    if cf:
        print("  class cost-mu shifts vs quiet (T M I R S):")
        for k in range(K):
            print(f"    class {k}: " + " ".join(f"{v:6.2f}" for v in C[k]))
    nat = 0.0
    for g, (tot, per) in enumerate(cost_by_size(q, lg, D, S, C)):
        nat += sm.N_BY_SIZE[g + 1] * tot
        print(f"  size {g + 1}: £/firm {tot:8,.0f}  (" + " ".join(f"{k} {v:,.0f}" for k, v in per.items()) + ")")
    print(f"  national £{nat / 1e9:.2f}bn")
    print("  shared sigma: " + " ".join(f"{ch} {q['s' + ch]:.2f}" for ch in f.CH))
    json.dump({"nll": float(best.fun), "npar": len(best.x), "K": K, "cf": cf,
               "national_bn": nat / 1e9, "x": best.x.tolist()},
              open(os.path.join(HERE, "build", f"size_frailty_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
