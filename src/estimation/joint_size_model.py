"""Which parameters need to vary by firm size? Joint five-channel model with size shifts.

Five-channel binary-frailty model with ransomware count (joint_five_channel.py), fitted over all
firms with size group g in {Micro, Small, Medium, Large}. Micro carries the base parameters;
Small/Medium/Large get additive shifts on selected blocks:
  rates : log attack rate per channel (5) + logit exposed share pi (1)      -> 18 params
  gates : logit success probability per channel (5)                           -> 15 params
  mu    : lognormal cost location per channel (5)                            -> 15 params
Sigma, NegBin shapes and frailty deltas are always shared. 'sep' fits the full 29-param model per
size group independently (everything varies).

Weights are normalised to mean 1 WITHIN each size group: the model conditions on size and
national totals scale by ONS counts per size, so weights only need to correct within size.

Models: pool | rates | rates_gates | rates_mu | sep:<size 1-4>
Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src:src/estimation \
     python3 src/estimation/joint_size_model.py MODEL
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import minimize

import joint_four_channel as j
import joint_five_channel as f

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [1, 2, 3, 4]
N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}
BLOCKS = {"pool": [], "rates": ["rates"], "rates_gates": ["rates", "gates"],
          "rates_mu": ["rates", "mu"]}
NBASE = 29


def logit(p):
    return np.log(p / (1 - p))


def size_q(q, sh, g):
    """Parameters for size index g (0 = Micro) given base q and shift dict sh."""
    if g == 0:
        return q
    qc = dict(q)
    if "rates" in sh:
        for c, ch in enumerate(f.CH):
            qc[f.RATE[ch]] = q[f.RATE[ch]] * np.exp(sh["rates"][g - 1, c])
        qc["pi"] = float(j.sig(logit(q["pi"]) + sh["rates"][g - 1, 5]))
    if "gates" in sh:
        for c, ch in enumerate(f.CH):
            qc["p" + ch] = float(j.sig(logit(q["p" + ch]) + sh["gates"][g - 1, c]))
    if "mu" in sh:
        for c, ch in enumerate(f.CH):
            qc["mu" + ch] = q["mu" + ch] + sh["mu"][g - 1, c]
    return qc


def unpack(x, blocks):
    q = f.unpack(x[:NBASE], False)
    sh, i = {}, NBASE
    for b in blocks:
        k = 6 if b == "rates" else 5
        sh[b] = np.clip(x[i:i + 3 * k].reshape(3, k), -8, 8)
        i += 3 * k
    return q, sh


class SizeLik:
    def __init__(self, d, F, blocks):
        self.blocks = blocks
        self.liks = []
        for s in SIZES:
            ds = d[d["sizeb"] == s].reset_index(drop=True)
            w = ds["weight"].fillna(ds["weight"].median()).values
            self.liks.append(f.Lik(ds, F, w / w.mean(), tied=False, use_rcount=True))

    def nll(self, x):
        q, sh = unpack(x, self.blocks)
        tot = 0.0
        for g, lik in enumerate(self.liks):
            tot += lik.nll(f.pack(size_q(q, sh, g), False))
        return tot


def cost_per_firm(q):
    out = {}
    for ch in f.CH:
        rq = q[f.RATE[ch]]
        ek = (1 - q["pi"]) * rq + q["pi"] * rq * np.exp(q["d" + ch])
        out[ch] = ek * q["p" + ch] * j.trunc_mean(q["mu" + ch], q["s" + ch])
    return out


def report(qs, nll, npar, tag):
    print(f"\n{tag}: -loglik {nll:.2f}  params {npar}")
    tot = 0.0
    for g, s in enumerate(SIZES):
        c = cost_per_firm(qs[g])
        cf = sum(c.values())
        tot += N_BY_SIZE[s] * cf
        print(f"  size {s}: pi {qs[g]['pi']:.2f}  £/firm {cf:8,.0f}  (" +
              " ".join(f"{k} {v:,.0f}" for k, v in c.items()) + ")")
    print(f"  national £{tot / 1e9:.2f}bn")
    return tot


def main():
    model = sys.argv[1]
    d = f.load_firms()
    F, _, _ = j.calibrate_freq(d)
    q0 = json.load(open(os.path.join(HERE, "build", "joint_five_channel_free_rcount_weighted.json")))
    x_base = f.pack(q0, False)
    out = os.path.join(HERE, "build", f"size_model_{model.replace(':', '')}.json")

    if model.startswith("sep"):
        s = int(model.split(":")[1])
        ds = d[d["sizeb"] == s].reset_index(drop=True)
        w = ds["weight"].fillna(ds["weight"].median()).values
        lik = f.Lik(ds, F, w / w.mean(), tied=False, use_rcount=True)
        nll_fn, x0, npar = lik.nll, x_base, NBASE
    else:
        blocks = BLOCKS[model]
        lik = SizeLik(d, F, blocks)
        nll_fn = lik.nll
        x0 = np.concatenate([x_base] + [np.zeros(18 if b == "rates" else 15) for b in blocks])
        npar = len(x0)

    best = None
    rng = np.random.default_rng(1)
    for start in range(2):
        xs = x0 + (rng.normal(0, 0.2, len(x0)) if start else 0)
        res = minimize(nll_fn, xs, method="L-BFGS-B", options=dict(maxiter=1500, maxfun=200_000))
        print(f"  {model} start {start}: -loglik {res.fun:.2f}  ({res.nit} it, {res.message})", flush=True)
        if best is None or res.fun < best.fun:
            best = res

    if model.startswith("sep"):
        q = f.unpack(best.x, False)
        c = cost_per_firm(q)
        print(f"\n{model}: -loglik {best.fun:.2f}  params {npar}  pi {q['pi']:.2f}  £/firm {sum(c.values()):,.0f}  (" +
              " ".join(f"{k} {v:,.0f}" for k, v in c.items()) + ")")
        json.dump({"nll": float(best.fun), "npar": npar, "cost_per_firm": float(sum(c.values())),
                   **{k: float(v) for k, v in q.items()}, **{f"cost_{k}": float(v) for k, v in c.items()}},
                  open(out, "w"), indent=1)
    else:
        q, sh = unpack(best.x, blocks)
        qs = [size_q(q, sh, g) for g in range(4)]
        nat = report(qs, best.fun, npar, model)
        for b, m in sh.items():
            print(f"  shifts[{b}] (rows Small/Medium/Large; cols {'/'.join(f.CH)}{'/pi' if b == 'rates' else ''}):")
            for r in m:
                print("    " + " ".join(f"{v:6.2f}" for v in r))
        json.dump({"nll": float(best.fun), "npar": npar, "national_bn": nat / 1e9,
                   "x": best.x.tolist(), "blocks": blocks}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
