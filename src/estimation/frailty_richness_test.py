"""Test binary frailty vs richer alternatives in the four-channel model.

Variants:
  A) Binary frailty (2 classes): current model, 23 params
  B) 3-class frailty (quiet / medium / hot): 23 + 5 = 28 params
     (extra pi_mid, extra 4 deltas for the middle class)
  C) Continuous (gamma) frailty: 23 - 4 + 1 = 20 params
     (single shared gamma shape α; each channel's rate multiplied by
      a common firm-level draw Z ~ Gamma(α, α) with E[Z]=1.
      Integrated analytically: NegBin marginal for Poisson channels,
      compound for NegBin channels.)

Actually C is hard to integrate analytically with the band structure.
So instead: continuous frailty via numerical quadrature over a small
number of mass points (Gauss-Hermite or discrete grid approximation
to a lognormal or gamma mixing distribution).

Simpler approach for C: K-class discrete approximation to a continuous
distribution, with K=5 or K=10 equi-probable classes. This nests
binary (K=2) and lets us see if more classes help.

Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src:src/estimation \
     python3 src/estimation/frailty_richness_test.py
"""

import json
import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

import joint_four_channel as j
import joint_four_channel_frailty as fr


# ---- 3-class frailty ----

def unpack3(x):
    """23 base + 5 (pi2, dT2, dM2, dI2, dS2) = 28 params total."""
    q = j.unpack(x[:18])
    sig = lambda v: float(1 / (1 + np.exp(-np.clip(v, -10, 10))))
    q["pi1"] = sig(x[18])        # P(quiet)... actually use softmax for 3 classes
    q["pi2"] = sig(x[19])
    # deltas for class 1 (middle)
    for i, ch in enumerate("TMIS"):
        q["d" + ch + "1"] = float(np.clip(x[20 + i], -6, 8))
    # deltas for class 2 (hot)
    for i, ch in enumerate("TMIS"):
        q["d" + ch] = float(np.clip(x[24 + i], -6, 8))
    return q


def class_weights_3(q):
    """Softmax-ish: pi1 = share of 'middle', pi2 = share of 'hot' among rest."""
    w_mid = q["pi1"]
    w_hot = (1 - q["pi1"]) * q["pi2"]
    w_quiet = 1 - w_mid - w_hot
    return max(w_quiet, 1e-6), max(w_mid, 1e-6), max(w_hot, 1e-6)


def class_params_3(q, cls):
    qc = dict(q)
    if cls == 1:  # middle
        for ch, key in fr.CHANNEL_RATE.items():
            qc[key] = q[key] * np.exp(q["d" + ch + "1"])
    elif cls == 2:  # hot
        for ch, key in fr.CHANNEL_RATE.items():
            qc[key] = q[key] * np.exp(q["d" + ch])
    return qc


class Frailty3Likelihood(fr.FrailtyLikelihood):
    def nll(self, x):
        q = unpack3(x)
        w0, w1, w2 = class_weights_3(q)
        v0 = self.firm_values(class_params_3(q, 0))
        v1 = self.firm_values(class_params_3(q, 1))
        v2 = self.firm_values(class_params_3(q, 2))
        total = 0.0
        for (key, fi, yi, li, w), a, b, c in zip(self.groups, v0, v1, v2):
            v = w0 * a + w1 * b + w2 * c
            total += (w * np.log(np.maximum(v, 1e-300))).sum()
        return -total if np.isfinite(total) else 1e15


def pack3(q):
    lg = lambda p: np.log(max(p, 1e-8) / max(1 - p, 1e-8))
    base = j.pack(q)
    return np.concatenate([base, [lg(q.get("pi1", 0.3)), lg(q.get("pi2", 0.3))],
                           [q.get("d" + ch + "1", 1.0) for ch in "TMIS"],
                           [q.get("d" + ch, 2.5) for ch in "TMIS"]])


# ---- K-class discrete frailty (approximation to continuous) ----

def unpack_K(x, K):
    """18 base + (K-1) log-weights + K*4 log-rate-multipliers.
    But that's a lot of params. Simpler: 18 base + 1 (log-variance of
    a lognormal mixing distribution) + 4 channel-specific loading factors.
    The K classes are fixed quantiles of LogNormal(0, σ²)."""
    q = j.unpack(x[:18])
    q["frailty_logsig"] = float(np.clip(x[18], -3, 3))
    for i, ch in enumerate("TMIS"):
        q["load_" + ch] = float(np.clip(x[19 + i], 0.01, 5))
    return q


def get_K_classes(q, K):
    """Return (weights, multipliers_per_channel) for K equi-probable classes."""
    sig = np.exp(q["frailty_logsig"])
    # quantiles of lognormal(0, sig^2) at midpoints of K equal-probability bins
    from scipy.stats import lognorm
    edges = np.linspace(0, 1, K + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    z_vals = lognorm.ppf(mids, sig)  # K values
    weights = np.ones(K) / K
    # per-channel multiplier = z^(loading)
    mults = {}
    for ch in "TMIS":
        load = q["load_" + ch]
        mults[ch] = z_vals ** load
    return weights, mults


def class_params_K(q, mults, k):
    qc = dict(q)
    for ch, key in fr.CHANNEL_RATE.items():
        qc[key] = q[key] * mults[ch][k]
    return qc


class FrailtyKLikelihood(fr.FrailtyLikelihood):
    def __init__(self, d, F, w, K=5):
        super().__init__(d, F, w)
        self.K = K

    def nll(self, x):
        q = unpack_K(x, self.K)
        weights, mults = get_K_classes(q, self.K)
        class_vals = [self.firm_values(class_params_K(q, mults, k)) for k in range(self.K)]
        total = 0.0
        for i, (key, fi, yi, li, w) in enumerate(self.groups):
            v = sum(weights[k] * class_vals[k][i] for k in range(self.K))
            total += (w * np.log(np.maximum(v, 1e-300))).sum()
        return -total if np.isfinite(total) else 1e15


def pack_K(q):
    base = j.pack(q)
    return np.concatenate([base, [q.get("frailty_logsig", 0.5)],
                           [q.get("load_" + ch, 1.0) for ch in "TMIS"]])


def fit(nll_fn, pack_fn, q0, label, n_starts=3, n_params=None):
    best = None
    rng = np.random.default_rng(42)
    x0 = pack_fn(q0)
    if n_params is None:
        n_params = len(x0)
    for start in range(n_starts):
        xs = x0 + (rng.normal(0, 0.3, len(x0)) if start > 0 else 0)
        res = minimize(nll_fn, xs, method="L-BFGS-B", options=dict(maxiter=500))
        res = minimize(nll_fn, res.x, method="Nelder-Mead",
                       options=dict(maxiter=8000, maxfev=8000, xatol=1e-4, fatol=1e-3))
        print(f"  {label} start {start}: -loglik {res.fun:.2f}")
        if best is None or res.fun < best.fun:
            best = res
    return best, n_params


def main():
    d = j.load_firms()
    F, _, _ = j.calibrate_freq(d)
    w = d["weight"].fillna(d["weight"].median()).values
    w = w / w.mean()

    here = os.path.dirname(os.path.abspath(__file__))
    q0 = json.load(open(os.path.join(here, "build", "joint_four_channel_frailty_weighted.json")))
    q0.pop("nll", None)

    # A) Binary (current)
    lik2 = fr.FrailtyLikelihood(d, F, w)
    best2, np2 = fit(lik2.nll, fr.pack, q0, "Binary(K=2)", n_starts=2, n_params=23)

    # B) 3-class
    lik3 = Frailty3Likelihood(d, F, w)
    q0_3 = dict(q0)
    q0_3["pi1"] = 0.3
    q0_3["pi2"] = 0.3
    for ch in "TMIS":
        q0_3["d" + ch + "1"] = q0["d" + ch] * 0.5
    best3, np3 = fit(lik3.nll, pack3, q0_3, "3-class", n_starts=3, n_params=28)

    # C) Continuous (K=5 lognormal quadrature)
    lik5 = FrailtyKLikelihood(d, F, w, K=5)
    q0_5 = dict(q0)
    q0_5["frailty_logsig"] = 0.5
    for ch in "TMIS":
        q0_5["load_" + ch] = 1.0
    best5, np5 = fit(lik5.nll, pack_K, q0_5, "Continuous(K=5)", n_starts=3, n_params=23)

    # D) Continuous (K=10)
    lik10 = FrailtyKLikelihood(d, F, w, K=10)
    best10, np10 = fit(lik10.nll, pack_K, q0_5, "Continuous(K=10)", n_starts=2, n_params=23)

    # Compare
    results = [("Binary (K=2)", best2, np2),
               ("3-class", best3, np3),
               ("Continuous (K=5)", best5, np5),
               ("Continuous (K=10)", best10, np10)]

    print("\n" + "=" * 70)
    print("FRAILTY RICHNESS COMPARISON")
    print("=" * 70)
    for label, res, np_ in results:
        print(f"  {label:25s}  -loglik = {res.fun:.2f}  ({np_} params)")

    # LR tests vs binary
    print(f"\n  LR tests vs binary (K=2, -loglik = {best2.fun:.2f}):")
    for label, res, np_ in results[1:]:
        if res.fun < best2.fun:
            lr = 2 * (best2.fun - res.fun)
            df = max(np_ - np2, 1)
            p = chi2.sf(lr, df)
            print(f"    {label:25s}  Δloglik = {best2.fun - res.fun:.2f}  "
                  f"LR = {lr:.1f}  df = {df}  p = {p:.2e}")
        else:
            print(f"    {label:25s}  WORSE by {res.fun - best2.fun:.2f}")

    # Cost comparison
    print(f"\n  Per-firm cost estimates:")
    for label, res, np_ in results:
        if "3-class" in label:
            q = unpack3(res.x)
            w0, w1, w2 = class_weights_3(q)
            total = 0
            for ch in "TMIS":
                rq = q[fr.CHANNEL_RATE[ch]]
                rates = [rq, rq * np.exp(q["d" + ch + "1"]), rq * np.exp(q["d" + ch])]
                ek = w0 * rates[0] + w1 * rates[1] + w2 * rates[2]
                p, mu, s = q["p" + ch], q["mu" + ch], q["s" + ch]
                total += ek * p * j.trunc_mean(mu, s)
        elif "Continuous" in label:
            K = 5 if "5" in label else 10
            q = unpack_K(res.x, K)
            weights, mults = get_K_classes(q, K)
            total = 0
            for ch in "TMIS":
                rq = q[fr.CHANNEL_RATE[ch]]
                ek = sum(weights[k] * rq * mults[ch][k] for k in range(K))
                p, mu, s = q["p" + ch], q["mu" + ch], q["s" + ch]
                total += ek * p * j.trunc_mean(mu, s)
        else:
            q = fr.unpack(res.x)
            total = 0
            for ch in "TMIS":
                rq = q[fr.CHANNEL_RATE[ch]]
                ek = (1 - q["pi"]) * rq + q["pi"] * rq * np.exp(q["d" + ch])
                p, mu, s = q["p" + ch], q["mu" + ch], q["s" + ch]
                total += ek * p * j.trunc_mean(mu, s)
        print(f"    {label:25s}  £{total:,.0f}/firm  -> £{total * 1_417_730 / 1e9:.2f}bn")


if __name__ == "__main__":
    main()
