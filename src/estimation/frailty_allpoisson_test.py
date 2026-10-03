"""Test whether frailty + all-Poisson is sufficient, or NegBin on M/S is needed.

Fits the frailty model in two variants on the pooled weighted sample:
  A) current: Poisson T/I, NegBin M/S  (18 + 5 = 23 params)
  B) all-Poisson: Poisson on all four   (16 + 5 = 21 params, rM/rS dropped)

If frailty absorbs the overdispersion that motivated NegBin, the likelihood
gap should be small relative to the 2 extra parameters.

Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src:src/estimation \
     python3 src/estimation/frailty_allpoisson_test.py
"""

import json
import os
import numpy as np
from scipy.optimize import minimize

import joint_four_channel as j
import joint_four_channel_frailty as fr


# --- All-Poisson variant: override pack/unpack to drop rM, rS ---

NAMES_AP = ["lamT", "pT", "muT", "sT", "lamM", "pM", "muM", "sM",
            "lamI", "pI", "muI", "sI", "lamS", "pS", "muS", "sS"]


def unpack_ap(x):
    e = lambda v, lo=-8, hi=8: float(np.clip(np.exp(v), np.exp(lo), np.exp(hi)))
    sig = lambda v: float(1 / (1 + np.exp(-np.clip(v, -10, 10))))
    mu = lambda v: float(np.clip(v, -2, 18))
    s = lambda v: float(np.clip(np.exp(v), 0.05, 8))
    q = dict(lamT=e(x[0]), pT=sig(x[1]), muT=mu(x[2]), sT=s(x[3]),
             mM=e(x[4], -6, 7), rM=1e6, pM=sig(x[5]), muM=mu(x[6]), sM=s(x[7]),
             lamI=e(x[8]), pI=sig(x[9]), muI=mu(x[10]), sI=s(x[11]),
             mS=e(x[12], -8, 6), rS=1e6, pS=sig(x[13]), muS=mu(x[14]), sS=s(x[15]))
    q["pi"] = sig(x[16])
    for i, ch in enumerate("TMIS"):
        q["d" + ch] = float(np.clip(x[17 + i], -6, 8))
    return q


def pack_ap(q):
    lg = lambda p: np.log(p / (1 - p))
    return np.array([np.log(q["lamT"]), lg(q["pT"]), q["muT"], np.log(q["sT"]),
                     np.log(q["mM"]), lg(q["pM"]), q["muM"], np.log(q["sM"]),
                     np.log(q["lamI"]), lg(q["pI"]), q["muI"], np.log(q["sI"]),
                     np.log(q["mS"]), lg(q["pS"]), q["muS"], np.log(q["sS"]),
                     lg(q["pi"])] + [q["d" + ch] for ch in "TMIS"])


class APFrailtyLikelihood(fr.FrailtyLikelihood):
    """Frailty likelihood with all-Poisson counts (rM, rS fixed at 1e6)."""
    def nll(self, x):
        q = unpack_ap(x)
        vq = self.firm_values(fr.class_params(q, False))
        ve = self.firm_values(fr.class_params(q, True))
        total = 0.0
        for (key, fi, yi, li, w), a, b in zip(self.groups, vq, ve):
            v = (1 - q["pi"]) * a + q["pi"] * b
            total += (w * np.log(np.maximum(v, 1e-300))).sum()
        return -total if np.isfinite(total) else 1e15


def fit_variant(lik, label, unpack_fn, pack_fn, nll_fn, q0, n_starts=3):
    best = None
    rng = np.random.default_rng(42)
    for start in range(n_starts):
        xs = pack_fn(q0) + (rng.normal(0, 0.3, len(pack_fn(q0))) if start > 0 else 0)
        res = minimize(nll_fn, xs, method="L-BFGS-B", options=dict(maxiter=500))
        res = minimize(nll_fn, res.x, method="Nelder-Mead",
                       options=dict(maxiter=8000, maxfev=8000, xatol=1e-4, fatol=1e-3))
        print(f"  {label} start {start}: -loglik {res.fun:.2f}")
        if best is None or res.fun < best.fun:
            best = res
    return best


def main():
    d = j.load_firms()
    F, _, _ = j.calibrate_freq(d)
    w = d["weight"].fillna(d["weight"].median()).values
    w = w / w.mean()

    # --- Variant A: current (NegBin M/S) ---
    lik_nb = fr.FrailtyLikelihood(d, F, w)
    here = os.path.dirname(os.path.abspath(__file__))
    q0_nb = json.load(open(os.path.join(here, "build", "joint_four_channel_frailty_weighted.json")))
    q0_nb.pop("nll", None)
    best_nb = fit_variant(lik_nb, "NegBin", fr.unpack, fr.pack, lik_nb.nll, q0_nb, n_starts=2)

    # --- Variant B: all-Poisson ---
    lik_ap = APFrailtyLikelihood(d, F, w)
    q0_ap = dict(q0_nb)
    q0_ap["rM"] = 1e6
    q0_ap["rS"] = 1e6
    best_ap = fit_variant(lik_ap, "AllPois", unpack_ap, pack_ap, lik_ap.nll, q0_ap, n_starts=3)

    # --- Compare ---
    q_nb = fr.unpack(best_nb.x)
    q_ap = unpack_ap(best_ap.x)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: frailty + NegBin(M,S) vs frailty + all-Poisson")
    print("=" * 70)
    print(f"  NegBin  -loglik = {best_nb.fun:.2f}  (23 params)")
    print(f"  AllPois -loglik = {best_ap.fun:.2f}  (21 params)")
    print(f"  Delta   = {best_ap.fun - best_nb.fun:.2f}  (2 extra params -> LR chi2(2) p < ...)")
    lr = 2 * (best_ap.fun - best_nb.fun)
    from scipy.stats import chi2
    pval = chi2.sf(lr, 2)
    print(f"  LR stat = {lr:.1f},  p = {pval:.2e}")
    print(f"\n  NegBin:  rM={q_nb['rM']:.3f}, rS={q_nb['rS']:.3f}")
    print(f"  (r→∞ = Poisson; small r = heavy overdispersion)")

    # cost comparison
    for label, q in [("NegBin", q_nb), ("AllPois", q_ap)]:
        total = 0
        for ch in "TMIS":
            rq = q[fr.CHANNEL_RATE[ch]]
            ek = (1 - q["pi"]) * rq + q["pi"] * rq * np.exp(q["d" + ch])
            p, mu, s = q["p" + ch], q["mu" + ch], q["s" + ch]
            total += ek * p * j.trunc_mean(mu, s)
        print(f"  {label} per-firm cost: £{total:,.0f}  -> national £{total * 1_417_730 / 1e9:.2f}bn")


if __name__ == "__main__":
    main()
