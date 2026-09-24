"""Joint four-channel fit with a binary firm-level frailty class.

Extends joint_four_channel.py. The independent-channel fit failed its moment checks:
attack types co-occur far more than independence allows (all three flags: obs 19% of
attacked vs 3% predicted). Here each firm is 'quiet' (1-pi) or 'exposed' (pi); each
channel's count rate differs by class (exposed rate = quiet rate * exp(delta_tau)).
Given the class, channels are independent. Success gates and cost distributions are
shared across classes. Likelihood = sum_c pi_c * L_c per firm.

Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src \
     python3 src/estimation/joint_four_channel_frailty.py [--weighted]
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import minimize

import joint_four_channel as j

CHANNEL_RATE = {"T": "lamT", "M": "mM", "I": "lamI", "S": "mS"}


def unpack(x):
    q = j.unpack(x[:18])
    q["pi"] = j.sig(x[18])
    for i, ch in enumerate("TMIS"):
        q["d" + ch] = float(np.clip(x[19 + i], -6, 8))
    return q


def pack(q):
    return np.concatenate([j.pack(q), [np.log(q["pi"] / (1 - q["pi"]))],
                           [q["d" + ch] for ch in "TMIS"]])


def class_params(q, exposed):
    qc = dict(q)
    if exposed:
        for ch, key in CHANNEL_RATE.items():
            qc[key] = q[key] * np.exp(q["d" + ch])
    return qc


class FrailtyLikelihood(j.Likelihood):
    def firm_values(self, q):
        P, SI, SS = j.tables(q)
        X = {(fi, fs): j.combine(j.restrict(SI, fi), j.restrict(SS, fs))
             for fi in (0, 1) for fs in (0, 1)}
        Fext = np.vstack([self.F.T, np.ones(j.NB)])
        out = []
        for (fP, fI, fS, nb), fi, yi, li, w in self.groups:
            if nb >= 0:
                Pr = np.zeros_like(P)
                Pr[nb] = P[nb] / j.WIDTH[nb]
            else:
                Pr = j.restrict(P, fP)
            C = j.combine(X[(fI, fS)], Pr)
            Q = np.einsum("fc,cyl->fyl", Fext, C)
            Q = np.concatenate([Q, Q.sum(1, keepdims=True)], 1)
            Q = np.concatenate([Q, Q.sum(2, keepdims=True)], 2)
            out.append(Q[fi, yi, li])
        return out

    def nll(self, x):
        q = unpack(x)
        vq = self.firm_values(class_params(q, False))
        ve = self.firm_values(class_params(q, True))
        total = 0.0
        for (key, fi, yi, li, w), a, b in zip(self.groups, vq, ve):
            v = (1 - q["pi"]) * a + q["pi"] * b
            total += (w * np.log(np.maximum(v, 1e-300))).sum()
        return -total if np.isfinite(total) else 1e15


def simulate(q, F, n, rng):
    n_e = rng.binomial(n, q["pi"])
    sims = [j.simulate(class_params(q, False), F, n - n_e, rng),
            j.simulate(class_params(q, True), F, n_e, rng)]
    sims[0]["cls"], sims[1]["cls"] = 0, 1
    return j.pd.concat(sims, ignore_index=True)


def report(q, d, F, label, nll):
    print("\n" + "=" * 78)
    print(f"FRAILTY FIT: {label}   -loglik = {nll:.2f}   firms = {len(d)}   pi(exposed) = {q['pi']:.3f}")
    print("=" * 78)
    rows = [("targeted T", "T", "Poisson", q["pT"], q["muT"], q["sT"]),
            ("mass M", "M", f"NegBin r={q['rM']:.3f}", q["pM"], q["muM"], q["sM"]),
            ("imperson. I", "I", "Poisson", q["pI"], q["muI"], q["sI"]),
            ("serious S", "S", f"NegBin r={q['rS']:.3f}", q["pS"], q["muS"], q["sS"])]
    total = 0
    print(f"  {'channel':>12s} {'rate quiet':>10s} {'rate exp.':>10s} {'E[K]/firm':>10s} {'count law':>16s}"
          f" {'p':>7s} {'E[C|succ]':>10s} {'E[cost/firm]':>13s}")
    for name, ch, law, p, mu, s in rows:
        rq = q[CHANNEL_RATE[ch]]
        re_ = rq * np.exp(q["d" + ch])
        ek = (1 - q["pi"]) * rq + q["pi"] * re_
        ec = j.trunc_mean(mu, s)
        total += ek * p * ec
        print(f"  {name:>12s} {rq:10.3f} {re_:10.3f} {ek:10.3f} {law:>16s} {p:7.4f} £{ec:9,.0f}"
              f" £{ek * p * ec:12,.1f}")
    print(f"  {'TOTAL':>12s}{'':>72s} £{total:12,.1f}  per firm (sample population)")
    print(f"  crude national (x 1,417,730 employer businesses, pooled sizes): £{total * 1_417_730 / 1e9:.2f}bn")
    j.moment_checks(d, simulate(q, F, 300_000, np.random.default_rng(1)))
    return total


def main():
    weighted = "--weighted" in sys.argv
    d = j.load_firms()
    F, _, _ = j.calibrate_freq(d)
    w = d["weight"].fillna(d["weight"].median()).values if weighted else np.ones(len(d))
    w = w / w.mean()
    lik = FrailtyLikelihood(d, F, w)

    base = os.path.join(os.path.dirname(__file__), "build",
                        f"joint_four_channel_{'weighted' if weighted else 'unweighted'}.json")
    q0 = {k: v for k, v in json.load(open(base)).items() if k != "nll"}
    best = None
    rng = np.random.default_rng(0)
    for start in range(4):
        qs = dict(q0)
        qs.update(pi=[0.3, 0.15, 0.5, 0.3][start])
        for ch, key in CHANNEL_RATE.items():
            qs[key] = q0[key] * 0.4
            qs["d" + ch] = [2.0, 3.0, 1.5, 2.5][start]
        xs = pack(qs) + (rng.normal(0, 0.3, 23) if start == 3 else 0)
        res = minimize(lik.nll, xs, method="L-BFGS-B", options=dict(maxiter=500))
        res = minimize(lik.nll, res.x, method="Nelder-Mead",
                       options=dict(maxiter=8000, maxfev=8000, xatol=1e-4, fatol=1e-3))
        print(f"  start {start}: -loglik {res.fun:.2f}")
        if best is None or res.fun < best.fun:
            best = res
    q = unpack(best.x)
    base_nll = json.load(open(base))["nll"]
    print(f"\nLR vs independent-channel fit: -loglik {base_nll:.2f} -> {best.fun:.2f} "
          f"(gain {base_nll - best.fun:.1f} for 5 extra params)")
    report(q, d, F, "survey-weighted" if weighted else "unweighted", best.fun)
    out = base.replace("joint_four_channel_", "joint_four_channel_frailty_")
    with open(out, "w") as fh:
        json.dump({k: float(v) for k, v in q.items()} | {"nll": float(best.fun)}, fh, indent=1)
    print(f"\nparams saved to {out}")


if __name__ == "__main__":
    main()
