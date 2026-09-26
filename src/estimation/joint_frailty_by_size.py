"""Frailty joint four-channel model fitted separately per firm-size group.

Pooled fits gave £2.34bn unweighted vs £0.86bn survey-weighted: the size mix of the
sample dominates the headline. Here each size group gets its own fit (survey-weighted
within group, so e.g. Small/Medium/Large are mixed in their population proportions
inside 'Rest'), and national = sum_g N_g * E[cost per firm | g].

The freq measurement model is calibrated once on the pooled sample.
Warm-started from the pooled frailty fit.

Run: OMP_NUM_THREADS=1 PYTHONPATH=/home/user/md-clean/src:src/estimation \
     python3 src/estimation/joint_frailty_by_size.py GROUP [--unweighted]
     GROUP in {micro, rest, small, medium, large}
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import minimize

import joint_four_channel as j
import joint_four_channel_frailty as fr

GROUPS = {"micro": [1], "rest": [2, 3, 4], "small": [2], "medium": [3], "large": [4]}
N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}


def per_firm_cost(q):
    out = {}
    for ch, (p, mu, s) in {"T": ("pT", "muT", "sT"), "M": ("pM", "muM", "sM"),
                           "I": ("pI", "muI", "sI"), "S": ("pS", "muS", "sS")}.items():
        rq = q[fr.CHANNEL_RATE[ch]]
        ek = (1 - q["pi"]) * rq + q["pi"] * rq * np.exp(q["d" + ch])
        out[ch] = ek * q[p] * j.trunc_mean(q[mu], q[s])
    return out


def weighted_checks(q, d, F, w):
    sim = fr.simulate(q, F, 300_000, np.random.default_rng(1))
    att = d["att"] == 1
    print("\nWEIGHTED CHECKS (observed survey-weighted vs simulated)")
    print(f"  prevalence: obs {np.average(att, weights=w):.3f}  "
          f"sim {(sim[['fP', 'fI', 'fS']].sum(axis=1) > 0).mean():.3f}")
    wa = w[att.values]
    fa = d.loc[att, ["fP", "fI", "fS"]].values
    sa = sim[sim[["fP", "fI", "fS"]].sum(axis=1) > 0][["fP", "fI", "fS"]].values
    for pat in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]:
        o = np.average((fa == pat).all(1), weights=wa)
        s = (sa == pat).all(1).mean()
        print(f"    {pat}: obs {o:.3f}  sim {s:.3f}")
    b = d.loc[att, "band"]
    ok = b.notna().values
    ob = np.array([np.average(b.values[ok] == k, weights=wa[ok]) for k in range(1, 11)])
    sb = sim.loc[sim["M"] > 0, "M"].value_counts(normalize=True).reindex(range(1, 11), fill_value=0).values
    print("  worst band | attacked: obs " + " ".join(f"{v:.3f}" for v in ob))
    print("                         sim " + " ".join(f"{v:.3f}" for v in sb))


def main():
    group = sys.argv[1]
    weighted = "--unweighted" not in sys.argv
    d_all = j.load_firms()
    F, _, _ = j.calibrate_freq(d_all)
    d = d_all[d_all["sizeb"].isin(GROUPS[group])].reset_index(drop=True)
    w = d["weight"].fillna(d["weight"].median()).values if weighted else np.ones(len(d))
    w = w / w.mean()
    lik = fr.FrailtyLikelihood(d, F, w)

    here = os.path.dirname(os.path.abspath(__file__))
    q0 = json.load(open(os.path.join(here, "build", "joint_four_channel_frailty_weighted.json")))
    q0.pop("nll")
    x0 = fr.pack(q0)
    best = None
    rng = np.random.default_rng(0)
    for start in range(2):
        xs = x0 if start == 0 else x0 + rng.normal(0, 0.4, len(x0))
        res = minimize(lik.nll, xs, method="L-BFGS-B", options=dict(maxiter=500))
        res = minimize(lik.nll, res.x, method="Nelder-Mead",
                       options=dict(maxiter=8000, maxfev=8000, xatol=1e-4, fatol=1e-3))
        print(f"  start {start}: -loglik {res.fun:.2f}")
        if best is None or res.fun < best.fun:
            best = res
    q = fr.unpack(best.x)
    n_att = int((d["att"] == 1).sum())
    print(f"\nGROUP {group} (sizes {GROUPS[group]}): firms={len(d)}, attacked={n_att}, "
          f"weighted={weighted}")
    fr.report(q, d, F, f"{group} {'weighted' if weighted else 'unweighted'}", best.fun)
    if weighted:
        weighted_checks(q, d, F, w)

    costs = per_firm_cost(q)
    total = sum(costs.values())
    n_pop = sum(N_BY_SIZE[s] for s in GROUPS[group])
    print(f"\nPER FIRM by channel: " + ", ".join(f"{k} £{v:,.0f}" for k, v in costs.items()) +
          f"  | total £{total:,.0f}")
    print(f"NATIONAL CONTRIBUTION: {n_pop:,} firms x £{total:,.0f} = £{n_pop * total / 1e9:.3f}bn")
    out = os.path.join(here, "build", f"joint_frailty_{group}_{'weighted' if weighted else 'unweighted'}.json")
    with open(out, "w") as fh:
        json.dump({k: float(v) for k, v in q.items()} |
                  {"nll": float(best.fun), "cost_per_firm": float(total), "n_pop": n_pop,
                   "national_bn": float(n_pop * total / 1e9)} |
                  {f"cost_{k}": float(v) for k, v in costs.items()}, fh, indent=1)


if __name__ == "__main__":
    main()
