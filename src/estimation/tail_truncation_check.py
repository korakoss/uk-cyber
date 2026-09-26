"""How much does capping per-attack cost at £500k matter in the size-split frailty fits?

Both the likelihood (band_cdf renormalises the lognormal onto bands 2..10) and the
estimate (trunc_mean cap=£500k) cut the fitted lognormal at £500k. This recomputes the
national total from the fitted parameters with the uncapped lognormal mean, and reports
P(cost > £500k | success) per channel. First-order only: parameters are not refitted.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/tail_truncation_check.py
"""

import json
import os
import numpy as np
from scipy.stats import norm

import joint_four_channel as j
import joint_frailty_by_size as bs

HERE = os.path.dirname(os.path.abspath(__file__))
CH = {"T": ("pT", "muT", "sT"), "M": ("pM", "muM", "sM"),
      "I": ("pI", "muI", "sI"), "S": ("pS", "muS", "sS")}


def main():
    for split in (["micro", "rest"], ["micro", "small", "medium", "large"]):
        tot_c, tot_u = 0.0, 0.0
        print("=" * 78)
        print(f"SPLIT: {' + '.join(split)}")
        print("=" * 78)
        for g in split:
            q = json.load(open(os.path.join(HERE, "build", f"joint_frailty_{g}_weighted.json")))
            n_pop = q["n_pop"]
            gc, gu = 0.0, 0.0
            for ch, (p, mu, s) in CH.items():
                rq = q[{"T": "lamT", "M": "mM", "I": "lamI", "S": "mS"}[ch]]
                ek = (1 - q["pi"]) * rq + q["pi"] * rq * np.exp(q["d" + ch])
                capped = j.trunc_mean(q[mu], q[s], cap=500_000)
                uncapped = j.trunc_mean(q[mu], q[s])
                p_over = norm.sf((np.log(500_000) - q[mu]) / q[s])
                gc += ek * q[p] * capped
                gu += ek * q[p] * uncapped
                print(f"  {g:>6s} {ch}: mu={q[mu]:5.2f} sigma={q[s]:4.2f}  P(>£500k|succ)={p_over:.5f}  "
                      f"E[C|succ] capped £{capped:9,.0f}  uncapped £{uncapped:11,.0f}")
            print(f"  {g:>6s} per firm: capped £{gc:,.0f}  uncapped £{gu:,.0f}  "
                  f"-> national £{n_pop * gc / 1e9:.3f}bn vs £{n_pop * gu / 1e9:.3f}bn")
            tot_c += n_pop * gc
            tot_u += n_pop * gu
        print(f"\n  NATIONAL: capped £{tot_c / 1e9:.3f}bn   uncapped £{tot_u / 1e9:.3f}bn   "
              f"(x{tot_u / tot_c:.2f})\n")


if __name__ == "__main__":
    main()
