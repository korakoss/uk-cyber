"""Do breached firms have multiple material episodes? Poisson-within-cell test.

Model. Per hit channel c, material episodes M_c ~ Poisson(lam_c * exp(b_tier[t] + b_size*Small+)); firm-level
Lam = sum over hit channels. Outcome flag (Q56A any outcome) = M >= 1, so P(outcome) = 1 - exp(-Lam).
Step 1: fit lam_c, b_tier (mid, high vs low), b_size on narrow firms (1-2 channels) from the outcome flag. Weighted.
Step 2: among narrow outcome firms, worst-incident cost = max over the M episodes (handling ignored: material dominates).
  Per-episode cost G: zero mass m0, else lognormal(mu_c, sigma); episode channel drawn in proportion to lam_c.
  M | M >= 1 ~ zero-truncated Poisson(kappa * Lam):  P(max <= x) = (exp(k Lam F(x)) - 1) / (exp(k Lam) - 1),
  F = sum_c (lam_c / Lam) F_c.  kappa -> 0 = exactly one episode; kappa = 1 = Poisson-within-cell.
  Profile kappa on a grid (G refitted at each kappa); compare loglik. Tightened worst-incident intervals.
Step 3: raw gradient: worst cost among outcome firms by implied E[M | M>=1] tercile, observed vs predicted (kappa = 0, 1).

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/breach_repeats.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import narrow_cost_model as ncm
from counts_given_frailty import tier_posteriors

CH = "PIRS"


def setup():
    idx, hits, lo, hi, oa, sm, w = ncm.setup()
    tier = tier_posteriors().argmax(1)
    H = np.zeros((len(oa), 4))
    for i, h in zip(idx, hits):
        H[i, h] = 1
    return idx, H, lo, hi, oa, sm, w, tier


def rates(p, H, sm, tier):
    lam = np.exp(p[:4])
    bt = np.r_[0.0, p[4:6]]
    return H * lam[None] * np.exp(bt[tier] + p[6] * sm)[:, None]


def fit_rates(idx, H, oa, sm, w, tier):
    def nll(p):
        L = rates(p, H[idx], sm[idx], tier[idx]).sum(1)
        pr = np.clip(1 - np.exp(-L), 1e-12, 1 - 1e-12)
        y = oa[idx]
        return -(w[idx] * (y * np.log(pr) + (1 - y) * np.log(1 - pr))).sum()
    o = minimize(nll, np.r_[np.full(4, -2.0), 0, 0, 0], method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
    return o.x


def max_cdf(x, lam_i, g, kappa):
    """P(worst <= x) for one firm: lam_i (4,) per-channel episode rates; g = (m0, mu(4), sigma)."""
    m0, mu, s = g
    L = lam_i.sum()
    if x <= 0:
        Fc = np.full(4, m0)
    else:
        Fc = m0 + (1 - m0) * norm.cdf((np.log(x) - mu) / s)
    F = (lam_i / L) @ Fc
    if kappa < 1e-6:
        return F
    kL = kappa * L
    return np.expm1(kL * F) / np.expm1(kL)


def unpack_g(q):
    return 1 / (1 + np.exp(-q[0])), q[1:5], np.exp(q[5])


def fit_cost(ids, LAM, lo, hi, w, kappa, q0):
    def nll(q):
        g = unpack_g(q)
        s = 0.0
        for i in ids:
            if hi[i] == 0:
                pr = max_cdf(0, LAM[i], g, kappa)
            else:
                pr = max_cdf(hi[i], LAM[i], g, kappa) - max_cdf(lo[i], LAM[i], g, kappa)
            s -= w[i] * np.log(max(pr, 1e-300))
        return s
    o = minimize(nll, q0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-5, "fatol": 1e-7})
    return o.x, -o.fun


def main():
    idx, H, lo, hi, oa, sm, w, tier = setup()
    p = fit_rates(idx, H, oa, sm, w, tier)
    print("step 1: episode rates per hit (low tier, Micro): " + ", ".join(f"{c} {v:.3f}" for c, v in zip(CH, np.exp(p[:4]))) +
          f";  tier mid x{np.exp(p[4]):.2f}, high x{np.exp(p[5]):.2f};  Small+ x{np.exp(p[6]):.2f}")
    LAM = np.zeros_like(H)
    LAM[idx] = rates(p, H[idx], sm[idx], tier[idx])
    L = LAM.sum(1)
    ids = idx[oa[idx] == 1]
    em = L[ids] / -np.expm1(-L[ids])
    print(f"outcome firms {len(ids)};  implied Lam: median {np.median(L[ids]):.2f}, max {L[ids].max():.2f};  "
          f"E[M|M>=1] (kappa=1): weighted mean {np.average(em, weights=w[ids]):.3f}, range {em.min():.2f}-{em.max():.2f}")
    print("  by tier (all narrow firms): " + ", ".join(
        f"{t} P(outcome) obs {np.average(oa[idx][tier[idx] == k], weights=w[idx][tier[idx] == k]):.3f} "
        f"pred {np.average(-np.expm1(-L[idx][tier[idx] == k]), weights=w[idx][tier[idx] == k]):.3f}"
        for k, t in enumerate(("low", "mid", "high"))))

    print("\nstep 2: kappa profile (G refitted each time)")
    q = np.r_[-1.0, np.full(4, 7.0), 0.6]
    res = {}
    for kappa in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        q, ll = fit_cost(ids, LAM, lo, hi, w, kappa, q)
        res[kappa] = (q, ll)
        g = unpack_g(q)
        print(f"  kappa {kappa:4.1f}  loglik {ll:8.2f}   m0 {g[0]:.2f}  medians " +
              " ".join(f"{c} £{np.exp(m):,.0f}" for c, m in zip(CH, g[1])) + f"  sigma {g[2]:.2f}")

    print("\nstep 3: worst cost among outcome firms by implied E[M|M>=1] tercile (weighted); obs vs pred kappa 0 / 1")
    cut = np.quantile(em, [1 / 3, 2 / 3])
    grp = np.digitize(em, cut)
    for k in range(3):
        sel = ids[grp == k]
        ws = w[sel]
        mid = np.where(hi[sel] == 0, 0, (lo[sel] + hi[sel]) / 2)
        line = f"  tercile {k} n {len(sel):3d}  E[M|M>=1] {em[grp == k].mean():.2f}  obs P(>=£500) {np.average(mid >= 500, weights=ws):.2f}" \
               f" P(>=£5k) {np.average(mid >= 5000, weights=ws):.2f}"
        for kappa in (0.0, 1.0):
            g = unpack_g(res[kappa][0])
            p5 = np.average([1 - max_cdf(500, LAM[i], g, kappa) for i in sel], weights=ws)
            p5k = np.average([1 - max_cdf(5000, LAM[i], g, kappa) for i in sel], weights=ws)
            line += f" | k{kappa:.0f}: {p5:.2f} {p5k:.2f}"
        print(line)

    print("\nstep 4: net expected material cost per breached firm = E[M|M>=1] x E[G] (weighted over outcome firms), by kappa")
    for kappa, (q, ll) in res.items():
        m0, mu, sg = unpack_g(q)
        eg = (LAM[ids] / L[ids][:, None]) @ ((1 - m0) * np.exp(mu + sg ** 2 / 2))
        kl = kappa * L[ids]
        emk = np.ones_like(kl) if kappa == 0 else kl / -np.expm1(-kl)
        print(f"  kappa {kappa:4.1f}  E[M|M>=1] {np.average(emk, weights=w[ids]):.2f}  E[G] £{np.average(eg, weights=w[ids]):,.0f}"
              f"  product £{np.average(emk * eg, weights=w[ids]):,.0f}  (loglik {ll:.2f})")


if __name__ == "__main__":
    main()
