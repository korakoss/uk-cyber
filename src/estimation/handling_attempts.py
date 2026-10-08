"""Attempts and clean-up only, on attacked firms WITHOUT a breach (W2 marker = 0): worst incident = largest clean-up.

Attempts: total attacks N in the year, constrained by the Q54 frequency band (all types together); within a band N is
integrated over three log-spaced points (once: 1; <monthly: 2,4,8; monthly: 8,12,18; weekly: 30,52,90; daily:
150,250,400; several a day: 500,1000,2500; missing -> <monthly), equal weights.
Per attempt: zero cost w.p. h0, else lognormal(mu + shifts for impersonation / ransomware / other-serious hit + Small+
 - beta log N, sigma).  Firm-level propensity: a share psi of firms has EVERY attempt free (always-free firms).
  P(worst <= x | N) = psi + (1 - psi) * F_att(x)^N,   F_att(0) = h0.
Hypotheses (all fitted by weighted ML on tightened intervals):
  H1 independent attempts, volume thinning: psi = 0, h0 = sigmoid(a + gam log N), beta = 0
  H2 firm heterogeneity: psi = sigmoid(p0 + p1 Small+), h0 constant, beta = 0
  H3 both: psi free, h0 = sigmoid(a + gam log N)
  H2b: H2 with beta free (cheaper attempts at high volume)
Implied yearly clean-up per firm = (1 - psi) * E[N * (1 - h0) * exp(mu - beta log N + s^2/2)]; national total for attacked
employer firms (non-breached firms' parameters applied to all attacked firms; ONS N; weighted as national_estimate).
Fit check: no-cost / <£500 / £500+ shares by frequency band, observed vs predicted.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/handling_attempts.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import joint_five_channel as f
import national_estimate as ne

PTS = {1: [1], 2: [2, 4, 8], 3: [8, 12, 18], 4: [30, 52, 90], 5: [150, 250, 400], 6: [500, 1000, 2500]}
sig = lambda z: 1 / (1 + np.exp(-z))


def build():
    pop = ne.population()
    fq = f.load_firms()["freq"].values
    band = np.where(np.isin(fq, [1, 2, 3, 4, 5, 6]), fq, 2).astype(int)
    br = (pop["oa"] == 1) | pop["soft"]
    m = pop["have"] & ~br
    idx = np.where(m)[0]
    G = np.zeros((len(pop["w"]), 3))
    for i in range(len(G)):
        p = PTS[band[i]]
        G[i, :len(p)] = p
        G[i, len(p):] = p[-1]          # pad 'once' with repeats of 1
    return pop, band, idx, G


def unpack(p, h):
    P = dict(mu=p[0], dI=p[1], dR=p[2], dS=p[3], dsm=p[4], s=np.exp(p[5]), a=p[6], gam=0.0, p0=-30.0, p1=0.0, beta=0.0)
    if h in ("H1", "H3"):
        P["gam"] = p[7]
    if h in ("H2", "H2b", "H3"):
        P["p0"], P["p1"] = p[8], p[9]
    if h == "H2b":
        P["beta"] = p[10]
    return P


def lik_terms(P, pop, rows, G):
    C, sm = pop["C"][rows], pop["sm"][rows]
    mu = P["mu"] + P["dI"] * C[:, 1] + P["dR"] * C[:, 2] + P["dS"] * C[:, 3] + P["dsm"] * sm
    psi = sig(P["p0"] + P["p1"] * sm)
    return mu, psi


def cdf(x, P, mu, psi, Gr):
    """P(worst <= x) per firm, averaging over the N points (Gr: (n,3))."""
    out = 0.0
    for j in range(3):
        n = Gr[:, j]
        h0 = sig(P["a"] + P["gam"] * np.log(n))
        if np.isscalar(x) or x.ndim == 0:
            xx = np.full(len(n), float(x))
        else:
            xx = x
        Fa = h0 + (1 - h0) * np.where(xx > 0, norm.cdf((np.log(np.maximum(xx, 1e-12)) - mu + P["beta"] * np.log(n)) / P["s"]), 0.0)
        out = out + (psi + (1 - psi) * np.exp(n * np.log(np.maximum(Fa, 1e-300)))) / 3
    return out


def nll(p, h, pop, idx, G):
    P = unpack(p, h)
    mu, psi = lik_terms(P, pop, idx, G)
    lo, hi = pop["lo"][idx], pop["hi"][idx]
    up = cdf(hi, P, mu, psi, G[idx])
    dn = cdf(lo, P, mu, psi, G[idx])
    pr = np.where(hi == 0, up, up - dn)
    w = pop["w"][idx] / pop["w"][idx].mean()
    return -(w * np.log(np.maximum(pr, 1e-300))).sum()


def fit(h, pop, idx, G, p0):
    free = {"H1": [0, 1, 2, 3, 4, 5, 6, 7], "H2": [0, 1, 2, 3, 4, 5, 6, 8, 9], "H2b": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10],
            "H3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}[h]
    full = p0.copy()

    def g(q):
        full[free] = q
        return nll(full, h, pop, idx, G)
    o = minimize(g, p0[free], method="L-BFGS-B", options={"maxiter": 20000})
    o = minimize(g, o.x, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
    o = minimize(g, o.x, method="L-BFGS-B", options={"maxiter": 20000})
    full[free] = o.x
    return full, -o.fun, len(free)


def yearly(P, pop, rows, G):
    mu, psi = lik_terms(P, pop, rows, G)
    tot = 0.0
    for j in range(3):
        n = G[rows, j]
        h0 = sig(P["a"] + P["gam"] * np.log(n))
        tot = tot + n * (1 - h0) * np.exp(mu - P["beta"] * np.log(n) + P["s"] ** 2 / 2) / 3
    return (1 - psi) * tot


def main():
    pop, band, idx, G = build()
    print(f"non-breached attacked firms with cost data {len(idx)}; frequency missing set to <monthly: "
          f"{int((~np.isin(f.load_firms()['freq'].values[idx], [1, 2, 3, 4, 5, 6])).sum())}")
    p0 = np.array([np.log(80), 0.0, 0.5, 0.5, 0.0, np.log(1.8), 0.9, 0.9, 0.0, 0.0, 0.0])
    res = {}
    for h in ("H1", "H2", "H2b", "H3"):
        p, ll, k = fit(h, pop, idx, G, p0 if h != "H3" else res["H2"][0].copy())
        res[h] = (p, ll, k)
        P = unpack(p, h)
        psi_m, psi_s = sig(P["p0"]), sig(P["p0"] + P["p1"])
        h0s = "/".join(f"{sig(P['a'] + P['gam'] * np.log(n)):.3f}" for n in (1, 12, 52, 1000))
        print(f"\n{h}: loglik {ll:.1f}  params {k}  AIC {2 * k - 2 * ll:.1f}")
        print(f"   always-free firms: Micro {psi_m:.2f}, Small+ {psi_s:.2f};  per-attempt P(zero) at N=1/12/52/1000: {h0s};"
              f"  beta {P['beta']:.2f}")
        print(f"   per-attempt cost if > 0: median £{np.exp(P['mu']):,.0f} (phishing-only Micro), shifts I/R/S x"
              f"{np.exp(P['dI']):.2f}/{np.exp(P['dR']):.2f}/{np.exp(P['dS']):.2f}, Small+ x{np.exp(P['dsm']):.2f}, sigma {P['s']:.2f}")
        Gr = np.array([[n] * 3 for n in (1, 12, 52, 300, 1000)], float)
        dummy = dict(C=np.tile([1, 0, 0, 0], (5, 1)), sm=np.zeros(5))
        yr = yearly(P, dummy, np.arange(5), Gr)
        print("   yearly clean-up, phishing-only Micro paying-or-not average, N=1/12/52/300/1000: " + " / ".join(f"£{v:,.0f}" for v in yr))
        vals = np.zeros(len(pop["w"]))
        hit = np.where(pop["hit"])[0]
        vals[hit] = yearly(P, pop, hit, G)
        tot = ne.national(vals, pop)[0]
        print(f"   national clean-up (all attacked employer firms): £{tot / 1e9:.2f}bn")
        mu, psi = lik_terms(P, pop, idx, G)
        line = []
        for b in range(1, 7):
            m = band[idx] == b
            if m.sum() < 8:
                continue
            ww = pop["w"][idx][m]
            midv = np.where(pop["hi"][idx][m] == 0, 0, (pop["lo"][idx][m] + pop["hi"][idx][m]) / 2)
            obs = [np.average(midv == 0, weights=ww), np.average((midv > 0) & (midv < 500), weights=ww)]
            c0 = cdf(0.0, P, mu[m], psi[m], G[idx][m])
            c5 = cdf(500.0, P, mu[m], psi[m], G[idx][m])
            pred = [np.average(c0, weights=ww), np.average(c5 - c0, weights=ww)]
            line.append(f"f{b} n{m.sum()} obs {obs[0]:.2f}/{obs[1]:.2f} pred {pred[0]:.2f}/{pred[1]:.2f}")
        print("   no-cost / <£500 by frequency band: " + ";  ".join(line))


if __name__ == "__main__":
    main()
