"""Narrow-firm cost model with Poisson material episodes (replaces the at-most-one-breach-per-channel version in
narrow_cost_model.py). Firms hit by 1-2 channels.

Generator, per firm (size s, frailty tier t), per hit channel c (P phishing, I impersonation, R ransomware, S other serious):
  handling H_c: 0 w.p. h0_c, else lognormal(mu_h_c + dh*Small+, s_h)            (one per hit channel-year)
  episodes M_c ~ Poisson(Lam_c), Lam_c = lam_c * tier_factor[t] * size_factor^Small+
  each episode X ~ G_c iid: 0 w.p. m0(s) (logit shift by size), else lognormal(mu_m_c + dm*Small+, s_m)
  annual = sum_c H_c + sum_c sum_k X_ck
Observed: outcome flag (Q56A any) = 1{sum M >= 1}; worst incident = max over all H and X (tightened interval).
Likelihood (Poisson merging + max of a Poisson number of iid draws), F = sum_c (Lam_c / Lam) F_mc:
  outcome 0:  P(worst <= x, M = 0)  = exp(-Lam) * prod_c F_hc(x)
  outcome 1:  P(worst <= x, M >= 1) = [exp(-Lam (1 - F(x))) - exp(-Lam)] * prod_c F_hc(x)
Survey-weighted (normalised). Checks: P(outcome) by tier; worst-cost shares by channel set x outcome; implied
expected annual cost per hit firm by channel (handling + Lam * E[G]).

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/poisson_cost_model.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import narrow_cost_model as ncm
from counts_given_frailty import tier_posteriors

CH = "PIRS"
GROUPS = ncm.GROUPS
sig = lambda z: 1 / (1 + np.exp(-z))


def setup():
    idx, hits, lo, hi, oa, sm, w = ncm.setup()
    tier = tier_posteriors().argmax(1)[idx]
    H = np.zeros((len(idx), 4))
    for k, h in enumerate(hits):
        H[k, h] = 1
    return dict(idx=idx, H=H, lo=lo[idx], hi=hi[idx], oa=oa[idx], sm=sm[idx], w=w[idx], tier=tier,
                sets=np.array(["".join(CH[c] for c in h) for h in hits]))


def unpack(p):
    return dict(lam=np.exp(p[0:4]), bt=np.r_[0.0, p[4:6]], bs=p[6], h0=sig(p[7:11]), mu_h=p[11:15], s_h=np.exp(p[15]),
                m0=p[16], dm0=p[17], mu_m=p[18:22], s_m=np.exp(p[22]), dh=p[23], dm=p[24])


def firm_terms(P, D):
    """Per-firm episode rates (n,4), m0 (n,), handling and material log-means (n,4)."""
    Lc = D["H"] * P["lam"][None] * np.exp(P["bt"][D["tier"]] + P["bs"] * D["sm"])[:, None]
    m0 = sig(P["m0"] + P["dm0"] * D["sm"])
    muh = P["mu_h"][None] + P["dh"] * D["sm"][:, None]
    mum = P["mu_m"][None] + P["dm"] * D["sm"][:, None]
    return Lc, m0, muh, mum


def joint_cdf(x, P, D, T, outcome):
    """P(worst <= x, outcome) per firm; x (n,) >= 0."""
    Lc, m0, muh, mum = T
    L = Lc.sum(1)
    lx = np.log(np.maximum(x, 1e-12))[:, None]
    pos = (x > 0)[:, None]
    Fh = P["h0"][None] + (1 - P["h0"][None]) * np.where(pos, norm.cdf((lx - muh) / P["s_h"]), 0.0)
    Fh = np.where(D["H"] == 1, Fh, 1.0).prod(1)
    Fm = m0[:, None] + (1 - m0[:, None]) * np.where(pos, norm.cdf((lx - mum) / P["s_m"]), 0.0)
    F = (Lc * Fm).sum(1) / L
    if outcome:
        return (np.exp(-L * (1 - F)) - np.exp(-L)) * Fh
    return np.exp(-L) * Fh


def loglik(p, D):
    P = unpack(p)
    T = firm_terms(P, D)
    o = D["oa"] == 1
    up = np.where(o, joint_cdf(D["hi"], P, D, T, True), joint_cdf(D["hi"], P, D, T, False))
    low = np.where(o, joint_cdf(D["lo"], P, D, T, True), joint_cdf(D["lo"], P, D, T, False))
    pr = np.where(D["hi"] == 0, up, up - low)
    return (D["w"] * np.log(np.maximum(pr, 1e-300))).sum()


def fit(D):
    p0 = np.r_[np.full(4, -2.0), 0, 0, 0, np.zeros(4), np.full(4, 5.0), 0.6, -1.0, 0.0, np.full(4, 7.0), 0.6, 0.0, 0.0]
    f = lambda p: -loglik(p, D)
    o = minimize(f, p0, method="L-BFGS-B", options={"maxiter": 20000})
    for _ in range(3):
        o = minimize(f, o.x, method="Nelder-Mead", options={"maxiter": 30000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(f, o.x, method="L-BFGS-B", options={"maxiter": 20000})
    return o.x, -o.fun


def report(p, ll, D):
    P = unpack(p)
    T = firm_terms(P, D)
    Lc, m0, muh, mum = T
    w = D["w"]
    print(f"firms {len(w)}  loglik {ll:.1f}  params {len(p)}")
    print("episode rate per hit (low tier, Micro): " + ", ".join(f"{c} {v:.3f}" for c, v in zip(CH, P["lam"])) +
          f";  tier mid x{np.exp(P['bt'][1]):.2f}, high x{np.exp(P['bt'][2]):.2f};  Small+ x{np.exp(P['bs']):.2f}")
    print("handling: P(zero) " + ", ".join(f"{c} {v:.2f}" for c, v in zip(CH, P["h0"])) + ";  median>0 " +
          ", ".join(f"{c} £{np.exp(v):,.0f}" for c, v in zip(CH, P["mu_h"])) + f";  sigma {P['s_h']:.2f};  Small+ amount x{np.exp(P['dh']):.2f}")
    print(f"episode cost: P(zero) Micro {sig(P['m0']):.2f} / Small+ {sig(P['m0'] + P['dm0']):.2f};  median>0 " +
          ", ".join(f"{c} £{np.exp(v):,.0f}" for c, v in zip(CH, P["mu_m"])) + f";  sigma {P['s_m']:.2f};  Small+ amount x{np.exp(P['dm']):.2f}")

    L = Lc.sum(1)
    pout = -np.expm1(-L)
    print("\nP(outcome) obs / pred:  by tier " + ", ".join(
        f"{t} {np.average(D['oa'][D['tier'] == k], weights=w[D['tier'] == k]):.3f}/{np.average(pout[D['tier'] == k], weights=w[D['tier'] == k]):.3f}"
        for k, t in enumerate(("low", "mid", "high"))) + ";  by size " + ", ".join(
        f"{t} {np.average(D['oa'][D['sm'] == k], weights=w[D['sm'] == k]):.3f}/{np.average(pout[D['sm'] == k], weights=w[D['sm'] == k]):.3f}"
        for k, t in enumerate(("Micro", "Small+"))))
    o = D["oa"] == 1
    em = L[o] / pout[o]
    print(f"E[episodes | breached]: weighted mean {np.average(em, weights=w[o]):.3f}, range {em.min():.2f}-{em.max():.2f}")

    print("\nexpected annual cost per hit firm, by channel (weighted over narrow firms hit by it): handling + episodes x E[G] = total")
    for c in range(4):
        m = D["H"][:, c] == 1
        eh = (1 - P["h0"][c]) * np.exp(muh[m, c] + P["s_h"] ** 2 / 2)
        eg = (1 - m0[m]) * np.exp(mum[m, c] + P["s_m"] ** 2 / 2)
        ep = Lc[m, c]
        print(f"  {CH[c]}  n {m.sum():4d}  handling £{np.average(eh, weights=w[m]):,.0f}  + episodes {np.average(ep, weights=w[m]):.3f}"
              f" x £{np.average(eg, weights=w[m]):,.0f}  = £{np.average(eh + ep * eg, weights=w[m]):,.0f}"
              f"   (Micro £{np.average((eh + ep * eg)[D['sm'][m] == 0], weights=w[m][D['sm'][m] == 0]):,.0f},"
              f" Small+ £{np.average((eh + ep * eg)[D['sm'][m] == 1], weights=w[m][D['sm'][m] == 1]):,.0f})")

    print("\nworst-cost shares obs vs pred (none / <£500 / £500-5k / £5k-20k / £20k+), by channel set x outcome")
    for key in sorted(set(zip(D["sets"], D["oa"].astype(int))), key=lambda k: -((D["sets"] == k[0]) & (D["oa"] == k[1])).sum()):
        m = (D["sets"] == key[0]) & (D["oa"] == key[1])
        if m.sum() < 8:
            continue
        Dm = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(w) else v) for k, v in D.items()}
        Tm = tuple(t[m] for t in T)
        mid = (Dm["lo"] + Dm["hi"]) / 2
        g = np.array([0 if h == 0 else next(j for j in range(1, 5) if GROUPS[j][0] <= x < GROUPS[j][1]) for h, x in zip(Dm["hi"], mid)])
        obs = np.array([Dm["w"][g == j].sum() for j in range(5)]) / Dm["w"].sum()
        n = len(Dm["w"])
        cs = [joint_cdf(np.full(n, b), P, Dm, Tm, bool(key[1])) for b in (0, 500, 5000, 20000, 1e12)]
        pr = np.diff(np.column_stack([np.zeros(n)] + cs), axis=1) / cs[-1][:, None]
        pred = (Dm["w"][:, None] * pr).sum(0) / Dm["w"].sum()
        print(f"  {key[0]:3s} outcome {key[1]}  n {n:4d}  obs " + " ".join(f"{v:.2f}" for v in obs) + "   pred " + " ".join(f"{v:.2f}" for v in pred))


def main():
    D = setup()
    p, ll = fit(D)
    np.save("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_params.npy", p)
    report(p, ll, D)


if __name__ == "__main__":
    main()
