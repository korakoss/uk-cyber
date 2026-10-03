"""Routine clean-up cost per ATTACK (not once per attack type per year).

Previous model: one handling draw per hit channel per year -> treats the worst handling as the total (wrong).
Here: a firm faces n attacks in the year, n from Q54 frequency (once 1; less than monthly 5; monthly 12; weekly 52;
daily 300; several a day 1000; don't know / missing -> 5, counted). Each attack has a clean-up cost: zero with prob
h0(n) = sigmoid(a0 + gam * log n), else lognormal(mu + channel shifts (I, R, S hit) + dh*Small+ - beta * log n, s_h).
beta > 0 lets heavily attacked firms handle each attack more cheaply. Breach episodes as in the main model (F2R:
Poisson breaches per hit channel, targeted-phishing rate, big vs ordinary by channels hit).
Observed worst = max over the n clean-ups and all breaches:  P(max handling <= x) = F_att(x)^n.
Yearly clean-up total per firm = n * E[per-attack cost].
1. Fit with beta free; compare loglik with the once-per-channel model (F2R).
2. Profile beta (fixed at a grid, everything else refitted): loglik and implied national clean-up total, national total.
3. Worst handling cost by frequency among non-breached firms, observed vs predicted, at the fitted beta.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/handling_volume.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import joint_five_channel as f
import national_estimate as ne
import poisson_cost_model as pcm

NMAP = {1: 1, 2: 5, 3: 12, 4: 52, 5: 300, 6: 1000}
sig = lambda z: 1 / (1 + np.exp(-z))


def attacks(pop):
    fq = f.load_firms()["freq"].values
    n = np.array([NMAP.get(int(v), np.nan) if not np.isnan(v) else np.nan for v in fq])
    miss = np.isnan(n) & pop["hit"]
    n = np.where(np.isnan(n), 5.0, n)
    return n, miss


def unpack(p):
    return dict(lam=np.exp(p[0:4]), bt=np.r_[0.0, p[4:6]], bs=p[6], m0=p[7], dm0=p[8], mu_m=p[9:13], s_m=np.exp(p[13]),
                dm=p[14], pi_a=p[15], pi_b=p[16], mu_big=p[17], s_big=np.exp(p[18]), t=p[19],
                a0=p[20], gam=p[21], mu=p[22], dch=np.r_[0.0, p[23:26]], beta=p[26], s_h=np.exp(p[27]), dh=p[28])


def from_f2r(q):
    P = pcm.unpack_b(q, "F2R")
    return np.r_[np.log(P["lam"]), P["bt"][1:], P["bs"], P["m0"], P["dm0"], P["mu_m"], np.log(P["s_m"]), P["dm"],
                 P["pi_a"], P["pi_b"], P["mu_big"], np.log(P["s_big"]), P["t"],
                 1.0, 0.0, np.log(80.0), 0.0, 0.5, 0.5, 0.3, np.log(1.9), 0.2]


def terms(P, D):
    e = D["nch"] - 1
    Lc = D["H"] * P["lam"][None] * np.exp(P["bt"][D["tier"]] + P["bs"] * D["sm"])[:, None]
    Lc[:, 0] *= np.exp(P["t"] * D["tg"])
    m0 = sig(P["m0"] + P["dm0"] * D["sm"])
    mum = P["mu_m"][None] + (P["dm"] * D["sm"])[:, None]
    pi = sig(P["pi_a"] + P["pi_b"] * e)
    ln = np.log(D["n"])
    h0 = sig(P["a0"] + P["gam"] * ln)
    mh = P["mu"] + D["H"][:, 1:] @ P["dch"][1:] + P["dh"] * D["sm"] - P["beta"] * ln
    return Lc, m0, mum, pi, h0, mh


def joint_cdf(x, P, D, T, outcome):
    Lc, m0, mum, pi, h0, mh = T
    L = Lc.sum(1)
    pos = x > 0
    lx = np.log(np.maximum(x, 1e-12))
    Fa = h0 + (1 - h0) * np.where(pos, norm.cdf((lx - mh) / P["s_h"]), 0.0)
    Fh = np.exp(D["n"] * np.log(np.maximum(Fa, 1e-300)))
    Fs = m0[:, None] + (1 - m0[:, None]) * np.where(pos[:, None], norm.cdf((lx[:, None] - mum) / P["s_m"]), 0.0)
    Fb = np.where(pos, norm.cdf((lx - P["mu_big"]) / P["s_big"]), 0.0)
    Fm = (1 - pi)[:, None] * Fs + (pi * Fb)[:, None]
    F = (Lc * Fm).sum(1) / L
    if outcome:
        return (np.exp(-L * (1 - F)) - np.exp(-L)) * Fh
    return np.exp(-L) * Fh


def loglik(p, D):
    P = unpack(p)
    T = terms(P, D)
    o = D["oa"] == 1
    up = np.where(o, joint_cdf(D["hi"], P, D, T, True), joint_cdf(D["hi"], P, D, T, False))
    lo = np.where(o, joint_cdf(D["lo"], P, D, T, True), joint_cdf(D["lo"], P, D, T, False))
    pr = np.where(D["hi"] == 0, up, up - lo)
    return (D["w"] * np.log(np.maximum(pr, 1e-300))).sum()


def fit(D, p0, fixed_beta=None):
    if fixed_beta is None:
        g = lambda q: -loglik(q, D)
        x0 = p0
    else:
        g = lambda q: -loglik(np.r_[q[:26], fixed_beta, q[26:]], D)
        x0 = np.r_[p0[:26], p0[27:]]
    o = minimize(g, x0, method="L-BFGS-B", options={"maxiter": 20000})
    for _ in range(2):
        o = minimize(g, o.x, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(g, o.x, method="L-BFGS-B", options={"maxiter": 20000})
    p = o.x if fixed_beta is None else np.r_[o.x[:26], fixed_beta, o.x[26:]]
    return p, -o.fun


def national_totals(p, pop, n):
    P = unpack(p)
    Dp = dict(H=pop["C"], nch=pop["nch"], tier=pop["tier"], sm=pop["sm"], tg=pop["tg"], n=n)
    Lc, m0, mum, pi, h0, mh = terms(P, Dp)
    handling = np.where(pop["hit"], n * (1 - h0) * np.exp(mh + P["s_h"] ** 2 / 2), 0.0)
    eo = (Lc * ((1 - pi) * (1 - m0))[:, None] * np.exp(mum + P["s_m"] ** 2 / 2)).sum(1)
    eb = Lc.sum(1) * pi * np.exp(P["mu_big"] + P["s_big"] ** 2 / 2)
    return ne.national(handling, pop)[0], ne.national(eo + eb, pop)[0]


def main():
    pop = ne.population()
    n, miss = attacks(pop)
    print(f"attacked firms {pop['hit'].sum()}; frequency missing/don't know {miss.sum()} (set to 5 attacks)")
    D = ne.fit_data(pop)
    D["n"] = n[D["idx"]]
    q = np.load(ne.SP + "national_main_F2R.npy")
    ll_old = pcm.loglik_b(q, D, "F2R")
    p0 = from_f2r(q)
    p, ll = fit(D, p0)
    P = unpack(p)
    np.save(ne.SP + "handling_volume.npy", p)
    print(f"\n1. per-attack clean-up model: loglik {ll:.1f}  vs once-per-attack-type model {ll_old:.1f}  (params 29 vs 31)")
    print(f"   per attack: P(zero) = sigmoid({P['a0']:.2f} + {P['gam']:.2f} log n) -> n=1 {sig(P['a0']):.2f}, n=52 {sig(P['a0'] + P['gam'] * np.log(52)):.3f},"
          f" n=1000 {sig(P['a0'] + P['gam'] * np.log(1000)):.4f}")
    print(f"   median if > 0: £{np.exp(P['mu']):,.0f} x n^-{P['beta']:.2f} (phishing-only, Micro); channel shifts I/R/S x" +
          "/".join(f"{np.exp(v):.2f}" for v in P["dch"][1:]) + f"; Small+ x{np.exp(P['dh']):.2f}; spread {P['s_h']:.2f}")
    for nn in (1, 12, 52, 300, 1000):
        h0 = sig(P["a0"] + P["gam"] * np.log(nn))
        per = (1 - h0) * np.exp(P["mu"] - P["beta"] * np.log(nn) + P["s_h"] ** 2 / 2)
        print(f"   n={nn:5d}: mean per attack £{per:,.2f}, yearly clean-up £{nn * per:,.0f} (phishing-only Micro)")
    h, b = national_totals(p, pop, n)
    print(f"   national: clean-up £{h / 1e9:.2f}bn + breaches £{b / 1e9:.2f}bn = £{(h + b) / 1e9:.2f}bn")

    print("\n2. profile over beta (others refitted)")
    pb = p
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        pb, llb = fit(D, pb, fixed_beta=beta)
        h, b = national_totals(pb, pop, n)
        print(f"   beta {beta:4.2f}  loglik {llb:8.1f}  national clean-up £{h / 1e9:6.2f}bn  breaches £{b / 1e9:.2f}bn  total £{(h + b) / 1e9:.2f}bn")

    print("\n3. non-breached firms: worst-incident shares (none / <£500 / £500+) by frequency, obs vs pred (fitted beta)")
    T = terms(P, D)
    nb = D["oa"] == 0
    for fqv, nn in NMAP.items():
        m = nb & (D["n"] == nn)
        if m.sum() < 10:
            continue
        Dm = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(D["w"]) else v) for k, v in D.items()}
        Tm = tuple(t[m] for t in T)
        k = m.sum()
        c0, c5, cinf = (joint_cdf(np.full(k, x), P, Dm, Tm, False) for x in (0, 500, 1e12))
        pred = np.array([(Dm["w"] * c0 / cinf).sum(), (Dm["w"] * (c5 - c0) / cinf).sum(), (Dm["w"] * (1 - c5 / cinf)).sum()]) / Dm["w"].sum()
        mid = np.where(Dm["hi"] == 0, 0, (Dm["lo"] + Dm["hi"]) / 2)
        obs = np.array([np.average(mid == 0, weights=Dm["w"]), np.average((mid > 0) & (mid < 500), weights=Dm["w"]), np.average(mid >= 500, weights=Dm["w"])])
        print(f"   freq {fqv} (n={nn:4d})  firms {k:3d}  obs " + " ".join(f"{v:.2f}" for v in obs) + "   pred " + " ".join(f"{v:.2f}" for v in pred))


if __name__ == "__main__":
    main()
