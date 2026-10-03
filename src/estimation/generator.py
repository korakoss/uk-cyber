"""Full generator: exposure model per size band + cost model with separate Small / Medium / Large effects.

Exposure: within each size band (Micro, Small, Medium, Large) a 3-class latent-class model on the 10 attack-type flags
(survey-weighted): class shares pi_bk, per-class type hit probabilities th_bkj; classes ordered by mean hit prob ->
frailty tiers low / mid / high. Channels: phishing, impersonation, ransomware own type; other serious = any of the 7
others. Given class, channels independent. Targeted phishing given phishing hit: weighted share per band x class.
Cost (same structure as the adopted model, breach marker = outcome flag or consequence marker): per hit channel a
handling cost (zero mass + lognormal) and Poisson breaches; rate lam_c x tier x band (x targeted for phishing); each
breach big with prob sigmoid(a + b (channels - 1) + band shift), big ~ lognormal, else ordinary channel episode
(zero mass by band, lognormal with band amount shift). Band effects (Small, Medium, Large vs Micro) on: breach rate,
handling amount, ordinary-breach zero share, ordinary amount, big-breach chance.
1. exposure fit per band: breadth 0-4 observed vs model.
2. cost fit: band effects separate (full) vs tied Small = Medium = Large (pooled); likelihood ratio.
3. national: full generator (exposure model x cost model, exact expectation over class x channel subset x targeted)
   vs plug-in of observed patterns with the same cost model; by band.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/generator.py
"""

import itertools
import os

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import national_estimate as ne
import poisson_cost_model as pcm
from exposure_check import chan_probs, bern_sum, obs_dists
from frailty_shape_flags import lca, patterns
from type_cooccurrence_structure import load

SP = ne.SP
sig = lambda z: 1 / (1 + np.exp(-z))
SUBSETS = [np.array(s, float) for s in itertools.product([0, 1], repeat=4)]


def exposure():
    X, w, size = load()
    fits, post = {}, np.zeros((len(X), 3))
    for b in (1, 2, 3, 4):
        gm = size == b
        P, c = patterns(X[gm], w[gm] / w[gm].mean())
        _, pi, th = lca(P, c, 3, np.random.default_rng(0), starts=30)
        o = np.argsort(th.mean(1))
        pi, th = pi[o], th[o]
        fits[b] = (pi, th)
        lp = np.log(pi)[None] + X[gm] @ np.log(th).T + (1 - X[gm]) @ np.log(1 - th).T
        p = np.exp(lp - lp.max(1, keepdims=True))
        post[gm] = p / p.sum(1, keepdims=True)
    return fits, post, X, w, size


# ---- cost model with band effects ----
def expand(q, tied):
    i = 0

    def take(k):
        nonlocal i
        v = q[i:i + k]
        i += k
        return v

    def band():
        v = take(1 if tied else 3)
        return np.repeat(v, 3) if tied else v
    P = dict(lam=np.exp(take(4)), bt=np.r_[0.0, take(2)], bs=band(), h0=sig(take(4)), mu_h=take(4), s_h=np.exp(take(1)[0]),
             dh=band(), m0=take(1)[0], dm0=band(), mu_m=take(4), s_m=np.exp(take(1)[0]), dm=band())
    P["pi_a"], P["pi_b"] = take(2)
    P["pb"] = band()
    P["mu_big"], ls = take(2)
    P["s_big"] = np.exp(ls)
    P["t"], P["k"] = take(2)
    return P


def start(tied):
    P = pcm.unpack_b(np.load(SP + "national_main_F2R.npy"), "F2R")
    b = (lambda v: [v]) if tied else (lambda v: [v] * 3)
    return np.r_[np.log(P["lam"]), P["bt"][1:], b(P["bs"]), np.log(P["h0"] / (1 - P["h0"])), P["mu_h"], np.log(P["s_h"]),
                 b(P["dh"]), P["m0"], b(P["dm0"]), P["mu_m"], np.log(P["s_m"]), b(P["dm"]), P["pi_a"], P["pi_b"], b(0.0),
                 P["mu_big"], np.log(P["s_big"]), P["t"], P["k"]]


def firm_terms(P, H, tier, B, tg):
    """H (n,4) channels hit, tier (n,), B (n,3) band one-hot (Small, Medium, Large), tg (n,)."""
    nch = H.sum(1)
    Lc = H * P["lam"][None] * np.exp(P["bt"][tier] + B @ P["bs"])[:, None]
    Lc[:, 0] *= np.exp(P["t"] * tg)
    h0 = sig(np.log(P["h0"] / (1 - P["h0"]))[None] + P["k"] * tg[:, None] * np.array([1.0, 0, 0, 0])[None])
    muh = P["mu_h"][None] + (B @ P["dh"])[:, None]
    m0 = sig(P["m0"] + B @ P["dm0"])
    mum = P["mu_m"][None] + (B @ P["dm"])[:, None]
    pi = sig(P["pi_a"] + P["pi_b"] * (nch - 1) + B @ P["pb"])
    return Lc, h0, muh, m0, mum, pi


def joint_cdf(x, P, H, T, outcome):
    Lc, h0, muh, m0, mum, pi = T
    L = Lc.sum(1)
    pos = (x > 0)[:, None]
    lx = np.log(np.maximum(x, 1e-12))[:, None]
    Fh = h0 + (1 - h0) * np.where(pos, norm.cdf((lx - muh) / P["s_h"]), 0.0)
    Fh = np.where(H == 1, Fh, 1.0).prod(1)
    Fs = m0[:, None] + (1 - m0[:, None]) * np.where(pos, norm.cdf((lx - mum) / P["s_m"]), 0.0)
    Fb = np.where(pos[:, 0], norm.cdf((lx[:, 0] - P["mu_big"]) / P["s_big"]), 0.0)
    F = (Lc * ((1 - pi)[:, None] * Fs + (pi * Fb)[:, None])).sum(1) / L
    return ((np.exp(-L * (1 - F)) - np.exp(-L)) if outcome else np.exp(-L)) * Fh


def loglik(q, D, tied):
    P = expand(q, tied)
    T = firm_terms(P, D["H"], D["tier"], D["B"], D["tg"])
    o = D["oa"] == 1
    up = np.where(o, joint_cdf(D["hi"], P, D["H"], T, True), joint_cdf(D["hi"], P, D["H"], T, False))
    lo = np.where(o, joint_cdf(D["lo"], P, D["H"], T, True), joint_cdf(D["lo"], P, D["H"], T, False))
    pr = np.where(D["hi"] == 0, up, up - lo)
    return (D["w"] * np.log(np.maximum(pr, 1e-300))).sum()


def fit(D, tied, q0):
    f = lambda q: -loglik(q, D, tied)
    o = minimize(f, q0, method="L-BFGS-B", options={"maxiter": 20000})
    for _ in range(2):
        o = minimize(f, o.x, method="Nelder-Mead", options={"maxiter": 30000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(f, o.x, method="L-BFGS-B", options={"maxiter": 20000})
    return o.x, -o.fun


def expected_cost(P, H, tier, B, tg):
    Lc, h0, muh, m0, mum, pi = firm_terms(P, H, tier, B, tg)
    eh = ((1 - h0) * np.exp(muh + P["s_h"] ** 2 / 2) * H).sum(1)
    eo = (Lc * ((1 - pi) * (1 - m0))[:, None] * np.exp(mum + P["s_m"] ** 2 / 2)).sum(1)
    eb = Lc.sum(1) * pi * np.exp(P["mu_big"] + P["s_big"] ** 2 / 2)
    return eh + eo + eb


def main():
    fits, post, X, w, size = exposure()
    tier = post.argmax(1)
    print("1. exposure per band (LCA K=3): breadth 0/1/2/3/4 observed | model, X2 (effective n)")
    for b, lab in zip((1, 2, 3, 4), ("Micro", "Small", "Medium", "Large")):
        pi, th = fits[b]
        cp = chan_probs(th)
        mb = sum(pi[k] * bern_sum(cp[k]) for k in range(3))
        ob, _, _, _, neff = obs_dists(X[size == b], w[size == b])
        print(f"   {lab:6s} classes " + "/".join(f"{v:.2f}" for v in pi) + "  " + " ".join(f"{v:.3f}" for v in ob) + " | " +
              " ".join(f"{v:.3f}" for v in mb) + f"   X2 {neff * ((ob - mb) ** 2 / np.maximum(mb, 1e-6)).sum():.1f}")

    pop = ne.population()
    pop["tier"] = tier
    D = ne.fit_data(pop)
    D["tier"] = tier[D["idx"]]
    Bfull = np.column_stack([size == 2, size == 3, size == 4]).astype(float)
    D["B"] = Bfull[D["idx"]]
    print("\n2. cost model with band effects")
    res = {}
    for tied in (True, False):
        q, ll = fit(D, tied, start(tied))
        res[tied] = (q, ll)
        P = expand(q, tied)
        lab = "pooled (Small=Medium=Large)" if tied else "separate bands"
        print(f"   {lab}: loglik {ll:.1f}, params {len(q)}")
        print("     band effects (Small/Medium/Large vs Micro): breach rate x" + "/".join(f"{np.exp(v):.2f}" for v in P["bs"]) +
              "; handling amount x" + "/".join(f"{np.exp(v):.2f}" for v in P["dh"]) +
              "; ordinary zero-share logit " + "/".join(f"{v:+.2f}" for v in P["dm0"]) +
              "; ordinary amount x" + "/".join(f"{np.exp(v):.2f}" for v in P["dm"]) +
              "; big-chance odds x" + "/".join(f"{np.exp(v):.2f}" for v in P["pb"]))
        print(f"     tiers mid/high x{np.exp(P['bt'][1]):.2f}/x{np.exp(P['bt'][2]):.2f}; P(big) by channels (Micro) " +
              "/".join(f"{sig(P['pi_a'] + P['pi_b'] * k):.3f}" for k in range(4)) + f"; big median £{np.exp(P['mu_big']):,.0f} sigma {P['s_big']:.2f}")
    lr = 2 * (res[False][1] - res[True][1])
    print(f"   LR separate vs pooled: {lr:.1f} on {len(res[False][0]) - len(res[True][0])} df")
    np.save(SP + "generator_cost_full.npy", res[False][0])
    np.save(SP + "generator_cost_pooled.npy", res[True][0])

    print("\n3. national: full generator vs observed-pattern plug-in (same cost model)")
    tgr = pop["tg"]
    for tied in (False, True):
        P = expand(res[tied][0], tied)
        vals = np.zeros(len(size))
        vals[pop["hit"]] = expected_cost(P, pop["C"][pop["hit"]], tier[pop["hit"]], Bfull[pop["hit"]], tgr[pop["hit"]])
        plug, plug_by = ne.national(vals, pop)
        gen_by = {}
        for b in (1, 2, 3, 4):
            pi, th = fits[b]
            cp = chan_probs(th)
            Bb = np.zeros((1, 3))
            if b > 1:
                Bb[0, b - 2] = 1
            gm = size == b
            tot = 0.0
            for k in range(3):
                ph = gm & (pop["C"][:, 0] == 1)
                wk = post[ph, k] * w[ph]
                ptg = (wk * tgr[ph]).sum() / max(wk.sum(), 1e-12)
                for S in SUBSETS:
                    pS = np.prod(np.where(S == 1, cp[k], 1 - cp[k]))
                    if S.sum() == 0 or pS < 1e-12:
                        continue
                    e0 = expected_cost(P, S[None], np.array([k]), Bb, np.array([0.0]))[0]
                    e1 = expected_cost(P, S[None], np.array([k]), Bb, np.array([1.0]))[0] if S[0] == 1 else e0
                    tot += pi[k] * pS * ((1 - ptg) * e0 + ptg * e1)
            gen_by[b] = tot * ne.N[b]
        gen = sum(gen_by.values())
        lab = "pooled" if tied else "separate"
        print(f"   [{lab}] generator £{gen / 1e9:.2f}bn vs plug-in £{plug / 1e9:.2f}bn")
        for b, sl in zip((1, 2, 3, 4), ("Micro", "Small", "Medium", "Large")):
            print(f"      {sl:6s} generator £{gen_by[b] / 1e9:.3f}bn (£{gen_by[b] / ne.N[b]:,.0f}/business)  plug-in £{plug_by[b] / 1e9:.3f}bn")


if __name__ == "__main__":
    main()
