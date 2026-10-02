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
    if len(p) == 29:  # channel-specific episode zero share (logit per channel + shared size shift)
        return dict(lam=np.exp(p[0:4]), bt=np.r_[0.0, p[4:6]], bs=p[6], h0=sig(p[7:11]), mu_h=p[11:15], s_h=np.exp(p[15]),
                    m0=p[25:29][None], dm0=p[17], mu_m=p[18:22], s_m=np.exp(p[22]), dh=p[23], dm=p[24])
    if len(p) == 28:  # channel-specific episode-cost sigma
        return dict(lam=np.exp(p[0:4]), bt=np.r_[0.0, p[4:6]], bs=p[6], h0=sig(p[7:11]), mu_h=p[11:15], s_h=np.exp(p[15]),
                    m0=p[16], dm0=p[17], mu_m=p[18:22], s_m=np.exp(p[22:26])[None], dh=p[26], dm=p[27])
    return dict(lam=np.exp(p[0:4]), bt=np.r_[0.0, p[4:6]], bs=p[6], h0=sig(p[7:11]), mu_h=p[11:15], s_h=np.exp(p[15]),
                m0=p[16], dm0=p[17], mu_m=p[18:22], s_m=np.exp(p[22]), dh=p[23], dm=p[24])


def firm_terms(P, D):
    """Per-firm episode rates (n,4), m0 (n,), handling and material log-means (n,4)."""
    Lc = D["H"] * P["lam"][None] * np.exp(P["bt"][D["tier"]] + P["bs"] * D["sm"])[:, None]
    m0 = sig(P["m0"] + P["dm0"] * D["sm"][:, None]) if np.ndim(P["m0"]) else sig(P["m0"] + P["dm0"] * D["sm"])
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
    m0 = m0 if m0.ndim == 2 else m0[:, None]
    Fm = m0 + (1 - m0) * np.where(pos, norm.cdf((lx - mum) / P["s_m"]), 0.0)
    F = (Lc * Fm).sum(1) / L
    if outcome:
        return (np.exp(-L * (1 - F)) - np.exp(-L)) * Fh
    return np.exp(-L) * Fh


def firm_ll(p, D):
    P = unpack(p)
    T = firm_terms(P, D)
    o = D["oa"] == 1
    up = np.where(o, joint_cdf(D["hi"], P, D, T, True), joint_cdf(D["hi"], P, D, T, False))
    low = np.where(o, joint_cdf(D["lo"], P, D, T, True), joint_cdf(D["lo"], P, D, T, False))
    pr = np.where(D["hi"] == 0, up, up - low)
    return np.log(np.maximum(pr, 1e-300))


def loglik(p, D):
    return (D["w"] * firm_ll(p, D)).sum()


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


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


# --- Episode-cost spread: channel-specific sigma, and which firms set it ------------------------------------------
# Breached-firm cells are middle-heavier than one shared sigma (~1.95) predicts. (1) refit with sigma per channel;
# (2) list the outcome firms the base model finds least likely; (3) refit base dropping the top-band (>= £100k) outcome
# firms and see sigma. Fit shares for breached cells printed for each.
# Run: ... poisson_cost_model.py sigma

def breached_fit(p, D, label):
    P = unpack(p)
    T = firm_terms(P, D)
    print(f"  [{label}] breached cells obs vs pred (none/<500/500-5k/5k-20k/20k+):")
    for st in ("PI", "P", "S", "PS"):
        m = (D["sets"] == st) & (D["oa"] == 1)
        Dm = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(D["w"]) else v) for k, v in D.items()}
        Tm = tuple(t[m] for t in T)
        mid = (Dm["lo"] + Dm["hi"]) / 2
        g = np.array([0 if h == 0 else next(j for j in range(1, 5) if GROUPS[j][0] <= x < GROUPS[j][1]) for h, x in zip(Dm["hi"], mid)])
        obs = np.array([Dm["w"][g == j].sum() for j in range(5)]) / Dm["w"].sum()
        n = m.sum()
        cs = [joint_cdf(np.full(n, b), P, Dm, Tm, True) for b in (0, 500, 5000, 20000, 1e12)]
        pr = np.diff(np.column_stack([np.zeros(n)] + cs), axis=1) / cs[-1][:, None]
        pred = (Dm["w"][:, None] * pr).sum(0) / Dm["w"].sum()
        print(f"     {st:3s} n {n:3d}  obs " + " ".join(f"{v:.2f}" for v in obs) + "   pred " + " ".join(f"{v:.2f}" for v in pred))


def sigma_check():
    D = setup()
    base = np.load("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_params.npy")
    ll0 = loglik(base, D)
    print(f"base: loglik {ll0:.1f}, shared episode sigma {np.exp(base[22]):.2f}")
    breached_fit(base, D, "base")

    f = lambda p: -loglik(p, D)
    p1 = np.r_[base[:22], np.full(4, base[22]), base[23:25]]
    o = minimize(f, p1, method="L-BFGS-B", options={"maxiter": 20000})
    for _ in range(2):
        o = minimize(f, o.x, method="Nelder-Mead", options={"maxiter": 30000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(f, o.x, method="L-BFGS-B", options={"maxiter": 20000})
    P1 = unpack(o.x)
    print(f"\n(1) channel-specific sigma: loglik {-o.fun:.1f} (+{-o.fun - ll0:.1f} for 3 params);  sigma " +
          ", ".join(f"{c} {v:.2f}" for c, v in zip(CH, P1["s_m"][0])) + ";  medians " +
          ", ".join(f"{c} £{np.exp(v):,.0f}" for c, v in zip(CH, P1["mu_m"])) +
          f";  zero Micro {sig(P1['m0']):.2f} / Small+ {sig(P1['m0'] + P1['dm0']):.2f}")
    breached_fit(o.x, D, "channel sigma")

    print("\n(2) outcome firms least likely under base (lowest log-lik):")
    fl = firm_ll(base, D)
    o_ = np.where(D["oa"] == 1)[0]
    for k in o_[np.argsort(fl[o_])][:8]:
        print(f"     row {D['idx'][k]:5d}  set {D['sets'][k]:3s}  {'Small+' if D['sm'][k] else 'Micro '}  weight {D['w'][k]:.2f}"
              f"  worst £{D['lo'][k]:,.0f}-{D['hi'][k]:,.0f}  loglik {fl[k]:.2f}")

    keep = ~((D["oa"] == 1) & (D["lo"] >= 100000))
    print(f"\n(3) drop top-band outcome firms: rows {list(D['idx'][~keep])}")
    Dk = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(D["w"]) else v) for k, v in D.items()}
    fk = lambda p: -loglik(p, Dk)
    o2 = minimize(fk, base, method="L-BFGS-B", options={"maxiter": 20000})
    o2 = minimize(fk, o2.x, method="Nelder-Mead", options={"maxiter": 30000, "xatol": 1e-6, "fatol": 1e-8})
    P2 = unpack(o2.x)
    print(f"     shared sigma {P2['s_m']:.2f};  medians " + ", ".join(f"{c} £{np.exp(v):,.0f}" for c, v in zip(CH, P2["mu_m"])) +
          f";  zero Micro {sig(P2['m0']):.2f} / Small+ {sig(P2['m0'] + P2['dm0']):.2f}")
    breached_fit(o2.x, Dk, "no top band")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "sigma":
    sigma_check()


# (4) channel-specific episode zero share: breached cells differ mainly in their zero share (P .37, S .40, PS .20, PI .01).
# Run: ... poisson_cost_model.py m0ch

def m0_channel():
    D = setup()
    base = np.load("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_params.npy")
    ll0 = loglik(base, D)
    f = lambda p: -loglik(p, D)
    p1 = np.r_[base, np.full(4, base[16])]
    o = minimize(f, p1, method="L-BFGS-B", options={"maxiter": 20000})
    for _ in range(2):
        o = minimize(f, o.x, method="Nelder-Mead", options={"maxiter": 30000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(f, o.x, method="L-BFGS-B", options={"maxiter": 20000})
    P = unpack(o.x)
    print(f"channel zero share: loglik {-o.fun:.1f} (+{-o.fun - ll0:.1f} for 3 params);  zero (Micro) " +
          ", ".join(f"{c} {v:.2f}" for c, v in zip(CH, sig(P["m0"][0]))) + f";  Small+ logit shift {P['dm0']:+.2f};  medians " +
          ", ".join(f"{c} £{np.exp(v):,.0f}" for c, v in zip(CH, P["mu_m"])) + f";  sigma {P['s_m']:.2f};  Small+ amount x{np.exp(P['dm']):.2f}")
    breached_fit(o.x, D, "channel zero share")
    np.save("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_params_m0ch.npy", o.x)


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "m0ch":
    m0_channel()


# --- Goodness of fit: parametric bootstrap ------------------------------------------------------------------------
# Simulate the fitted generator for the same 806 firms (their hits, tier, size, weights): handling per hit channel,
# Poisson episodes, iid episode costs, outcome = any episode, worst = max. Statistic per cell (channel set x outcome,
# cells with >= 8 observed firms): weighted Pearson X2 over the 5 worst-cost groups, effective n = (sum w)^2 / sum w^2.
# Compare observed X2 (total, breached cells, each cell) with the simulated distribution. Parameters not refitted per
# replicate (makes the test conservative: simulated X2 slightly too large).
# Run: ... poisson_cost_model.py gof

def simulate(P, D, rng):
    Lc, m0, muh, mum = firm_terms(P, D)
    n = len(D["w"])
    hcost = np.where(rng.random((n, 4)) < P["h0"][None], 0.0, np.exp(muh + P["s_h"] * rng.standard_normal((n, 4))))
    worst = np.where(D["H"] == 1, hcost, 0.0).max(1)
    M = rng.poisson(Lc)
    for i, j in zip(*np.nonzero(M)):
        x = np.where(rng.random(M[i, j]) < m0[i], 0.0, np.exp(mum[i, j] + P["s_m"] * rng.standard_normal(M[i, j])))
        worst[i] = max(worst[i], x.max())
    return worst, (M.sum(1) >= 1).astype(int)


def cell_stats(group, out, D, cells):
    st = {}
    for key in cells:
        m = (D["sets"] == key[0]) & (out == key[1])
        if m.sum() == 0:
            st[key] = 0.0
            continue
        w = D["w"][m]
        neff = w.sum() ** 2 / (w ** 2).sum()
        obs = np.array([w[group[m] == j].sum() for j in range(5)]) / w.sum()
        st[key] = (obs, neff)
    return st


def gof(R=400):
    D = setup()
    P = unpack(np.load("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_params.npy"))
    rng = np.random.default_rng(3)
    edges = [0, 500, 5000, 20000]
    to_group = lambda v: np.where(v <= 0, 0, np.searchsorted(edges, v, side="right"))
    mid = np.where(D["hi"] == 0, 0.0, (D["lo"] + D["hi"]) / 2)
    gobs = to_group(mid)
    cells = [k for k in set(zip(D["sets"], D["oa"].astype(int))) if ((D["sets"] == k[0]) & (D["oa"] == k[1])).sum() >= 8]
    sims = [simulate(P, D, rng) for _ in range(R)]
    gs = [(to_group(wv), o) for wv, o in sims]
    # expected shares per cell from the simulations (pooled)
    exp_ = {}
    for key in cells:
        num = np.zeros(5)
        den = 0.0
        for g, o in gs:
            m = (D["sets"] == key[0]) & (o == key[1])
            num += np.array([D["w"][m & (g == j)].sum() for j in range(5)])
            den += D["w"][m].sum()
        exp_[key] = np.maximum(num / den, 1e-4)

    def x2(g, o):
        out = {}
        for key, v in cell_stats(g, o, D, cells).items():
            if v == 0.0:
                out[key] = 0.0
            else:
                obs, neff = v
                out[key] = neff * ((obs - exp_[key]) ** 2 / exp_[key]).sum()
        return out
    xo = x2(gobs, D["oa"].astype(int))
    xs = [x2(g, o) for g, o in gs]
    tot_o, tot_s = sum(xo.values()), np.array([sum(x.values()) for x in xs])
    br = [k for k in cells if k[1] == 1]
    bo, bs = sum(xo[k] for k in br), np.array([sum(x[k] for k in br) for x in xs])
    print(f"replicates {R}; cells {len(cells)} ({len(br)} breached)")
    print(f"total X2 obs {tot_o:.1f}  sim median {np.median(tot_s):.1f}  95% {np.percentile(tot_s, 95):.1f}  p {np.mean(tot_s >= tot_o):.3f}")
    print(f"breached cells X2 obs {bo:.1f}  sim median {np.median(bs):.1f}  95% {np.percentile(bs, 95):.1f}  p {np.mean(bs >= bo):.3f}")
    for key in sorted(cells, key=lambda k: -xo[k]):
        s = np.array([x[key] for x in xs])
        print(f"  {key[0]:3s} outcome {key[1]}  X2 obs {xo[key]:6.1f}  sim median {np.median(s):5.1f}  p {np.mean(s >= xo[key]):.3f}"
              f"   obs " + " ".join(f"{v:.2f}" for v in cell_stats(gobs, D['oa'].astype(int), D, [key])[key][0]) +
              "  exp " + " ".join(f"{v:.2f}" for v in exp_[key]))


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "gof":
    gof()


# --- Details of the one misfitting cell (IS, no outcome flag) -----------------------------------------------------
# Per firm: size, weight, types hit, worst band and tightened interval, components, outcomes, impacts, restore,
# external reporting, model log-lik. Run: ... poisson_cost_model.py iscell

def is_cell():
    import pandas as pd
    from latent_on_streams import aligned_raw
    from type_cooccurrence_structure import load, SHORT
    D = setup()
    p = np.load("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_params.npy")
    fl = firm_ll(p, D)
    X, _, size = load()
    r = aligned_raw()
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values
    olab = {1: "systems corrupted", 2: "personal data", 3: "files lost", 4: "temp access loss", 5: "assets/IP",
            6: "money stolen", 7: "services down", 8: "3rd-party loss", 11: "paid attackers", 12: "devices damaged",
            13: "accounts misused"}
    ilab = {1: "staff stopped", 2: "revenue loss", 3: "staff time", 4: "recovery costs", 5: "new measures", 7: "fines",
            8: "reputation", 9: "no service", 10: "deterred", 13: "complaints", 14: "compensation"}
    rlab = {1: "none", 2: "<1 day", 3: "1d-1wk", 4: "1wk-1mo", 5: "1mo+", 6: "not yet"}
    comp = ["damagedirsx_bands", "damagedirlx_bands", "damagestaffx_bands", "damageindx_bands"]
    m = np.where((D["sets"] == "IS") & (D["oa"] == 0))[0]
    print(f"IS outcome-0 cell: {len(m)} firms, total weight {D['w'][m].sum():.2f}")
    for k in m[np.argsort(-D["hi"][m])]:
        i = D["idx"][k]
        types = [SHORT[j] for j in range(X.shape[1]) if X[i, j] == 1]
        oc = [olab[j] for j in olab if num(f"outcome{j}")[i] == 1]
        im = [ilab[j] for j in ilab if num(f"impact{j}")[i] == 1]
        rest = num("restore")[i]
        cp = [num(c)[i] for c in comp]
        print(f"\nrow {i}: {['', 'Micro', 'Small', 'Medium', 'Large'][int(size[i])]}, weight {D['w'][k]:.2f} (share of cell "
              f"{D['w'][k] / D['w'][m].sum():.2f}), worst £{D['lo'][k]:,.0f}-{D['hi'][k]:,.0f}, model loglik {fl[k]:.2f}")
        print(f"  types: {', '.join(types)};  outcomes: {', '.join(oc) or '-'};  impacts: {', '.join(im) or '-'}")
        print(f"  restore: {rlab.get(int(rest), rest) if not np.isnan(rest) else 'NA'};  reported externally: {num('reporta')[i]};  "
              "components (ext-during/ext-after/staff/disruption, band): " +
              " / ".join(str(int(x)) if not np.isnan(x) and x < 997 else "NA" for x in cp))


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "iscell":
    is_cell()


# --- Widened breach marker ----------------------------------------------------------------------------------------
# The outcome flag misses contained incidents (breach_measures.py leak). Refit with breach marker =
# W1: outcome OR (external payments > 0 OR staff stopped OR restore >= 1 day OR recovery costs OR revenue loss)
# W2: outcome OR (staff stopped OR restore >= 1 day OR recovery costs OR revenue loss)   [external payments excluded]
# Compare per-hit expected cost (handling + episodes x E[G]) with the base fit.
# Run: ... poisson_cost_model.py widen

def per_hit(p, D):
    P = unpack(p)
    Lc, m0, muh, mum = firm_terms(P, D)
    out = []
    for c in range(4):
        m = D["H"][:, c] == 1
        eh = (1 - P["h0"][c]) * np.exp(muh[m, c] + P["s_h"] ** 2 / 2)
        eg = (1 - m0[m]) * np.exp(mum[m, c] + P["s_m"] ** 2 / 2)
        out.append((np.average(eh, weights=D["w"][m]), np.average(Lc[m, c], weights=D["w"][m]),
                    np.average(eg, weights=D["w"][m]), np.average(eh + Lc[m, c] * eg, weights=D["w"][m])))
    return out


def widen():
    import pandas as pd
    from latent_on_streams import aligned_raw
    D = setup()
    r = aligned_raw()
    nm = lambda c: pd.to_numeric(r[c], errors="coerce").values[D["idx"]]
    ext = ((nm("damagedirsx_bands") > 1) & (nm("damagedirsx_bands") <= 13)) | ((nm("damagedirlx_bands") > 1) & (nm("damagedirlx_bands") <= 13))
    soft = np.isin(nm("restore"), [3, 4, 5, 6]) | (nm("impact1") == 1) | (nm("impact4") == 1) | (nm("impact2") == 1)
    base = np.load("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_params.npy")
    res = {"base (outcome)": (base, D)}
    for lab, mk in (("W1 outcome|any marker", ext | soft), ("W2 outcome|soft marker", soft)):
        Dw = dict(D)
        Dw["oa"] = ((D["oa"] == 1) | mk).astype(float)
        print(f"\n===== {lab}: breached firms {int(Dw['oa'].sum())} (was {int(D['oa'].sum())})")
        p, ll = fit(Dw)
        report(p, ll, Dw)
        res[lab] = (p, Dw)
    print("\n===== per-hit expected annual cost: handling + episodes x E[G] = total")
    for lab, (p, Dx) in res.items():
        print(f"  {lab:24s} " + "   ".join(f"{c} £{h:,.0f}+{e:.3f}x£{g:,.0f}=£{t:,.0f}" for c, (h, e, g, t) in zip(CH, per_hit(p, Dx))))


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "widen":
    widen()


# --- All firms, with exposure breadth amplifying breach severity (explanation A, 2026-10-02) ----------------------
# Same generator, now on every hit firm (1-4 channels), breach marker W2. Breadth = number of channels hit (nch).
#   F0: no breadth effect (narrow model applied to everyone).
#   F1: episode cost scale rises with breadth: log-median + g*(nch-1); zero share logit + g0*(nch-1).
#   F2: each episode is BIG with prob pi = sigmoid(a + b*(nch-1)); big episodes ~ lognormal(mu_big, s_big), no zero mass;
#       otherwise the usual channel episode cost.
# Compare loglik; worst-cost shares by breadth x breached, obs vs pred; expected cost per breach and per hit firm by breadth.
# Run: ... poisson_cost_model.py breadth

def setup_all():
    import pandas as pd
    from latent_on_streams import aligned_raw
    from material_breach import data
    from broad_interaction import channels, tightened
    X, size, band, D_, oa, w = data()
    C = channels(X)
    nch = C.sum(1)
    lo, hi = tightened(band)
    r = aligned_raw()
    nm = lambda c: pd.to_numeric(r[c], errors="coerce").values
    soft = np.isin(nm("restore"), [3, 4, 5, 6]) | (nm("impact1") == 1) | (nm("impact4") == 1) | (nm("impact2") == 1)
    m = ~np.isnan(band) & ~np.isnan(oa) & (nch >= 1)
    idx = np.where(m)[0]
    tier = tier_posteriors().argmax(1)
    return dict(idx=idx, H=C[idx], lo=lo[idx], hi=hi[idx], oa=((oa == 1) | soft)[idx].astype(float), sm=(size >= 2)[idx].astype(float),
                w=w[idx] / w[idx].mean(), tier=tier[idx], nch=nch[idx],
                sets=np.array(["".join(CH[c] for c in np.where(C[i] == 1)[0]) for i in idx]))


def unpack_b(p, form):
    P = unpack(p[:25])
    P["g"] = P["g0"] = 0.0
    P["pi_a"], P["pi_b"], P["mu_big"], P["s_big"] = -50.0, 0.0, 0.0, 1.0
    if form == "F1":
        P["g"], P["g0"] = p[25], p[26]
    if form == "F2":
        P["pi_a"], P["pi_b"], P["mu_big"], P["s_big"] = p[25], p[26], p[27], np.exp(p[28])
    return P


def joint_cdf_b(x, P, D, outcome):
    e = D["nch"] - 1
    Lc = D["H"] * P["lam"][None] * np.exp(P["bt"][D["tier"]] + P["bs"] * D["sm"])[:, None]
    L = Lc.sum(1)
    m0 = sig(P["m0"] + P["dm0"] * D["sm"] + P["g0"] * e)
    muh = P["mu_h"][None] + P["dh"] * D["sm"][:, None]
    mum = P["mu_m"][None] + (P["dm"] * D["sm"] + P["g"] * e)[:, None]
    pi = sig(P["pi_a"] + P["pi_b"] * e)
    lx = np.log(np.maximum(x, 1e-12))[:, None]
    pos = (x > 0)[:, None]
    Fh = P["h0"][None] + (1 - P["h0"][None]) * np.where(pos, norm.cdf((lx - muh) / P["s_h"]), 0.0)
    Fh = np.where(D["H"] == 1, Fh, 1.0).prod(1)
    Fs = m0[:, None] + (1 - m0[:, None]) * np.where(pos, norm.cdf((lx - mum) / P["s_m"]), 0.0)
    Fb = np.where(pos[:, 0], norm.cdf((lx[:, 0] - P["mu_big"]) / P["s_big"]), 0.0)
    Fm = (1 - pi)[:, None] * Fs + (pi * Fb)[:, None]
    F = (Lc * Fm).sum(1) / L
    if outcome:
        return (np.exp(-L * (1 - F)) - np.exp(-L)) * Fh
    return np.exp(-L) * Fh


def loglik_b(p, D, form):
    P = unpack_b(p, form)
    o = D["oa"] == 1
    up = np.where(o, joint_cdf_b(D["hi"], P, D, True), joint_cdf_b(D["hi"], P, D, False))
    low = np.where(o, joint_cdf_b(D["lo"], P, D, True), joint_cdf_b(D["lo"], P, D, False))
    pr = np.where(D["hi"] == 0, up, up - low)
    return (D["w"] * np.log(np.maximum(pr, 1e-300))).sum()


def fit_b(D, form, p0):
    f = lambda p: -loglik_b(p, D, form)
    o = minimize(f, p0, method="L-BFGS-B", options={"maxiter": 20000})
    for _ in range(3):
        o = minimize(f, o.x, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(f, o.x, method="L-BFGS-B", options={"maxiter": 20000})
    return o.x, -o.fun


def breadth():
    D = setup_all()
    base = np.load("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_params_w2.npy")
    print(f"firms {len(D['w'])}; by channels hit: " + ", ".join(f"{k}: {(D['nch'] == k).sum()}" for k in (1, 2, 3, 4)) +
          f"; breached (W2) {int(D['oa'].sum())}")
    starts = {"F0": base, "F1": np.r_[base, 0.3, -0.3], "F2": np.r_[base, -3.0, 0.8, np.log(20000), np.log(1.2)]}
    fits = {}
    for form in ("F0", "F1", "F2"):
        p, ll = fit_b(D, form, starts[form])
        fits[form] = (p, ll)
        P = unpack_b(p, form)
        extra = {"F0": "", "F1": f"  cost scale x{np.exp(P['g']):.2f} per extra channel, zero-share logit {P['g0']:+.2f} per extra channel",
                 "F2": f"  P(big) by channels 1/2/3/4: " + "/".join(f"{sig(P['pi_a'] + P['pi_b'] * k):.3f}" for k in range(4)) +
                       f";  big episode median £{np.exp(P['mu_big']):,.0f}, sigma {P['s_big']:.2f}"}[form]
        print(f"\n{form}: loglik {ll:.1f}  params {len(p)}{extra}")
        print("   rates per hit " + ", ".join(f"{c} {v:.3f}" for c, v in zip(CH, P["lam"])) +
              f"; episode medians " + ", ".join(f"{c} £{np.exp(v):,.0f}" for c, v in zip(CH, P["mu_m"])) + f", sigma {P['s_m']:.2f}")
        print("   worst-cost shares (none/<500/500-5k/5k-20k/20k+) by channels x breached, obs | pred:")
        for k in (1, 2, 3, 4):
            for o in (0, 1):
                m = (D["nch"] == k) & (D["oa"] == o)
                Dm = {kk: (v[m] if isinstance(v, np.ndarray) and len(v) == len(D["w"]) else v) for kk, v in D.items()}
                n = m.sum()
                mid = (Dm["lo"] + Dm["hi"]) / 2
                g = np.array([0 if h == 0 else next(j for j in range(1, 5) if GROUPS[j][0] <= x < GROUPS[j][1]) for h, x in zip(Dm["hi"], mid)])
                obs = np.array([Dm["w"][g == j].sum() for j in range(5)]) / Dm["w"].sum()
                cs = [joint_cdf_b(np.full(n, b), P, Dm, bool(o)) for b in (0, 500, 5000, 20000, 1e12)]
                pr = np.diff(np.column_stack([np.zeros(n)] + cs), axis=1) / cs[-1][:, None]
                pred = (Dm["w"][:, None] * pr).sum(0) / Dm["w"].sum()
                print(f"     {k} ch, breached {o}  n {n:4d}  " + " ".join(f"{v:.2f}" for v in obs) + " | " + " ".join(f"{v:.2f}" for v in pred))
        print("   expected cost per breach (episode mean) and per hit firm (handling + episodes), by channels:")
        for k in (1, 2, 3, 4):
            m = D["nch"] == k
            e = k - 1
            m0 = sig(P["m0"] + P["dm0"] * D["sm"][m] + P["g0"] * e)
            pi = sig(P["pi_a"] + P["pi_b"] * e)
            Lc = D["H"][m] * P["lam"][None] * np.exp(P["bt"][D["tier"][m]] + P["bs"] * D["sm"][m])[:, None]
            eg_c = (1 - m0)[:, None] * np.exp(P["mu_m"][None] + (P["dm"] * D["sm"][m] + P["g"] * e)[:, None] + P["s_m"] ** 2 / 2)
            eg_c = (1 - pi) * eg_c + pi * np.exp(P["mu_big"] + P["s_big"] ** 2 / 2)
            eh = ((1 - P["h0"])[None] * np.exp(P["mu_h"][None] + P["dh"] * D["sm"][m][:, None] + P["s_h"] ** 2 / 2) * D["H"][m]).sum(1)
            ep = (Lc * eg_c).sum(1)
            perbreach = ep / np.maximum(Lc.sum(1), 1e-12)
            print(f"     {k} ch  n {m.sum():4d}  mean cost per breach £{np.average(perbreach, weights=D['w'][m]):,.0f}"
                  f"  expected episodes {np.average(Lc.sum(1), weights=D['w'][m]):.2f}"
                  f"  annual per firm £{np.average(eh + ep, weights=D['w'][m]):,.0f}")
    np.save("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_breadth_fits.npy",
            np.array([fits[k][0] for k in ("F0", "F1", "F2")], dtype=object), allow_pickle=True)


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "breadth":
    breadth()
