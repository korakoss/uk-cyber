"""Tail by extrapolating the fitted model's curve, with a sanity check against the largest known 2025 UK incidents.

Model: unified Poisson cost model with breadth (poisson_cost_model.py breadth; F2 main, F1 sensitivity). Every episode's
cost distribution is extrapolated as fitted (lognormal; F2: big episodes lognormal(mu_big, s_big)).
National: per size band, expected episodes per business = survey-weighted mean over ALL firms in the band (unhit = 0;
hit firms without cost data assumed like hit firms with it), scaled by ONS N(size).
For a loss threshold x, national expected number of episodes costing > x per year:
    Lam_nat(x) = sum_size N_s * mean_firms_s[ sum_c Lam_c * P(episode cost > x) ]
and, since episodes are Poisson, P(largest single loss in the UK this year > x) = 1 - exp(-Lam_nat(x)).
Thresholds include the scale of the 2025 M&S (~£300m reported profit hit) and JLR (~£1.9bn estimated UK economic
impact) incidents - figures approximate, used only as reference sizes.
Also: national expected annual cost and the share coming from single losses above £500k / £5m.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/tail_extrapolation.py
"""

import numpy as np
from scipy.stats import norm

import poisson_cost_model as pcm
from broad_interaction import channels
from counts_given_frailty import tier_posteriors
from material_breach import data

N = {1: 1150875, 2: 220085, 3: 38435, 4: 8335}
SL = {1: "Micro", 2: "Small", 3: "Medium", 4: "Large"}
sig = lambda z: 1 / (1 + np.exp(-z))


def per_firm(P, C, nch, tier, sm):
    """Per firm: episode rates (n,4), m0, small log-means (n,4), pi (n,)."""
    e = nch - 1
    Lc = C * P["lam"][None] * np.exp(P["bt"][tier] + P["bs"] * sm)[:, None]
    m0 = sig(P["m0"] + P["dm0"] * sm + P["g0"] * e)
    mum = P["mu_m"][None] + (P["dm"] * sm + P["g"] * e)[:, None]
    pi = sig(P["pi_a"] + P["pi_b"] * e)
    return Lc, m0, mum, pi


def exceed(x, P, m0, mum, pi):
    """P(episode cost > x), (n,4)."""
    lx = np.log(x)
    small = (1 - m0)[:, None] * norm.sf((lx - mum) / P["s_m"])
    big = norm.sf((lx - P["mu_big"]) / P["s_big"])
    return (1 - pi)[:, None] * small + pi[:, None] * big


def partial_mean_above(x, P, m0, mum, pi):
    """E[cost * 1(cost > x)] per episode, (n,4) (lognormal partial expectation)."""
    lx = np.log(x)
    s = P["s_m"]
    small = (1 - m0)[:, None] * np.exp(mum + s ** 2 / 2) * norm.sf((lx - mum - s ** 2) / s)
    b = P["s_big"]
    big = np.exp(P["mu_big"] + b ** 2 / 2) * norm.sf((lx - P["mu_big"] - b ** 2) / b)
    return (1 - pi)[:, None] * small + pi[:, None] * big


def main():
    X, size, band, D, oa, w = data()
    C = channels(X)
    nch = C.sum(1)
    tier = tier_posteriors().argmax(1)
    sm = (size >= 2).astype(float)
    hit = nch >= 1
    have = hit & ~np.isnan(band) & ~np.isnan(oa)
    fits = np.load("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_breadth_fits.npy", allow_pickle=True)
    thresholds = [5e5, 5e6, 5e7, 3e8, 1e9, 1.9e9]
    for form, p in (("F2", fits[2]), ("F1", fits[1])):
        P = pcm.unpack_b(np.asarray(p, float), form)
        Lc, m0, mum, pi = per_firm(P, C, nch, tier, sm)
        eh = ((1 - P["h0"])[None] * np.exp(P["mu_h"][None] + P["dh"] * sm[:, None] + P["s_h"] ** 2 / 2) * C).sum(1)
        mean_ep = (1 - pi)[:, None] * (1 - m0)[:, None] * np.exp(mum + P["s_m"] ** 2 / 2) + pi[:, None] * np.exp(P["mu_big"] + P["s_big"] ** 2 / 2)
        print(f"\n===== {form}" + (f": big episodes median £{np.exp(P['mu_big']):,.0f}, sigma {P['s_big']:.2f}" if form == "F2" else
                                  f": cost scale x{np.exp(P['g']):.2f} per extra channel, sigma {P['s_m']:.2f}"))

        def nat(vals):
            """National total of a per-firm quantity: sum_s N_s * weighted mean over all firms in band s (unhit = 0)."""
            tot, by = 0.0, {}
            for s in (1, 2, 3, 4):
                a = size == s
                scale = w[a & hit].sum() / w[a & have].sum()
                v = (w[a & have] * vals[a & have]).sum() * scale / w[a].sum() * N[s]
                by[s] = v
                tot += v
            return tot, by
        ep_tot, ep_by = nat(Lc.sum(1))
        print(f"  episodes (breaches) per year, UK: {ep_tot:,.0f}  (" + ", ".join(f"{SL[s]} {v:,.0f}" for s, v in ep_by.items()) + ")")
        if form == "F2":
            bt, bb = nat((Lc * pi[:, None]).sum(1))
            print(f"  of which big: {bt:,.0f}  (" + ", ".join(f"{SL[s]} {v:,.0f}" for s, v in bb.items()) + ")")
        cost_tot, cost_by = nat(eh + (Lc * mean_ep).sum(1))
        print(f"  expected annual cost, UK: £{cost_tot / 1e9:.2f}bn  (" + ", ".join(f"{SL[s]} £{v / 1e9:.2f}bn" for s, v in cost_by.items()) + ")")
        for x in (5e5, 5e6):
            above, _ = nat((Lc * partial_mean_above(x, P, m0, mum, pi)).sum(1))
            print(f"    share from single losses above £{x / 1e6:g}m: {above / cost_tot:.2f}")
        print("  single losses above threshold, UK per year: expected number | P(at least one) | expected years between")
        for x in thresholds:
            lam, by = nat((Lc * exceed(x, P, m0, mum, pi)).sum(1))
            yrs = 1 / lam if lam > 0 else np.inf
            tag = {3e8: "  <- M&S-scale", 1.9e9: "  <- JLR-scale"}.get(x, "")
            print(f"    > £{x / 1e6:>6,.1f}m  {lam:12.4g}  {-np.expm1(-lam):.3g}  {yrs:12,.3g}{tag}")
        lam_j, by_j = nat((Lc * exceed(1.9e9, P, m0, mum, pi)).sum(1))
        print("    JLR-scale expected per year by size: " + ", ".join(f"{SL[s]} {v:.3g}" for s, v in by_j.items()))


if __name__ == "__main__":
    main()
