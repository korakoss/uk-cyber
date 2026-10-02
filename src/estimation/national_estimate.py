"""National annual cost of cybercrime to UK businesses (employer frame) from the unified generator.

Main model F2R (poisson_cost_model.py): per firm, hit channels (survey's observed mix, reweighted); per hit channel a
handling cost and Poisson material breaches (breach marker W2); targeted-phishing firms get a higher phishing breach rate
and more often a handling cost; each breach is big with a chance rising with channels hit (big-breach lognormal),
otherwise an ordinary channel-specific episode; annual cost = sum. Tails = fitted curves extrapolated (pinned
simplification: no firm-finance scaling, Large-firm catastrophes understated).
National: per size band, expected annual cost per business = survey-weighted mean over ALL firms in the band (unhit = 0;
hit firms without cost data assumed like hit firms with it), x ONS N(size). Analytic expectations.

main:  fit F2R; national total by size and component (handling / ordinary breaches / big breaches), share from single
       losses above £500k / £5m; sensitivities (refits): F1R (cost-scale breadth form), no targeted split (F2), breach
       marker outcome-only and W1, firm 1380 excluded; empirical baseline (worst incident as annual cost, weighted
       midpoints). Repeat-rate sensitivity not refit: in-sample cost per breached firm was invariant to it (+-2%).
boot:  bootstrap - resample all survey firms within size band (multiplicities on weights), refit F2R from the main
       estimate, recompute national total. B replicates, appended to a log.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/national_estimate.py main|boot [B]
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

import poisson_cost_model as pcm
from breach_measures import num as cnum
from broad_interaction import channels, tightened
from counts_given_frailty import tier_posteriors
from latent_on_streams import aligned_raw
from material_breach import data

SP = "/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/"
N = {1: 1150875, 2: 220085, 3: 38435, 4: 8335}
SL = {1: "Micro", 2: "Small", 3: "Medium", 4: "Large"}
sig = lambda z: 1 / (1 + np.exp(-z))


def population():
    X, size, band, D, oa, w = data()
    r = aligned_raw()
    nm = lambda c: pd.to_numeric(r[c], errors="coerce").values
    C = channels(X)
    nch = C.sum(1)
    lo, hi = tightened(band)
    soft = np.isin(nm("restore"), [3, 4, 5, 6]) | (nm("impact1") == 1) | (nm("impact4") == 1) | (nm("impact2") == 1)
    ext = ((nm("damagedirsx_bands") > 1) & (nm("damagedirsx_bands") <= 13)) | ((nm("damagedirlx_bands") > 1) & (nm("damagedirlx_bands") <= 13))
    t = cnum(r, "phishcon")
    cb = nm("phishcon_bands")
    tg = (np.where(~np.isnan(t), t >= 1, (cb >= 2) & (cb <= 9)) & (C[:, 0] == 1)).astype(float)
    return dict(C=C, nch=nch, size=size, band=band, oa=oa, w=w, lo=lo, hi=hi, soft=soft, ext=ext, tg=tg,
                tier=tier_posteriors().argmax(1), sm=(size >= 2).astype(float),
                hit=nch >= 1, have=(nch >= 1) & ~np.isnan(band) & ~np.isnan(oa))


def fit_data(pop, marker="W2", mult=None, drop=()):
    k = np.ones(len(pop["w"])) if mult is None else mult
    have = pop["have"] & (k > 0)
    for i in drop:
        have[i] = False
    idx = np.where(have)[0]
    o = pop["oa"] == 1
    br = {"outcome": o, "W2": o | pop["soft"], "W1": o | pop["soft"] | pop["ext"]}[marker]
    ww = pop["w"][idx] * k[idx]
    return dict(idx=idx, H=pop["C"][idx], lo=pop["lo"][idx], hi=pop["hi"][idx], oa=br[idx].astype(float),
                sm=pop["sm"][idx], w=ww / ww.mean(), tier=pop["tier"][idx], nch=pop["nch"][idx], tg=pop["tg"][idx])


def quick_fit(D, form, p0, thorough=True):
    f = lambda p: -pcm.loglik_b(p, D, form)
    o = minimize(f, p0, method="L-BFGS-B", options={"maxiter": 20000})
    rounds = 2 if thorough else 1
    for _ in range(rounds):
        o = minimize(f, o.x, method="Nelder-Mead", options={"maxiter": 20000 if thorough else 4000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(f, o.x, method="L-BFGS-B", options={"maxiter": 20000})
    return o.x, -o.fun


def per_firm_cost(P, pop):
    """Expected annual cost per firm (all rows): handling, ordinary breaches, big breaches; and partial means above x."""
    C, e, sm, tg = pop["C"], pop["nch"] - 1, pop["sm"], pop["tg"]
    Lc = C * P["lam"][None] * np.exp(P["bt"][pop["tier"]] + P["bs"] * sm)[:, None]
    Lc[:, 0] *= np.exp(P["t"] * tg)
    h0 = sig(np.log(P["h0"] / (1 - P["h0"]))[None] + P["k"] * tg[:, None] * np.array([1.0, 0, 0, 0])[None])
    eh = ((1 - h0) * np.exp(P["mu_h"][None] + P["dh"] * sm[:, None] + P["s_h"] ** 2 / 2) * C).sum(1)
    m0 = sig(P["m0"] + P["dm0"] * sm + P["g0"] * e)
    mum = P["mu_m"][None] + (P["dm"] * sm + P["g"] * e)[:, None]
    pi = sig(P["pi_a"] + P["pi_b"] * e)
    eo = (Lc * ((1 - pi) * (1 - m0))[:, None] * np.exp(mum + P["s_m"] ** 2 / 2)).sum(1)
    eb = Lc.sum(1) * pi * np.exp(P["mu_big"] + P["s_big"] ** 2 / 2)

    def above(x):
        lx = np.log(x)
        so = (1 - m0)[:, None] * np.exp(mum + P["s_m"] ** 2 / 2) * norm.sf((lx - mum - P["s_m"] ** 2) / P["s_m"])
        sb = np.exp(P["mu_big"] + P["s_big"] ** 2 / 2) * norm.sf((lx - P["mu_big"] - P["s_big"] ** 2) / P["s_big"])
        return (Lc * ((1 - pi)[:, None] * so + (pi * sb)[:, None])).sum(1)
    return dict(handling=eh, ordinary=eo, big=eb, episodes=Lc.sum(1), above=above)


def national(vals, pop, mult=None, drop=()):
    k = np.ones(len(pop["w"])) if mult is None else mult
    w = pop["w"] * k
    have = pop["have"].copy()
    for i in drop:
        have[i] = False
    tot, by = 0.0, {}
    for s in (1, 2, 3, 4):
        a = pop["size"] == s
        scale = w[a & pop["hit"]].sum() / w[a & have].sum()
        v = (w[a & have] * vals[a & have]).sum() * scale / w[a].sum() * N[s]
        by[s] = v
        tot += v
    return tot, by


def total_of(P, pop, mult=None, drop=()):
    c = per_firm_cost(P, pop)
    return national(c["handling"] + c["ordinary"] + c["big"], pop, mult, drop)[0]


def main():
    pop = population()
    fits = np.load(SP + "poisson_breadth_fits.npy", allow_pickle=True)
    pF1, pF2 = np.asarray(fits[1], float), np.asarray(fits[2], float)
    D = fit_data(pop)
    p, ll = quick_fit(D, "F2R", np.r_[pF2, 0.39, -0.83])
    np.save(SP + "national_main_F2R.npy", p)
    P = pcm.unpack_b(p, "F2R")
    print(f"MAIN F2R: loglik {ll:.1f}; targeted phishing rate x{np.exp(P['t']):.2f}; P(big | breach) by channels "
          + "/".join(f"{sig(P['pi_a'] + P['pi_b'] * k):.3f}" for k in range(4)) + f"; big median £{np.exp(P['mu_big']):,.0f}, sigma {P['s_big']:.2f}")
    c = per_firm_cost(P, pop)
    comp = {k: national(c[k], pop) for k in ("handling", "ordinary", "big", "episodes")}
    tot = sum(comp[k][0] for k in ("handling", "ordinary", "big"))
    print(f"\nNATIONAL expected annual cost: £{tot / 1e9:.2f}bn;  breaches per year {comp['episodes'][0]:,.0f}")
    print("  by component: " + ", ".join(f"{k} £{comp[k][0] / 1e9:.2f}bn ({comp[k][0] / tot:.0%})" for k in ("handling", "ordinary", "big")))
    for s in (1, 2, 3, 4):
        st = sum(comp[k][1][s] for k in ("handling", "ordinary", "big"))
        print(f"  {SL[s]:6s} £{st / 1e9:.3f}bn ({st / tot:.0%});  per business £{st / N[s]:,.0f};  breaches {comp['episodes'][1][s]:,.0f}")
    for x in (5e5, 5e6):
        print(f"  share from single losses above £{x / 1e6:g}m: {national(c['above'](x), pop)[0] / tot:.2f}")

    print("\nSENSITIVITIES (refit each)")
    res = {"main F2R": tot}
    q, _ = quick_fit(D, "F1R", np.r_[pF1, 0.39, -0.83])
    res["breadth form: cost scale (F1R)"] = total_of(pcm.unpack_b(q, "F1R"), pop)
    res["no targeted-phishing split (F2)"] = total_of(pcm.unpack_b(pF2, "F2"), pop) if True else None
    for mk in ("outcome", "W1"):
        q, _ = quick_fit(fit_data(pop, marker=mk), "F2R", p)
        res[f"breach marker {mk}"] = total_of(pcm.unpack_b(q, "F2R"), pop)
    q, _ = quick_fit(fit_data(pop, drop=(1380,)), "F2R", p)
    res["firm 1380 excluded"] = total_of(pcm.unpack_b(q, "F2R"), pop, drop=(1380,))
    for k, v in res.items():
        print(f"  {k:34s} £{v / 1e9:.2f}bn  ({v / tot:.2f}x main)")

    mid = np.where(pop["hi"] == 0, 0.0, (pop["lo"] + pop["hi"]) / 2)
    emp = national(np.nan_to_num(mid), pop)[0]
    print(f"\nEMPIRICAL BASELINE (each firm's worst incident taken as its annual cost, interval midpoints): £{emp / 1e9:.2f}bn")


def boot(B):
    pop = population()
    p0 = np.load(SP + "national_main_F2R.npy")
    rng = np.random.default_rng(int.from_bytes(os.urandom(4), "little"))
    n = len(pop["w"])
    out = open(SP + "national_boot.log", "a")
    for b in range(B):
        k = np.zeros(n)
        for s in (1, 2, 3, 4):
            ids = np.where(pop["size"] == s)[0]
            np.add.at(k, rng.choice(ids, len(ids)), 1)
        D = fit_data(pop, mult=k)
        q, _ = quick_fit(D, "F2R", p0, thorough=False)
        t = total_of(pcm.unpack_b(q, "F2R"), pop, mult=k)
        out.write(f"{t:.6e}\n")
        out.flush()


if __name__ == "__main__":
    if sys.argv[1] == "main":
        main()
    elif sys.argv[1] == "boot":
        boot(int(sys.argv[2]) if len(sys.argv) > 2 else 100)


def boot_summary():
    """Summarise the bootstrap log: percentiles, share of replicates far above the main estimate."""
    x = np.loadtxt(SP + "national_boot.log") / 1e9
    print(f"replicates {len(x)}: median £{np.median(x):.2f}bn; 50% £{np.percentile(x, 25):.2f}-{np.percentile(x, 75):.2f}bn;"
          f" 90% £{np.percentile(x, 5):.2f}-{np.percentile(x, 95):.2f}bn; 95% £{np.percentile(x, 2.5):.2f}-{np.percentile(x, 97.5):.2f}bn")
    print(f"  min £{x.min():.2f}bn, max £{x.max():.2f}bn; share > £3bn {np.mean(x > 3):.3f}, > £5bn {np.mean(x > 5):.3f}")


if __name__ == "__main__" and sys.argv[1] == "bootsum":
    boot_summary()


# --- #11 part 2: attacked firms without usable cost answers -------------------------------------------------------
# 'have' = attacked with worst-incident band and outcome answer. Compare attacked firms without (missing) vs with, on
# what they did answer; then recompute the national total imputing each missing firm from the fitted model given its
# own channels, tier, size and targeted-phishing status, instead of scaling up the answering firms.
# Run: ... national_estimate.py missing

def missing():
    pop = population()
    r = aligned_raw()
    nm = lambda c: pd.to_numeric(r[c], errors="coerce").values
    hit, have, w, size = pop["hit"], pop["have"], pop["w"], pop["size"]
    miss = hit & ~have
    print(f"attacked firms {hit.sum()}: with cost data {have.sum()}, without {miss.sum()} "
          f"(weighted share without {w[miss].sum() / w[hit].sum():.3f})")
    print("  why missing: no worst band " + str(int((miss & np.isnan(pop['band'])).sum())) + ", no outcome answer "
          + str(int((miss & np.isnan(pop['oa'])).sum())))
    oa_ans = ~np.isnan(pop["oa"])
    feats = {"Micro": size == 1, "Small": size == 2, "Medium": size == 3, "Large": size == 4,
             "channels hit (mean)": pop["nch"], "3+ channels": pop["nch"] >= 3,
             "phishing": pop["C"][:, 0] == 1, "impersonation": pop["C"][:, 1] == 1, "ransomware": pop["C"][:, 2] == 1,
             "other serious": pop["C"][:, 3] == 1, "targeted phishing": pop["tg"] == 1,
             "tier mid": pop["tier"] == 1, "tier high": pop["tier"] == 2,
             "soft breach marker": pop["soft"], "attack freq weekly+": nm("freq") >= 4}
    print(f"  {'':22s} {'with':>7s} {'without':>8s}   (weighted)")
    for k, v in feats.items():
        v = v.astype(float)
        print(f"  {k:22s} {np.average(v[have], weights=w[have]):7.3f} {np.average(v[miss], weights=w[miss]):8.3f}")
    m = miss & oa_ans
    print(f"  outcome flag (where answered): with {np.average(pop['oa'][have], weights=w[have]):.3f}, without "
          f"{np.average(pop['oa'][m], weights=w[m]) if m.any() else float('nan'):.3f} (n {m.sum()})")
    p = np.load(SP + "national_main_F2R.npy")
    c = per_firm_cost(pcm.unpack_b(p, "F2R"), pop)
    v = c["handling"] + c["ordinary"] + c["big"]
    base, by_b = national(v, pop)
    tot, by = 0.0, {}
    for s in (1, 2, 3, 4):
        a = size == s
        val = (w[a & hit] * v[a & hit]).sum() / w[a].sum() * N[s]
        by[s] = val
        tot += val
    print(f"\nnational: scaling answering firms £{base / 1e9:.3f}bn;  model-imputing missing firms £{tot / 1e9:.3f}bn ({tot / base:.3f}x)")
    for s in (1, 2, 3, 4):
        a = size == s
        print(f"  {SL[s]:6s} missing weight share {w[a & miss].sum() / w[a & hit].sum():.3f};  model cost per firm: answering "
              f"£{np.average(v[a & have], weights=w[a & have]):,.0f}, missing £{np.average(v[a & miss], weights=w[a & miss]) if (a & miss).any() else float('nan'):,.0f};"
              f"  total £{by_b[s] / 1e9:.3f}bn -> £{by[s] / 1e9:.3f}bn")


if __name__ == "__main__" and sys.argv[1] == "missing":
    missing()


# Missing firms imputed conditional on their own breach marker (W2: outcome flag or soft marker, which they answered):
# E[annual | no breach] = handling; E[annual | breached] = handling + Lam E[G] / (1 - exp(-Lam)).
# Run: ... national_estimate.py missing2

def missing2():
    pop = population()
    hit, have, w, size = pop["hit"], pop["have"], pop["w"], pop["size"]
    miss = hit & ~have
    p = np.load(SP + "national_main_F2R.npy")
    c = per_firm_cost(pcm.unpack_b(p, "F2R"), pop)
    L = c["episodes"]
    eg = (c["ordinary"] + c["big"])
    br = ((pop["oa"] == 1) | pop["soft"])
    v = c["handling"] + eg
    vc = np.where(br, c["handling"] + eg / np.maximum(-np.expm1(-L), 1e-12), c["handling"])
    print(f"missing firms {miss.sum()}: breached (W2) {int((miss & br).sum())} ({np.average(br[miss], weights=w[miss]):.3f} weighted);"
          f" model P(breach) for them {np.average(-np.expm1(-L[miss]), weights=w[miss]):.3f}")
    base = national(v, pop)[0]
    tot = 0.0
    for s in (1, 2, 3, 4):
        a = size == s
        val = ((w[a & have] * v[a & have]).sum() + (w[a & miss] * vc[a & miss]).sum()) / w[a].sum() * N[s]
        tot += val
    print(f"national: scaling £{base / 1e9:.3f}bn;  imputing missing firms given their breach marker £{tot / 1e9:.3f}bn ({tot / base:.3f}x)")


if __name__ == "__main__" and sys.argv[1] == "missing2":
    missing2()
