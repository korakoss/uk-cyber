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


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


# --- Direct signals of repeat breaches (2026-10-02) ---------------------------------------------------------------
# A. Per-type yearly cost (asked of firms with >= 1 success in takeover / DoS / malware / ransomware) is a YEARLY TOTAL
#    for that channel; the worst incident is a single incident. Firms with exactly 1 success in the channel cannot have
#    repeats (material episodes are a subset of successes), so their per-type / worst ratio shows the reporting gap alone
#    (recall, concept). Firms with 2+ successes: if they really had several costly episodes, the ratio shifts up.
#    Upper-bound prediction: every success is an episode; extra episodes are draws from that channel's episode-cost
#    distribution (poisson_cost_model base fit) truncated below the worst incident; reported = (resampled 1-success
#    ratio) x yearly sum. Firms whose worst incident is that channel only.
# B. Distinct consequences among breached firms (outcome types + restore >= 1 day, staff stopped, recovery costs,
#    revenue loss): with repeats, firms with more expected episodes should report more distinct consequences at the same
#    worst cost. Weighted least squares of count on log expected episodes + worst-cost group dummies.
# Run: ... breach_repeats.py direct

def direct():
    import pandas as pd
    from latent_on_streams import aligned_raw
    from material_breach import data
    from broad_interaction import tightened
    import joint_five_channel as f
    from breach_measures import num as cnum
    import poisson_cost_model as pcm
    rng = np.random.default_rng(5)
    TB = {1: (0, 100), 2: (100, 250), 3: (250, 500), 4: (500, 1000), 5: (1000, 2000), 6: (2000, 5000), 7: (5000, 10000),
          8: (10000, 20000), 9: (20000, 50000), 10: (50000, 100000), 11: (100000, 250000), 12: (250000, 500000)}
    X, size, band, D, oa, w = data()
    r = aligned_raw()
    nm = lambda c: pd.to_numeric(r[c], errors="coerce").values
    disr = f.load_firms()["disrupta"].values
    lo, hi = tightened(band)
    wmid = np.where(hi == 0, 0.0, (lo + hi) / 2)
    P = pcm.unpack(np.load("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_params.npy"))
    rows = []
    for lab, cv, sv, codes, ch in (("takeover", "tkvrcost_bands", "tkvrsuc", [11], 3), ("DoS", "doscost_bands", "dossoft", [3], 3),
                                   ("malware", "viruscost_bands", "virussoft", [2], 3), ("ransomware", "ranscost_bands", "ranssoft", [1], 2)):
        c, s = nm(cv), cnum(r, sv)
        for i in np.where((c >= 1) & (c <= 12) & ~np.isnan(wmid) & (wmid > 0) & np.isin(disr, codes) & (s >= 1))[0]:
            t = TB[int(c[i])]
            rows.append(dict(lab=lab, ch=ch, n=int(s[i]), tlo=t[0], thi=t[1], tmid=(t[0] + t[1]) / 2, wlo=lo[i], whi=hi[i],
                             wmid=wmid[i], w=w[i], sm=float(size[i] >= 2)))
    R = pd.DataFrame(rows)
    R["ratio"] = R.tmid / R.wmid
    R["higher"] = R.tlo >= R.whi
    print("A. per-type yearly cost vs worst incident (worst = that channel), by success count")
    for g, m in (("1 success", R.n == 1), ("2+ successes", R.n >= 2)):
        x = R[m]
        print(f"  {g:13s} n {len(x):2d}  median ratio {x.ratio.median():.2f}  share ratio >= 1 {np.mean(x.ratio >= 1):.2f}"
              f"  clearly higher {int(x.higher.sum())}  floor '<£100' {int((x.thi == 100).sum())}  (by channel: " +
              ", ".join(f"{l} {int((x.lab == l).sum())}" for l in ("takeover", "DoS", "malware", "ransomware")) + ")")
    one, two = R[R.n == 1], R[R.n >= 2]
    print("  2+ firms (success count / per-type band / worst band):")
    for _, x in two.iterrows():
        print(f"     {x.lab:10s} n {x.n:3d}  type £{x.tlo:,.0f}-{x.thi:,.0f}  worst £{x.wlo:,.0f}-{x.whi:,.0f}  ratio {x.ratio:.2f}")
    if len(one) and len(two):
        rho = one.ratio.values
        sims_med, sims_hi, sims_ge1 = [], [], []
        for _ in range(2000):
            rep = []
            for _, x in two.iterrows():
                m0 = 1 / (1 + np.exp(-(P["m0"] + P["dm0"] * x.sm)))
                mu = P["mu_m"][x.ch] + P["dm"] * x.sm
                k = min(x.n, 20) - 1
                extra = 0.0
                if k > 0:
                    z = rng.random(k) >= m0
                    u = rng.random(k) * norm.cdf((np.log(x.wmid) - mu) / P["s_m"])
                    extra = np.where(z, np.exp(mu + P["s_m"] * norm.ppf(np.maximum(u, 1e-12))), 0.0).sum()
                rep.append(rng.choice(rho) * (x.wmid + extra) / x.wmid)
            rep = np.array(rep)
            sims_med.append(np.median(rep))
            sims_ge1.append(np.mean(rep >= 1))
        print(f"  prediction for 2+ if every success were a costly episode: median ratio {np.median(sims_med):.2f}"
              f" [90%: {np.percentile(sims_med, 5):.2f}-{np.percentile(sims_med, 95):.2f}], share >= 1 {np.median(sims_ge1):.2f}"
              f" [90%: {np.percentile(sims_ge1, 5):.2f}-{np.percentile(sims_ge1, 95):.2f}]")
        print(f"  no-repeat prediction for 2+ = 1-success distribution: median {np.median(rho):.2f}, share >= 1 {np.mean(rho >= 1):.2f}")

    print("\nB. distinct consequences among breached firms vs expected episodes (narrow firms, base Poisson fit)")
    Dn = pcm.setup()
    p0 = np.load("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/poisson_params.npy")
    Lc = pcm.firm_terms(pcm.unpack(p0), Dn)[0]
    L = Lc.sum(1)
    idx = Dn["idx"]
    cons = np.column_stack([nm(f"outcome{j}") == 1 for j in (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13)] +
                           [np.isin(nm("restore"), [3, 4, 5, 6]), nm("impact1") == 1, nm("impact4") == 1, nm("impact2") == 1]).sum(1)[idx]
    soft = (np.isin(nm("restore"), [3, 4, 5, 6]) | (nm("impact1") == 1) | (nm("impact4") == 1) | (nm("impact2") == 1))[idx]
    br = (Dn["oa"] == 1) | soft
    em = L / -np.expm1(-L)
    mid = np.where(Dn["hi"] == 0, 0.0, (Dn["lo"] + Dn["hi"]) / 2)
    grp = np.digitize(mid, [1, 500, 5000])
    m = br
    Xd = np.column_stack([np.log(em[m])] + [(grp[m] == g).astype(float) for g in range(4)])
    ww = Dn["w"][m]
    beta = np.linalg.lstsq(Xd * np.sqrt(ww)[:, None], cons[m] * np.sqrt(ww), rcond=None)[0]
    res = cons[m] - Xd @ beta
    cov = np.linalg.inv((Xd * ww[:, None]).T @ Xd) * np.average(res ** 2, weights=ww) * ww.sum() / (ww.sum() - Xd.shape[1])
    print(f"  breached firms {m.sum()}; mean consequences {np.average(cons[m], weights=ww):.2f}; E[episodes|breached] range {em[m].min():.2f}-{em[m].max():.2f}")
    print(f"  slope of consequences on log E[episodes | breached], given worst-cost group: {beta[0]:.2f} (se {np.sqrt(cov[0, 0]):.2f})")
    print(f"  Poisson-repeat expectation for comparison: one extra episode adds up to ~mean consequences per episode;"
          f" log E range {np.log(em[m]).min():.2f}-{np.log(em[m]).max():.2f}")
    cut = np.quantile(em[m], [1 / 3, 2 / 3])
    for k, g in enumerate(np.digitize(em[m], cut) == t for t in range(3)):
        print(f"    tercile {k}: E[episodes|breached] {em[m][g].mean():.2f}  mean consequences {np.average(cons[m][g], weights=ww[g]):.2f}"
              f"  mean worst-cost group {np.average(grp[m][g], weights=ww[g]):.2f}  n {g.sum()}")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "direct":
    direct()
