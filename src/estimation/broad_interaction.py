"""Do broad compromises (firms hit by 3+ channels) cost more than the per-channel model predicts? Two parts.

(a) Breach occurrence. Channels P, I, R, S (flags). Material breach = Q56A any outcome. Independence across channels:
    P(no material breach) = prod over hit channels (1 - q_c). Fit q_c by ML on firms hit by 1 or 2 channels, predict
    P(any material) for 3- and 4-channel firms; compare with observed (also by tier). Weighted.
(b) Breach size. Material firms' worst-incident cost on TIGHTENED intervals (total band intersected with the summed
    component range); zero cost as point mass; positive part interval-censored lognormal with shifts for the worst
    channel (D), Small+, and broad (3+ channels); common sigma. LR test and size of the broad shift.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/broad_interaction.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from counts_given_frailty import tier_posteriors, TIER
from latent_on_streams import aligned_raw
from material_breach import data, WB, NAMES
from type_cooccurrence_structure import SHORT


def channels(X):
    P = X[:, 0] == 1
    I = X[:, SHORT.index("Imper")] == 1
    R = X[:, SHORT.index("Ransm")] == 1
    S = np.delete(X, [0, SHORT.index("Imper"), SHORT.index("Ransm")], axis=1).max(1) == 1
    return np.column_stack([P, I, R, S]).astype(float)


def tightened(band):
    r = aligned_raw()
    comp = ["damagedirsx_bands", "damagedirlx_bands", "damagestaffx_bands", "damageindx_bands"]
    V = np.column_stack([pd.to_numeric(r[c], errors="coerce").values for c in comp])
    full = ((V >= 1) & (V <= 13)).all(1) & ~np.isnan(band)
    lo = np.array([WB[int(b)][0] if not np.isnan(b) else np.nan for b in band])
    hi = np.array([WB[int(b)][1] if not np.isnan(b) else np.nan for b in band])
    for i in np.where(full)[0]:
        slo = sum(WB[int(V[i, j])][0] for j in range(4))
        shi = sum(WB[int(V[i, j])][1] for j in range(4))
        if slo <= hi[i] and shi >= lo[i]:
            lo[i], hi[i] = max(lo[i], slo), min(hi[i], shi)
    return lo, hi


def part_a(C, oa, w, tier):
    nch = C.sum(1)
    ok = ~np.isnan(oa) & (nch >= 1)
    fitm = ok & (nch <= 2)
    y, Cf, wf = oa[fitm], C[fitm], w[fitm] / w[fitm].mean()

    def nll(z):
        q = 1 / (1 + np.exp(-z))
        p_none = np.exp(Cf @ np.log(1 - q))
        return -(wf * (y * np.log(np.maximum(1 - p_none, 1e-12)) + (1 - y) * np.log(np.maximum(p_none, 1e-12)))).sum()
    z = minimize(nll, np.full(4, -2.0), method="BFGS").x
    q = 1 / (1 + np.exp(-z))
    print("(a) material-breach rate per channel (fit on 1-2 channel firms): " + ", ".join(f"{n} {v:.3f}" for n, v in zip("PIRS", q)))
    pred = 1 - np.exp(C @ np.log(1 - q))
    print("    by number of channels hit: observed vs predicted P(any material breach)")
    for k in (1, 2, 3, 4):
        m = ok & (nch == k)
        print(f"      {k} channel(s)  n {m.sum():4d}  obs {np.average(oa[m], weights=w[m]):.3f}  pred {np.average(pred[m], weights=w[m]):.3f}"
              + ("   (fit sample)" if k <= 2 else "   (out of sample)"))
    print("    3+ channels by tier:")
    for c in range(3):
        m = ok & (nch >= 3) & (tier == c)
        if m.sum():
            print(f"      tier {TIER[c]:4s} n {m.sum():3d}  obs {np.average(oa[m], weights=w[m]):.3f}  pred {np.average(pred[m], weights=w[m]):.3f}")
    # bootstrap the 3+ gap
    rng = np.random.default_rng(0)
    idx3 = np.where(ok & (nch >= 3))[0]
    gaps = []
    for _ in range(500):
        b = rng.choice(idx3, len(idx3))
        gaps.append(np.average(oa[b], weights=w[b]) - np.average(pred[b], weights=w[b]))
    lo, hi = np.percentile(gaps, [5, 95])
    print(f"    3+ channels: obs - pred {np.mean(gaps):+.3f}  90% [{lo:+.3f}, {hi:+.3f}] (resampling firms; q held fixed)")


def part_b(C, band, D, oa, w, size):
    lo, hi = tightened(band)
    nch = C.sum(1)
    m = (oa == 1) & ~np.isnan(band) & ~np.isnan(D)
    pos = m & (band >= 2)
    ch = (D[pos] - 1).astype(int)
    L, H = np.log(np.maximum(lo[pos], 1)), np.log(hi[pos])
    sm, br, ww = (size[pos] >= 2).astype(float), (nch[pos] >= 3).astype(float), w[pos] / w[pos].mean()
    E = np.eye(4)[ch]

    def fit(use_broad):
        def nll(p):
            mu = E @ p[:4] + p[4] * sm + (p[5] * br if use_broad else 0)
            s = np.exp(p[6])
            return -(ww * np.log(np.maximum(norm.cdf((H - mu) / s) - norm.cdf((L - mu) / s), 1e-300))).sum()
        o = minimize(nll, np.r_[np.full(4, 7.0), 0.3, 0.0, 0.7], method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(nll, o.x, method="BFGS")
        return -o.fun, o.x
    l0, p0 = fit(False)
    l1, p1 = fit(True)
    print(f"\n(b) material breaches with positive cost: {pos.sum()} (broad 3+ channels: {int(br.sum())})")
    print(f"    broad shift on log cost {p1[5]:+.2f} -> cost x{np.exp(p1[5]):.2f};  LR p {chi2.sf(2 * (l1 - l0), 1):.3f};  sigma {np.exp(p1[6]):.2f}")
    print("    channel medians (Micro, non-broad): " + ", ".join(f"{n} £{np.exp(v):,.0f}" for n, v in zip(NAMES, p1[:4])))
    zero = m & (band == 1)
    print(f"    zero-cost share among material breaches: broad {np.average(zero[m & (nch >= 3)], weights=w[m & (nch >= 3)]):.2f}  "
          f"non-broad {np.average(zero[m & (nch < 3)], weights=w[m & (nch < 3)]):.2f}")


def main():
    X, size, band, D, oa, w = data()
    C = channels(X)
    tier = tier_posteriors().argmax(1)
    part_a(C, oa, w, tier)
    part_b(C, band, D, oa, w, size)


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def combined_check():
    """Can multiplicity alone explain the broad-firm cost excess? Model: each hit channel c independently yields a
    material breach with prob q_c (part a); each breach costs 0 with prob z, else lognormal(mu_c + delta*Small+, sigma).
    Worst = max over the firm's breaches. Conditional on >= 1 material breach:
        P(worst <= x) = [prod_c (1 - q_c + q_c F_c(x)) - prod_c (1 - q_c)] / [1 - prod_c (1 - q_c)].
    Fit (mu_c, delta, sigma, z) on material firms hit by 1-2 channels (tightened intervals), predict the worst-cost
    distribution of 3+ channel material firms, compare with observed. No interaction term."""
    X, size, band, D, oa, w = data()
    C = channels(X)
    nch = C.sum(1)
    lo, hi = tightened(band)
    ok = (oa == 1) & ~np.isnan(band) & (nch >= 1)
    # q from part (a)
    fitq = ~np.isnan(oa) & (nch >= 1) & (nch <= 2)
    yq, Cq, wq = oa[fitq], C[fitq], w[fitq] / w[fitq].mean()
    zq = minimize(lambda z: -(wq * (yq * np.log(np.maximum(1 - np.exp(Cq @ np.log(1 - 1 / (1 + np.exp(-z)))), 1e-12)) +
                                    (1 - yq) * (Cq @ np.log(1 - 1 / (1 + np.exp(-z)))))).sum(), np.full(4, -2.0), method="BFGS").x
    q = 1 / (1 + np.exp(-zq))
    sm = (size >= 2).astype(float)

    def cdf_worst(x, Ci, smi, p):
        mu, delta, s, z = p[:4], p[4], np.exp(p[5]), 1 / (1 + np.exp(-p[6]))
        if x <= 0:
            F = np.full(4, z)
        else:
            F = z + (1 - z) * norm.cdf((np.log(x) - (mu + delta * smi)) / s)
        prod_all = np.prod(np.where(Ci == 1, 1 - q + q * F, 1.0))
        prod0 = np.prod(np.where(Ci == 1, 1 - q, 1.0))
        return (prod_all - prod0) / (1 - prod0)

    def interval_prob(i, p):
        L, H = lo[i], hi[i]
        if H == 0:
            return cdf_worst(0, C[i], sm[i], p)
        return cdf_worst(H, C[i], sm[i], p) - (cdf_worst(L, C[i], sm[i], p) if L > 0 else cdf_worst(0, C[i], sm[i], p) * 0)

    fit_idx = np.where(ok & (nch <= 2))[0]
    wf = w[fit_idx] / w[fit_idx].mean()

    def nll(p):
        return -sum(wf[k] * np.log(max(interval_prob(i, p), 1e-300)) for k, i in enumerate(fit_idx))
    o = minimize(nll, np.r_[np.full(4, 6.0), 0.3, 0.7, -1.5], method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-4, "fatol": 1e-6})
    p = o.x
    print(f"fit on {len(fit_idx)} material firms with 1-2 channels: medians " +
          ", ".join(f"{n} £{np.exp(v):,.0f}" for n, v in zip("PIRS", p[:4])) +
          f"; size x{np.exp(p[4]):.2f}; sigma {np.exp(p[5]):.2f}; zero share {1 / (1 + np.exp(-p[6])):.2f}")
    groups = [(0, 0), (1, 500), (500, 5000), (5000, 20000), (20000, 1e9)]
    labs = "none / <£500 / £500-5k / £5k-20k / £20k+"
    print(f"\nworst-incident cost of material firms: observed vs predicted ({labs})")
    for k_lab, sel in (("1-2 channels (fit)", ok & (nch <= 2)), ("3 channels", ok & (nch == 3)), ("4 channels", ok & (nch == 4)),
                       ("3+ channels", ok & (nch >= 3))):
        idx = np.where(sel)[0]
        ww = w[idx]
        obs, pred = np.zeros(5), np.zeros(5)
        for i, wi in zip(idx, ww):
            mid = (lo[i] + hi[i]) / 2
            g = 0 if hi[i] == 0 else next(j for j, (a, b) in enumerate(groups) if j > 0 and a <= mid < b)
            obs[g] += wi
            cs = [cdf_worst(0, C[i], sm[i], p)] + [cdf_worst(b, C[i], sm[i], p) for _, b in groups[1:-1]] + [1.0]
            pred += wi * np.diff(np.r_[0.0, cs])
        obs, pred = obs / ww.sum(), pred / ww.sum()
        print(f"  {k_lab:20s} n {len(idx):3d}  obs  " + " ".join(f"{v:.2f}" for v in obs))
        print(f"  {'':20s}        pred " + " ".join(f"{v:.2f}" for v in pred) +
              f"   P(>=£5k) obs {obs[3:].sum():.2f} pred {pred[3:].sum():.2f}")


if __name__ == "__main__" and "combined" in __import__("sys").argv:
    combined_check()


def anatomy():
    """What are costly broad compromises? Attacked firms with a worst-incident band, three groups:
    NC narrow (1-2 channels) & worst >= £5k;  BC broad (3+ channels) & worst >= £5k;  BL broad & worst < £1k.
    Rows (weighted shares unless stated): firm size; frequency answer; chains (Q64B: phishing worst incident 'resulted in'
    ..., asked only when the worst incident was phishing); frauds attributed to a cause (Q88D); number of channels with a
    recorded success; outcomes / impacts counts; restore >= 1 day; reported to police / Action Fraud / NCA / ICO / NCSC."""
    X, size, band, D, oa, w = data()
    r = aligned_raw()
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values
    C = channels(X)
    nch = C.sum(1)
    ok = ~np.isnan(band) & (nch >= 1)
    groups = {"NC narrow >=£5k": ok & (nch <= 2) & (band >= 6), "BC broad >=£5k": ok & (nch >= 3) & (band >= 6),
              "BL broad <£1k": ok & (nch >= 3) & (band <= 3)}
    def wmean(v, m):
        mm = m & ~np.isnan(v)
        return np.average(v[mm], weights=w[mm]) if mm.any() else float("nan")
    rows = []
    rows.append(("n firms", {g: int(m.sum()) for g, m in groups.items()}))
    for k, lab in ((1, "size Micro"), (2, "size Small"), (3, "size Medium"), (4, "size Large")):
        rows.append((lab, {g: wmean((size == k).astype(float), m) for g, m in groups.items()}))
    fq = num("freq") if "freq" in r else np.full(len(X), np.nan)
    import joint_five_channel as jf
    fq = jf.load_firms()["freq"].values
    for k, lab in ((1, "freq once"), (2, "freq <monthly"), (3, "freq monthly"), (4, "freq weekly+")):
        v = np.where(np.isnan(fq), np.nan, (fq >= 4) if k == 4 else (fq == k)).astype(float)
        rows.append((lab, {g: wmean(v, m) for g, m in groups.items()}))
    ph_worst = D == 1
    chain_items = {1: "ransomware", 2: "malware", 3: "DoS", 4: "bank hacking", 5: "impersonation", 8: "outsider access",
                   10: "takeover", 11: "money moved", 13: "fake invoice paid", 14: "other attacks"}
    anych = np.zeros(len(X))
    for j in chain_items:
        anych = np.maximum(anych, (num(f"disruptphish{j}") == 1).astype(float))
    rows.append(("[worst=phishing] n", {g: int((m & ph_worst).sum()) for g, m in groups.items()}))
    rows.append(("[worst=phishing] led to another attack/fraud", {g: wmean(np.where(ph_worst, anych, np.nan), m) for g, m in groups.items()}))
    cont = np.column_stack([num(c) for c in ("fraudconta", "fraudcontb", "fraudcontc", "fraudcontd", "fraudconte",
                                              "fraudcontf", "fraudcontg", "fraudconth", "fraudconti")])
    anyfraud = (np.nansum(np.where((cont >= 1) & (cont < 997), cont, 0), 1) >= 1).astype(float)
    rows.append(("fraud attributed to an attack (Q88D)", {g: wmean(anyfraud, m) for g, m in groups.items()}))
    succ = np.column_stack([(num(c) >= 1) & (num(c) < 997) for c in ("phisheng", "tkvrsuc", "dossoft", "virussoft", "ranssoft")])
    nsucc = succ.sum(1).astype(float)
    for k, lab in ((0, "channels with a success: 0"), (1, "channels with a success: 1"), (2, "channels with a success: 2+")):
        v = (nsucc >= 2) if k == 2 else (nsucc == k)
        rows.append((lab, {g: wmean(v.astype(float), m) for g, m in groups.items()}))
    oc = np.column_stack([num(f"outcome{j}") == 1 for j in (1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13)]).sum(1).astype(float)
    im = np.column_stack([num(f"impact{j}") == 1 for j in (1, 2, 4, 7, 8, 9, 10, 13, 14)]).sum(1).astype(float)
    rows.append(("mean # material outcomes", {g: wmean(oc, m) for g, m in groups.items()}))
    rows.append(("mean # impacts (excl. staff time, new measures)", {g: wmean(im, m) for g, m in groups.items()}))
    for j, lab in ((1, "outcome: systems corrupted"), (2, "outcome: personal data"), (6, "outcome: money stolen"),
                   (7, "outcome: services down"), (4, "outcome: temp access loss"), (11, "outcome: paid attackers")):
        rows.append((lab, {g: wmean((num(f"outcome{j}") == 1).astype(float), m) for g, m in groups.items()}))
    rest = num("restore")
    rows.append(("restore >= 1 day", {g: wmean(np.where((rest >= 1) & (rest < 997), (rest >= 3).astype(float), np.nan), m) for g, m in groups.items()}))
    rep = np.column_stack([num(f"reportb{j}") == 1 for j in (1, 9, 12, 24, 48)]).any(1).astype(float)
    rows.append(("reported to police/AF/NCA/ICO/NCSC", {g: wmean(rep, m) for g, m in groups.items()}))
    names = list(groups)
    print(f"{'':48s}" + "".join(f"{g:>18s}" for g in names))
    for lab, vals in rows:
        print(f"{lab:48s}" + "".join(f"{vals[g]:>18.2f}" if isinstance(vals[g], float) else f"{vals[g]:>18d}" for g in names))


if __name__ == "__main__" and "anatomy" in __import__("sys").argv:
    anatomy()


def broad_structure():
    """Within broad (3+ channel) firms with a material breach: does the cost distribution vary by channel combination,
    by number of channels, by size? Tightened intervals; zero cost as weighted share; positive part interval-censored
    lognormal. Base: one mu, one sigma. Added one at a time: ransomware present, 4 channels (vs 3), Small+, worst
    channel (D) dummies. LR tests (weighted pseudo-likelihood, approximate). Also per-combination tabulation."""
    X, size, band, D, oa, w = data()
    C = channels(X)
    nch = C.sum(1)
    lo, hi = tightened(band)
    m = (oa == 1) & ~np.isnan(band) & (nch >= 3)
    print(f"broad material firms: {m.sum()};  zero-cost share {np.average(band[m] == 1, weights=w[m]):.2f}")
    combo = np.array(["".join(c for c, f in zip("PIRS", row) if f) for row in C.astype(int)])
    print("\nby channel combination: n, weighted shares <£500 / £500-5k / £5k-20k / £20k+, weighted mean of interval midpoints")
    for cb in sorted(set(combo[m]), key=lambda s: -(combo[m] == s).sum()):
        mm = m & (combo == cb)
        mid = (lo[mm] + hi[mm]) / 2
        sh = [np.average((mid >= a) & (mid < b), weights=w[mm]) for a, b in ((0, 500), (500, 5000), (5000, 20000), (20000, 1e9))]
        print(f"  {cb:5s} n {mm.sum():3d}  " + " ".join(f"{v:.2f}" for v in sh) + f"   mean £{np.average(mid, weights=w[mm]):,.0f}")
    pos = m & (band >= 2)
    L, H = np.log(np.maximum(lo[pos], 1)), np.log(hi[pos])
    ww = w[pos] / w[pos].mean()
    covs = {"ransomware present": C[pos, 2], "4 channels": (nch[pos] == 4).astype(float), "Small+": (size[pos] >= 2).astype(float),
            "worst channel (D) dummies": np.column_stack([(D[pos] == k).astype(float) for k in (2, 3, 4)])}

    def fit(Z):
        Z = np.column_stack([np.ones(pos.sum())] + ([Z] if Z is not None else []))
        def nll(p):
            mu, s = Z @ p[:-1], np.exp(p[-1])
            return -(ww * np.log(np.maximum(norm.cdf((H - mu) / s) - norm.cdf((L - mu) / s), 1e-300))).sum()
        o = minimize(nll, np.r_[8.0, np.zeros(Z.shape[1] - 1), 0.5], method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
        o = minimize(nll, o.x, method="BFGS")
        return -o.fun, o.x
    l0, p0 = fit(None)
    print(f"\npositive-cost broad breaches n {pos.sum()}: base median £{np.exp(p0[0]):,.0f}, sigma {np.exp(p0[-1]):.2f}, "
          f"mean (positive part) £{np.exp(p0[0] + np.exp(p0[-1]) ** 2 / 2):,.0f}")
    for nm, Z in covs.items():
        l1, p1 = fit(Z)
        k = 1 if Z.ndim == 1 else Z.shape[1]
        eff = ", ".join(f"x{np.exp(v):.2f}" for v in p1[1:1 + k])
        print(f"  + {nm:28s} effect {eff:24s} LR p {chi2.sf(2 * (l1 - l0), k):.3f}   sigma {np.exp(p1[-1]):.2f}")


if __name__ == "__main__" and "bstruct" in __import__("sys").argv:
    broad_structure()
