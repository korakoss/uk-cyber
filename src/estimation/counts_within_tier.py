"""Step 4 of the counts argument: channel count shape once the shared latent is factored out.

Tiers come from the count-agnostic flag model of Step 3 (3-class latent class model,
frailty_shape_flags.py), fitted within size group. Leave-one-out: for phishing, tiers use the
OTHER types' flags only, excluding phishing AND impersonation (co-labelled with phishing); for
ransomware, excluding ransomware AND malware. Within each tier x size group, the exact count
distribution (phishing count N; ransomware count) among firms that report it:
  mean, variance, dispersion (var/mean), quantiles, and zero-truncated Poisson vs NegBin fits
  (counts are only observed for flagged firms, so N >= 1).
If the latent explained all spread, within-tier counts would look Poisson. Tier assignment is
uncertain (reported as mean posterior certainty); misassignment mixes tiers and inflates spread,
so leftover overdispersion is an upper bound on channel-private spread.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/counts_within_tier.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

import joint_four_channel as j
import joint_five_channel as f
from type_cooccurrence_structure import load
from frailty_shape_flags import lca, patterns

TIER = ["low", "mid", "high"]


def posteriors(Xo, w, rng, K=3):
    P, c = patterns(Xo, w)
    _, pi, th = lca(P, c, K, rng, starts=20)
    order = np.argsort(th.mean(1))
    pi, th = pi[order], th[order]
    lp = np.log(pi)[None] + Xo @ np.log(th).T + (1 - Xo) @ np.log(1 - th).T
    post = np.exp(lp - lp.max(1, keepdims=True))
    return post / post.sum(1, keepdims=True), pi


def fit_trunc(x, w):
    x = x.astype(int)

    def nll_p(v):
        lam = np.exp(v[0])
        lp = poisson.logpmf(x, lam) - np.log1p(-np.exp(-lam))
        return -(w * lp).sum()

    def nll_nb(v):
        m, r = np.exp(v)
        p = r / (r + m)
        lp = nbinom.logpmf(x, r, p) - np.log1p(-np.clip(nbinom.pmf(0, r, p), 0, 1 - 1e-12))
        return -(w * lp).sum()

    rp = minimize(nll_p, [np.log(max(np.average(x, weights=w), 1.01))], method="Nelder-Mead")
    rn = min((minimize(nll_nb, [np.log(max(np.average(x, weights=w), 1.01)), np.log(r0)], method="Nelder-Mead")
              for r0 in (0.1, 0.5, 2.0)), key=lambda r: r.fun)
    return rp.fun, rn.fun, np.exp(rn.x[1])


def analyse(d, X, w, size, target, exclude, label, rng):
    keep_cols = [i for i in range(X.shape[1]) if i not in exclude]
    print("\n" + "=" * 96)
    print(label)
    print("=" * 96)
    print(f"  {'size':>6s} {'tier':>5s} {'tier share':>10s} {'n':>4s} {'certainty':>9s} {'mean':>7s} {'var/mean':>9s}"
          f" {'q25/50/75/90':>18s}   {'NB r':>6s} {'LL gain NB vs Pois':>18s}")
    for slab, sizes in (("Micro", [1]), ("Rest", [2, 3, 4])):
        m = np.isin(size, sizes)
        ww = w[m] / w[m].mean()
        post, pi = posteriors(X[m][:, keep_cols], ww, rng)
        tier = post.argmax(1)
        cert = post.max(1)
        cnt = target[m]
        for t in range(3):
            sel = (tier == t) & ~np.isnan(cnt)
            if sel.sum() < 4:
                print(f"  {slab:>6s} {TIER[t]:>5s} {pi[t]:10.2f} {sel.sum():4d}   (too few)")
                continue
            x, wx = cnt[sel], ww[sel]
            mu = np.average(x, weights=wx)
            var = np.average((x - mu) ** 2, weights=wx)
            q = np.percentile(np.repeat(x, np.maximum(1, np.round(wx * 10).astype(int))), [25, 50, 75, 90])
            llp, lln, r = fit_trunc(x, wx)
            print(f"  {slab:>6s} {TIER[t]:>5s} {pi[t]:10.2f} {sel.sum():4d} {cert[sel].mean():9.2f} {mu:7.1f} {var / mu:9.1f}"
                  f"   {'/'.join(f'{v:.0f}' for v in q):>16s}   {r:6.2f} {llp - lln:18.1f}")


def main():
    X, w, size = load()
    d = f.load_firms()
    rng = np.random.default_rng(0)
    N = np.where(d["fP"].values == 1, d["N"].values, np.nan)
    analyse(d, X, w, size, N, exclude=[0, 1],
            label="PHISHING COUNT within tiers (tiers from flags excluding phishing + impersonation)", rng=rng)
    NR = d["NR"].values
    analyse(d, X, w, size, NR, exclude=[2, 3],
            label="RANSOMWARE COUNT within tiers (tiers from flags excluding ransomware + malware)", rng=rng)


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def poisson_prediction_test():
    """Sharper Step-4 test. Tiers from the Step-3 flag LCA on ALL 10 types (within size).
    Under Poisson-within-tier, a tier's hit probability p pins its mean: lambda = -ln(1-p), so
    counts among hit firms follow zero-truncated Poisson(lambda) -- a prediction with nothing
    fitted to counts. Compare with observed counts (phishing N; ransomware count).
    Co-labelling (more phishing incidents -> more impersonation ticks -> pushed to higher tiers)
    biases the low tier towards FEWER high counts, i.e. against finding overdispersion."""
    X, w, size = load()
    d = f.load_firms()
    rng = np.random.default_rng(0)
    for chan, col, cnt in (("PHISHING", 0, np.where(d["fP"].values == 1, d["N"].values, np.nan)),
                           ("RANSOMWARE", 2, d["NR"].values)):
        print("\n" + "=" * 100)
        print(f"{chan}: observed counts among hit firms vs Poisson prediction lambda = -ln(1 - p_tier)")
        print("=" * 100)
        print(f"  {'size':>6s} {'tier':>5s} {'share':>6s} {'p_hit':>6s} {'lambda':>7s} {'n':>4s} {'cert':>5s}"
              f"  {'P(1)':>11s} {'P(2-3)':>11s} {'P(4-10)':>11s} {'P(>10)':>11s} {'mean':>13s}   (obs / Poisson)")
        for slab, sizes in (("Micro", [1]), ("Rest", [2, 3, 4])):
            m = np.isin(size, sizes)
            ww = w[m] / w[m].mean()
            P, c = patterns(X[m], ww)
            _, pi, th = lca(P, c, 3, rng, starts=20)
            order = np.argsort(th.mean(1))
            pi, th = pi[order], th[order]
            Xm = X[m]
            lp = np.log(pi)[None] + Xm @ np.log(th).T + (1 - Xm) @ np.log(1 - th).T
            post = np.exp(lp - lp.max(1, keepdims=True))
            post /= post.sum(1, keepdims=True)
            tier, cert = post.argmax(1), post.max(1)
            x_all = cnt[m]
            for t in range(3):
                p = th[t, col]
                lam = -np.log(1 - p)
                sel = (tier == t) & ~np.isnan(x_all)
                if sel.sum() == 0:
                    continue
                x, wx = x_all[sel], ww[sel]
                k = np.arange(0, 5000)
                pk = poisson.pmf(k, lam)
                pk[0] = 0
                pk /= pk.sum()
                bins = [(1, 1), (2, 3), (4, 10), (11, 10**9)]
                obs = [np.average((x >= a) & (x <= b), weights=wx) for a, b in bins]
                pred = [pk[(k >= a) & (k <= b)].sum() for a, b in bins]
                print(f"  {slab:>6s} {TIER[t]:>5s} {pi[t]:6.2f} {p:6.3f} {lam:7.3f} {sel.sum():4d} {cert[sel].mean():5.2f}  " +
                      " ".join(f"{o:5.2f}/{q:5.2f}" for o, q in zip(obs, pred)) +
                      f"  {np.average(x, weights=wx):6.1f}/{(k * pk).sum():5.2f}")


if __name__ == "__main__" and "poisson" in __import__("sys").argv:
    poisson_prediction_test()


def _tiers(X, w, rng):
    P, c = patterns(X, w)
    _, pi, th = lca(P, c, 3, rng, starts=20)
    order = np.argsort(th.mean(1))
    pi, th = pi[order], th[order]
    lp = np.log(pi)[None] + X @ np.log(th).T + (1 - X) @ np.log(1 - th).T
    post = np.exp(lp - lp.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    return post.argmax(1), post.max(1), pi, th


def poisson_single_channel_test():
    """Test 2: Poisson for impersonation / other serious via the FREQUENCY answer of firms hit by
    ONLY that channel (their answer reflects that channel's count). Within tier, Poisson(lambda =
    -ln(1-p_tier)) predicts P(K>=2 | K>=1) and P(K>=12 | K>=1); compare with shares answering
    'more than once' (freq>=2) and 'monthly or more often' (freq>=3). Phishing-only firms as a
    reference where Poisson is known to fail. Frequency answers are noisy (some 'once' firms report
    many phishing attacks), so this is coarse."""
    X, w, size = load()
    d = f.load_firms()
    rng = np.random.default_rng(1)
    freq = d["freq"].values
    fl = {k: d[k].values == 1 for k in ("fP", "fI", "fR", "fS")}
    only = {"phishing-only (reference)": (fl["fP"] & ~fl["fI"] & ~fl["fR"] & ~fl["fS"], 0),
            "impersonation-only": (fl["fI"] & ~fl["fP"] & ~fl["fR"] & ~fl["fS"], 1),
            "other-serious-only": (fl["fS"] & ~fl["fP"] & ~fl["fI"] & ~fl["fR"], None)}
    print("\n" + "=" * 100)
    print("TEST 2: single-channel firms' frequency answer vs Poisson at their tier's rate")
    print("=" * 100)
    print(f"  {'group':>26s} {'size':>6s} {'tier':>5s} {'n':>4s} {'lambda':>7s}   "
          f"{'>once obs/Pois':>15s}   {'monthly+ obs/Pois':>18s}")
    for slab, sizes in (("Micro", [1]), ("Rest", [2, 3, 4])):
        m = np.isin(size, sizes)
        ww = w[m] / w[m].mean()
        tier, cert, pi, th = _tiers(X[m], ww, rng)
        for lab, (mask, col) in only.items():
            mm = mask[m] & ~np.isnan(freq[m])
            for t in range(3):
                sel = mm & (tier == t)
                if sel.sum() < 5:
                    continue
                # other serious = any of the serious subtypes except ransomware (cols 3..9)
                p = th[t, col] if col is not None else 1 - np.prod(1 - th[t, 3:])
                lam = -np.log(1 - p)
                p1 = 1 - np.exp(-lam)
                ge2 = (p1 - lam * np.exp(-lam)) / p1
                ge12 = poisson.sf(11, lam) / p1
                fq, wx = freq[m][sel], ww[sel]
                print(f"  {lab:>26s} {slab:>6s} {TIER[t]:>5s} {sel.sum():4d} {lam:7.3f}   "
                      f"{np.average(fq >= 2, weights=wx):6.2f}/{ge2:6.3f}   {np.average(fq >= 3, weights=wx):8.2f}/{ge12:8.1e}")


def negbin_fit_test():
    """Test 4: NegBin acceptability. Within size group, tiers from the flag LCA; phishing count
    N (>=1, reporters) modelled as T + M with T ~ Poisson(lT_t), M ~ NegBin(mean mM_t, shape r),
    r SHARED across tiers (model assumption: the latent scales the mean, not the shape);
    zero-truncated; weighted MLE. Goodness of fit on coarse bins with Kish n_eff. Also: implied
    P(N>=1) per tier vs the tier's phishing hit probability from the flags (consistency).
    Ransomware: NegBin alone, shared r (few firms)."""
    from scipy.stats import chi2 as chi2d
    X, w, size = load()
    d = f.load_firms()
    rng = np.random.default_rng(2)
    K = np.arange(0, 3001)
    BINS = [(1, 1), (2, 3), (4, 10), (11, 50), (51, 10**9)]

    def pmf_TM(lT, mM, r):
        pT = poisson.pmf(K, lT)
        pM = nbinom.pmf(K, r, r / (r + mM))
        return np.convolve(pT, pM)[:len(K)]

    def pmf_NB(mM, r):
        return nbinom.pmf(K, r, r / (r + mM))

    for chan, cnt_all, col, kind in (("PHISHING (targeted Poisson + mass NegBin)",
                                      np.where(d["fP"].values == 1, d["N"].values, np.nan), 0, "TM"),
                                     ("RANSOMWARE (NegBin)", d["NR"].values, 2, "NB")):
        print("\n" + "=" * 100)
        print(f"TEST 4: {chan}, shape r shared across tiers, within size group")
        print("=" * 100)
        for slab, sizes in (("Micro", [1]), ("Rest", [2, 3, 4])):
            m = np.isin(size, sizes)
            ww = w[m] / w[m].mean()
            tier, cert, pi, th = _tiers(X[m], ww, rng)
            cnt = cnt_all[m]
            tiers = [t for t in range(3) if ((tier == t) & ~np.isnan(cnt)).sum() >= 3]
            data = {t: (np.minimum(cnt[(tier == t) & ~np.isnan(cnt)], K[-1]).astype(int),
                        ww[(tier == t) & ~np.isnan(cnt)]) for t in tiers}

            def unpack(v):
                r = np.exp(v[0])
                per = {}
                for i, t in enumerate(tiers):
                    if kind == "TM":
                        per[t] = pmf_TM(np.exp(v[1 + 2 * i]), np.exp(v[2 + 2 * i]), r)
                    else:
                        per[t] = pmf_NB(np.exp(v[1 + i]), r)
                return r, per

            def nll(v):
                _, per = unpack(v)
                tot = 0.0
                for t in tiers:
                    x, wx = data[t]
                    pk = per[t]
                    tot -= (wx * (np.log(np.maximum(pk[x], 1e-300)) - np.log(max(1 - pk[0], 1e-300)))).sum()
                return tot

            npt = 2 if kind == "TM" else 1
            best = None
            for r0 in (0.05, 0.2, 1.0):
                v0 = [np.log(r0)]
                for t in tiers:
                    mu = np.average(data[t][0], weights=data[t][1])
                    v0 += ([np.log(0.5), np.log(mu)] if kind == "TM" else [np.log(mu)])
                res = minimize(nll, v0, method="Nelder-Mead", options=dict(maxiter=20000, maxfev=20000,
                                                                            xatol=1e-5, fatol=1e-6))
                best = res if best is None or res.fun < best.fun else best
            r, per = unpack(best.x)
            print(f"  {slab}: shared r = {r:.3f}")
            print(f"    {'tier':>5s} {'n':>4s} {'n_eff':>6s}  " + "  ".join(f"{('%d' % a if a == b else '%d-%s' % (a, b if b < 10**9 else '+')):>11s}"
                                                                      for a, b in BINS) +
                  f"  {'GOF p':>6s}  {'P(N>=1) fit/flags':>18s}" + ("  targeted/mass mean" if kind == "TM" else ""))
            for i, t in enumerate(tiers):
                x, wx = data[t]
                pk = per[t].copy()
                p1 = 1 - pk[0]
                pk[0] = 0
                pk /= pk.sum()
                obs = np.array([np.average((x >= a) & (x <= b), weights=wx) for a, b in BINS])
                exp = np.array([pk[(K >= a) & (K <= min(b, K[-1]))].sum() for a, b in BINS])
                n_eff = wx.sum() ** 2 / (wx ** 2).sum()
                x2 = n_eff * ((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
                dof = max(len(BINS) - 1 - npt, 1)
                extra = ""
                if kind == "TM":
                    extra = f"  {np.exp(best.x[1 + 2 * i]):6.2f}/{np.exp(best.x[2 + 2 * i]):7.1f}"
                print(f"    {TIER[t]:>5s} {len(x):4d} {n_eff:6.0f}  " +
                      "  ".join(f"{o:5.2f}/{e:5.2f}" for o, e in zip(obs, exp)) +
                      f"  {chi2d.sf(x2, dof):6.2f}  {p1:8.3f}/{th[t, col]:8.3f}" + extra)


if __name__ == "__main__" and "tests24" in __import__("sys").argv:
    poisson_single_channel_test()
    negbin_fit_test()


def poisson_single_channel_test_v2():
    """Test 2, corrected: the frequency answer is not a count (firms with exactly 1 phishing attack
    answer 'once' only ~37% of the time). Poisson prediction pushed through the calibrated
    count -> answer mapping F (calibrate_freq, from firms with exact phishing counts):
    P(freq | K>=1) = sum_K Pois(K; lambda) F[bucket(K), freq] / P(K>=1), lambda = -ln(1 - p_tier).
    Compare full answer distribution of single-channel firms. Caveat: F is calibrated on phishing."""
    from scipy.stats import chi2 as chi2d
    X, w, size = load()
    d = f.load_firms()
    F, tab, _ = j.calibrate_freq(d)
    rng = np.random.default_rng(1)
    freq = d["freq"].values
    fl = {k: d[k].values == 1 for k in ("fP", "fI", "fR", "fS")}
    only = {"phishing-only (ref)": (fl["fP"] & ~fl["fI"] & ~fl["fR"] & ~fl["fS"], 0),
            "impersonation-only": (fl["fI"] & ~fl["fP"] & ~fl["fR"] & ~fl["fS"], 1),
            "other-serious-only": (fl["fS"] & ~fl["fP"] & ~fl["fI"] & ~fl["fR"], None)}
    Kc = np.arange(1, j.KMAX + 1)
    print("\n" + "=" * 100)
    print("TEST 2 (corrected): single-channel firms' frequency answers vs Poisson through the count->answer map")
    print("  answers: once | <monthly | monthly | weekly | daily | several/day   (obs / Poisson-predicted)")
    print("=" * 100)
    for slab, sizes in (("Micro", [1]), ("Rest", [2, 3, 4])):
        m = np.isin(size, sizes)
        ww = w[m] / w[m].mean()
        tier, cert, pi, th = _tiers(X[m], ww, rng)
        for lab, (mask, col) in only.items():
            mm = mask[m] & ~np.isnan(freq[m])
            for t in range(3):
                sel = mm & (tier == t)
                if sel.sum() < 8:
                    continue
                p = th[t, col] if col is not None else 1 - np.prod(1 - th[t, 3:])
                lam = -np.log(1 - p)
                pk = poisson.pmf(Kc, lam)
                pred = (pk[:, None] * F[j.BUCKET_OF_K[Kc]]).sum(0) / pk.sum()
                fq, wx = freq[m][sel].astype(int), ww[sel]
                obs = np.array([np.average(fq == k, weights=wx) for k in range(1, 7)])
                n_eff = wx.sum() ** 2 / (wx ** 2).sum()
                x2 = n_eff * ((obs - pred) ** 2 / np.maximum(pred, 1e-9)).sum()
                print(f"  {lab:>20s} {slab:>5s} {TIER[t]:>4s} n={sel.sum():3d} lam={lam:.3f}  " +
                      " ".join(f"{o:.2f}/{q:.2f}" for o, q in zip(obs, pred)) + f"   p={chi2d.sf(x2, 5):.3f}")


def negbin_fit_test_v2():
    """Test 4, corrected: fits constrained to the flags. Per tier t, P(N>=1 | t) is fixed at the flag
    hit probability p_t (count-agnostic, Step 3). Phishing N = T + M, T ~ Poisson(lT_t), M ~ NegBin
    (mean mM_t, shape r shared across tiers); mM_t is solved from the constraint
    exp(-lT_t) * (r/(r+mM_t))^r = 1 - p_t. Fitted to reported counts (zero-truncated), weighted;
    goodness of fit on coarse bins (Kish n_eff). Ransomware: NegBin alone, same constraint.
    Caveats: count reporting is higher for costly firms; counts heap on round numbers."""
    from scipy.stats import chi2 as chi2d
    from scipy.optimize import brentq
    X, w, size = load()
    d = f.load_firms()
    rng = np.random.default_rng(2)
    K = np.arange(0, 3001)
    BINS = [(1, 1), (2, 3), (4, 10), (11, 50), (51, 10**9)]

    def solve_m(p, lT, r):
        target = (1 - p) / np.exp(-lT)
        if target >= 1:
            return None
        g = lambda lm: r * np.log(r / (r + np.exp(lm))) - np.log(target)
        if g(-20) * g(20) > 0:
            return None                      # this shape cannot reach the tier's hit probability
        return brentq(g, -20, 20)

    for chan, cnt_all, col, kind in (("PHISHING (targeted Poisson + mass NegBin)",
                                      np.where(d["fP"].values == 1, d["N"].values, np.nan), 0, "TM"),
                                     ("RANSOMWARE (NegBin)", d["NR"].values, 2, "NB")):
        print("\n" + "=" * 100)
        print(f"TEST 4 (corrected): {chan}; P(N>=1) per tier fixed at flag hit prob; shape r shared")
        print("=" * 100)
        for slab, sizes in (("Micro", [1]), ("Rest", [2, 3, 4])):
            m = np.isin(size, sizes)
            ww = w[m] / w[m].mean()
            tier, cert, pi, th = _tiers(X[m], ww, rng)
            cnt = cnt_all[m]
            tiers = [t for t in range(3) if ((tier == t) & ~np.isnan(cnt)).sum() >= 3 and th[t, col] < 0.999]
            data = {t: (np.minimum(cnt[(tier == t) & ~np.isnan(cnt)], K[-1]).astype(int),
                        ww[(tier == t) & ~np.isnan(cnt)]) for t in tiers}

            def pmfs(v):
                r = np.exp(v[0])
                out = {}
                for i, t in enumerate(tiers):
                    p = th[t, col]
                    lT = 0.0
                    if kind == "TM":
                        lT = -np.log(1 - p) * j.sig(v[1 + i])      # targeted takes a share of P(hit)
                    lm = solve_m(p, lT, r)
                    if lm is None:
                        return None
                    pm = nbinom.pmf(K, r, r / (r + np.exp(lm)))
                    out[t] = (np.convolve(poisson.pmf(K, lT), pm)[:len(K)] if kind == "TM" else pm,
                              lT, np.exp(lm))
                return r, out

            def nll(v):
                res = pmfs(v)
                if res is None:
                    return 1e12
                _, out = res
                tot = 0.0
                for t in tiers:
                    x, wx = data[t]
                    pk = out[t][0]
                    tot -= (wx * (np.log(np.maximum(pk[x], 1e-300)) - np.log(max(1 - pk[0], 1e-300)))).sum()
                return tot

            best = None
            for r0 in (0.02, 0.1, 0.5, 2.0):
                for s0 in ((-1.0, 1.0) if kind == "TM" else (0.0,)):
                    v0 = [np.log(r0)] + ([s0] * len(tiers) if kind == "TM" else [])
                    res = minimize(nll, v0, method="Nelder-Mead",
                                   options=dict(maxiter=20000, maxfev=20000, xatol=1e-5, fatol=1e-6))
                    best = res if best is None or res.fun < best.fun else best
            r, out = pmfs(best.x)
            print(f"  {slab}: shared r = {r:.3f}")
            for t in tiers:
                x, wx = data[t]
                pk = out[t][0].copy()
                pk[0] = 0
                pk /= pk.sum()
                obs = np.array([np.average((x >= a) & (x <= b), weights=wx) for a, b in BINS])
                exp = np.array([pk[(K >= a) & (K <= min(b, K[-1]))].sum() for a, b in BINS])
                n_eff = wx.sum() ** 2 / (wx ** 2).sum()
                x2 = n_eff * ((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
                extra = f"  targeted/mass mean {out[t][1]:.2f}/{out[t][2]:.1f}" if kind == "TM" else f"  mean {out[t][2]:.2f}"
                print(f"    {TIER[t]:>4s} n={len(x):3d} p_hit={th[t, col]:.3f}  " +
                      " ".join(f"{o:.2f}/{e:.2f}" for o, e in zip(obs, exp)) +
                      f"   GOF p={chi2d.sf(x2, len(BINS) - 2):.2f}{extra}")


if __name__ == "__main__" and "tests24v2" in __import__("sys").argv:
    poisson_single_channel_test_v2()
    negbin_fit_test_v2()
