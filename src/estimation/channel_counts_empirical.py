"""Empirical per-type attack counts for the non-phishing channels.

Per-type count questions in the survey: ranssoft (ransomware), doscount (DoS), tkvrcount (account takeover),
hackcount (hacking / unauthorised access). Exact answer where >= 0, else the '_bands' answer (1=None,
2=1, 3=2-3, 4=4-5, 5=6-10, 6=11-20, 7=21-50, 8=51-100, 9=100+). Tabulated among firms flagged with the
matching type(s): coverage, exact value:#firms, band:#firms for the rest. Impersonation has no count
question: global freq answer among impersonation-only firms shown instead (1 once .. 6 several/day).
Unweighted firm counts.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/channel_counts_empirical.py
"""

import numpy as np
import pandas as pd

import joint_five_channel as f
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load, SHORT

VARS = [("ranssoft", ["Ransm"]), ("doscount", ["DoS"]), ("tkvrcount", ["Takov"]),
        ("hackcount", ["BankH", "AcOut", "AcStf"])]
BAND = {1: "0", 2: "1", 3: "2-3", 4: "4-5", 5: "6-10", 6: "11-20", 7: "21-50", 8: "51-100", 9: "100+"}


def main():
    X, w, size = load()
    r = aligned_raw()
    num = lambda c: pd.to_numeric(r[c], errors="coerce")
    for var, types in VARS:
        hit = X[:, [SHORT.index(t) for t in types]].max(1) == 1
        ex = num(var).where(lambda s: (s >= 0) & (s < 997)).values
        bd = num(var + "_bands").where(lambda s: s.between(1, 9)).values
        print(f"\n{var}  (flagged {'/'.join(types)}: {hit.sum()} firms; exact {int((hit & ~np.isnan(ex)).sum())}, "
              f"band only {int((hit & np.isnan(ex) & ~np.isnan(bd)).sum())}, missing {int((hit & np.isnan(ex) & np.isnan(bd)).sum())})")
        v = ex[hit & ~np.isnan(ex)]
        print("  exact:  " + "  ".join(f"{int(x)}:{int((v == x).sum())}" for x in np.unique(v)))
        b = bd[hit & np.isnan(ex) & ~np.isnan(bd)]
        if len(b):
            print("  bands:  " + "  ".join(f"{BAND[int(x)]}:{int((b == x).sum())}" for x in np.unique(b)))
        nh = (~hit) & ~np.isnan(ex) & (ex > 0)
        print(f"  (firms NOT flagged but reporting count>0: {int(nh.sum())})")

    d = f.load_firms()
    fq = d["freq"].values
    io = (X[:, SHORT.index("Imper")] == 1) & (X.sum(1) == 1) & ~np.isnan(fq)
    print(f"\nImpersonation-only firms, global freq answer (n={io.sum()}):")
    lab = {1: "once", 2: "<monthly", 3: "monthly", 4: "weekly", 5: "daily", 6: "several/day"}
    for g, gm in (("Micro", size == 1), ("Small+", size >= 2)):
        m = io & gm
        print(f"  {g:7s} n={m.sum():3d}  " + "  ".join(f"{lab[k]}:{int((fq[m] == k).sum())}" for k in range(1, 7)))


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def ransomware_sum_vs_soft():
    """Is Cybercrime_ranssum the positive part of ranssoft? Firm-by-firm match among ransomware-flagged firms."""
    X, w, size = load()
    r = aligned_raw()
    num = lambda c: pd.to_numeric(r[c], errors="coerce")
    hit = X[:, SHORT.index("Ransm")] == 1
    soft = num("ranssoft").where(lambda s: (s >= 0) & (s < 997)).values
    rsum = num("Cybercrime_ranssum").where(lambda s: s >= 0).values
    print(f"flagged {hit.sum()}; ranssoft answered {int((hit & ~np.isnan(soft)).sum())}, >0 {int((hit & (soft > 0)).sum())}; "
          f"ranssum present {int((hit & ~np.isnan(rsum)).sum())}, >0 {int((hit & (rsum > 0)).sum())}")
    both = hit & ~np.isnan(soft) & ~np.isnan(rsum)
    print(f"both present {both.sum()}, equal {int((soft[both] == rsum[both]).sum())}")
    print(f"ranssoft>0 without ranssum {int((hit & (soft > 0) & np.isnan(rsum)).sum())}; "
          f"ranssum without ranssoft>0 {int((hit & ~np.isnan(rsum) & ~(soft > 0)).sum())}")
    print(f"ranssum present among NON-flagged firms: {int((~hit & ~np.isnan(rsum)).sum())}")


if __name__ == "__main__" and "ranssum" in __import__("sys").argv:
    ransomware_sum_vs_soft()


def ranssum_by_size():
    """Cybercrime_ranssum values by size band (1 Micro .. 4 Large), unweighted firm counts."""
    X, w, size = load()
    r = aligned_raw()
    rsum = pd.to_numeric(r["Cybercrime_ranssum"], errors="coerce").where(lambda s: s >= 0).values
    hit = X[:, SHORT.index("Ransm")] == 1
    for s, lab in zip(range(1, 5), ["Micro", "Small", "Medium", "Large"]):
        m = (size == s) & ~np.isnan(rsum)
        v = rsum[m]
        print(f"{lab:6s} flagged {int((hit & (size == s)).sum()):3d}  with ranssum {m.sum():3d}   "
              + "  ".join(f"{int(x)}:{int((v == x).sum())}" for x in np.unique(v)))


if __name__ == "__main__" and "ranssize" in __import__("sys").argv:
    ranssum_by_size()


def ranssum_poisson_check(drop=(100,)):
    """Zero-truncated Poisson for Cybercrime_ranssum (all firms have >= 1), sizes pooled, dropping given values.
    MLE lambda; expected vs observed per value bin; parametric-bootstrap p for (a) Pearson X2, (b) max count."""
    from scipy.optimize import brentq
    from scipy.stats import poisson
    r = aligned_raw()
    v = pd.to_numeric(r["Cybercrime_ranssum"], errors="coerce").where(lambda s: s >= 1).dropna().values
    v = v[~np.isin(v, drop)]
    n, mean = len(v), v.mean()
    lam = brentq(lambda l: l / (1 - np.exp(-l)) - mean, 1e-6, 100)
    bins = [(1, 1), (2, 2), (3, 4), (5, 10**6)]

    def pk(a, b):
        return (poisson.cdf(b, lam) - poisson.cdf(a - 1, lam)) / (1 - np.exp(-lam))

    exp = np.array([n * pk(a, b) for a, b in bins])
    obs = np.array([((v >= a) & (v <= b)).sum() for a, b in bins])
    x2 = ((obs - exp) ** 2 / exp).sum()
    rng = np.random.default_rng(0)
    sims_x2, sims_max = [], []
    for _ in range(20000):
        s = rng.poisson(lam, 5 * n)
        s = s[s > 0][:n]
        o = np.array([((s >= a) & (s <= b)).sum() for a, b in bins])
        sims_x2.append(((o - exp) ** 2 / exp).sum())
        sims_max.append(s.max())
    print(f"dropped {drop}: n={n}, mean {mean:.2f}, var {v.var(ddof=1):.2f}, ZT-Poisson lambda {lam:.2f}")
    for (a, b), o, e in zip(bins, obs, exp):
        print(f"  {str(a) if a == b else (f'{a}-{b}' if b < 10**6 else f'{a}+'):>5s}  obs {o:3d}  exp {e:6.2f}")
    print(f"  X2 {x2:.1f}, bootstrap p {np.mean(np.array(sims_x2) >= x2):.4f};  "
          f"max {v.max():g}, P(max >= obs) {np.mean(np.array(sims_max) >= v.max()):.5f}")


if __name__ == "__main__" and "ranspois" in __import__("sys").argv:
    ranssum_poisson_check((100,))
    ranssum_poisson_check((100, 24))


def ranssum_logseries_check(drop=()):
    """Log-series law for Cybercrime_ranssum (>= 1), sizes pooled. MLE theta (mean equation); obs vs expected per
    bin; parametric-bootstrap p for Pearson X2 and for the max count."""
    from scipy.optimize import brentq
    from scipy.stats import logser
    r = aligned_raw()
    v = pd.to_numeric(r["Cybercrime_ranssum"], errors="coerce").where(lambda s: s >= 1).dropna().values
    v = v[~np.isin(v, drop)]
    n, mean = len(v), v.mean()
    th = brentq(lambda t: -t / ((1 - t) * np.log(1 - t)) - mean, 1e-9, 1 - 1e-12)
    bins = [(1, 1), (2, 2), (3, 4), (5, 10), (11, 10**6)]
    pk = lambda a, b: logser.cdf(b, th) - logser.cdf(a - 1, th)
    exp = np.array([n * pk(a, b) for a, b in bins])
    obs = np.array([((v >= a) & (v <= b)).sum() for a, b in bins])
    x2 = ((obs - exp) ** 2 / exp).sum()
    rng = np.random.default_rng(0)
    sx, sm = [], []
    for _ in range(20000):
        s = logser.rvs(th, size=n, random_state=rng)
        o = np.array([((s >= a) & (s <= b)).sum() for a, b in bins])
        sx.append(((o - exp) ** 2 / exp).sum())
        sm.append(s.max())
    print(f"dropped {drop}: n={n}, mean {mean:.2f}, log-series theta {th:.3f}")
    for (a, b), o, e in zip(bins, obs, exp):
        lab = str(a) if a == b else (f"{a}-{b}" if b < 10**6 else f"{a}+")
        print(f"  {lab:>5s}  obs {o:3d}  exp {e:6.2f}")
    print(f"  X2 {x2:.1f}, bootstrap p {np.mean(np.array(sx) >= x2):.3f};  max {v.max():g}, "
          f"P(max >= obs) {np.mean(np.array(sm) >= v.max()):.3f}")


if __name__ == "__main__" and "ranslogser" in __import__("sys").argv:
    ranssum_logseries_check(())
    ranssum_logseries_check((100,))


def ranssum_zipf_oils(drop=()):
    """Zipf P(k) = k^-a / zeta(a) and one-inflated log-series P(1) = pi + (1-pi) LS(1), P(k>1) = (1-pi) LS(k),
    for Cybercrime_ranssum (>= 1), sizes pooled. MLE; obs vs expected per bin; log-lik and AIC (log-series
    included for reference); parametric-bootstrap p for Pearson X2 (parameters refitted per replicate) and
    for the max count."""
    from scipy.optimize import minimize, minimize_scalar
    from scipy.stats import logser, zipf
    r = aligned_raw()
    v = pd.to_numeric(r["Cybercrime_ranssum"], errors="coerce").where(lambda s: s >= 1).dropna().values.astype(int)
    v = v[~np.isin(v, drop)]
    n = len(v)
    bins = [(1, 1), (2, 2), (3, 4), (5, 10), (11, 10**6)]
    tf = lambda t: 1 / (1 + np.exp(-t))

    def fit_zipf(x):
        a = minimize_scalar(lambda a: -zipf.logpmf(x, a).sum(), bounds=(1.01, 10), method="bounded").x
        return (a,), zipf.logpmf(x, a).sum()

    def fit_ls(x):
        t = minimize_scalar(lambda u: -logser.logpmf(x, tf(u)).sum(), bounds=(-10, 15), method="bounded").x
        return (tf(t),), logser.logpmf(x, tf(t)).sum()

    def oils_logpmf(x, pi, th):
        ls = logser.pmf(x, th)
        return np.log(np.where(x == 1, pi + (1 - pi) * ls, (1 - pi) * ls))

    def fit_oils(x):
        o = minimize(lambda z: -oils_logpmf(x, tf(z[0]), tf(z[1])).sum(), [0.0, 2.0], method="Nelder-Mead")
        return (tf(o.x[0]), tf(o.x[1])), -o.fun

    def cdf(model, k, p):
        if model == "zipf":
            return zipf.cdf(k, *p)
        if model == "logser":
            return logser.cdf(k, *p)
        pi, th = p
        return np.where(k >= 1, pi + (1 - pi) * logser.cdf(k, th), 0.0)

    def rvs(model, p, size, rng):
        if model == "zipf":
            return zipf.rvs(*p, size=size, random_state=rng)
        if model == "logser":
            return logser.rvs(*p, size=size, random_state=rng)
        pi, th = p
        s = logser.rvs(th, size=size, random_state=rng)
        return np.where(rng.random(size) < pi, 1, s)

    def expected(model, p, m):
        return np.array([m * (cdf(model, b, p) - cdf(model, a - 1, p)) for a, b in bins])

    obs = np.array([((v >= a) & (v <= b)).sum() for a, b in bins])
    rng = np.random.default_rng(0)
    print(f"\ndropped {drop}: n={n}")
    for name, fit, k in (("logser", fit_ls, 1), ("zipf", fit_zipf, 1), ("oils", fit_oils, 2)):
        p, ll = fit(v)
        e = expected(name, p, n)
        x2 = ((obs - e) ** 2 / e).sum()
        sx, sm = [], []
        for _ in range(2000):
            s = rvs(name, p, n, rng)
            ps, _ = fit(s)
            es = expected(name, ps, n)
            os_ = np.array([((s >= a) & (s <= b)).sum() for a, b in bins])
            sx.append(((os_ - es) ** 2 / es).sum())
            sm.append(s.max())
        par = ", ".join(f"{x:.3f}" for x in p)
        print(f"  {name:6s} params ({par})  loglik {ll:7.2f}  AIC {2 * k - 2 * ll:6.2f}  "
              f"X2 {x2:5.1f} p {np.mean(np.array(sx) >= x2):.3f}  P(max>={v.max()}) {np.mean(np.array(sm) >= v.max()):.3f}")
        print("         obs " + " ".join(f"{o:5d}" for o in obs) + "   (1 / 2 / 3-4 / 5-10 / 11+)")
        print("         exp " + " ".join(f"{x:5.1f}" for x in e))


if __name__ == "__main__" and "ranszipf" in __import__("sys").argv:
    ranssum_zipf_oils(())
    ranssum_zipf_oils((100,))
