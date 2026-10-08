"""How many breaches per firm-year? Bounding repeats from cost distributions (2026-10-06/07).

Idea: the worst incident is the largest of the year's breach costs. If breached firms in some group have more
breaches, their worst cost moves up (largest of N draws) and 'no cost' gets rarer.

Set-up (plain, few assumptions):
- Single-breach cost distribution G = worst-cost band distribution of breached firms attacked ONCE in the year
  (one attack -> one breach). Each breach elsewhere is assumed to be a draw from the same G.
- Breach counts: negative binomial with mean m_g and shape k (k = infinity is plain Poisson; small k = firms differ a
  lot in breach-proneness). For each group g, m_g is set so that P(N >= 1) equals the group's observed breached share.
- Predicted worst-cost distribution for breached firms in group g: sum_n P(N = n | N >= 1) * [G cdf]^n.
- Compare with the observed worst-cost bands of breached firms in groups that should have more breaches
  (attacked more often / more phishing). Scan k; report fit (log-likelihood), mean breaches per breached firm,
  and the extra cost from repeats (expected sum / expected worst, using band midpoints).
NOTE: this bounds repeats of breaches that cost like the single-breach distribution G. Extra breaches that are
much cheaper than G would barely move the worst and are not bounded here (they also add little cost).
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

rng = np.random.default_rng(5)
df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
df['c'] = df['damage_bands'].astype(int)
ps = num('Cybercrime_phishsum')
MID = np.array([0, 50, 300, 750, 3000, 7500, 15000, 35000, 75000, 300000, 750000, 3e6, 7.5e6])  # bands 1..13
NB = 13


def band_pmf(s):
    p = np.array([s.weight[s.c == b].sum() for b in range(1, NB + 1)])
    return p / p.sum()


base = df[(df.freq == 1) & (df.breach == 1)]
G = band_pmf(base)
Gcdf = np.cumsum(G)
print(f'single-breach cost distribution G from breached firms attacked once: n={len(base)}')
print('   G (none/<100/100-500/500-1k/1k-5k/5k+): ' +
      ' '.join(f'{x:.2f}' for x in list(G[:5]) + [G[5:].sum()]))

GROUPS = [('attacked <monthly-monthly', df.freq.isin([2, 3])),
          ('attacked weekly+', df.freq.isin([4, 5, 6])),
          ('phishing count 2-20', ps.between(2, 20)),
          ('phishing count >20', ps > 20)]


def n_dist(p, k, nmax=60):
    """P(N=n | N>=1), n=1..nmax, for NegBin with shape k (np.inf = Poisson) and P(N>=1)=p."""
    if np.isinf(k):
        m = -np.log(1 - p)
        from scipy.stats import poisson
        pm = poisson.pmf(np.arange(nmax + 1), m)
    else:
        m = k * ((1 - p) ** (-1 / k) - 1)
        from scipy.stats import nbinom
        pm = nbinom.pmf(np.arange(nmax + 1), k, k / (k + m))
    pn = pm[1:] / pm[1:].sum()
    return pn, m


def pred_pmf(pn):
    cdf = sum(pn[i] * Gcdf ** (i + 1) for i in range(len(pn)))
    return np.diff(np.concatenate([[0], cdf]))


def sum_over_max(pn, sims=20000):
    """Expected sum of N breach costs / expected worst, by simulation with band midpoints."""
    n = rng.choice(np.arange(1, len(pn) + 1), size=sims, p=pn)
    draws = [MID[rng.choice(NB, size=x, p=G)] for x in n]
    tot = np.array([d.sum() for d in draws])
    mx = np.array([d.max() for d in draws])
    return tot.mean() / mx.mean()


KS = [np.inf, 5, 1, 0.5, 0.2, 0.1, 0.05]
print('\nk = inf is plain Poisson; smaller k = more firm-to-firm variation in breach-proneness (more repeats)')
for lab, m in GROUPS:
    g_all = df[m]
    p = np.average(g_all.breach, weights=g_all.weight)
    gb = g_all[g_all.breach == 1]
    obs = band_pmf(gb)
    w = (gb.weight / gb.weight.mean()).values
    print(f'\n=== {lab}: breached share {p:.2f}, breached firms n={len(gb)} ===')
    print(f"   observed worst: none {obs[0]:.2f}  >=£500 {obs[3:].sum():.2f}  >=£5k {obs[5:].sum():.2f}")
    print(f"   {'k':>5s} {'mean N|breached':>16s} {'none':>6s} {'>=£500':>7s} {'>=£5k':>6s} {'loglik':>8s} {'sum/worst':>10s}")
    rows = []
    for k in KS:
        pn, mm = n_dist(p, k)
        pr = pred_pmf(pn)
        ll = float((w * np.log(np.clip(pr[gb.c.values - 1], 1e-12, 1))).sum())
        rows.append((k, (np.arange(1, len(pn) + 1) * pn).sum(), pr, ll, sum_over_max(pn)))
    best = max(r[3] for r in rows)
    for k, mn, pr, ll, som in rows:
        print(f"   {str(k):>5s} {mn:16.2f} {pr[0]:6.2f} {pr[3:].sum():7.2f} {pr[5:].sum():6.2f} "
              f"{ll - best:8.1f} {som:10.2f}")
print('\n(loglik is relative to the best k in that group; a drop of about 2 or more counts against that k)')
