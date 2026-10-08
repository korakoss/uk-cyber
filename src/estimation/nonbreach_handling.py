"""Yearly cost of attacks that did NOT breach: a simple generative model (2026-10-08, user-agreed design).
Per not-breached attacked firm:
  - N attacks a year, from its frequency band (rough counts below; a second mapping as a check)
  - number of costly handlings K ~ Poisson(m), m = exp(c0 + cz*(size-1)) * N^beta
      beta = 1: every attack carries the same chance of costly handling; beta = 0: a fixed number whatever the volume
  - each costly handling costs a lognormal amount (median exp(mu + gz*(size-1)), log-sd sd)
  - observed: the worst handling's cost band ('no cost' if K = 0); not observed: the yearly sum
Worst-band probabilities: P(worst <= x) = exp(-m * P(cost > x)).
Fit by maximum likelihood on the 738 not-breached firms (survey weights; unweighted as a check).
Output: settings, the likelihood at fixed beta values, fit check by frequency band, and the implied yearly total
(m x mean cost) next to the mean worst, by frequency band and size.
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

UNW = 'unweighted' in sys.argv
df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
nb = df[df.breach == 0]
C = nb.damage_bands.values.astype(int) - 1
Z = nb.sizeb.values.astype(int) - 1
F = nb.freq.values.astype(int)
W = np.ones(len(nb)) if UNW else (nb.weight / nb.weight.mean()).values
FQ = {1: 'once', 2: '<monthly', 3: 'monthly', 4: 'weekly', 5: 'daily', 6: 'several/day'}
MAPS = {'main': {1: 1, 2: 5, 3: 12, 4: 52, 5: 250, 6: 1000},
        'check': {1: 1, 2: 3, 3: 10, 4: 30, 5: 150, 6: 500}}
LOGE = np.log(np.array([100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 500000, 1e6, 5e6]))
NAMES = ['c0', 'cz', 'beta', 'mu', 'lsd', 'gz']
MID = np.array([0, 50, 300, 750, 3000, 7500, 15000, 35000, 75000, 300000, 750000, 3e6, 7.5e6])


def band_probs(m, mu, sd):
    """Worst-band probabilities (13) for Poisson(m) handlings with lognormal(mu, sd) costs; m, mu arrays (n,)."""
    surv = 1 - norm.cdf((LOGE[None, :] - mu[:, None]) / sd)          # P(cost > edge), n x 11
    cdf = np.exp(-m[:, None] * surv)                                  # P(worst <= edge)
    cdf = np.concatenate([np.exp(-m)[:, None], cdf, np.ones((len(m), 1))], axis=1)   # P(none), edges, 1
    return np.concatenate([np.exp(-m)[:, None], np.diff(cdf, axis=1)], axis=1)


def model(th, N):
    m = np.exp(th[0] + th[1] * Z) * N ** th[2]
    mu = th[3] + th[5] * Z
    return m, mu, np.exp(th[4])


def nll(th, N, beta=None):
    if beta is not None:
        th = np.r_[th[:2], beta, th[2:]]
    m, mu, sd = model(th, N)
    p = band_probs(m, mu, sd)[np.arange(len(C)), C]
    return -(W * np.log(np.clip(p, 1e-300, None))).sum()


def fit(N, beta=None):
    best = None
    for b0 in [0.0, 0.5]:
        for c00 in [-1.0, -3.0]:
            x0 = np.array([c00, 0.2, b0, np.log(150), 0.3, 0.3])
            if beta is not None:
                x0 = np.delete(x0, 2)
            r = minimize(nll, x0, args=(N, beta), method='Nelder-Mead',
                         options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-8})
            if best is None or r.fun < best.fun:
                best = r
    th = best.x if beta is None else np.r_[best.x[:2], beta, best.x[2:]]
    return th, best.fun


print(f'{"UNWEIGHTED" if UNW else "survey weights"}; not-breached firms n = {len(nb)}')
for mlab, mp in MAPS.items():
    N = np.array([mp[f] for f in F], dtype=float)
    th, f0 = fit(N)
    m, mu, sd = model(th, N)
    print(f'\n##### attacks per year by band ({mlab}): ' + ', '.join(f'{FQ[k]} {v}' for k, v in mp.items()))
    print('settings: ' + ', '.join(f'{k} {v:.3f}' for k, v in zip(NAMES, th)))
    print(f'   costly handlings per year, Micro attacked once: {np.exp(th[0]):.2f}; x{np.exp(th[1]):.2f} per size step; '
          f'grows as N^{th[2]:.2f}')
    print(f'   cost per costly handling (Micro): median £{np.exp(th[3]):,.0f}, log-sd {sd:.2f}, '
          f'mean £{np.exp(th[3] + sd ** 2 / 2):,.0f}; x{np.exp(th[5]):.2f} per size step')
    print('   log-likelihood with beta held at: ' + '   '.join(
        f'{b}: {f0 - fit(N, b)[1]:6.2f}' for b in [0.0, 0.1, 0.25, 0.5, 1.0]))

    P = band_probs(m, mu, sd)
    total = m * np.exp(mu + sd ** 2 / 2)
    # expected worst (exact, by simulation with the same model; not band midpoints, which overstate wide bands)
    rng = np.random.default_rng(23)
    S = 4000
    worst = np.zeros(len(m))
    for i in range(len(m)):
        k = rng.poisson(m[i], S)
        x = np.exp(mu[i] + sd * rng.standard_normal((S, max(k.max(), 1))))
        x[np.arange(x.shape[1])[None, :] >= k[:, None]] = 0.0
        worst[i] = x.max(axis=1).mean()
    print('\n   by frequency band: observed vs fitted share with no cost; fitted mean worst vs mean yearly total')
    for f, lab in FQ.items():
        k = F == f
        ww = W[k] / W[k].sum()
        print(f'   {lab:12s} n={k.sum():3d}  no cost obs {(ww * (C[k] == 0)).sum():.2f} fit {(ww * P[k, 0]).sum():.2f}   '
              f'£1k+ obs {(ww * (C[k] >= 4)).sum():.3f} fit {(ww * P[k, 4:].sum(axis=1)).sum():.3f}   '
              f'handlings/yr {(ww * m[k]).sum():5.2f}   mean worst £{(ww * worst[k]).sum():6,.0f}   '
              f'mean total £{(ww * total[k]).sum():7,.0f}')
    print('   by size:')
    for z, lab in enumerate(['Micro', 'Small', 'Medium', 'Large']):
        k = Z == z
        ww = W[k] / W[k].sum()
        print(f'   {lab:12s} n={k.sum():3d}  no cost obs {(ww * (C[k] == 0)).sum():.2f} fit {(ww * P[k, 0]).sum():.2f}   '
              f'mean worst £{(ww * worst[k]).sum():6,.0f}   mean total £{(ww * total[k]).sum():7,.0f}')
    ww = W / W.sum()
    print(f'   all: mean worst £{(ww * worst).sum():,.0f}, mean yearly total £{(ww * total).sum():,.0f} '
          f'(observed mean worst, band midpoints, £{(ww * MID[C]).sum():,.0f})')
