"""The top end of the cost range, pooled over all costly firms; and the size effect with fairer weights (2026-10-07).

Part 1. One shared steepness for the top end.
  Assume that above £20k, worst-incident costs follow a power law: P(X > x | X > £20k) = (x / £20k)^-alpha
  (smaller alpha = heavier top end). Then, for firms at £20k+, the split across the bands
  £20k-50k / £50k-100k / £100k-500k / £500k+ depends on alpha only (not on firm size or the share of firms that
  get there). Fit alpha to all attacked firms at £20k+ (26 firms), whatever the cost part (spread, ordinary
  breach, non-breach).
  Shown with three weightings: survey weights; weights balanced within size band (each band's firms sum to its
  firm count); no weights. Plus: likelihood profile, bootstrap, dropping each firm in turn, spread vs other firms.
Part 2. Does spread cost grow with size? Same mixture as spread_cost_shapes.py (ordinary part from all narrow
  breached firms), spread cost = g^(size-1) x curve, but with weights balanced within each size band so that each
  band's firms set that band's level. Curves: lognormal, and lognormal up to £20k + power law with alpha fixed at
  the pooled values from part 1. Also the plain share at £20k+ among costly (>= £5k) spread-marked firms by size.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

rng = np.random.default_rng(19)
df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
TYPES = ['type6', 'type5', 'type1', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
INSIDE = ['type1', 'type4', 'type8', 'type7']
df['k'] = (df[TYPES] == 1).sum(axis=1)
df['n_inside'] = (df[INSIDE] == 1).sum(axis=1)
df['c'] = df.damage_bands.astype(int)
df['spread'] = ((df.breach == 1) & (df.k >= 4) & (df.n_inside >= 1) & (df.c >= 6)).astype(int)
# weights balanced within size band: each band's weights sum to its firm count
df['w_size'] = df.weight * df.groupby('sizeb').weight.transform(lambda w: len(w) / w.sum())

# ---------------- Part 1
U = 20000.0
TOP_EDGES = np.array([50000, 100000, 500000])          # upper edges of bands 8, 9, 10; band 11+ is 500k+


def top_probs(a):
    s = (TOP_EDGES / U) ** (-a)                         # survival at the edges
    return np.array([1 - s[0], s[0] - s[1], s[1] - s[2], s[2]])


def top_cat(c):
    return np.clip(c - 8, 0, 3)                         # band 8 -> 0, 9 -> 1, 10 -> 2, 11+ -> 3


def nll_alpha(a, cats, w):
    return -(w * np.log(np.clip(top_probs(a)[cats], 1e-12, 1))).sum()


def fit_alpha(s, wcol):
    cats = top_cat(s.c.values)
    w = np.ones(len(s)) if wcol is None else (s[wcol] / s[wcol].mean()).values
    r = minimize_scalar(lambda la: nll_alpha(np.exp(la), cats, w), bounds=(np.log(0.05), np.log(20)), method='bounded')
    return np.exp(r.x), r.fun, cats, w


top = df[df.c >= 8].copy()
print(f'=== Part 1. Firms at £20k+: n = {len(top)}  (band 8: {(top.c == 8).sum()}, 9: {(top.c == 9).sum()}, '
      f'10: {(top.c == 10).sum()}, 11+: {(top.c >= 11).sum()})')
print('   by size: ' + ', '.join(f'{z}: {(top.sizeb == z).sum()}' for z in [1, 2, 3, 4]) +
      f';  spread {top.spread.sum()}, other {len(top) - top.spread.sum()}')
WEIGHTINGS = [('survey weights', 'weight'), ('balanced within size', 'w_size'), ('no weights', None)]
for wlab, wcol in WEIGHTINGS:
    a, f, cats, w = fit_alpha(top, wcol)
    prof = []
    for at in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        prof.append(f'{at}: {-nll_alpha(at, cats, w) + f:5.2f}')
    bs = [fit_alpha(top.sample(len(top), replace=True, random_state=rng), wcol)[0] for _ in range(1000)]
    loo = [fit_alpha(top.drop(i), wcol)[0] for i in top.index]
    pr = top_probs(a)
    print(f'\n-- {wlab}: alpha {a:.2f}  bootstrap 90% [{np.percentile(bs, 5):.2f}-{np.percentile(bs, 95):.2f}]  '
          f'dropping each firm in turn: {min(loo):.2f}-{max(loo):.2f}')
    obs = np.array([w[cats == i].sum() for i in range(4)]) / w.sum()
    print('   share 20-50k / 50-100k / 100-500k / 500k+:  observed ' + ' '.join(f'{x:.2f}' for x in obs) +
          '   fitted ' + ' '.join(f'{x:.2f}' for x in pr))
    print('   loglik relative to best at alpha = ' + '   '.join(prof))
    for glab, m in [('spread firms', top.spread == 1), ('other firms', top.spread == 0),
                    ('Micro+Small', top.sizeb <= 2), ('Medium+Large', top.sizeb >= 3)]:
        s = top[m]
        print(f'   {glab:13s} n={len(s):2d}  alpha {fit_alpha(s, wcol)[0]:.2f}')
    top_micro = top[(top.sizeb == 1) & (top.c >= 10)].index
    print(f'   without the {len(top_micro)} Micro firms at £100k+: alpha {fit_alpha(top.drop(top_micro), wcol)[0]:.2f}')

print('\nmean above £20k for a power law capped at C:  E[min(X, C) | X > 20k]')
for a in [0.7, 1.0, 1.5, 2.0, 3.0]:
    vals = []
    for C in [5e5, 1e6, 5e6]:
        m = U + (U * np.log(C / U) if abs(a - 1) < 1e-9 else U / (1 - a) * ((C / U) ** (1 - a) - 1))
        vals.append(f'£{m:,.0f}')
    print(f'   alpha {a}: cap £500k {vals[0]}, £1m {vals[1]}, £5m {vals[2]}')

# ---------------- Part 2
print('\n=== Part 2. Does spread cost grow with size? ===')
sp = df[df.spread == 1]
print('Plain look: marked spread firms (all >= £5k), share at £20k+ and at £100k+, weights balanced within size')
for lab, m in [('Micro', sp.sizeb == 1), ('Small', sp.sizeb == 2), ('Medium', sp.sizeb == 3), ('Large', sp.sizeb == 4)]:
    s = sp[m]
    print(f'   {lab:7s} n={len(s):2d}  £20k+ {np.average(s.c >= 8, weights=s.weight):.2f}  '
          f'£100k+ {np.average(s.c >= 10, weights=s.weight):.2f}  (unweighted counts at £20k+: {int((s.c >= 8).sum())})')

EDGES = np.array([100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 500000, 1e6, 5e6])
CATS = [[1], [2], [3, 4], [5], [6], [7], [8], [9], [10], [11, 12, 13]]
CATM = np.array([[1.0 if bd in cs else 0.0 for bd in range(1, 14)] for cs in CATS])
cat_of = {bd: i for i, cs in enumerate(CATS) for bd in cs}
df['cat'] = df.c.map(cat_of)
b = df[df.breach == 1]
bi = b[(b.k >= 4) & (b.n_inside >= 1)]
O_all = b[b.k <= 3]
O = np.array([O_all.weight[O_all.cat == i].sum() for i in range(len(CATS))]) / O_all.weight.sum()


def cats_from_cdf(cdf_edges):
    cdf = np.concatenate([[0.0], cdf_edges, [1.0]])
    return CATM @ np.concatenate([[0.0], np.diff(cdf)])


def curve_cdf(name, th, x, a):
    m, s = np.exp(th[0]), np.exp(min(th[1], np.log(5.0)))
    if name == 'lognormal':
        return norm.cdf((np.log(x) - np.log(m)) / s)
    q = 1 / (1 + np.exp(-th[2]))
    m = min(m, 1e6)
    lo = norm.cdf((np.log(np.minimum(x, U)) - np.log(m)) / s) / norm.cdf((np.log(U) - np.log(m)) / s)
    return np.where(x <= U, (1 - q) * lo, 1 - q * (np.maximum(x, U) / U) ** (-a))


def fit_size(s, name, scale, wcol, a=None):
    w = (s[wcol] / s[wcol].mean()).values
    cats = s.cat.values.astype(int)
    size = s.sizeb.values.astype(int)
    k = 2 if name == 'lognormal' else 3

    def nll(x):
        e = 1 / (1 + np.exp(-x[0]))
        g = np.exp(x[-1]) if scale else 1.0
        ll = 0.0
        for z in range(1, 5):
            mz = size == z
            p = (1 - e) * O + e * cats_from_cdf(curve_cdf(name, x[1:1 + k], EDGES / g ** (z - 1), a))
            ll += (w[mz] * np.log(np.clip(p[cats[mz]], 1e-12, 1))).sum()
        return -ll

    best = None
    for m0 in [8000, 15000]:
        for e0 in [0.0, 1.0]:
            x0 = [e0, np.log(m0), np.log(1.0)] + ([-1.5] if k == 3 else []) + ([0.3] if scale else [])
            r = minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 6000, 'xatol': 1e-5, 'fatol': 1e-7})
            if best is None or r.fun < best.fun:
                best = r
    return best


a_pool = {wlab: fit_alpha(top, wcol)[0] for wlab, wcol in WEIGHTINGS}
print('\nMixture fit on the 53 broad breached firms with an inside type, weights balanced within size')
for name, a in [('lognormal', None), ('power-law top, alpha = pooled (balanced weights)', a_pool['balanced within size']),
                ('power-law top, alpha = pooled (survey weights)', a_pool['survey weights'])]:
    kind = 'lognormal' if a is None else 'spliced'
    r0 = fit_size(bi, kind, False, 'w_size', a)
    r1 = fit_size(bi, kind, True, 'w_size', a)
    print(f'   {name}: loglik same-for-all-sizes {-r0.fun:.2f}, with size step {-r1.fun:.2f} '
          f'(gain {r0.fun - r1.fun:.2f}); size step x{np.exp(r1.x[-1]):.2f} per band '
          f'(Large vs Micro x{np.exp(3 * r1.x[-1]):.1f}); e {1 / (1 + np.exp(-r1.x[0])):.2f}')
    g = np.exp(r1.x[-1])
    prof = []
    for gt in [1.0, 1.25, 1.5, 2.0, 3.0]:
        w = (bi.w_size / bi.w_size.mean()).values

        def nll_fixed(x, gt=gt):
            e = 1 / (1 + np.exp(-x[0]))
            k = 2 if kind == 'lognormal' else 3
            ll = 0.0
            for z in range(1, 5):
                mz = bi.sizeb.values == z
                p = (1 - e) * O + e * cats_from_cdf(curve_cdf(kind, x[1:1 + k], EDGES / gt ** (z - 1), a))
                ll += (w[mz] * np.log(np.clip(p[bi.cat.values[mz].astype(int)], 1e-12, 1))).sum()
            return -ll
        rf = minimize(nll_fixed, r1.x[:-1], method='Nelder-Mead', options={'maxiter': 6000})
        prof.append(f'x{gt}: {r1.fun - rf.fun:5.2f}')
    print('      loglik relative to best, size step held at ' + '   '.join(prof))
