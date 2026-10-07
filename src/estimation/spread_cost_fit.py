"""Fit the cost of spread breaches without the £5k cutoff of the marker (2026-10-07).
Firms: breached, 4+ types, at least one inside type (n about 53), cheap and costly alike.
Model: their worst-cost band is a mix:
  - with chance 1-e an ordinary breach, cost bands as in a reference group of ordinary breaches
    (main: narrow breached firms with an inside type; check: all narrow breached firms)
  - with chance e a spread breach, cost from a smooth lognormal (median m, spread s), cut into the survey bands.
Why the lognormal: with a free-form spread distribution, any e from 0 up to a maximum fits equally well, so some
shape has to be assumed. Lognormal is the plain 'smooth, skewed, positive' choice; it also gives spread breaches
no 'no cost' answers. Fit by maximum likelihood (survey weights scaled to sum to n). 90% intervals by bootstrap.
Then the same fit split by number of inside types (1 vs 2+), with the reference group held fixed.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

rng = np.random.default_rng(13)
df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
TYPES = ['type6', 'type5', 'type1', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
INSIDE = ['type1', 'type4', 'type8', 'type7']
df['k'] = (df[TYPES] == 1).sum(axis=1)
df['n_inside'] = (df[INSIDE] == 1).sum(axis=1)
b = df[df.breach == 1].copy()
b['c'] = b.damage_bands.astype(int)

# survey bands 1..13: band 1 = no cost; upper edges in £ of bands 2..12
EDGES = np.array([100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 500000, 1e6, 5e6])
LOGE = np.log(EDGES)
# collapse to categories for the fit (small n)
CATS = [[1], [2], [3, 4], [5], [6], [7], [8, 9], [10, 11, 12, 13]]
CLAB = ['none', '<100', '100-1k', '1k-5k', '5k-10k', '10k-20k', '20k-100k', '100k+']
CATM = np.array([[1.0 if bd in cs else 0.0 for bd in range(1, 14)] for cs in CATS])
cat_of = {bd: i for i, cs in enumerate(CATS) for bd in cs}
b['cat'] = b.c.map(cat_of)
Z = rng.standard_normal(200000)


def cat_dist(s):
    p = np.array([s.weight[s.cat == i].sum() for i in range(len(CATS))])
    return p / p.sum()


def lognorm_cats(m, sd):
    """Category probabilities of a lognormal with median m (£) and log-sd sd. No mass on 'none'."""
    cdf = np.concatenate([[0.0], norm.cdf((LOGE - np.log(m)) / sd), [1.0]])
    band = np.concatenate([[0.0], np.diff(cdf)])
    return CATM @ band


def fit(s, O, starts=None):
    w = (s.weight / s.weight.mean()).values
    cats = s.cat.values.astype(int)

    def nll(th):
        e = 1 / (1 + np.exp(-th[0]))
        p = (1 - e) * O + e * lognorm_cats(np.exp(th[1]), np.exp(th[2]))
        return -(w * np.log(np.clip(p[cats], 1e-12, 1))).sum()

    if starts is None:
        starts = [np.array([np.log(e0 / (1 - e0)), np.log(m0), np.log(1.2)]) for e0 in [0.3, 0.6, 0.85]
                  for m0 in [5000, 15000, 40000]]
    best = None
    for x0 in starts:
        r = minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    return 1 / (1 + np.exp(-best.x[0])), np.exp(best.x[1]), np.exp(best.x[2]), best.x


def mean_ln(m, sd, cap=None):
    if cap is None:
        return m * np.exp(sd ** 2 / 2)
    return np.minimum(np.exp(np.log(m) + sd * Z), cap).mean()


def report(lab, s, O, boot=200):
    e, m, sd, x0 = fit(s, O)
    bs = []
    for _ in range(boot):
        r = s.sample(len(s), replace=True, random_state=rng)
        bs.append(fit(r, O, starts=[x0, x0 + [1, 0, 0], x0 - [1, 0, 0]])[:3])
    bs = np.array(bs)
    q = lambda a: f'[{np.percentile(a, 5):.2f}-{np.percentile(a, 95):.2f}]'
    qi = lambda a: f'[{np.percentile(a, 5):,.0f}-{np.percentile(a, 95):,.0f}]'
    cap1 = np.array([mean_ln(x[1], x[2], 1e6) for x in bs])
    print(f'{lab}  (n={len(s)})')
    print(f'   spread share e {e:.2f} {q(bs[:, 0])}')
    print(f'   spread cost: median £{m:,.0f} {qi(bs[:, 1])}, log-sd {sd:.2f} {q(bs[:, 2])}')
    print(f'   mean: uncapped £{mean_ln(m, sd):,.0f}; capped at £1m £{mean_ln(m, sd, 1e6):,.0f} {qi(cap1)}; '
          f'capped at £500k £{mean_ln(m, sd, 5e5):,.0f}')
    fitted = (1 - e) * O + e * lognorm_cats(m, sd)
    print('   ' + ''.join(f'{c:>9s}' for c in CLAB))
    print('obs' + ''.join(f'{x:9.2f}' for x in cat_dist(s)))
    print('fit' + ''.join(f'{x:9.2f}' for x in fitted))


bi = b[(b.k >= 4) & (b.n_inside >= 1)]
REFS = [('narrow breached with an inside type', b[(b.k <= 3) & (b.n_inside >= 1)]),
        ('all narrow breached', b[b.k <= 3])]
for rlab, r in REFS:
    O = cat_dist(r)
    print(f'\n##### Ordinary reference: {rlab} (n={len(r)})')
    print('   ' + ''.join(f'{c:>9s}' for c in CLAB))
    print('   ' + ''.join(f'{x:9.2f}' for x in O))
    report('All broad breached firms with an inside type', bi, O)
    report('  1 inside type', bi[bi.n_inside == 1], O)
    report('  2+ inside types', bi[bi.n_inside >= 2], O)

# --- 2026-10-07: the lognormal fits above are pulled by a pile-up in the £10k-20k band. Plain look at the raw data:
# unweighted counts and weighted shares by band for marked spread firms (broad, inside type, >= £5k), the weight
# carried by the £10k-20k firms, and the weighted mean using band midpoints (with and without the top firm).
MID = {6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000, 11: 750000, 12: 3e6, 13: 7.5e6}
sp = bi[bi.c >= 6]
print('\n##### Raw: marked spread firms (n=%d)' % len(sp))
for bd in range(6, 14):
    s = sp[sp.c == bd]
    print(f'   band {bd:2d} (mid £{MID[bd]:>9,.0f}): n={len(s):2d}  weighted share {s.weight.sum() / sp.weight.sum():.2f}  '
          f'sizes {sorted(s.sizeb.astype(int).tolist())}')
cm = sp.c.map(MID)
print(f'   weighted mean (midpoints) £{np.average(cm, weights=sp.weight):,.0f}; unweighted £{cm.mean():,.0f}')
top = (cm * sp.weight).idxmax()
print(f'   top firm carries {(cm * sp.weight)[top] / (cm * sp.weight).sum():.2f} of the weighted mean; mean without it '
      f'£{np.average(cm.drop(top), weights=sp.weight.drop(top)):,.0f}')
print('   weighted mean by size: ' + ', '.join(
    f'{lab} £{np.average(cm[m], weights=sp.weight[m]):,.0f} (n={int(m.sum())})'
    for lab, m in [('Micro', sp.sizeb == 1), ('Small', sp.sizeb == 2), ('Medium+Large', sp.sizeb >= 3)]))
