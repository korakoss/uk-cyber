"""Spread breach cost: fit a few candidate shapes and compare fit and mean (2026-10-07).
Same set-up as spread_cost_fit.py: the 53 breached firms with 4+ types and an inside type are a mix of ordinary
breaches (cost bands as in a reference group of narrow breached firms) and spread breaches (share e, cost from the
candidate shape). All bands enter the fit, including the empty bands above £500k (no firm chose them).
Candidate shapes for the spread cost (each is a smooth curve; they differ in how heavy the top end is):
  A. lognormal (median m, log-sd s)                                       -- the thinnest top end
  B. log-t: like A but with a Student-t in log scale (extra 'df'; small df = heavier top end)
  C. lognormal up to £20k, power law above it: share q above £20k, P(X > x | X > 20k) = (x / 20k)^-alpha
     (small alpha = heavy top end; alpha <= 1 means the uncapped mean is infinite)
Means are reported capped at a largest single loss of £500k / £1m / £5m (the uncapped mean is infinite or
dominated by the cap for B and for C with small alpha).
Also: for C, how the fit changes as alpha is held at fixed values (what top-end heaviness the data allows).
90% intervals by bootstrap (resampling firms) for e and the £1m-capped mean.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as tdist

rng = np.random.default_rng(17)
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

EDGES = np.array([100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 500000, 1e6, 5e6])  # upper edges, bands 2..12
CATS = [[1], [2], [3, 4], [5], [6], [7], [8], [9], [10], [11, 12, 13]]
CLAB = ['none', '<100', '100-1k', '1k-5k', '5k-10k', '10k-20k', '20k-50k', '50k-100k', '100k-500k', '500k+']
CATM = np.array([[1.0 if bd in cs else 0.0 for bd in range(1, 14)] for cs in CATS])
cat_of = {bd: i for i, cs in enumerate(CATS) for bd in cs}
b['cat'] = b.c.map(cat_of)
U = 20000.0
CAPS = [5e5, 1e6, 5e6]
Z = rng.standard_normal(400000)
CH = rng.chisquare(1, 400000)  # reused for log-t draws (scaled per df below)


def cats_from_cdf(cdf_edges):
    cdf = np.concatenate([[0.0], cdf_edges, [1.0]])
    band = np.concatenate([[0.0], np.diff(cdf)])  # band 1 ('none') gets no mass
    return CATM @ band


# --- shapes: parameter vector -> (category probs, capped-mean function)
def ln_cdf(x, m, s):
    return norm.cdf((np.log(x) - np.log(m)) / s)


def shape_A(th):
    m, s = np.exp(th[0]), np.exp(th[1])
    return cats_from_cdf(ln_cdf(EDGES, m, s))


def shape_B(th):
    m, s, nu = np.exp(th[0]), np.exp(th[1]), 0.5 + np.exp(th[2])
    return cats_from_cdf(tdist.cdf((np.log(EDGES) - np.log(m)) / s, nu))


def spliced_cdf(x, m, s, q, a):
    x = np.asarray(x, float)
    below = (1 - q) * ln_cdf(np.minimum(x, U), m, s) / ln_cdf(U, m, s)
    above = 1 - q * (np.maximum(x, U) / U) ** (-a)
    return np.where(x <= U, below, above)


def body_params(th):
    """Body median and log-sd, clipped (median <= £1m, log-sd <= 5) to keep the numbers finite. The fit tends to
    push the body towards 'rising up to £20k', which these limits still allow."""
    return np.exp(min(th[0], np.log(1e6))), np.exp(min(th[1], np.log(5.0)))


def shape_C(th, a_fixed=None):
    m, s = body_params(th)
    q = 1 / (1 + np.exp(-th[2]))
    a = a_fixed if a_fixed is not None else np.exp(th[3])
    return cats_from_cdf(spliced_cdf(EDGES, m, s, q, a))


def capped_mean(name, th, cap):
    if name == 'A':
        m, s = np.exp(th[0]), np.exp(th[1])
        return np.minimum(np.exp(np.log(m) + s * Z), cap).mean()
    if name == 'B':
        m, s, nu = np.exp(th[0]), np.exp(th[1]), 0.5 + np.exp(th[2])
        tz = tdist.ppf(norm.cdf(Z), nu)
        return np.minimum(np.exp(np.log(m) + s * tz), cap).mean()
    m, s = body_params(th)
    q = 1 / (1 + np.exp(-th[2]))
    a = np.exp(th[3])
    lx = np.linspace(0, np.log(U), 4000)              # E[X | X < U] by numerical integration in log scale
    wgt = norm.pdf((lx - np.log(m)) / s)
    body = (np.exp(lx) * wgt).sum() / wgt.sum()
    tail = U + (U * np.log(cap / U) if abs(a - 1) < 1e-9 else U / (1 - a) * ((cap / U) ** (1 - a) - 1))
    return (1 - q) * body + q * tail


SHAPES = {'A': (shape_A, 2), 'B': (shape_B, 3), 'C': (shape_C, 4)}


def starts_for(name):
    base = [[np.log(m0), np.log(s0)] for m0 in [8000, 14000, 25000] for s0 in [0.4, 1.0]]
    if name == 'A':
        return [x for x in base]
    if name == 'B':
        return [x + [np.log(nu)] for x in base for nu in [1.0, 5.0]]
    return [x + [q, np.log(a)] for x in base for q in [-2.0, -0.5] for a in [0.7, 1.5]]


def fit(s, O, name, starts=None, a_fixed=None):
    w = (s.weight / s.weight.mean()).values
    cats = s.cat.values.astype(int)
    f = SHAPES[name][0]

    def nll(th):
        e = 1 / (1 + np.exp(-th[0]))
        sp = f(th[1:]) if a_fixed is None else f(th[1:], a_fixed)
        p = (1 - e) * O + e * sp
        return -(w * np.log(np.clip(p[cats], 1e-12, 1))).sum()

    if starts is None:
        st = starts_for(name)
        if a_fixed is not None:
            st = [x[:3] for x in st]
        starts = [np.array([e0] + x) for e0 in [0.0, 1.5] for x in st]
    best = None
    for x0 in starts:
        r = minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 6000, 'xatol': 1e-5, 'fatol': 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    return best


def describe(name, x):
    th = x[1:]
    if name == 'A':
        return f'median £{np.exp(th[0]):,.0f}, log-sd {np.exp(th[1]):.2f}'
    if name == 'B':
        return f'median £{np.exp(th[0]):,.0f}, log-scale {np.exp(th[1]):.2f}, df {0.5 + np.exp(th[2]):.2f}'
    m, sd = body_params(th)
    return (f'body median £{m:,.0f}, log-sd {sd:.2f} (clipped), share above £20k '
            f'{1 / (1 + np.exp(-th[2])):.2f}, alpha {np.exp(th[3]):.2f}')


def cat_dist(s):
    p = np.array([s.weight[s.cat == i].sum() for i in range(len(CATS))])
    return p / p.sum()


bi = b[(b.k >= 4) & (b.n_inside >= 1)]
REFS = [('narrow breached with an inside type', b[(b.k <= 3) & (b.n_inside >= 1)]),
        ('all narrow breached', b[b.k <= 3])]
BOOT = 100
import sys
SHAPES_TO_RUN = sys.argv[1:] or ['A', 'B', 'C']
for rlab, r in (REFS if any(n in SHAPES for n in SHAPES_TO_RUN) else []):
    O = cat_dist(r)
    print(f'\n##### Ordinary reference: {rlab} (n={len(r)})')
    print('        ' + ''.join(f'{c:>10s}' for c in CLAB))
    print('obs     ' + ''.join(f'{x:10.2f}' for x in cat_dist(bi)))
    for name in [n for n in SHAPES_TO_RUN if n in SHAPES]:
        res = fit(bi, O, name)
        x = res.x
        e = 1 / (1 + np.exp(-x[0]))
        k = SHAPES[name][1] + 1
        th = x[1:]
        means = [capped_mean(name, th, c) for c in CAPS]
        bs = []
        for _ in range(BOOT):
            rs = bi.sample(len(bi), replace=True, random_state=rng)
            rb = fit(rs, O, name, starts=[x, x + np.r_[1.0, np.zeros(len(x) - 1)]]).x
            bs.append((1 / (1 + np.exp(-rb[0])), capped_mean(name, rb[1:], 1e6)))
        bs = np.array(bs)
        f = SHAPES[name][0]
        fitted = (1 - e) * O + e * f(th)
        print(f'{name}: loglik {-res.fun:7.2f}  AIC {2 * k + 2 * res.fun:6.1f}   spread share e {e:.2f} '
              f'[{np.percentile(bs[:, 0], 5):.2f}-{np.percentile(bs[:, 0], 95):.2f}]')
        print(f'   {describe(name, x)}')
        print(f'   mean capped at £500k £{means[0]:,.0f}, £1m £{means[1]:,.0f} '
              f'[{np.percentile(bs[:, 1], 5):,.0f}-{np.percentile(bs[:, 1], 95):,.0f}], £5m £{means[2]:,.0f}')
        print('fit     ' + ''.join(f'{v:10.2f}' for v in fitted))

    if 'C' not in SHAPES_TO_RUN:
        continue
    print('\n  C with alpha held fixed (loglik relative to the best; a drop of about 2 or more counts against it):')
    best_ll = -fit(bi, O, 'C').fun
    for a in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        res = fit(bi, O, 'C', a_fixed=a)
        th = np.r_[res.x[1:4], np.log(a)]
        print(f'   alpha {a:3.1f}: loglik {-res.fun - best_ll:6.2f}   share above £20k {1 / (1 + np.exp(-res.x[3])):.2f}   '
              f'mean capped £1m £{capped_mean("C", th, 1e6):,.0f}, £5m £{capped_mean("C", th, 5e6):,.0f}')


# ---------------------------------------------------------------------------------------------------------------
# 2026-10-07: two checks on what drives the heavy top end (F28).
#  (1) let spread cost grow with firm size: cost = g^(size-1) x (curve), one extra setting g (size 1..4);
#  (2) drop the one Micro firm at £100k-500k that carries most of the weighted top end.
# Ordinary reference: all narrow breached firms, pooled over sizes (too few to split by size; caveat).
# No bootstrap here; compare fits by log-likelihood, and show the alpha profile for shape C in each setting.
def run_checks():
    O = cat_dist(b[b.k <= 3])
    top_micro = bi[(bi.sizeb == 1) & (bi.c >= 10)].index
    print(f'\n\n##### Checks: size scaling and dropping the top Micro firm (n dropped = {len(top_micro)})')

    def curve_cdf(name, th, x, a_fixed=None):
        if name == 'A':
            return ln_cdf(x, np.exp(th[0]), np.exp(th[1]))
        if name == 'B':
            return tdist.cdf((np.log(x) - th[0]) / np.exp(th[1]), 0.5 + np.exp(th[2]))
        m, s = body_params(th)
        q = 1 / (1 + np.exp(-th[2]))
        a = a_fixed if a_fixed is not None else np.exp(th[3])
        return spliced_cdf(x, m, s, q, a)

    npar = {'A': 2, 'B': 3, 'C': 4}

    def fit2(s, name, scale, a_fixed=None):
        w = (s.weight / s.weight.mean()).values
        cats = s.cat.values.astype(int)
        size = s.sizeb.values.astype(int)
        k = npar[name] - (1 if a_fixed is not None else 0)

        def nll(x):
            e = 1 / (1 + np.exp(-x[0]))
            th = x[1:1 + k]
            g = np.exp(x[-1]) if scale else 1.0
            ll = 0.0
            for z in range(1, 5):
                mz = size == z
                if not mz.any():
                    continue
                sp = cats_from_cdf(curve_cdf(name, th, EDGES / g ** (z - 1), a_fixed))
                p = (1 - e) * O + e * sp
                ll += (w[mz] * np.log(np.clip(p[cats[mz]], 1e-12, 1))).sum()
            return -ll

        best = None
        for x0 in starts_for(name):
            x0 = list(x0)[:k]
            for e0 in [0.0, 1.5]:
                for g0 in ([0.0, 0.7] if scale else [None]):
                    st = np.array([e0] + x0 + ([g0] if scale else []))
                    r = minimize(nll, st, method='Nelder-Mead',
                                 options={'maxiter': 8000, 'xatol': 1e-5, 'fatol': 1e-7})
                    if best is None or r.fun < best.fun:
                        best = r
        return best

    def mean_by_size(name, x, scale, cap=1e6, a_fixed=None):
        k = npar[name] - (1 if a_fixed is not None else 0)
        th = np.array(x[1:1 + k])
        if a_fixed is not None:
            th = np.r_[th, np.log(a_fixed)]
        g = np.exp(x[-1]) if scale else 1.0
        return [g ** (z - 1) * capped_mean(name, th, cap / g ** (z - 1)) for z in range(1, 5)]

    for lab, s in [('all 53 firms', bi), ('without the top Micro firm', bi.drop(top_micro))]:
        print(f'\n=== {lab} (n={len(s)}) ===')
        for scale in [False, True]:
            print(f'-- {"cost grows with size" if scale else "same cost curve for all sizes"}')
            for name in ['A', 'B', 'C']:
                r = fit2(s, name, scale)
                x = r.x
                k = npar[name] + 1 + (1 if scale else 0)
                ms = mean_by_size(name, x, scale)
                gtxt = f'  size step x{np.exp(x[-1]):.2f}' if scale else ''
                print(f'   {name}: loglik {-r.fun:7.2f}  AIC {2 * k + 2 * r.fun:6.1f}  e {1 / (1 + np.exp(-x[0])):.2f}{gtxt}'
                      f'   mean (cap £1m) Micro/Small/Medium/Large: ' + ' / '.join(f'£{v:,.0f}' for v in ms))
            print('   C with alpha fixed (loglik vs best C; mean cap £1m Micro / Large):')
            best_ll = -fit2(s, 'C', scale).fun
            for a in [0.7, 1.0, 1.5, 2.0, 3.0]:
                r = fit2(s, 'C', scale, a_fixed=a)
                ms = mean_by_size('C', r.x, scale, a_fixed=a)
                print(f'      alpha {a:3.1f}: {-r.fun - best_ll:6.2f}   £{ms[0]:,.0f} / £{ms[3]:,.0f}')


if 'checks' in SHAPES_TO_RUN:
    run_checks()
