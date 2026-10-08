"""Which attack types reach a firm: one hidden 'how targeted' score (2026-10-08, user-agreed design).
All surveyed businesses (questtype 1, n about 2,180), including those that ticked no type (not attacked).
Model: each firm has a hidden score E ~ Normal(mean, 1); the chance of ticking type t is logistic(a_t + b_t * E).
Size enters in one of two ways:
  - 'shared': size shifts the score's mean (bigger firms are more targeted overall), mean = g * (size - 1)
  - 'per type': each type has its own size effect, logistic(a_t + b_t * E + c_t * (size - 1))
2026-10-08 addition: '+spread' versions also let the score's spread grow with size: sd = exp(h * (size - 1)).
2026-10-08 addition: '+link' versions let one incident tick two related boxes: an impersonation incident also ticks
phishing with chance q1 (spoofed email), a ransomware incident also ticks malware with chance q2.
Fitted by maximum likelihood (the hidden score is integrated out over a grid), survey weights.
Checks (model simulated vs observed, weighted): share attacked by size, number of types ticked by size,
how often pairs of types occur together (ratio to what independent types would give).
Note: ticks here include the footprint of spreading breaches (inside types); this part treats all ticks alike.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

df = pd.read_csv('data/proc/data.csv', low_memory=False)
TYPES = {'type6': 'phishing', 'type5': 'impersonation', 'type2': 'malware', 'type16': 'takeover', 'type3': 'DoS',
         'type1': 'ransomware', 'type4': 'bank hacking', 'type8': 'outsider access', 'type7': 'staff access',
         'type15': 'eavesdropping', 'type9': 'other'}
df = df[(df.questtype == 1) & df[list(TYPES)].notna().all(axis=1)].copy()
X = (df[list(TYPES)] == 1).values.astype(float)
Z = df.sizeb.values.astype(int) - 1
W = (df.weight / df.weight.mean()).values
n, T = X.shape
GH_x, GH_w = np.polynomial.hermite_e.hermegauss(41)        # grid for a standard normal score
GH_w = GH_w / GH_w.sum()
print(f'businesses: {n}; ticked any type: {int((X.sum(axis=1) > 0).sum())}')


IPH, IIM, IMW, IRW = 0, 1, 2, 5                              # column positions: phishing, impersonation, malware, ransomware


def links(th, mode):
    return (expit(th[-2]), expit(th[-1])) if mode.endswith('+link') else (0.0, 0.0)


def unpack(th, mode):
    """returns a, b, score mean shift g, per-type size effects c, score spread growth h"""
    if mode.endswith('+link'):
        th = th[:-2]
    a, b = th[:T], th[T:2 * T]
    h = th[-1] if mode.endswith('+spread') else 0.0
    if mode.startswith('shared'):
        return a, b, th[2 * T], np.zeros(T), h
    return a, b, 0.0, th[2 * T:3 * T], h


def loglik_each(th, mode):
    a, b, g, c, h = unpack(th, mode)
    E = GH_x[None, :] * np.exp(h * Z[:, None]) + g * Z[:, None]   # n x grid
    eta = a[None, None, :] + b[None, None, :] * E[:, :, None] + c[None, None, :] * Z[:, None, None]
    p = expit(eta)                                           # n x grid x T
    q1, q2 = links(th, mode)
    p[:, :, IPH] = 1 - (1 - p[:, :, IPH]) * (1 - q1 * X[:, None, IIM])
    p[:, :, IMW] = 1 - (1 - p[:, :, IMW]) * (1 - q2 * X[:, None, IRW])
    lp = np.where(X[:, None, :] == 1, np.log(np.clip(p, 1e-12, 1)), np.log(np.clip(1 - p, 1e-12, 1))).sum(axis=2)
    m = lp.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log((np.exp(lp - m) * GH_w[None, :]).sum(axis=1)))


def fit(mode):
    k = 2 * T + (1 if mode.startswith('shared') else T) + (1 if mode.endswith('+spread') else 0) + \
        (2 if mode.endswith('+link') else 0)
    x0 = np.r_[np.log(np.clip(np.average(X, axis=0, weights=W), 0.01, 0.9)) - 1.0, np.full(T, 1.0),
               np.full(k - 2 * T, 0.2)]
    r = minimize(lambda th: -(W * loglik_each(th, mode)).sum(), x0, method='L-BFGS-B', options={'maxiter': 3000})
    return r.x, r.fun, k


def simulate(th, mode, reps=200, seed=1):
    rng = np.random.default_rng(seed)
    a, b, g, c, h = unpack(th, mode)
    E = rng.standard_normal((reps, n)) * np.exp(h * Z[None, :]) + g * Z[None, :]
    p = expit(a[None, None, :] + b[None, None, :] * E[:, :, None] + c[None, None, :] * Z[None, :, None])
    S = (rng.random(p.shape) < p).astype(float)               # reps x n x T
    q1, q2 = links(th, mode)
    S[:, :, IPH] = np.maximum(S[:, :, IPH], S[:, :, IIM] * (rng.random(S.shape[:2]) < q1))
    S[:, :, IMW] = np.maximum(S[:, :, IMW], S[:, :, IRW] * (rng.random(S.shape[:2]) < q2))
    return S


SIZES = ['Micro', 'Small', 'Medium', 'Large']
res = {}
import sys
MODES = sys.argv[1:] or ['shared', 'per type']
for mode in MODES:
    th, f, k = fit(mode)
    res[mode] = th
    print(f'\n##### size {mode}: -loglik {f:.1f}, settings {k}, AIC {2 * k + 2 * f:.1f}')
    a, b, g, c, h = unpack(th, mode)
    print('   type: chance at an average score (Micro) / slope on the score' +
          ('' if mode.startswith('shared') else ' / size odds per step'))
    for i, lab in enumerate(TYPES.values()):
        print(f'   {lab:16s} {expit(a[i]):.3f}  slope {b[i]:5.2f}' + ('' if mode.startswith('shared') else f'  x{np.exp(c[i]):.2f}'))
    if mode.startswith('shared'):
        print(f'   size shifts the score by {g:.2f} per size step')
    if mode.endswith('+link'):
        q1, q2 = links(th, mode)
        print(f'   an impersonation incident also ticks phishing: {q1:.2f}; a ransomware incident also ticks malware: {q2:.2f}')
    if mode.endswith('+spread'):
        print(f'   score spread x{np.exp(h):.2f} per size step (Large vs Micro x{np.exp(3 * h):.2f})')

    S = simulate(th, mode)
    ks = S.sum(axis=2)                                        # reps x n
    kobs = X.sum(axis=1)
    print('   share attacked (any type) and number of types, observed vs model, by size:')
    for z in range(4):
        m = Z == z
        ww = W[m] / W[m].sum()
        obs = [(ww * (kobs[m] == j)).sum() for j in range(4)] + [(ww * (kobs[m] >= 4)).sum()]
        mod = [(ww[None, :] * (ks[:, m] == j)).sum(axis=1).mean() for j in range(4)] + \
              [(ww[None, :] * (ks[:, m] >= 4)).sum(axis=1).mean()]
        print(f'   {SIZES[z]:7s} n={m.sum():4d}  0/1/2/3/4+ types  obs ' + ' '.join(f'{x:.3f}' for x in obs) +
              '   model ' + ' '.join(f'{x:.3f}' for x in mod))

    ww = W / W.sum()
    po = (ww[:, None] * X).sum(axis=0)
    pm = (ww[None, :, None] * S).sum(axis=1).mean(axis=0)
    rows = []
    names = list(TYPES.values())
    for i in range(T):
        for j in range(i + 1, T):
            jo = (ww * X[:, i] * X[:, j]).sum()
            jm = (ww[None, :] * S[:, :, i] * S[:, :, j]).sum(axis=1).mean()
            nij = int((X[:, i] * X[:, j]).sum())
            rows.append((names[i], names[j], nij, jo / (po[i] * po[j]), jm / (pm[i] * pm[j])))
    rows.sort(key=lambda r: -r[2])
    print('   pairs (most common first): firms with both, observed ratio to independence, model ratio')
    for r in rows[:14]:
        print(f'      {r[0]:14s} + {r[1]:15s} n={r[2]:3d}  obs x{r[3]:5.1f}  model x{r[4]:5.1f}')
    big = [r for r in rows if r[2] >= 5]
    lo, lm = np.log([r[3] for r in big]), np.log([r[4] for r in big])
    print(f'   over the {len(big)} pairs with 5+ firms: median obs x{np.exp(np.median(lo)):.1f}, '
          f'model x{np.exp(np.median(lm)):.1f}; correlation of log ratios {np.corrcoef(lo, lm)[0, 1]:.2f}')
