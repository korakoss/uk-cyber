"""Spreading as a hidden event, fitted to all attacked firms (2026-10-08). User design.
No hard 'spread' marker. A breach may spread (unobserved). If it does:
  - it may add inside-type ticks (ransomware, bank hacking, outsider access, staff access) not already there;
  - its cost comes from a separate (higher) cost curve;
  - it tends to have more consequences (Q56A outcome items).
For each firm we add up the 'spread' and 'not spread' versions, weighted by their chances; settings are chosen to make
everything observed most likely. Firms not breached cannot spread; they inform the background tick rates and the
clean-up cost curve.

Per firm (observed: size band z = 1..4, breach marker B, the 4 inside ticks, number of OTHER ticks K (phishing,
impersonation, malware, DoS, takeover, eavesdropping, other; taken as given, a rough measure of how heavily the firm
is attacked), worst-cost band, number of outcomes for breached firms):
  - background chance of inside tick j: logistic(a_j + b*K + c*(z-1) + d*B)
  - spread (breached firms only): chance logistic(s0 + s1*(z-1)); if spread, tick j is added with chance r_j
  - cost: not breached:    'no cost' with chance p0n, else lognormal(mu_n, sd_n)
          ordinary breach: 'no cost' with chance p0o, else lognormal(mu_o, sd_o)
          spread breach:   lognormal(mu_s, sd_s)
          all costs shifted by a shared factor g per size step (log cost + log g * (z-1))
  - outcomes (breached firms): Poisson with mean lam_o (ordinary) or lam_s (spread)
One breach per breached firm (simplification). Plain lognormal curves; checked afterwards against observed bands.
Weights: survey weights scaled to sum to n (main); unweighted run as a check (pass 'unweighted').
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import norm

UNW = 'unweighted' in sys.argv
df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
INSIDE = {'type1': 'ransom', 'type4': 'bankhack', 'type8': 'outsider', 'type7': 'staff'}
OTHER = ['type6', 'type5', 'type2', 'type3', 'type16', 'type15', 'type9']
I = (df[list(INSIDE)] == 1).values.astype(float)               # n x 4
K = (df[OTHER] == 1).sum(axis=1).values.astype(float)
Z = df.sizeb.values.astype(int) - 1                             # 0..3
B = df.breach.values.astype(int)
C = df.damage_bands.values.astype(int) - 1                      # 0..12
NOUT = sum((num(f'outcome{j}') == 1).astype(int) for j in [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]).values
W = np.ones(len(df)) if UNW else (df.weight / df.weight.mean()).values
n = len(df)
LOGE = np.log(np.array([100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 500000, 1e6, 5e6]))  # bands 2..12 tops

NAMES = (['a_' + v for v in INSIDE.values()] + ['b_other', 'c_size', 'd_breach'] + ['r_' + v for v in INSIDE.values()] +
         ['s0', 's1', 'p0n', 'mu_n', 'lsd_n', 'p0o', 'mu_o', 'lsd_o', 'mu_s', 'lsd_s', 'lg_size', 'llam_o', 'llam_s'])
IX = {k: i for i, k in enumerate(NAMES)}


def band_probs(p0, mu, sd, shift):
    """13 band probabilities for: 'no cost' with chance p0, else lognormal(mu + shift, sd)."""
    cdf = norm.cdf((LOGE - mu - shift) / sd)
    pos = np.diff(np.concatenate([[0.0], cdf, [1.0]]))
    return np.concatenate([[p0], (1 - p0) * pos])


def parts(th):
    g = lambda k: th[IX[k]]
    a = np.array([g('a_' + v) for v in INSIDE.values()])
    pb = expit(a[None, :] + g('b_other') * K[:, None] + g('c_size') * Z[:, None] + g('d_breach') * B[:, None])
    r = expit(np.array([g('r_' + v) for v in INSIDE.values()]))
    ps = 1 - (1 - pb) * (1 - r[None, :])
    tick_ns = np.prod(np.where(I == 1, pb, 1 - pb), axis=1)
    tick_s = np.prod(np.where(I == 1, ps, 1 - ps), axis=1)
    s = expit(g('s0') + g('s1') * Z)
    lgz = g('lg_size')
    cn, co, cs = np.zeros(n), np.zeros(n), np.zeros(n)
    for z in range(4):
        m = Z == z
        cn[m] = band_probs(expit(g('p0n')), g('mu_n'), np.exp(g('lsd_n')), lgz * z)[C[m]]
        co[m] = band_probs(expit(g('p0o')), g('mu_o'), np.exp(g('lsd_o')), lgz * z)[C[m]]
        cs[m] = band_probs(0.0, g('mu_s'), np.exp(g('lsd_s')), lgz * z)[C[m]]
    pois = lambda lam: np.exp(NOUT * np.log(lam) - lam - gammaln(NOUT + 1))
    oo, os_ = pois(np.exp(g('llam_o'))), pois(np.exp(g('llam_s')))
    L_ns = tick_ns * np.where(B == 1, co * oo, cn)
    L_s = tick_s * cs * os_
    return s, L_ns, L_s


def nll(th):
    s, L_ns, L_s = parts(th)
    L = np.where(B == 1, (1 - s) * L_ns + s * L_s, L_ns)
    return -(W * np.log(np.clip(L, 1e-300, None))).sum()


x0 = np.zeros(len(NAMES))
init = {'a_ransom': -3, 'a_bankhack': -3, 'a_outsider': -4, 'a_staff': -4, 'b_other': 0.5, 'c_size': 0.2,
        'd_breach': 1.0, 'r_ransom': -1, 'r_bankhack': -1, 'r_outsider': -2, 'r_staff': -2, 's0': -2, 's1': 0.3,
        'p0n': 0.5, 'mu_n': np.log(150), 'lsd_n': 0.3, 'p0o': -1.5, 'mu_o': np.log(600), 'lsd_o': 0.5,
        'mu_s': np.log(15000), 'lsd_s': 0.0, 'lg_size': 0.2, 'llam_o': 0.0, 'llam_s': 1.2}
for k, v in init.items():
    x0[IX[k]] = v
best = None
for jit in range(6):
    st = x0 + (0 if jit == 0 else np.random.default_rng(jit).normal(0, 0.5, len(x0)))
    r = minimize(nll, st, method='L-BFGS-B', options={'maxiter': 5000})
    r = minimize(nll, r.x, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-6, 'fatol': 1e-8})
    if best is None or r.fun < best.fun:
        best = r
th = best.x
g = lambda k: th[IX[k]]
print(f'{"UNWEIGHTED" if UNW else "survey weights"}: n = {n}, breached {B.sum()}; -loglik {best.fun:.2f}')

s, L_ns, L_s = parts(th)
post = np.where(B == 1, s * L_s / ((1 - s) * L_ns + s * L_s), 0.0)
wt = df.weight.values
print('\n=== Spreading ===')
for z, lab in enumerate(['Micro', 'Small', 'Medium', 'Large']):
    m = (B == 1) & (Z == z)
    print(f'   {lab:7s} chance a breach spreads {expit(g("s0") + g("s1") * z):.2f}   '
          f'expected spread firms in sample {post[m].sum():5.1f} of {m.sum()} breached')
print(f'   all breached firms (weighted): spread share {np.average(post[B == 1], weights=wt[B == 1]):.2f}; '
      f'expected spread firms in sample {post.sum():.1f}')
print('   chance spreading adds each inside type (if not already there): ' +
      ', '.join(f'{v} {expit(g("r_" + v)):.2f}' for v in INSIDE.values()))
print(f'   background inside ticks: per extra other-type tick x{np.exp(g("b_other")):.2f} odds, '
      f'per size step x{np.exp(g("c_size")):.2f}, breached firms x{np.exp(g("d_breach")):.2f}')
print(f'   outcomes per breached firm: ordinary {np.exp(g("llam_o")):.2f}, spread {np.exp(g("llam_s")):.2f}')

print('\n=== Cost curves (Micro level; x{:.2f} per size step) ==='.format(np.exp(g('lg_size'))))
for lab, p0, mu, lsd in [('not breached', expit(g('p0n')), g('mu_n'), g('lsd_n')),
                          ('ordinary breach', expit(g('p0o')), g('mu_o'), g('lsd_o')),
                          ('spread breach', 0.0, g('mu_s'), g('lsd_s'))]:
    sd = np.exp(lsd)
    print(f'   {lab:16s} no cost {p0:.2f}; otherwise median £{np.exp(mu):,.0f}, log-sd {sd:.2f}, '
          f'mean £{(1 - p0) * np.exp(mu + sd ** 2 / 2):,.0f}')

print('\n=== Who looks spread: posterior chance vs the old 34-firm marker ===')
k_all = I.sum(axis=1) + K
old = (B == 1) & (k_all >= 4) & (I.sum(axis=1) >= 1) & (C >= 5)
for lab, m in [('old marker yes', old), ('old marker no, breached', (B == 1) & ~old)]:
    p = post[m]
    print(f'   {lab:25s} n={m.sum():3d}  mean posterior {p.mean():.2f}  >0.5: {(p > 0.5).sum()}  sum {p.sum():.1f}')
BL = ['none', '<100', '100-500', '500-1k', '1k-5k', '5k-10k', '10k-20k', '20k-50k', '50k-100k', '100k-500k']
print('   breached firms with posterior > 0.5 by cost band: ' +
      ', '.join(f'{BL[b]} {int(((post > 0.5) & (C == b)).sum())}' for b in range(10) if ((post > 0.5) & (C == b)).sum()))

print('\n=== Checks: predicted vs observed (weighted shares; counts for top bands) ===')
pred = np.zeros((n, 13))
for z in range(4):
    m = Z == z
    bn = band_probs(expit(g('p0n')), g('mu_n'), np.exp(g('lsd_n')), g('lg_size') * z)
    bo = band_probs(expit(g('p0o')), g('mu_o'), np.exp(g('lsd_o')), g('lg_size') * z)
    bs = band_probs(0.0, g('mu_s'), np.exp(g('lsd_s')), g('lg_size') * z)
    sz = expit(g('s0') + g('s1') * z)
    pred[m] = np.where(B[m, None] == 1, (1 - sz) * bo + sz * bs, bn)
GR = [('none', [0]), ('<500', [1, 2]), ('500-5k', [3, 4]), ('5k-20k', [5, 6]), ('20k-100k', [7, 8]), ('100k-500k', [9]),
      ('500k+', [10, 11, 12])]
for lab, m in [('breached', B == 1), ('not breached', B == 0)]:
    ww = wt[m] / wt[m].sum()
    print(f'   {lab}:')
    print('      ' + ''.join(f'{g_:>11s}' for g_, _ in GR))
    print('  obs ' + ''.join(f'{(ww * np.isin(C[m], bb)).sum():11.3f}' for _, bb in GR))
    print('  fit ' + ''.join(f'{(ww[:, None] * pred[m][:, bb]).sum():11.3f}' for _, bb in GR))
    print(f'      firms at £100k+: observed {int((C[m] >= 9).sum())}, expected {pred[m][:, 9:].sum():.1f}; '
          f'at £20k+: observed {int((C[m] >= 7).sum())}, expected {pred[m][:, 7:].sum():.1f}')
# inside-tick counts among breached firms
sz = expit(g('s0') + g('s1') * Z)
a = np.array([g('a_' + v) for v in INSIDE.values()])
pb = expit(a[None, :] + g('b_other') * K[:, None] + g('c_size') * Z[:, None] + g('d_breach') * B[:, None])
ps = 1 - (1 - pb) * (1 - expit(np.array([g('r_' + v) for v in INSIDE.values()]))[None, :])


def count_dist(p):
    d = np.zeros((len(p), 5))
    d[:, 0] = 1
    for j in range(4):
        d = d * (1 - p[:, [j]]) + np.concatenate([np.zeros((len(p), 1)), d[:, :-1]], axis=1) * p[:, [j]]
    return d


mb = B == 1
pd_ = (1 - sz[mb, None]) * count_dist(pb[mb]) + sz[mb, None] * count_dist(ps[mb])
obs = np.bincount(I[mb].sum(axis=1).astype(int), minlength=5)
print('   number of inside ticks, breached firms (counts): obs ' + ' '.join(f'{x}' for x in obs) +
      '   fit ' + ' '.join(f'{x:.1f}' for x in pd_.sum(axis=0)))
mn = B == 0
obs = np.bincount(I[mn].sum(axis=1).astype(int), minlength=5)
print('   number of inside ticks, non-breached firms:     obs ' + ' '.join(f'{x}' for x in obs) +
      '   fit ' + ' '.join(f'{x:.1f}' for x in count_dist(pb[mn]).sum(axis=0)))
obs = np.bincount(NOUT[mb], minlength=12)[:8]
po = lambda lam: np.exp(np.arange(8) * np.log(lam) - lam - gammaln(np.arange(8) + 1))
fit_o = ((1 - sz[mb, None]) * po(np.exp(g('llam_o')))[None, :] + sz[mb, None] * po(np.exp(g('llam_s')))[None, :]).sum(axis=0)
print('   outcomes per breached firm 0..7 (counts): obs ' + ' '.join(f'{x}' for x in obs) +
      '   fit ' + ' '.join(f'{x:.0f}' for x in fit_o))
print('\nall settings: ' + ', '.join(f'{k} {v:.2f}' for k, v in zip(NAMES, th)))
