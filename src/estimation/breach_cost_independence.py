"""Cost comparison for breached firms using the per-type breach chances fitted on all firms (F13), 2026-10-05.

Question: do breached broad firms cost more than independent breaches would give, once each type's breach chance
AND each type's breach cost are allowed for? (Replaces the single-group-baseline comparisons in F8 / F11.)

Set-up (breached firms with a cost answer):
- Which of a firm's ticked types were breached is unknown. Under independence, given that the firm was breached,
  each subset B of its ticked types has probability  prod_{t in B} b_t * prod_{t not in B} (1 - b_t), rescaled to
  exclude the empty set. b_t are the F13 per-type breach chances (held fixed).
- Each breach costs: zero with probability z, otherwise a lognormal draw with a type-group median and one shared
  spread. Worst incident = the largest breach cost (non-breach clean-up costs are ignored here; F8 shows they rarely
  reach £500).
- Breadth term: every breach cost is multiplied by exp(beta * (k - 1)), k = number of types ticked.
  beta = 0 is 'independent breaches, no breadth effect'.
Fit by weighted likelihood on the cost bands, with beta fixed at 0 and with beta free. Compare, and show observed vs
predicted shares (>= £500, >= £5k) by breadth under beta = 0.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

rng = np.random.default_rng(4)
df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
TYPES = ['type6', 'type5', 'type1', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
NAMES = ['phishing', 'impersonation', 'ransomware', 'malware', 'DoS', 'bank hacking', 'takeover', 'outsider access',
         'staff access', 'eavesdropping', 'other']
# per-type breach chances from type_breach_rates.py (F13, fit on all firms)
B_RATE = np.array([0.06, 0.09, 0.35, 0.19, 0.51, 0.59, 0.36, 0.52, 0.08, 0.07, 0.25])
# cost groups: each type gets its own median except the four thin access/other types, which share one
GROUP = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7])
GNAMES = ['phishing', 'impersonation', 'ransomware', 'malware', 'DoS', 'bank hacking', 'takeover', 'access/other']
NG = GROUP.max() + 1

d = df[(df.breach == 1) & df['damage_bands'].isin(range(1, 14))].copy()
H = (d[TYPES] == 1).values
d['k'] = H.sum(axis=1)
d = d[d.k > 0]
H = H[d.index.isin(d.index)] if False else (d[TYPES] == 1).values
band = d['damage_bands'].astype(int).values
w = (d['weight'] / d['weight'].mean()).values
k = d['k'].values
LO = {1: 0, 2: 0, 3: 100, 4: 500, 5: 1000, 6: 5000, 7: 10000, 8: 20000, 9: 50000, 10: 100000, 11: 500000,
      12: 1e6, 13: 5e6}
HI = {1: 0, 2: 100, 3: 500, 4: 1000, 5: 5000, 6: 10000, 7: 20000, 8: 50000, 9: 100000, 10: 500000, 11: 1e6,
      12: 5e6, 13: np.inf}
lo = np.array([LO[b] for b in band], float)
hi = np.array([HI[b] for b in band], float)
print(f'breached firms with cost answer: {len(d)}')

# flatten (firm, breached subset) rows
rows_f, rows_M, rows_p = [], [], []
for i in range(len(d)):
    ticked = np.where(H[i])[0]
    subs = []
    for r in range(1, len(ticked) + 1):
        for S in itertools.combinations(ticked, r):
            m = np.zeros(len(TYPES), bool)
            m[list(S)] = True
            p = np.prod(np.where(m[ticked], B_RATE[ticked], 1 - B_RATE[ticked]))
            subs.append((m, p))
    tot = sum(p for _, p in subs)
    for m, p in subs:
        rows_f.append(i)
        rows_M.append(m)
        rows_p.append(p / tot)
rows_f = np.array(rows_f)
rows_M = np.array(rows_M, float)
rows_p = np.array(rows_p)
print(f'(firm, breached-subset) rows: {len(rows_f)}')


def log_cdf_terms(x, mu_t, sig, z):
    """log P(one breach of each type <= x), per row and type; x per row."""
    with np.errstate(divide='ignore'):
        lx = np.log(np.maximum(x, 1e-300))
    F = np.where(x[:, None] > 0, norm.cdf((lx[:, None] - mu_t[None, :]) / sig), 0.0)
    F = np.where(np.isinf(x)[:, None], 1.0, F)
    return np.log(np.clip(z + (1 - z) * F, 1e-300, 1))


def unpack(theta, beta_free):
    mu_g = theta[:NG]
    sig = np.exp(theta[NG])
    z = 1 / (1 + np.exp(-theta[NG + 1]))
    beta = theta[NG + 2] if beta_free else 0.0
    return mu_g, sig, z, beta


def row_band_prob(theta, beta_free, x_lo, x_hi, zero_band):
    mu_g, sig, z, beta = unpack(theta, beta_free)
    mu_t = mu_g[GROUP]
    shift = beta * (k[rows_f] - 1)
    # costs scale by exp(shift): equivalent to evaluating at x * exp(-shift)
    xl = x_lo[rows_f] * np.exp(-shift)
    xh = x_hi[rows_f] * np.exp(-shift)
    cl = np.exp((rows_M * log_cdf_terms(xl, mu_t, sig, z)).sum(axis=1))
    ch = np.exp((rows_M * log_cdf_terms(xh, mu_t, sig, z)).sum(axis=1))
    zero = np.exp(rows_M.sum(axis=1) * np.log(z))              # all breaches cost zero
    p = np.where(zero_band[rows_f], zero, np.where(x_lo[rows_f] == 0, ch - zero, ch - cl))
    return p


zero_band = band == 1


def nll(theta, beta_free):
    pr = row_band_prob(theta, beta_free, lo, hi, zero_band)
    firm_p = np.bincount(rows_f, weights=rows_p * pr, minlength=len(d))
    return -(w * np.log(np.clip(firm_p, 1e-300, 1))).sum()


theta0 = np.concatenate([np.full(NG, np.log(1000)), [np.log(2.0)], [-1.0]])
f0 = minimize(nll, theta0, args=(False,), method='L-BFGS-B')
f1 = minimize(nll, np.concatenate([f0.x, [0.0]]), args=(True,), method='L-BFGS-B')
print(f'\nlog-likelihood: no breadth term {-f0.fun:.1f}; with breadth term {-f1.fun:.1f}; '
      f'gain {f0.fun - f1.fun:.1f} for 1 extra parameter')
for lab, f, bf in [('no breadth term', f0, False), ('with breadth term', f1, True)]:
    mu_g, sig, z, beta = unpack(f.x, bf)
    print(f'-- {lab}: spread {sig:.2f}, zero-cost share per breach {z:.2f}, breadth factor per extra type '
          f'x{np.exp(beta):.2f}')
    print('   median cost per breach: ' + ', '.join(f'{n} £{np.exp(m):,.0f}' for n, m in zip(GNAMES, mu_g)))

# bootstrap interval for the breadth factor (resample firms, refit with breadth term)
bb = []
for _ in range(60):
    idx = rng.integers(0, len(d), len(d))
    cnt = np.bincount(idx, minlength=len(d)).astype(float)
    w_save = w.copy()
    w = w_save * cnt
    fb = minimize(nll, f1.x, args=(True,), method='L-BFGS-B')
    bb.append(np.exp(unpack(fb.x, True)[3]))
    w = w_save
print(f'breadth factor bootstrap 90% interval: x{np.percentile(bb, 5):.2f} to x{np.percentile(bb, 95):.2f}')

# observed vs predicted shares by breadth, under the no-breadth fit (independent breaches, own type costs)
print('\n=== Breached firms: observed vs predicted under independence (each type its own breach chance and cost) ===')
inf = np.full(len(d), np.inf)


def p_at_least(theta, bf, x):
    xs = np.full(len(d), x, float)
    pr = row_band_prob(theta, bf, xs, inf, np.zeros(len(d), bool))      # P(worst in (x, inf))
    return np.bincount(rows_f, weights=rows_p * pr, minlength=len(d))


p500_0, p5k_0 = p_at_least(f0.x, False, 500), p_at_least(f0.x, False, 5000)
p500_1, p5k_1 = p_at_least(f1.x, True, 500), p_at_least(f1.x, True, 5000)
R = (d['type1'] == 1).values
print(f"{'types':6s} {'n':>4s} {'>=£500 obs':>11s} {'pred':>6s} {'pred+b':>7s} {'>=£5k obs':>10s} {'pred':>6s} {'pred+b':>7s}")
for lab, m in [('1', k == 1), ('2', k == 2), ('3', k == 3), ('4', k == 4), ('5+', k >= 5),
               ('3+ no ransomware', (k >= 3) & ~R), ('3+ ransomware', (k >= 3) & R)]:
    ww = w[m]
    print(f'{lab:6s} {m.sum():4d} {np.average(band[m] >= 4, weights=ww):11.2f} {np.average(p500_0[m], weights=ww):6.2f} '
          f'{np.average(p500_1[m], weights=ww):7.2f} {np.average(band[m] >= 6, weights=ww):10.2f} '
          f'{np.average(p5k_0[m], weights=ww):6.2f} {np.average(p5k_1[m], weights=ww):7.2f}')
print('(pred = independence, no breadth term; pred+b = with breadth term)')
