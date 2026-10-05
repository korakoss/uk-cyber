"""Per-TYPE breach chances inferred from all firms (2026-10-05).
Independence at type level: a firm escapes breach only if it escapes on every type it ticked:
    P(no breach | ticked types) = product over ticked types t of q_t
Fit the 11 escape chances q_t by weighted likelihood on all attacked businesses (breach marker as in
breach_cost.py). Report each type's breach chance (1 - q_t) with a bootstrap 90% interval.
Then check where the fit fails: observed vs fitted breached share by number of types ticked.
Second fit: using only firms with 2+ types, then predict the single-type firms (are they odd?).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

rng = np.random.default_rng(3)
df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
TYPES = {'type6': 'phishing', 'type5': 'impersonation', 'type1': 'ransomware', 'type2': 'malware', 'type3': 'DoS',
         'type4': 'bank hacking', 'type16': 'takeover', 'type8': 'outsider access', 'type7': 'staff access',
         'type15': 'eavesdropping', 'type9': 'other'}
H = (df[list(TYPES)] == 1).values.astype(float)
df['k'] = H.sum(axis=1)
keep = df['k'].values > 0
df, H = df[keep].copy(), H[keep]
y = df['breach'].values
w = (df['weight'] / df['weight'].mean()).values


def fit(Hm, ym, wm):
    def nll(z):
        p_no = np.exp(Hm @ (-np.exp(z)))
        p = np.clip(1 - p_no, 1e-9, 1 - 1e-9)
        return -(wm * (ym * np.log(p) + (1 - ym) * np.log(1 - p))).sum()
    z0 = np.full(Hm.shape[1], np.log(0.2))
    z = minimize(nll, z0, method='L-BFGS-B', bounds=[(-12, 4)] * Hm.shape[1]).x
    return np.exp(-np.exp(z))                       # escape chances q_t


q = fit(H, y, w)
boot = []
for _ in range(200):
    i = rng.integers(0, len(df), len(df))
    boot.append(fit(H[i], y[i], w[i]))
boot = np.array(boot)
print('=== Per-type breach chance (fitted on all firms), bootstrap 90% interval; n firms ticking ===')
for j, name in enumerate(TYPES.values()):
    lo, hi = np.percentile(1 - boot[:, j], [5, 95])
    print(f'{name:16s} {1 - q[j]:5.2f}  [{lo:.2f}, {hi:.2f}]   n={int(H[:, j].sum())}')

fitted = 1 - np.exp(H @ np.log(q))


def by_k(fitted_vals, lab):
    print(f'\n=== {lab}: breached share by number of types ticked, observed vs fitted ===')
    for k, kl in [(1, '1'), (2, '2'), (3, '3'), (4, '4'), (5, '5+')]:
        m = (df.k.values == k) if k < 5 else (df.k.values >= 5)
        o = np.average(y[m], weights=w[m])
        f = np.average(fitted_vals[m], weights=w[m])
        print(f'{kl:3s} types  n={m.sum():4d}  observed {o:.2f}  fitted {f:.2f}  gap {o - f:+.2f}')


by_k(fitted, 'Fit on all firms')

multi = df.k.values >= 2
q2 = fit(H[multi], y[multi], w[multi])
print('\n=== Fit on firms with 2+ types only: per-type breach chance ===')
for j, name in enumerate(TYPES.values()):
    print(f'{name:16s} {1 - q2[j]:5.2f}   (all-firm fit {1 - q[j]:.2f})')
f2 = 1 - np.exp(H @ np.log(q2))
by_k(f2, 'Fit on 2+ type firms, predicting all')
print('\nsingle-type firms by type: observed breached vs predicted from the 2+ fit')
for j, name in enumerate(TYPES.values()):
    m = (df.k.values == 1) & (H[:, j] == 1)
    if m.sum() >= 3:
        print(f'{name:16s} n={m.sum():3d}  observed {np.average(y[m], weights=w[m]):.2f}  predicted {1 - q2[j]:.2f}')
