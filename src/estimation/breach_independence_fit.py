"""Breach prevalence vs independence, using ALL firms rather than single-group firms as the baseline (2026-10-05).
Under independence a firm hit by a set of groups escapes breach only if it escapes on every channel:
    P(no breach | set) = product over groups g in set of q_g
Fit the three escape chances q_P, q_I, q_S to all seven sets at once (weighted binomial likelihood),
then compare observed vs fitted breached share per set. Bootstrap over firms (refitting each time) gives
an interval for observed minus fitted in each set.
Groups: P phishing, I impersonation, S everything else (incl. ransomware). Breach marker as in breach_cost.py.
Attacked businesses (valid freq), pooled and within Micro / Small+.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

rng = np.random.default_rng(1)
df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
OTHER = ['type1', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
df['P'] = (df['type6'] == 1).astype(int)
df['I'] = (df['type5'] == 1).astype(int)
df['S'] = (df[OTHER] == 1).any(axis=1).astype(int)
df['set'] = df.apply(lambda r: ''.join(g for g in 'PIS' if r[g] == 1), axis=1)
df = df[df['set'] != ''].copy()
SETS = ['P', 'I', 'S', 'PI', 'PS', 'IS', 'PIS']


def fit(d):
    H = d[['P', 'I', 'S']].values.astype(float)
    y = d['breach'].values
    w = d['weight'].values / d['weight'].mean()

    def nll(z):
        logq = -np.exp(z)                      # log escape chance per group, < 0
        p_no = np.exp(H @ logq)
        p = np.clip(1 - p_no, 1e-9, 1 - 1e-9)
        return -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum()

    z = minimize(nll, np.log([0.1, 0.1, 0.5]), method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8}).x
    return np.exp(-np.exp(z))                  # q_P, q_I, q_S


def table(d, q):
    out = {}
    for st in SETS:
        s = d[d['set'] == st]
        if len(s) == 0:
            continue
        fitted = 1 - np.prod([q['PIS'.index(g)] for g in st])
        out[st] = (len(s), np.average(s['breach'], weights=s['weight']), fitted)
    return out


for lab, m in [('all sizes', df.sizeb > 0), ('Micro', df.sizeb == 1), ('Small+', df.sizeb >= 2)]:
    d = df[m]
    q = fit(d)
    t = table(d, q)
    boots = {st: [] for st in t}
    for _ in range(300):
        b = d.iloc[rng.integers(0, len(d), len(d))]
        qb = fit(b)
        tb = table(b, qb)
        for st in t:
            if st in tb:
                boots[st].append(tb[st][1] - tb[st][2])
    print(f'\n=== {lab}: per-channel breach chance (1 - q): P {1 - q[0]:.3f}  I {1 - q[1]:.3f}  S {1 - q[2]:.3f} ===')
    print(f"{'set':5s} {'n':>4s} {'observed':>9s} {'fitted':>7s} {'obs - fit':>10s} {'90% interval':>16s}")
    for st, (n, o, f) in t.items():
        lo, hi = np.percentile(boots[st], [5, 95])
        print(f'{st:5s} {n:4d} {o:9.2f} {f:7.2f} {o - f:+10.2f}   [{lo:+.2f}, {hi:+.2f}]')

# --- 2026-10-05: how solid is it? (1) S rate implied separately by each S-containing set, given the fitted P and I
# rates; (2) how many S types firms in each set ticked (S lumps several types into one channel).
print('\n=== S breach chance implied by each S-containing set (all sizes; P, I rates from the joint fit) ===')
q = fit(df)
for st in ['S', 'PS', 'IS', 'PIS']:
    s = df[df['set'] == st]
    o = np.average(s['breach'], weights=s['weight'])
    other = np.prod([q['PIS'.index(g)] for g in st if g != 'S'])
    print(f'{st:4s} n={len(s):3d}  observed breached {o:.2f}  implied S chance {1 - (1 - o) / other:.2f}')
df['nS'] = (df[OTHER] == 1).sum(axis=1)
print('\n=== number of S types ticked, by set (weighted share) ===')
for st in ['S', 'PS', 'IS', 'PIS']:
    s = df[df['set'] == st]
    print(f'{st:4s} ' + '  '.join(f'{k}: {s.loc[s.nS == k, "weight"].sum() / s.weight.sum():.2f}' for k in [1, 2])
          + f'  3+: {s.loc[s.nS >= 3, "weight"].sum() / s.weight.sum():.2f}   mean {np.average(s.nS, weights=s.weight):.2f}')
