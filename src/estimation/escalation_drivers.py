"""Escalation: what sets the chance a breach spreads, and how many inside types it reaches (2026-10-07).
Spread marker (F23, user decision): breached, 4+ types, at least one inside type (ransomware, bank hacking,
outsider access, staff access), worst >= £5k. Breached firms with a cost answer. Shares weighted; n unweighted;
bootstrap 90% intervals (resampling firms).
1a. Spread share of breached firms by size.
1b. Spread share by the type of the worst incident (disrupta), as a rough stand-in for the entry type.
    Caveat: for a spread incident the firm may name where it ended up, not how it started.
    Also: which non-inside types spread firms tick, vs broad cheap firms (same breadth precondition).
2.  Number of inside types (and of all types) among spread firms, vs broad cheap firms as the 'ordinary' reference,
    overall and by size.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
TYPES = {'type6': 'phish', 'type5': 'imperson', 'type1': 'ransom', 'type2': 'malware', 'type3': 'DoS',
         'type4': 'bankhack', 'type16': 'takeover', 'type8': 'outsider', 'type7': 'staff', 'type15': 'eavesdrop',
         'type9': 'other'}
for c, n in TYPES.items():
    df[n] = (df[c] == 1).astype(int)
INSIDE = ['ransom', 'bankhack', 'outsider', 'staff']
OUTSIDE = [t for t in TYPES.values() if t not in INSIDE]
df['k'] = df[list(TYPES.values())].sum(axis=1)
df['n_inside'] = df[INSIDE].sum(axis=1)
df['costly'] = (df.damage_bands >= 6).astype(int)
df['broad_in'] = ((df.k >= 4) & (df.n_inside >= 1)).astype(int)
df['spread'] = (df.broad_in * df.costly).astype(int)
b = df[df.breach == 1].copy()


def ci(s, col, reps=2000):
    if len(s) < 3:
        return ''
    v = [np.average(r[col], weights=r.weight) for r in
         (s.sample(len(s), replace=True, random_state=rng) for _ in range(reps))]
    return f'[{np.percentile(v, 5):.2f}-{np.percentile(v, 95):.2f}]'


def row(lab, s, col='spread'):
    if len(s) == 0:
        return f'{lab:26s} n=  0'
    return f'{lab:26s} n={len(s):3d}  {np.average(s[col], weights=s.weight):.2f} {ci(s, col)}'


print(f'breached firms: {len(b)}; spread: {b.spread.sum()}')
print('\n=== 1a. Spread share of breached firms, by size ===')
SIZES = [('Micro', b.sizeb == 1), ('Small', b.sizeb == 2), ('Medium+Large', b.sizeb >= 3), ('all', b.sizeb > 0)]
for lab, m in SIZES:
    print(row(lab, b[m]))
print('split into two steps: reaching 4+ types with an inside type, then costly given that')
for lab, m in SIZES:
    s = b[m]
    print('  ' + row(lab + ' broad+inside', s, 'broad_in') + '   |  ' +
          row('costly given broad+inside', s[s.broad_in == 1], 'costly'))

print('\n=== 1b. Spread share of breached firms, by type of worst incident (disrupta) ===')
DIS = {6: 'phishing', 5: 'impersonation', 1: 'ransomware', 2: 'malware', 3: 'DoS', 4: 'bank hacking',
       11: 'takeover', 9: 'outsider access', 7: 'staff access', 10: 'eavesdropping', 12: 'other'}
for code, name in DIS.items():
    s = b[b.disrupta == code]
    if len(s):
        print(row(name, s) + f'   spread firms naming it: {int(s.spread.sum())}')
print('\nworst incident named by spread firms: inside type ' +
      f'{np.average(b[b.spread == 1].disrupta.isin([1, 4, 9, 7]), weights=b[b.spread == 1].weight):.2f}, '
      f'broad cheap firms: {np.average(b[(b.broad_in == 1) & (b.costly == 0)].disrupta.isin([1, 4, 9, 7]), weights=b[(b.broad_in == 1) & (b.costly == 0)].weight):.2f}')
print('\nnon-inside types ticked: spread vs broad cheap (share of firms)')
sp, bc = b[b.spread == 1], b[(b.broad_in == 1) & (b.costly == 0)]
for t in OUTSIDE:
    print(f'  {t:10s} {np.average(sp[t], weights=sp.weight):.2f}  vs  {np.average(bc[t], weights=bc.weight):.2f}')

print('\n=== 2. Number of inside types (share of firms) ===')
for lab, s in [('spread', sp), ('broad cheap', bc), ('spread, Micro', sp[sp.sizeb == 1]),
               ('spread, Small', sp[sp.sizeb == 2]), ('spread, Medium+Large', sp[sp.sizeb >= 3])]:
    w = s.weight / s.weight.sum()
    print(f'{lab:22s} n={len(s):3d}  1: {w[s.n_inside == 1].sum():.2f}  2: {w[s.n_inside == 2].sum():.2f}  '
          f'3+: {w[s.n_inside >= 3].sum():.2f}  mean {np.average(s.n_inside, weights=s.weight):.2f}')
print('\nNumber of all types (share of firms)')
for lab, s in [('spread', sp), ('broad cheap', bc)]:
    w = s.weight / s.weight.sum()
    print(f'{lab:22s} 4: {w[s.k == 4].sum():.2f}  5: {w[s.k == 5].sum():.2f}  6: {w[s.k == 6].sum():.2f}  '
          f'7+: {w[s.k >= 7].sum():.2f}  mean {np.average(s.k, weights=s.weight):.2f}')
print('\nwhich inside types spread firms tick vs broad cheap')
for t in INSIDE:
    print(f'  {t:10s} {np.average(sp[t], weights=sp.weight):.2f}  vs  {np.average(bc[t], weights=bc.weight):.2f}')
