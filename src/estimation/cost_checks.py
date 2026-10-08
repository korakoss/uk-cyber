"""Two deferred checks (2026-10-08). Plain cross-tabs. Attacked firms with a cost answer.
1. Does an ordinary (not spread) breach cost depend on the type it came through?
   Ordinary breaches: breached, not marked as spread (spread marker: 4+ types, inside type, >= £5k; also excluding
   other broad firms with an inside type, which may be spread breaches below £5k). Entry type: the type of the worst
   incident (disrupta); also firms that ticked a single type.
2. Does cost grow with firm size? For three groups (not breached / ordinary breach / spread-marked), cost bands by
   size; and within ordinary breaches whose worst incident was phishing or impersonation (same entry type).
Shares weighted within each cell; n unweighted.
"""
import numpy as np
import pandas as pd

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
broad_in = (df.k >= 4) & (df.n_inside >= 1)
df['spread'] = ((df.breach == 1) & broad_in & (df.c >= 6)).astype(int)
df['grp'] = np.select([df.breach == 0, (df.breach == 1) & ~broad_in, df.spread == 1],
                      ['not breached', 'ordinary breach', 'spread'], 'other')
MID = np.array([0, 50, 300, 750, 3000, 7500, 15000, 35000, 75000, 300000, 750000, 3e6, 7.5e6])
df['mid'] = MID[df.c - 1]
BL = ['none', '<100', '100-500', '500-1k', '1k-5k', '5k-10k', '10k-20k', '20k-50k', '50k-100k', '100k-500k']


def row(lab, s):
    if len(s) == 0:
        return f'   {lab:24s} n=  0'
    w = s.weight / s.weight.sum()
    o = s.sort_values('c')
    med = BL[int(o.c[(o.weight.cumsum() / o.weight.sum()) >= 0.5].iloc[0]) - 1]
    mean_trim = np.average(s.mid[s.c <= 9], weights=s.weight[s.c <= 9]) if (s.c <= 9).any() else np.nan
    return (f'   {lab:24s} n={len(s):3d}  none {w[s.c == 1].sum():.2f}  <£500 {w[s.c.isin([2, 3])].sum():.2f}  '
            f'£500-5k {w[s.c.isin([4, 5])].sum():.2f}  £5k+ {w[s.c >= 6].sum():.2f}   median {med:9s} '
            f'mean (midpoints, excl. £100k+) £{mean_trim:,.0f}  [£100k+ firms: {int((s.c >= 10).sum())}]')


DIS = {6: 'phishing', 5: 'impersonation', 2: 'malware', 11: 'takeover', 4: 'bank hacking', 3: 'DoS',
       1: 'ransomware', 9: 'outsider access', 7: 'staff access', 12: 'other'}
TN = {'type6': 'phishing', 'type5': 'impersonation', 'type2': 'malware', 'type16': 'takeover', 'type4': 'bank hacking',
      'type3': 'DoS', 'type1': 'ransomware'}
o = df[df.grp == 'ordinary breach']
print(f'=== 1. Ordinary breaches (n={len(o)}): cost by entry type ===')
print('-- by type of the worst incident:')
for code, lab in DIS.items():
    print(row(lab, o[o.disrupta == code]))
print('-- firms that ticked a single type:')
for c, lab in TN.items():
    print(row(lab, o[(o.k == 1) & (o[c] == 1)]))
print('-- phishing/impersonation only (any combination of the two) vs any other type present:')
pi = o[TYPES].eq(1)[['type6', 'type5']].sum(axis=1) == o.k
print(row('phishing/imp. only', o[pi]))
print(row('other type present', o[~pi]))

print('\n=== 2. Cost by firm size ===')
SZ = [(1, 'Micro'), (2, 'Small'), (3, 'Medium'), (4, 'Large')]
for g in ['not breached', 'ordinary breach', 'spread']:
    print(f'-- {g}:')
    for z, lab in SZ:
        print(row(lab, df[(df.grp == g) & (df.sizeb == z)]))
print('-- ordinary breaches whose worst incident was phishing or impersonation (same entry type):')
pp = o[o.disrupta.isin([5, 6])]
for z, lab in SZ:
    print(row(lab, pp[pp.sizeb == z]))
print('-- ordinary breaches with phishing/impersonation only:')
for z, lab in SZ:
    print(row(lab, o[pi & (o.sizeb == z)]))
