"""Is the 'breached broad firms cost more than independence' excess just ransomware? (2026-10-05)
Plain cross-tab, breached firms only (breach marker as in breach_cost.py):
  rows    = ransomware ticked or not
  columns = breadth measured WITHOUT the ransomware tick (number of other types ticked)
Cells: n, share with worst >= £500, share >= £5k (weighted).
If breadth matters with ransomware held fixed (both rows rise left to right), there is a breadth fact beyond
ransomware. If only the ransomware row is high and the other row is flat, it is mostly ransomware.
Also the same table for NOT breached firms, and a list of the ransomware firms with narrow breadth.
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
df['c'] = df['damage_bands'].astype(int)
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
NON_R = ['type6', 'type5', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
df['R'] = (df['type1'] == 1).astype(int)
df['k'] = (df[NON_R] == 1).sum(axis=1)          # types ticked other than ransomware
df['kb'] = pd.cut(df['k'], [-1, 1, 2, 3, 99], labels=['0-1', '2', '3', '4+'])
COLS = ['0-1', '2', '3', '4+']


def cell(s):
    if len(s) == 0:
        return f"{'-':>22s}"
    w = s.weight / s.weight.sum()
    return f'{len(s):3d}  {w[s.c >= 4].sum():.2f} / {w[s.c >= 6].sum():.2f}    '


for lab, d in [('BREACHED firms', df[df.breach == 1]), ('NOT breached firms', df[df.breach == 0])]:
    print(f'\n=== {lab}: n   share >= £500 / >= £5k, by number of types ticked other than ransomware ===')
    print(f"{'':16s}" + ''.join(f'{c:>22s}' for c in COLS))
    for r, rl in [(0, 'no ransomware'), (1, 'ransomware')]:
        print(f'{rl:16s}' + ''.join(f'{cell(d[(d.R == r) & (d.kb == c)]):>22s}' for c in COLS))

BL = {1: 'none', 2: '<100', 3: '100-500', 4: '500-1k', 5: '1k-5k', 6: '5k-10k', 7: '10k-20k', 8: '20k-50k',
      9: '50k-100k', 10: '100k-500k'}
print('\n=== Ransomware firms (all), worst cost band by breadth without ransomware (counts) ===')
r = df[df.R == 1]
print(pd.crosstab(r['c'].map(BL), r['kb']).reindex(list(BL.values())).fillna(0).astype(int))
print('breached share of ransomware firms by breadth:',
      {c: f'{r[r.kb == c].breach.mean():.2f} (n={len(r[r.kb == c])})' for c in COLS})
