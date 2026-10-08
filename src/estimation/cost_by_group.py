"""Worst-incident cost bands for three groups (2026-10-07): spread breaches (marker: breached, 4+ types, inside type,
>= £5k), breached but not spread, attacked but not breached. Attacked firms with a cost answer.
Counts unweighted; shares weighted.
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
df['spread'] = ((df.breach == 1) & (df.k >= 4) & (df.n_inside >= 1) & (df.c >= 6)).astype(int)
G = [('spread', df.spread == 1), ('breached, not spread', (df.breach == 1) & (df.spread == 0)),
     ('not breached', df.breach == 0)]
BL = ['none', '<100', '100-500', '500-1k', '1k-5k', '5k-10k', '10k-20k', '20k-50k', '50k-100k', '100k-500k']
print(f"{'band':10s}" + ''.join(f'{g:>26s}' for g, _ in G))
for bd in range(1, 11):
    cells = []
    for _, m in G:
        s = df[m]
        cells.append(f'{int((s.c == bd).sum()):4d}  {s.weight[s.c == bd].sum() / s.weight.sum():5.3f}')
    print(f'{BL[bd - 1]:10s}' + ''.join(f'{c:>26s}' for c in cells))
print(f"{'total n':10s}" + ''.join(f'{int(m.sum()):>26d}' for _, m in G))
