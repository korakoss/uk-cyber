"""Fresh look (2026-10-05) at three core variables and how they relate:
  sizeb        - firm size band (Micro / Small / Medium / Large)
  freq         - how often attacked in the last 12 months (1 = once ... 6 = several times a day)
  damage_bands - cost of the single worst incident (1 = no cost ... 13 = £5m+)
Businesses only. Shares are weighted by the survey weight; n = unweighted counts.
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[df['questtype'] == 1].copy()  # private-sector businesses

SIZE = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}
FREQ = {1: 'once', 2: '<monthly', 3: 'monthly', 4: 'weekly', 5: 'daily', 6: 'several/day'}
COST = {1: 'none', 2: '<100', 3: '100-500', 4: '500-1k', 5: '1k-5k', 6: '5k-10k',
        7: '10k-20k', 8: '20k-50k', 9: '50k-100k', 10: '100k-500k', 11: '500k-1m', 12: '1m-5m', 13: '5m+'}
# rough cost groups so the tables stay readable
COSTGRP = {1: 'none', 2: '<500', 3: '<500', 4: '500-5k', 5: '500-5k', 6: '5k-50k', 7: '5k-50k', 8: '5k-50k',
           9: '50k+', 10: '50k+', 11: '50k+', 12: '50k+', 13: '50k+'}

df['size'] = df['sizeb'].map(SIZE)
df['f'] = df['freq'].where(df['freq'].isin(FREQ.keys()))
df['c'] = df['damage_bands'].where(df['damage_bands'].isin(COST.keys()))
w = df['weight']


def wshare(sub, col, levels):
    tot = sub['weight'].sum()
    return [sub.loc[sub[col] == l, 'weight'].sum() / tot for l in levels]


def table(rows, rowcol, rowlevels, rowlabels, col, levels, labels, data):
    print(f"{'':12s} {'n':>5s} " + ' '.join(f'{l:>9s}' for l in labels))
    for rl, lab in zip(rowlevels, rowlabels):
        sub = data[data[rowcol] == rl]
        if len(sub) == 0:
            continue
        print(f'{lab:12s} {len(sub):5d} ' + ' '.join(f'{s:9.2f}' for s in wshare(sub, col, levels)))
    print(f"{'all':12s} {len(data):5d} " + ' '.join(f'{s:9.2f}' for s in wshare(data, col, levels)))


sizes = ['Micro', 'Small', 'Medium', 'Large']

print('=== 1. Sample and share attacked, by size ===')
for s in sizes:
    sub = df[df['size'] == s]
    att = sub['f'].notna()
    print(f'{s:7s} businesses {len(sub):5d}  attacked {att.sum():5d}  '
          f'weighted share attacked {sub.loc[att, "weight"].sum() / sub["weight"].sum():.2f}')

att = df[df['f'].notna()].copy()
print(f'\nattacked with a usable cost answer: {att["c"].notna().sum()} of {len(att)}')

print('\n=== 2. How often attacked, by size (attacked firms; weighted shares) ===')
table(None, 'size', sizes, sizes, 'f', list(FREQ), list(FREQ.values()), att)

attc = att[att['c'].notna()].copy()
attc['cg'] = attc['c'].map(COSTGRP)
CG = ['none', '<500', '500-5k', '5k-50k', '50k+']

print('\n=== 3. Worst-incident cost, by size (attacked firms with cost answer) ===')
table(None, 'size', sizes, sizes, 'cg', CG, CG, attc)

print('\n=== 4. Worst-incident cost, by how often attacked ===')
for grp, members in [('Micro', ['Micro']), ('Small+', ['Small', 'Medium', 'Large']), ('all sizes', sizes)]:
    print(f'-- {grp}')
    table(None, 'f', list(FREQ), list(FREQ.values()), 'cg', CG, CG, attc[attc['size'].isin(members)])

print('\n=== 5. Reverse: how often attacked, by worst-incident cost ===')
for grp, members in [('Micro', ['Micro']), ('Small+', ['Small', 'Medium', 'Large'])]:
    print(f'-- {grp}')
    table(None, 'cg', CG, CG, 'f', list(FREQ), list(FREQ.values()), attc[attc['size'].isin(members)])

print('\n=== 6. Raw counts of the top cost bands, by size ===')
top = attc[attc['c'] >= 8]
print(pd.crosstab(top['c'].map(COST), top['size']).reindex(columns=sizes, fill_value=0))
