"""Yearly cost of attacks that did NOT breach: what the raw data shows (2026-10-08).
Only the worst incident is priced (damage_bands). Other relevant items:
  - attack frequency (freq: once ... several times a day)
  - parts of the worst incident's cost: staff time (damagestaffx_bands), external payments during / after
    (damagedirsx_bands / damagedirlx_bands), damage or disruption (damageindx_bands)
  - yearly totals: crimecost_bands (all cyber crimes incl. fraud), notfraudcost_bands (all cyber crimes except fraud),
    and per type: ranscost, hackcost, tkvrcost, doscost, viruscost (all on the same 13-band scale)
Attacked firms with a cost answer, NOT breached (breach marker as usual). Counts unweighted; shares weighted.
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
df['c'] = df.damage_bands.astype(int)
nb = df[df.breach == 0].copy()
BL = ['none', '<100', '100-500', '500-1k', '1k-5k', '5k-10k', '10k-20k', '20k-50k', '50k-100k', '100k-500k',
      '500k-1m', '1m-5m', '5m+']
FQ = {1: 'once', 2: '<monthly', 3: 'monthly', 4: 'weekly', 5: 'daily', 6: 'several/day'}
print(f'not-breached attacked firms with a cost answer: {len(nb)}')


def dist_row(lab, s, col='c', bands=range(1, 8)):
    w = s.weight / s.weight.sum()
    cells = ' '.join(f'{w[s[col] == b].sum():7.2f}' for b in bands)
    rest = w[s[col] >= max(bands) + 1].sum()
    return f'{lab:14s} n={len(s):3d} {cells} {rest:7.3f}'


print('\n=== 1. Worst-incident cost by attack frequency (not breached) ===')
print(f"{'':20s}" + ' '.join(f'{b:>7s}' for b in BL[:7]) + '  higher')
for f, lab in FQ.items():
    print(dist_row(lab, nb[nb.freq == f]))
print(dist_row('all', nb))
print('by size:')
for z, lab in [(1, 'Micro'), (2, 'Small'), (3, 'Medium'), (4, 'Large')]:
    print(dist_row(lab, nb[nb.sizeb == z]))

print('\n=== 2. Parts of the worst incident cost (not breached, worst cost > 0) ===')
pos = nb[nb.c >= 2]
for col, lab in [('damagestaffx_bands', 'staff time'), ('damagedirsx_bands', 'external during'),
                 ('damagedirlx_bands', 'external after'), ('damageindx_bands', 'damage/disruption')]:
    v = num(col).loc[pos.index]
    ok = v.isin(range(1, 14))
    s = pos[ok]
    vv = v[ok]
    w = s.weight / s.weight.sum()
    print(f'{lab:18s} answered {ok.sum():3d}/{len(pos)}   none {w[vv == 1].sum():.2f}  <£100 {w[vv == 2].sum():.2f}  '
          f'£100-500 {w[vv == 3].sum():.2f}  £500+ {w[vv >= 4].sum():.2f}')

print('\n=== 3. Yearly totals vs the worst incident (not breached) ===')
for col, lab in [('crimecost_bands', 'all cyber crimes incl. fraud'), ('notfraudcost_bands', 'all cyber crimes excl. fraud')]:
    v = num(col)
    ok = v.loc[nb.index].isin(range(1, 14))
    s = nb[ok].copy()
    s['tot'] = v.loc[s.index].astype(int)
    print(f'-- {lab}: answered by {ok.sum()} of {len(nb)}')
    for flab, m in [('all', s.index == s.index), ('once', s.freq == 1), ('<monthly-monthly', s.freq.isin([2, 3])),
                    ('weekly+', s.freq >= 4)]:
        t = s[m]
        if len(t) == 0:
            continue
        w = t.weight / t.weight.sum()
        print(f'   {flab:17s} n={len(t):3d}  total higher {w[t.tot > t.c].sum():.2f}  same {w[t.tot == t.c].sum():.2f}  '
              f'lower {w[t.tot < t.c].sum():.2f}   total none {w[t.tot == 1].sum():.2f}, worst none {w[t.c == 1].sum():.2f}')
    print('   who answers (share of not-breached firms answering), by frequency: ' +
          ', '.join(f'{FQ[f]} {ok[nb.freq == f].mean():.2f}' for f in FQ))

print('\n=== 4. Per-type yearly totals for not-breached firms ticking the type ===')
for col, tick, lab in [('ranscost_bands', 'type1', 'ransomware'), ('hackcost_bands', 'type8', 'hacking (outsider)'),
                       ('tkvrcost_bands', 'type16', 'takeover'), ('doscost_bands', 'type3', 'DoS'),
                       ('viruscost_bands', 'type2', 'malware')]:
    v = num(col)
    t = nb[nb[tick] == 1]
    ok = v.loc[t.index].isin(range(1, 14))
    s = t[ok]
    vv = v.loc[s.index].astype(int)
    if len(s) == 0:
        print(f'{lab:18s} ticking {len(t)}, answered 0')
        continue
    w = s.weight / s.weight.sum()
    print(f'{lab:18s} ticking {len(t):3d}, answered {len(s):3d}:  none {w[vv == 1].sum():.2f}  <£100 {w[vv == 2].sum():.2f}  '
          f'£100-1k {w[vv.isin([3, 4])].sum():.2f}  £1k+ {w[vv >= 5].sum():.2f}   total > worst {w[vv > s.c].sum():.2f}')
