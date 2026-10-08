"""Evidence on how many breaches a firm has in a year (2026-10-06). Plain tables.
1. Per-type success counts (phishing engaged / successful takeovers / successful DoS / successful malware /
   ransom demands): share 0 / 1 / 2 / 3+ among firms ticking the type, breached vs not breached.
   'Success' is a looser concept than the breach marker, so 2+ successes is an upper bound on 2+ breaches.
2. Yearly total vs worst incident: crimecost_bands (total cost of all cyber crimes incl. fraud, same 13-band scale
   as damage_bands) compared with the worst-incident band. Higher total band = a second costly incident
   (or a broader cost concept); same band = consistent with one incident; lower = concept mismatch / recall.
Attacked businesses with a cost answer. Shares weighted; n unweighted.
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
df['costly'] = (df['damage_bands'] >= 6).astype(int)
TYPES = ['type6', 'type5', 'type1', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
df['k'] = (df[TYPES] == 1).sum(axis=1)

print('=== 1. Per-type success counts among firms ticking the type ===')
ITEMS = [('phishing engaged', 'type6', 'phisheng'), ('successful takeover', 'type16', 'tkvrsuc'),
         ('successful DoS', 'type3', 'dossoft'), ('successful malware', 'type2', 'virussoft'),
         ('ransom demanded', 'type1', 'ranssoft')]
for lab, tick, col in ITEMS:
    v = num(col)
    print(f'-- {lab} ({col})')
    for blab, bm in [('not breached', df.breach == 0), ('breached', df.breach == 1),
                     ('breached, costly', (df.breach == 1) & (df.costly == 1))]:
        s = df[(df[tick] == 1) & bm & (v >= 0)]
        if len(s) == 0:
            continue
        x, w = v[s.index], s.weight / s.weight.sum()
        pos = x >= 1
        two_given_one = w[x >= 2].sum() / w[pos].sum() if w[pos].sum() > 0 else np.nan
        print(f'   {blab:17s} n={len(s):3d}  0: {w[x == 0].sum():.2f}  1: {w[x == 1].sum():.2f}  '
              f'2: {w[x == 2].sum():.2f}  3+: {w[x >= 3].sum():.2f}   share 2+ among those with 1+: {two_given_one:.2f}')

print('\n=== 2. Yearly total (crimecost_bands) vs worst incident (damage_bands) ===')
cc = num('crimecost_bands')
ok = cc.isin(range(1, 14))
d = df[ok].copy()
d['tot'] = cc[ok].astype(int)
d['rel'] = np.select([d.tot > d.damage_bands, d.tot == d.damage_bands], ['total higher', 'same band'], 'total lower')
print(f'firms answering both: {len(d)} (of {len(df)})')
for lab, m in [('all', d.index == d.index), ('breached', d.breach == 1), ('not breached', d.breach == 0),
               ('breached, narrow (1-3 types)', (d.breach == 1) & (d.k <= 3)),
               ('breached, broad (4+)', (d.breach == 1) & (d.k >= 4)),
               ('worst >= £5k', d.damage_bands >= 6)]:
    s = d[m]
    if len(s) == 0:
        continue
    w = s.weight / s.weight.sum()
    print(f'{lab:30s} n={len(s):3d}  total higher {w[s.rel == "total higher"].sum():.2f}  '
          f'same {w[s.rel == "same band"].sum():.2f}  lower {w[s.rel == "total lower"].sum():.2f}')
print('\nwho answered crimecost (share answering, by group):')
for lab, m in [('breached', df.breach == 1), ('not breached', df.breach == 0), ('worst >= £5k', df.costly == 1)]:
    print(f'   {lab:14s} {ok[m].mean():.2f} of {m.sum()}')
BL = {1: 'none', 2: '<100', 3: '100-500', 4: '500-1k', 5: '1k-5k', 6: '5k-10k', 7: '10k-20k', 8: '20k-50k',
      9: '50k-100k', 10: '100k-500k', 11: '500k-1m', 12: '1m-5m', 13: '5m+'}
print('\nfirms where the total is HIGHER than the worst (worst -> total):')
for _, r in d[d.rel == 'total higher'].sort_values('damage_bands').iterrows():
    print(f"   worst {BL[int(r.damage_bands)]:9s} -> total {BL[int(r.tot)]:9s}  breached {int(r.breach)}  types {int(r.k)}  "
          f"size {int(r.sizeb)}")
