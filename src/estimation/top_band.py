"""The top of the cost range: who the most costly firms are (2026-10-07). Plain look first.
Attacked firms with a cost answer.
1. Share of firms at £20k+, £50k+, £100k+ (band 10 = £100k-500k; bands 11-13 above £500k exist, none chosen),
   by size; n unweighted, shares weighted.
2. How much of the weighted mean worst cost (band midpoints) comes from the £100k+ firms, by size.
3. Each firm at £50k+: size, weight, types, breach / spread marker, attack frequency, worst-incident type,
   yearly total band (crimecost_bands).
"""
import numpy as np
import pandas as pd

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
df['k'] = df[list(TYPES.values())].sum(axis=1)
df['n_inside'] = df[INSIDE].sum(axis=1)
df['c'] = df.damage_bands.astype(int)
df['spread'] = ((df.breach == 1) & (df.k >= 4) & (df.n_inside >= 1) & (df.c >= 6)).astype(int)
MID = np.array([0, 50, 300, 750, 3000, 7500, 15000, 35000, 75000, 300000, 750000, 3e6, 7.5e6])
df['mid'] = MID[df.c - 1]
SIZES = [('Micro', df.sizeb == 1), ('Small', df.sizeb == 2), ('Medium', df.sizeb == 3), ('Large', df.sizeb == 4),
         ('all', df.sizeb > 0)]

print('=== 1. Share of firms at the top (weighted; counts in brackets) ===')
for lab, m in SIZES:
    s = df[m]
    w = s.weight / s.weight.sum()
    cells = [f'{t}: {w[s.c >= b].sum():.3f} ({int((s.c >= b).sum())})' for t, b in
             [('£20k+', 8), ('£50k+', 9), ('£100k+', 10)]]
    print(f'{lab:7s} n={len(s):4d}  ' + '   '.join(cells))

print('\n=== 2. Share of the weighted mean worst cost from £100k+ firms (band midpoints) ===')
for lab, m in SIZES:
    s = df[m]
    tot = (s.mid * s.weight).sum()
    top = (s.mid * s.weight)[s.c >= 10].sum()
    print(f'{lab:7s} mean £{tot / s.weight.sum():8,.0f}   from £100k+ firms: {top / tot:.2f}   '
          f'mean without them £{(tot - top) / s.weight.sum():7,.0f}')

print('\n=== 3. Firms at £50k+ ===')
BL = {9: '50k-100k', 10: '100k-500k'}
FQ = {1: 'once', 2: '<monthly', 3: 'monthly', 4: 'weekly', 5: 'daily', 6: 'several/day'}
DIS = {6: 'phishing', 5: 'imperson', 1: 'ransom', 2: 'malware', 3: 'DoS', 4: 'bankhack', 11: 'takeover',
       9: 'outsider', 7: 'staff', 10: 'eavesdrop', 12: 'other'}
cc = num('crimecost_bands')
CB = {1: 'none', 2: '<100', 3: '100-500', 4: '500-1k', 5: '1k-5k', 6: '5k-10k', 7: '10k-20k', 8: '20k-50k',
      9: '50k-100k', 10: '100k-500k', 11: '500k-1m', 12: '1m-5m', 13: '5m+'}
for _, r in df[df.c >= 9].sort_values(['c', 'sizeb']).iterrows():
    ty = ','.join(t for t in TYPES.values() if r[t] == 1)
    tot = CB.get(int(cc[_]), '-') if cc[_] == cc[_] else '-'
    print(f"{BL[int(r.c)]:10s} size {int(r.sizeb)}  weight {r.weight:6.2f}  breach {int(r.breach)}  spread {int(r.spread)}  "
          f"freq {FQ[int(r.freq)]:11s} worst {DIS.get(int(r.disrupta), str(r.disrupta)):9s} total {tot:9s} types {ty}")

# --- 2026-10-07: which firms do the spread-cost fits miss at the top? Among the 53 broad breached firms with an
# inside type (the set used in spread_cost_shapes.py): cost by size, and the firms at £50k+ with their weights.
bi = df[(df.breach == 1) & (df.k >= 4) & (df.n_inside >= 1)]
print('\n=== 4. Broad breached firms with an inside type, by size ===')
for lab, m in SIZES:
    s = bi[m.loc[bi.index]]
    if len(s) == 0:
        continue
    w = s.weight / s.weight.sum()
    print(f'{lab:7s} n={len(s):3d}  weight share {s.weight.sum() / bi.weight.sum():.2f}  '
          f'>=£5k {w[s.c >= 6].sum():.2f}  >=£20k {w[s.c >= 8].sum():.2f}  >=£100k {w[s.c >= 10].sum():.2f}  '
          f'(firms >=£20k: {int((s.c >= 8).sum())})')
print('firms at £20k+ in this set (band, size, weight, share of the set\'s weight):')
for _, r in bi[bi.c >= 8].sort_values(['c', 'sizeb']).iterrows():
    print(f'   band {int(r.c):2d}  size {int(r.sizeb)}  weight {r.weight:5.2f}  ({r.weight / bi.weight.sum():.3f})')
