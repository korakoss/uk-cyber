"""Is the cost of spread breaches about the same whatever inside types are involved? (2026-10-07)
Old-notes claim: homogeneous, except bank hacking is costlier. Fresh check.
Spread breaches (working proxy): breached firms with 4+ types AND at least one inside type
(ransomware, bank hacking, outsider access, staff access). Narrow breached firms with an inside type shown for comparison.
For each inside type: costly share (worst >= £5k), share >= £20k, median band, with vs without the type.
Bootstrap 90% intervals (resampling firms, weighted) for the with-minus-without gap in costly share.
Also: by number of inside types, and by size. Small n throughout.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
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
df['costly'] = (df.damage_bands >= 6).astype(int)
df['big'] = (df.damage_bands >= 8).astype(int)
BL = {1: 'none', 2: '<100', 3: '100-500', 4: '500-1k', 5: '1k-5k', 6: '5k-10k', 7: '10k-20k', 8: '20k-50k',
      9: '50k-100k', 10: '100k-500k', 11: '500k-1m', 12: '1m-5m', 13: '5m+'}


def wmedian_band(s):
    o = s.sort_values('damage_bands')
    c = o.weight.cumsum() / o.weight.sum()
    return BL[int(o.damage_bands[c >= 0.5].iloc[0])]


def line(lab, s):
    if len(s) == 0:
        return f'{lab:24s} n=  0'
    w = s.weight
    return (f'{lab:24s} n={len(s):3d}  costly {np.average(s.costly, weights=w):.2f}  '
            f'>=£20k {np.average(s.big, weights=w):.2f}  median {wmedian_band(s)}')


def gap_ci(s, col, reps=2000):
    a, b = s[s[col] == 1], s[s[col] == 0]
    if len(a) < 3 or len(b) < 3:
        return ''
    g = []
    for _ in range(reps):
        ra, rb = a.sample(len(a), replace=True, random_state=rng), b.sample(len(b), replace=True, random_state=rng)
        g.append(np.average(ra.costly, weights=ra.weight) - np.average(rb.costly, weights=rb.weight))
    d = np.average(a.costly, weights=a.weight) - np.average(b.costly, weights=b.weight)
    return f'gap {d:+.2f} [{np.percentile(g, 5):+.2f}, {np.percentile(g, 95):+.2f}]'


b = df[df.breach == 1]
groups = [('SPREAD: breached, 4+ types, inside type', b[(b.k >= 4) & (b.n_inside >= 1)]),
          ('NARROW: breached, 1-3 types, inside type', b[(b.k <= 3) & (b.n_inside >= 1)])]
for title, s in groups:
    print(f'\n=== {title} ===')
    print(line('all', s))
    for t in INSIDE:
        print(line(f'  with {t}', s[s[t] == 1]))
        print(line(f'  without {t}', s[s[t] == 0]) + '   ' + gap_ci(s, t))
    print('by number of inside types:')
    for n in [1, 2, 3]:
        print(line(f'  {n}{"+" if n == 3 else ""} inside', s[s.n_inside == n] if n < 3 else s[s.n_inside >= 3]))
    print('by size:')
    for lab, m in [('  Micro', s.sizeb == 1), ('  Small+', s.sizeb >= 2)]:
        print(line(lab, s[m]))

print('\n=== SPREAD firms with exactly one inside type: which one ===')
s = groups[0][1]
for t in INSIDE:
    print(line(f'  only {t}', s[(s.n_inside == 1) & (s[t] == 1)]))

print('\n=== SPREAD firms: individual rows (inside types, total types, size, worst band) ===')
for _, r in s.sort_values('damage_bands').iterrows():
    ins = '+'.join(t for t in INSIDE if r[t] == 1)
    print(f'  {ins:30s} types {int(r.k):2d}  size {int(r.sizeb)}  worst {BL[int(r.damage_bands)]}')
