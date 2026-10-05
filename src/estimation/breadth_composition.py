"""Is the breadth uplift in breached firms spread over many type combinations, or carried by something specific?
(2026-10-05) Plain cross-tabs, breached firms with a cost answer. 'Costly' = worst incident >= £5k.
1. Which types are present in breached broad (4+ types) firms, costly vs not.
2. For each type T: costly share among breached firms, by breadth counted WITHOUT T, with T present vs absent.
   If one type carries the uplift, the 'absent' row stays flat with breadth.
3. The type combinations of costly broad firms, listed.
4. Size and attack frequency in costly vs not-costly broad breached firms.
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
N = list(TYPES.values())
df['k'] = df[N].sum(axis=1)
df['costly'] = (df['damage_bands'] >= 6).astype(int)
b = df[df.breach == 1].copy()
print(f'breached firms with cost answer: {len(b)}; with 4+ types: {(b.k >= 4).sum()}')


def ws(s, col='costly'):
    return np.average(s[col], weights=s.weight) if len(s) else np.nan


print('\n=== 1. Breached firms with 4+ types: share ticking each type, costly vs not (unweighted counts) ===')
bb = b[b.k >= 4]
c1, c0 = bb[bb.costly == 1], bb[bb.costly == 0]
print(f'costly n={len(c1)}, not costly n={len(c0)}')
for n in N:
    print(f'{n:10s} costly {c1[n].mean():.2f} ({c1[n].sum():2d})   not costly {c0[n].mean():.2f} ({c0[n].sum():2d})')

print('\n=== 2. Costly share by breadth WITHOUT type T, T present vs absent (breached firms; n in brackets) ===')
for n in N:
    if b[n].sum() < 15:
        continue
    kk = b.k - b[n]
    cells = []
    for pres in [0, 1]:
        row = []
        for lo, hi, lab in [(0, 1, '0-1'), (2, 2, '2'), (3, 99, '3+')]:
            s = b[(b[n] == pres) & (kk >= lo) & (kk <= hi)]
            row.append(f'{lab}: {ws(s):.2f} ({len(s):3d})' if len(s) else f'{lab}:  -  (  0)')
        cells.append('  '.join(row))
    print(f'{n:10s} without: {cells[0]}\n{"":10s} with:    {cells[1]}')

print('\n=== 3. Type combinations of COSTLY breached firms with 4+ types ===')
combo = c1[N].apply(lambda r: '+'.join(x for x in N if r[x] == 1), axis=1)
for cmb, cnt in combo.value_counts().items():
    print(f'{cnt:2d}  {cmb}')
print('\n--- and of NOT costly breached firms with 4+ types ---')
combo0 = c0[N].apply(lambda r: '+'.join(x for x in N if r[x] == 1), axis=1)
for cmb, cnt in combo0.value_counts().items():
    print(f'{cnt:2d}  {cmb}')

print('\n=== 4. Size and frequency, breached firms with 4+ types, costly vs not (weighted shares) ===')
for lab, s in [('costly', c1), ('not costly', c0)]:
    w = s.weight / s.weight.sum()
    print(f'{lab:11s} Small+ {w[s.sizeb >= 2].sum():.2f}  Medium/Large {w[s.sizeb >= 3].sum():.2f}  '
          f'attacked once {w[s.freq == 1].sum():.2f}  weekly+ {w[s.freq >= 4].sum():.2f}')

print('\n=== 5. Costly share by breadth when SEVERAL types are removed (breached firms; breadth = all ticked types) ===')
for lab, drop in [('no ransomware, no bank hacking', ['ransom', 'bankhack']),
                  ('none of ransomware / bank hacking / outsider / staff access', ['ransom', 'bankhack', 'outsider', 'staff']),
                  ('at least one of ransomware / bank hacking / outsider / staff', None)]:
    if drop is not None:
        s = b[(b[drop].sum(axis=1) == 0)]
    else:
        s = b[(b[['ransom', 'bankhack', 'outsider', 'staff']].sum(axis=1) > 0)]
    print(f'-- {lab}')
    for lo, hi, kl in [(1, 1, '1'), (2, 2, '2'), (3, 3, '3'), (4, 99, '4+')]:
        t = s[(s.k >= lo) & (s.k <= hi)]
        if len(t):
            print(f'   {kl:3s} types n={len(t):3d}  costly {ws(t):.2f}  >= £500 {np.average(t.damage_bands >= 4, weights=t.weight):.2f}')
