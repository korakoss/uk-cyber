"""Fresh look (2026-10-05): worst-incident cost vs how many attack types a firm ticked.
Attacked businesses with a cost answer. Shares weighted; n unweighted.
'Serious' = any type other than phishing and impersonation.
Also: does the extra cost of broad firms come with more attacks (exposure) or not?
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
TYPES = ['type6', 'type5', 'type1', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
SERIOUS = ['type1', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
df['ntypes'] = (df[TYPES] == 1).sum(axis=1)
df['nser'] = (df[SERIOUS] == 1).sum(axis=1)
df['c'] = df['damage_bands']
df['grp_size'] = np.where(df['sizeb'] == 1, 'Micro', 'Small+')
MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000}
df['cmid'] = df['c'].map(MID)
print(f'attacked businesses with a cost answer: {len(df)}')


def w(s):
    return s['weight'].sum()


def row(lab, s):
    if len(s) == 0:
        return
    tot = w(s)
    contrib = s['cmid'] * s['weight']
    print(f'{lab:16s} {len(s):4d} {w(s[s.c == 1]) / tot:6.2f} {w(s[s.c.isin([2, 3])]) / tot:6.2f} '
          f'{w(s[s.c.isin([4, 5])]) / tot:7.2f} {w(s[s.c >= 6]) / tot:6.2f} '
          f'{contrib.sum() / tot:9,.0f} {contrib.max() / contrib.sum() if contrib.sum() else 0:8.2f}')


HEAD = (f"{'':16s} {'n':>4s} {'none':>6s} {'<500':>6s} {'500-5k':>7s} {'5k+':>6s} {'mean £':>9s} "
        f"{'top firm':>8s}")

print('\n=== A. Worst-incident cost by number of types ticked ===')
print('(mean £ from band midpoints; "top firm" = share of that mean from the single biggest firm)')
for g in ['Micro', 'Small+', None]:
    sub = df if g is None else df[df.grp_size == g]
    print(f'-- {g or "all sizes"}\n' + HEAD)
    for k, lab in [(1, '1'), (2, '2'), (3, '3'), (4, '4+')]:
        row(lab, sub[sub.ntypes == k] if k < 4 else sub[sub.ntypes >= 4])

print('\n=== B. By number of SERIOUS types ticked (phishing/impersonation ignored) ===')
for g in ['Micro', 'Small+']:
    sub = df[df.grp_size == g]
    print(f'-- {g}\n' + HEAD)
    for k, lab in [(0, '0'), (1, '1'), (2, '2'), (3, '3+')]:
        row(lab, sub[sub.nser == k] if k < 3 else sub[sub.nser >= 3])

print('\n=== C. Share of total weighted cost (band midpoints) by number of types ticked ===')
tot = (df['cmid'] * df['weight']).sum()
for k, lab in [(1, '1'), (2, '2'), (3, '3'), (4, '4+')]:
    s = df[df.ntypes == k] if k < 4 else df[df.ntypes >= 4]
    print(f'{lab:3s} types: {w(s) / w(df):5.2f} of attacked firms, {(s.cmid * s.weight).sum() / tot:5.2f} of cost')

print('\n=== D. Within breadth, does attack frequency matter? share with worst >= £5k ===')
FG = [([1], 'once'), ([2, 3], '<monthly-monthly'), ([4, 5, 6], 'weekly+')]
print(f"{'':10s}" + ''.join(f'{lab:>20s}' for _, lab in FG))
for k, lab in [(1, '1 type'), (2, '2 types'), (3, '3+ types')]:
    s = df[df.ntypes == k] if k < 3 else df[df.ntypes >= 3]
    cells = []
    for fs, _ in FG:
        t = s[s.freq.isin(fs)]
        cells.append(f'{w(t[t.c >= 6]) / w(t):10.2f} (n={len(t):3d})' if len(t) else f"{'-':>20s}")
    print(f'{lab:10s}' + ''.join(f'{c:>20s}' for c in cells))
