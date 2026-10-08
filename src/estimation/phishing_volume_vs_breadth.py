"""Fresh check (2026-10-05): do firms with more phishing also have other attack types present?
Plain cross-tabs, no model. Phishing-ticking attacked businesses. Shares weighted; n unweighted.
Phishing volume measures:
  - Cybercrime_phishsum: total number of phishing crimes (exact count; asked of a subset)
  - phishcon_bands: number of phishing emails containing personal details ("targeted")
  - freq: how often attacked overall (all types, so only a rough phishing-volume proxy)
'Serious' = any type other than phishing and impersonation.
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & (df['type6'] == 1)].copy()
SERIOUS = ['type1', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
df['nser'] = (df[SERIOUS] == 1).sum(axis=1)
df['imp'] = (df['type5'] == 1).astype(int)
df['anyser'] = (df['nser'] >= 1).astype(int)
df['grp_size'] = np.where(df['sizeb'] == 1, 'Micro', 'Small+')
df['n'] = df['Cybercrime_phishsum'].where(df['Cybercrime_phishsum'] >= 0)
print(f'phishing-ticking attacked businesses: {len(df)}; exact phishing count answered: {df["n"].notna().sum()}')


def wavg(x, w):
    return np.average(x, weights=w) if len(x) else np.nan


def show(lab, s):
    print(f'{lab:14s} {len(s):4d} {wavg(s.anyser, s.weight):9.2f} {wavg(s.nser, s.weight):9.2f} '
          f'{wavg(s.imp, s.weight):9.2f}')


HEAD = f"{'':14s} {'n':>4s} {'any serious':>9s} {'# serious':>9s} {'imperson':>9s}"

print('\n=== A. Other types present, by exact phishing count ===')
NB = [(1, 1, '1'), (2, 5, '2-5'), (6, 20, '6-20'), (21, 100, '21-100'), (101, 1e9, '>100')]
for g in ['Micro', 'Small+', None]:
    sub = df[df['n'].notna()] if g is None else df[df['n'].notna() & (df.grp_size == g)]
    print(f'-- {g or "all sizes"}\n' + HEAD)
    for lo, hi, lab in NB:
        show(lab, sub[(sub.n >= lo) & (sub.n <= hi)])

print('\n=== B. Other types present, by number of targeted phishing emails ===')
TB = [([1], 'none'), ([2], '1'), ([3, 4], '2-5'), ([5, 6], '6-20'), ([7, 8], '21-100'), ([9], '>100')]
for g in ['Micro', 'Small+']:
    sub = df[df.grp_size == g]
    print(f'-- {g}\n' + HEAD)
    for codes, lab in TB:
        show(lab, sub[sub.phishcon_bands.isin(codes)])

print('\n=== C. Other types present, by overall attack frequency ===')
FREQ = {1: 'once', 2: '<monthly', 3: 'monthly', 4: 'weekly', 5: 'daily', 6: 'several/day'}
for g in ['Micro', 'Small+']:
    sub = df[df.grp_size == g]
    print(f'-- {g}\n' + HEAD)
    for f, lab in FREQ.items():
        show(lab, sub[sub.freq == f])

print('\n=== D. Reverse: exact phishing count by what else is present (weighted median / 75th pct) ===')


def wq(x, w, q):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, c, x)


df['other'] = np.select([df.nser >= 2, df.nser == 1, df.imp == 1], ['2+ serious', '1 serious', 'imperson only'],
                        'phishing only')
for g in ['Micro', 'Small+']:
    print(f'-- {g}')
    for o in ['phishing only', 'imperson only', '1 serious', '2+ serious']:
        s = df[(df.grp_size == g) & (df.other == o) & df.n.notna()]
        if len(s):
            print(f'{o:14s} n={len(s):3d}  median {wq(s.n, s.weight, .5):6.1f}  75th {wq(s.n, s.weight, .75):6.1f}')

print('\n=== E. Who answered the exact count? (coverage by what else is present) ===')
for o in ['phishing only', 'imperson only', '1 serious', '2+ serious']:
    s = df[df.other == o]
    print(f'{o:14s} answered {s.n.notna().sum():3d} of {len(s):3d}')
