"""Per-type count data for the 'inside' types: coverage and a first look (2026-10-05).
Idea (user): if costly broad profiles come from one intrusion spreading, the extra types in those firms should
show LOW counts (the one spread event) compared with a baseline (the same type in narrow firms, or in cheap
broad firms), where the tick reflects regular separate attacks.
Count variables (codebook wording):
  ransomware: Cybercrime_ranssum (total ransomware crimes), ranssoft (attacks with a ransom demanded)
  malware:    Cybercrime_virussum, virussoft (successful)
  hacking / unauthorised access: hackcount (attempts, EXCLUDING instances that led to fraud/ransomware),
              Cybercrime_hacksum
  takeover incl. online bank accounts: tkvrcount (attempts, EXCLUDING instances that led to fraud/ransomware),
              tkvrsuc (successful)
  DoS:        doscount (no exclusion), Cybercrime_dossum
Breached firms with a cost answer; groups: narrow (1-3 types) vs broad (4+), cheap vs costly (>= £5k).
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
df['k'] = df[list(TYPES.values())].sum(axis=1)
df['costly'] = (df['damage_bands'] >= 6).astype(int)
df['grp'] = np.select([(df.k <= 3) & (df.costly == 0), (df.k <= 3) & (df.costly == 1),
                       (df.k >= 4) & (df.costly == 0), (df.k >= 4) & (df.costly == 1)],
                      ['narrow cheap', 'narrow costly', 'broad cheap', 'broad costly'], 'x')
b = df[df.breach == 1]
G = ['narrow cheap', 'narrow costly', 'broad cheap', 'broad costly']

PAIRS = [('ransom', 'Cybercrime_ranssum'), ('ransom', 'ranssoft'), ('malware', 'Cybercrime_virussum'),
         ('bankhack', 'tkvrcount'), ('takeover', 'tkvrcount'), ('outsider', 'hackcount'), ('staff', 'hackcount'),
         ('outsider', 'Cybercrime_hacksum'), ('DoS', 'doscount')]
print('For firms ticking the type: answered n / ticked n; median and share with count <= 1 among answers')
for t, col in PAIRS:
    v = pd.to_numeric(b[col], errors='coerce')
    ok = v >= 0
    print(f'\n{t:9s} {col}')
    for g in G:
        s = (b[t] == 1) & (b.grp == g)
        a = s & ok
        if s.sum() == 0:
            continue
        vals = v[a]
        med = np.median(vals) if len(vals) else np.nan
        low = (vals <= 1).mean() if len(vals) else np.nan
        zero = (vals == 0).mean() if len(vals) else np.nan
        print(f'   {g:14s} answered {a.sum():3d}/{s.sum():3d}   median {med:6.1f}   share 0: {zero:.2f}   '
              f'share <= 1: {low:.2f}')
