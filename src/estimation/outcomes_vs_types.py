"""Do the outcome items (Q56A, what happened as a result of the year's breaches) line up with the four
'inside' types (ransomware, bank hacking, outsider access, staff access)? (2026-10-05)
Breached firms with a cost answer. Shares weighted; n unweighted.
1. Type x outcome: share with each outcome, firms ticking the type vs breached firms not ticking it.
2. For each of the four types: share with its 'matching' outcome, narrow (1-3 types) vs broad (4+).
3. Number of distinct outcomes, by narrow/broad x inside-type x costly.
Note: Q56A covers all breaches in the year, not only the worst incident.
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
OUT = {1: 'sys corrupted', 2: 'personal data', 3: 'files lost', 4: 'temp no access', 5: 'assets/IP',
       6: 'money stolen', 7: 'services down', 8: '3rd-party loss', 11: 'paid attackers', 12: 'devices damaged',
       13: 'accounts misused'}
for j, n in OUT.items():
    df[n] = (num(f'outcome{j}') == 1).astype(int)
O = list(OUT.values())
df['n_out'] = df[O].sum(axis=1)
INSIDE = ['ransom', 'bankhack', 'outsider', 'staff']
df['inside'] = (df[INSIDE].sum(axis=1) > 0).astype(int)
df['costly'] = (df['damage_bands'] >= 6).astype(int)
b = df[df.breach == 1].copy()


def ws(s, col):
    return np.average(s[col], weights=s.weight) if len(s) else np.nan


print(f'breached firms with cost answer: {len(b)}')
print('\n=== 1. Share with each outcome: firms ticking the type vs breached firms not ticking it ===')
print(f"{'':16s}" + ''.join(f'{t:>18s}' for t in INSIDE))
for o in O:
    cells = []
    for t in INSIDE:
        cells.append(f'{ws(b[b[t] == 1], o):.2f} vs {ws(b[b[t] == 0], o):.2f}')
    print(f'{o:16s}' + ''.join(f'{c:>18s}' for c in cells))
print('n ticking: ' + ', '.join(f'{t} {int(b[t].sum())}' for t in INSIDE))

MATCH = {'ransom': ['sys corrupted', 'files lost', 'temp no access', 'paid attackers'],
         'bankhack': ['money stolen', 'accounts misused'],
         'outsider': ['personal data', 'files lost', 'assets/IP', 'sys corrupted'],
         'staff': ['personal data', 'files lost', 'assets/IP', 'sys corrupted']}
print('\n=== 2. Share with a matching outcome, by narrow (1-3) / broad (4+) and costly / not ===')
for t, outs in MATCH.items():
    b['_m'] = (b[outs].sum(axis=1) > 0).astype(int)
    row = []
    for lab, m in [('narrow', b.k <= 3), ('broad', b.k >= 4)]:
        for cl in [0, 1]:
            s = b[(b[t] == 1) & m & (b.costly == cl)]
            row.append(f"{lab} {'costly' if cl else 'cheap'}: {ws(s, '_m'):.2f} ({len(s)})" if len(s) else
                       f"{lab} {'costly' if cl else 'cheap'}:  -  (0)")
    print(f'{t:9s} [{", ".join(outs)}]\n          ' + '   '.join(row))

print('\n=== 3. Mean number of distinct outcomes (breached firms) ===')
for lab, m in [('narrow 1-3', b.k <= 3), ('broad 4+', b.k >= 4)]:
    for ins in [0, 1]:
        for cl in [0, 1]:
            s = b[m & (b.inside == ins) & (b.costly == cl)]
            if len(s):
                print(f"{lab:10s} {'inside type' if ins else 'no inside type':15s} {'costly' if cl else 'cheap':7s} "
                      f"n={len(s):3d}  mean outcomes {ws(s, 'n_out'):.2f}  money stolen {ws(s, 'money stolen'):.2f}  "
                      f"sys corrupted {ws(s, 'sys corrupted'):.2f}")
