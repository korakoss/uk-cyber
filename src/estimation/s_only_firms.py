"""Look at the firms whose only ticks are 'other' types (S-only), firm by firm (2026-10-05).
Question: is something odd about them, given their very high breach rate (61%)?
Prints: each S-only firm (size, weight, types ticked, attack frequency, worst cost, worst type, breach items);
then a summary comparing what sets the breach marker in S-only vs broader firms.
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
TYPES = {'type1': 'ransom', 'type2': 'malware', 'type3': 'DoS', 'type4': 'bankhack', 'type16': 'takeover',
         'type8': 'outsider', 'type7': 'staff', 'type15': 'eavesdrop', 'type9': 'other'}
OUT = {1: 'systems corrupted', 2: 'personal data', 3: 'files lost', 4: 'temp access loss', 5: 'assets/IP',
       6: 'money stolen', 7: 'services down', 8: '3rd-party loss', 11: 'paid attackers', 12: 'devices damaged',
       13: 'accounts misused'}
df['nS'] = (df[list(TYPES)] == 1).sum(axis=1)
df['P'] = df['type6'] == 1
df['I'] = df['type5'] == 1
df['set'] = np.select([~df.P & ~df.I & (df.nS > 0), (df.P | df.I) & (df.nS > 0)], ['S only', 'S + P/I'], 'no S')
df['b_out'] = num('outcome_any') == 1
df['b_rest'] = num('restore').isin([3, 4, 5, 6])
df['b_imp'] = (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)
df['breach'] = df.b_out | df.b_rest | df.b_imp
FREQ = {1: 'once', 2: '<monthly', 3: 'monthly', 4: 'weekly', 5: 'daily', 6: 'sev/day'}
BL = {1: 'none', 2: '<100', 3: '100-500', 4: '500-1k', 5: '1k-5k', 6: '5k-10k', 7: '10k-20k', 8: '20k-50k',
      9: '50k-100k', 10: '100k-500k'}
SIZE = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}

s = df[df.set == 'S only'].sort_values(['breach', 'damage_bands'])
print(f'S-only firms: {len(s)}\n')
for _, r in s.iterrows():
    types = ','.join(v for k, v in TYPES.items() if r[k] == 1)
    outs = ','.join(v for k, v in OUT.items() if num(f'outcome{k}')[_] == 1)
    soft = ','.join(x for x, f in [('restore>=1d', r.b_rest), ('impact', r.b_imp)] if f)
    print(f"{SIZE.get(r.sizeb, '?'):6s} w={r.weight:5.2f} {types:22s} {FREQ[r.freq]:9s} "
          f"cost {BL.get(r.damage_bands, 'na'):9s} breach={int(r.breach)} outcomes: {outs or '-'}; {soft or ''}")

print('\n=== Summary (weighted shares) ===')
print(f"{'':10s} {'n':>4s} {'Small+':>7s} {'once':>6s} {'breached':>9s} {'by outcome':>11s} {'soft only':>10s} "
      f"{'nS mean':>8s}")
for lab in ['S only', 'S + P/I']:
    d = df[df.set == lab]
    w = d.weight / d.weight.sum()
    print(f'{lab:10s} {len(d):4d} {w[d.sizeb >= 2].sum():7.2f} {w[d.freq == 1].sum():6.2f} {w[d.breach].sum():9.2f} '
          f'{w[d.b_out].sum():11.2f} {w[d.breach & ~d.b_out].sum():10.2f} {np.average(d.nS, weights=d.weight):8.2f}')
print('\nPer S type: breached share among firms ticking it, S-only vs S + P/I (n in brackets)')
for k, v in TYPES.items():
    cells = []
    for lab in ['S only', 'S + P/I']:
        d = df[(df.set == lab) & (df[k] == 1)]
        cells.append(f'{np.average(d.breach, weights=d.weight):.2f} ({len(d):3d})' if len(d) else '   -     ')
    print(f'{v:10s} ' + '   '.join(cells))
