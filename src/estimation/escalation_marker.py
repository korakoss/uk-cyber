"""Marking spread breaches (2026-10-07). User's suggestion: mark them mainly by the cost uplift, with breadth
(4+ types and an inside type) as a precondition. Then check whether other spread indicators line up with it.
Groups of breached firms with a cost answer:
  broad costly  = 4+ types, inside type, worst >= £5k   (candidate 'spread')
  broad cheap   = 4+ types, inside type, worst <  £5k
  narrow inside = 1-3 types, inside type (reference: ordinary breaches involving an inside type)
  narrow other  = 1-3 types, no inside type
Indicators (none uses cost):
  - number of different outcomes (Q56A), share with 3+
  - restore took a day or more; staff stopped / revenue loss / recovery costs (impact1/2/4)
  - linkage items: impersonation involved access/takeover; impersonation used info from the breach; any fraud
  - inside-type counts: share of ticked inside types with a count <= 1 (one event, not repeated attacks)
Then: a mixture reading. If broad+inside breaches are a mix of ordinary ones (cost like 'narrow inside') and spread
ones, what share must be spread to give the observed costly share?
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
df['costly'] = (df.damage_bands >= 6).astype(int)
OUT = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]
df['n_out'] = sum((num(f'outcome{j}') == 1).astype(int) for j in OUT)
df['out3'] = (df.n_out >= 3).astype(int)
df['restore_day'] = num('restore').isin([3, 4, 5, 6]).astype(int)
df['n_impact'] = sum((num(f'impact{j}') == 1).astype(int) for j in [1, 2, 4])
ih, it, f4 = num('impersonationhack'), num('impersonationtkvr'), num('fraud4_comb2')
df['imp_access'] = np.where(ih.isin([1, 2, 3]) | it.isin([1, 2, 3]), (ih.isin([1, 2]) | it.isin([1, 2])).astype(float), np.nan)
df['imp_breachinfo'] = np.where(f4.isin([0, 1]), (f4 == 1).astype(float), np.nan)
fr_ok = num('fraud1') >= 0
df['fraud'] = np.where(fr_ok, ((num('fraud1') > 0) | (num('fraud2') > 0) | (num('fraud3') > 0)).astype(float), np.nan)
CNT = {'ransom': 'Cybercrime_ranssum', 'bankhack': 'tkvrcount', 'outsider': 'Cybercrime_hacksum',
       'staff': 'Cybercrime_hacksum'}
low, ans = 0, 0
for t, col in CNT.items():
    v = num(col)
    a = (df[t] == 1) & (v >= 0)
    low = low + (a & (v <= 1)).astype(int)
    ans = ans + a.astype(int)
df['low_count'] = np.where(ans > 0, low / ans.replace(0, 1), np.nan)

b = df[df.breach == 1]
G = [('broad costly', b[(b.k >= 4) & (b.n_inside >= 1) & (b.costly == 1)]),
     ('broad cheap', b[(b.k >= 4) & (b.n_inside >= 1) & (b.costly == 0)]),
     ('narrow inside', b[(b.k <= 3) & (b.n_inside >= 1)]),
     ('narrow other', b[(b.k <= 3) & (b.n_inside == 0)])]


def ws(s, col):
    s = s[s[col].notna()]
    return (np.average(s[col], weights=s.weight), len(s)) if len(s) else (np.nan, 0)


print('=== 1. Spread indicators by group (weighted share or mean; n answering) ===')
print(f"{'':34s}" + ''.join(f'{g:>18s}' for g, _ in G))
print(f"{'n firms':34s}" + ''.join(f'{len(s):18d}' for _, s in G))
for col, lab in [('n_out', 'mean number of outcomes'), ('out3', '3+ outcomes'),
                 ('restore_day', 'restore took a day or more'), ('n_impact', 'mean impact items (of 3)'),
                 ('imp_access', 'impersonation involved access'), ('imp_breachinfo', 'impers. used breach info'),
                 ('fraud', 'any fraud'), ('low_count', 'inside ticks with count <= 1'),
                 ('n_inside', 'mean number of inside types')]:
    cells = []
    for _, s in G:
        v, n = ws(s, col)
        cells.append(f'{v:.2f} ({n})' if n else '-')
    print(f'{lab:34s}' + ''.join(f'{c:>18s}' for c in cells))

print('\n=== 2. Mixture reading ===')
BL = ['none', '<100', '100-500', '500-1k', '1k-5k', '5k-10k', '10k-20k', '20k-50k', '50k-100k', '100k+']
def dist(s):
    c = s.damage_bands.clip(upper=10)
    return np.array([s.weight[c == i].sum() for i in range(1, 11)]) / s.weight.sum()
bi = b[(b.k >= 4) & (b.n_inside >= 1)]
for lab, s in [('broad+inside (all)', bi), ('broad cheap', G[1][1]), ('narrow inside', G[2][1]), ('narrow other', G[3][1])]:
    print(f'{lab:20s} ' + ' '.join(f'{x:5.2f}' for x in dist(s)))
print(f"{'':20s} " + ' '.join(f'{x:>5s}' for x in BL))
p_b, p_o = np.average(bi.costly, weights=bi.weight), np.average(G[2][1].costly, weights=G[2][1].weight)
for p_s in [1.0, 0.9, 0.8]:
    e = (p_b - p_o) / (p_s - p_o)
    print(f'if spread breaches are costly {p_s:.0%} of the time: spread share of broad+inside = {e:.2f}')
print('below £5k, compare broad cheap with narrow inside: if alike, broad cheap firms look like ordinary breaches.')
