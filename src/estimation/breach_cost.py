"""Fresh look (2026-10-05) at breaches and cost. Plain tables, no model.
Breach marker (old 'W2', user decision 2026-10-02): outcome_any (Q56A: something happened as a result,
e.g. systems corrupted, data lost, money stolen) OR restore took >= 1 day (Q71) OR staff stopped from
working / revenue loss / recovery costs (Q57 impact1 / impact2 / impact4).
NOTE: Q56A and Q57 ask about ALL breaches in the last 12 months, not only the worst one.
Attacked businesses with a cost answer. Shares weighted; n unweighted.
Questions:
 1. Is a breach costlier than a non-breach, for every type?
 2. Among breached firms, is there a two-cluster ('big' vs 'ordinary') structure in cost,
    and how does the distribution change from narrow to broad firms: whole shift, or a new high bump?
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
df['c'] = df['damage_bands'].astype(int)
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
df['outcome_only'] = (num('outcome_any') == 1).astype(int)
SERIOUS = ['type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
df['gP'] = (df['type6'] == 1).astype(int)
df['gI'] = (df['type5'] == 1).astype(int)
df['gR'] = (df['type1'] == 1).astype(int)
df['gS'] = (df[SERIOUS] == 1).any(axis=1).astype(int)
df['ngroups'] = df[['gP', 'gI', 'gR', 'gS']].sum(axis=1)
df['size'] = np.where(df['sizeb'] == 1, 'Micro', 'Small+')


def w(s):
    return s['weight'].sum()


def row(lab, s):
    if len(s) == 0:
        print(f'{lab:28s}    0')
        return
    t = w(s)
    print(f'{lab:28s} {len(s):4d} {w(s[s.c == 1]) / t:6.2f} {w(s[s.c.isin([2, 3])]) / t:6.2f} '
          f'{w(s[s.c.isin([4, 5])]) / t:7.2f} {w(s[s.c >= 6]) / t:6.2f}')


HEAD = f"{'':28s} {'n':>4s} {'none':>6s} {'<500':>6s} {'500-5k':>7s} {'5k+':>6s}"

print(f'attacked firms with cost answer: {len(df)}; breached: {df.breach.sum()} '
      f'(weighted {w(df[df.breach == 1]) / w(df):.2f}); outcome item alone: {df.outcome_only.sum()}')

print('\n=== 1a. Breached vs not, by TYPE OF THE WORST INCIDENT (disrupta) ===')
DIS = {6: 'phishing', 5: 'impersonation', 1: 'ransomware', 2: 'malware', 3: 'DoS', 4: 'bank hacking',
       11: 'takeover', 7: 'staff access', 9: 'outsider access', 12: 'other'}
print(HEAD)
for code, name in DIS.items():
    s = df[df.disrupta == code]
    row(f'{name}: not breached', s[s.breach == 0])
    row(f'{name}: breached', s[s.breach == 1])

print('\n=== 1b. Breached vs not, firms hit by ONE type group only ===')
print(HEAD)
for g, name in [('gP', 'phishing only'), ('gI', 'impersonation only'), ('gR', 'ransomware only'),
                ('gS', 'other serious only')]:
    s = df[(df[g] == 1) & (df.ngroups == 1)]
    row(f'{name}: not breached', s[s.breach == 0])
    row(f'{name}: breached', s[s.breach == 1])

print('\n=== 1c. Same with the stricter marker (outcome item only), worst-incident type ===')
print(HEAD)
for code, name in DIS.items():
    s = df[df.disrupta == code]
    row(f'{name}: no outcome', s[s.outcome_only == 0])
    row(f'{name}: outcome', s[s.outcome_only == 1])

BL = {1: 'none', 2: '<100', 3: '100-500', 4: '500-1k', 5: '1k-5k', 6: '5k-10k', 7: '10k-20k', 8: '20k-50k',
      9: '50k-100k', 10: '100k-500k'}
print('\n=== 2a. Full band distribution of worst cost among BREACHED firms, by number of type groups ===')
print('(weighted shares; unweighted counts in brackets)')
groups = [('1 group', [1]), ('2 groups', [2]), ('3 groups', [3]), ('4 groups', [4])]
print(f"{'band':10s}" + ''.join(f'{g:>16s}' for g, _ in groups))
br = df[df.breach == 1]
for b, lab in BL.items():
    cells = []
    for _, ks in groups:
        s = br[br.ngroups.isin(ks)]
        cells.append(f'{w(s[s.c == b]) / w(s):.2f} ({(s.c == b).sum():3d})')
    print(f'{lab:10s}' + ''.join(f'{c:>16s}' for c in cells))
print(f"{'n':10s}" + ''.join(f'{len(br[br.ngroups.isin(ks)]):>16d}' for _, ks in groups))

print('\n=== 2b. Same for NOT breached firms (for comparison) ===')
nb = df[df.breach == 0]
print(f"{'band':10s}" + ''.join(f'{g:>16s}' for g, _ in groups))
for b, lab in BL.items():
    cells = []
    for _, ks in groups:
        s = nb[nb.ngroups.isin(ks)]
        cells.append(f'{w(s[s.c == b]) / w(s):.2f} ({(s.c == b).sum():3d})' if len(s) else '-')
    print(f'{lab:10s}' + ''.join(f'{c:>16s}' for c in cells))
print(f"{'n':10s}" + ''.join(f'{len(nb[nb.ngroups.isin(ks)]):>16d}' for _, ks in groups))

print('\n=== 2c. Breached firms: quantiles of worst cost band (weighted), narrow (1-2 groups) vs broad (3-4) ===')


def wq_band(s, q):
    o = s.sort_values('c')
    cw = o['weight'].cumsum() / o['weight'].sum()
    return BL[int(o['c'].values[np.searchsorted(cw.values, q)])]


for lab, ks in [('narrow 1-2', [1, 2]), ('broad 3-4', [3, 4])]:
    for sz in ['Micro', 'Small+']:
        s = br[br.ngroups.isin(ks) & (br['size'] == sz)]
        print(f'{lab:11s} {sz:6s} n={len(s):3d}  ' + '  '.join(f'q{int(q * 100)}: {wq_band(s, q):>9s}'
                                                         for q in [.1, .25, .5, .75, .9]))
