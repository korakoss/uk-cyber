"""Two checks to separate (a) 'breadth = real exposure that makes breaches worse' from
(b) 'breadth = footprint of one big compromise' (2026-10-05). Breached firms with a cost answer.
Groups: narrow (1-3 types) / broad (4+) x cheap / costly (worst >= £5k).

1. Direct linkage items (the survey's own 'this led to that' questions):
   - impersonationhack / impersonationtkvr (Q53B/C): did impersonation involve unauthorised access / account takeover
     (asked of impersonation tickers)
   - fraud4_comb2 (Q88A): impersonation using information obtained through the initial breach
   - fraud1-3 (Q88A): money moved out / card details misused / paid on fake info; fraudconta..i (Q88D): of the
     frauds, how many were the direct result of each attack type
   (b) predicts these links are much more common in costly broad firms than in cheap broad firms.
2. Exposure on things a breach cannot create: phishing received (exact count, targeted count), DoS,
   overall attack frequency, impersonation that did NOT involve access/takeover (outside spoofing).
   (a) predicts costly broad firms are more exposed on these than cheap broad firms; (b) predicts no difference.
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
G = ['narrow cheap', 'narrow costly', 'broad cheap', 'broad costly']
b = df[df.breach == 1].copy()
print('breached firms per group: ' + ', '.join(f'{g} {(b.grp == g).sum()}' for g in G))


def share(col_ok, col_yes, lab):
    out = []
    for g in G:
        s = b[(b.grp == g) & col_ok]
        if len(s) == 0:
            out.append(f'{g}: -')
            continue
        out.append(f'{g}: {np.average(col_yes[s.index], weights=s.weight):.2f} ({len(s)})')
    print(f'{lab:46s} ' + '   '.join(out))


print('\n=== 1. Direct linkage items: share yes (n asked/answered) ===')
ih, it = num('impersonationhack'), num('impersonationtkvr')
share(ih.loc[b.index].isin([1, 2, 3]), ih.isin([1, 2]).astype(int), 'impersonation involved unauthorised access')
share(it.loc[b.index].isin([1, 2, 3]), it.isin([1, 2]).astype(int), 'impersonation involved account takeover')
f4 = num('fraud4_comb2')
share(f4.loc[b.index].isin([0, 1]), (f4 == 1).astype(int), 'impersonation using info from the breach')
fr = (num('fraud1') > 0) | (num('fraud2') > 0) | (num('fraud3') > 0)
fr_ok = num('fraud1').notna() & (num('fraud1') >= 0)
share(fr_ok.loc[b.index], fr.astype(int), 'any fraud (money out / cards / paid on fake info)')
CONT = {'a': 'ransomware', 'b': 'malware', 'c': 'DoS', 'd': 'bank hacking', 'e': 'phishing', 'f': 'staff access',
        'g': 'outsider access', 'h': 'eavesdropping', 'i': 'takeover'}
cont = pd.DataFrame({v: (num(f'fraudcont{k}') > 0).astype(int) for k, v in CONT.items()})
cont_ok = pd.concat([num(f'fraudcont{k}') >= 0 for k in CONT], axis=1).any(axis=1)
share(cont_ok.loc[b.index], (cont.sum(axis=1) >= 1).astype(int), 'fraud attributed to some attack type')
share(cont_ok.loc[b.index], (cont.sum(axis=1) >= 2).astype(int), 'fraud attributed to 2+ attack types')
print('\nfraud source types (counts of firms), costly broad vs cheap broad, among answering:')
for g in ['broad cheap', 'broad costly', 'narrow cheap', 'narrow costly']:
    idx = b.index[(b.grp == g) & cont_ok.loc[b.index]]
    print(f'  {g:14s} n={len(idx):3d}  ' + ', '.join(f'{v} {int(cont.loc[idx, v].sum())}' for v in CONT.values()
                                                   if cont.loc[idx, v].sum() > 0))

print('\n=== 2. Exposure on things a breach cannot create ===')
ps = num('Cybercrime_phishsum')
pcb = num('phishcon_bands')
for lab, ok, val in [
        ('phishing ticked', pd.Series(True, index=df.index), df.phish),
        ('phishing count > 20 (of answering phishing firms)', ps >= 0, (ps > 20).astype(int)),
        ('targeted phishing >= 6 (of answering phishing firms)', pcb.isin(range(1, 10)), (pcb >= 5).astype(int)),
        ('DoS ticked', pd.Series(True, index=df.index), df.DoS),
        ('outside spoofing only (imperson., no access/takeover)', ih.isin([1, 2, 3]) & it.isin([1, 2, 3]),
         ((ih == 3) & (it == 3)).astype(int)),
        ('attacked weekly or more', pd.Series(True, index=df.index), (df.freq >= 4).astype(int)),
        ('attacked once', pd.Series(True, index=df.index), (df.freq == 1).astype(int))]:
    share(ok.loc[b.index], val, lab)
print('\nmedian exact phishing count (answering):  ' + '   '.join(
    f'{g}: {np.median(ps[b.index[(b.grp == g) & (ps.loc[b.index] >= 0)]]):.0f} '
    f'({int(((b.grp == g) & (ps.loc[b.index] >= 0)).sum())})' for g in G))
