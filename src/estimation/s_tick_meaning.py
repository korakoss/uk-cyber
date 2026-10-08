"""Does an 'other types' (S) tick mean something different in S-only firms vs broad firms? (2026-10-05)
Per-type success questions exist for four S types:
  malware  virussoft_comb  (any successful malware attack)
  takeover tkvrsuc_comb    (any successful takeover)
  DoS      dossoft_comb    (any successful DoS attack)
  ransom   ranssoft_comb   (any attack with a ransom demanded)
Step 1 (this file, first part): coverage - how many ticking firms answered, in S-only vs broad firms.
Step 2: success share per tick, S-only vs broad. If S ticks in broad firms are mostly unsuccessful attempts while
S-only ticks are mostly successful, the S-only firms are a 'noticed because harmed' group and the wrong baseline.
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7))].copy()
OTHER = ['type1', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
df['P'] = (df['type6'] == 1).astype(int)
df['I'] = (df['type5'] == 1).astype(int)
df['S'] = (df[OTHER] == 1).any(axis=1).astype(int)
df['set'] = df.apply(lambda r: ''.join(g for g in 'PIS' if r[g] == 1), axis=1)
df['where'] = np.select([df['set'] == 'S', df['set'].isin(['PS', 'IS']), df['set'] == 'PIS'],
                        ['S only', 'S + one other', 'P + I + S'], 'no S')

ITEMS = [('malware', 'type2', 'virussoft_comb'), ('takeover', 'type16', 'tkvrsuc_comb'),
         ('DoS', 'type3', 'dossoft_comb'), ('ransomware', 'type1', 'ranssoft_comb')]
print(f"{'type':10s} {'where':14s} {'ticked':>6s} {'answered':>8s} {'success share (weighted)':>25s}")
for name, tick, col in ITEMS:
    for wh in ['S only', 'S + one other', 'P + I + S']:
        s = df[(df[tick] == 1) & (df['where'] == wh)]
        a = s[s[col].isin([0, 1])]
        share = np.average(a[col], weights=a.weight) if len(a) else np.nan
        print(f'{name:10s} {wh:14s} {len(s):6d} {len(a):8d} {share:25.2f}')

print('\n--- pooled over malware / takeover / DoS: firm has any success among its ticked types (answered firms) ---')
for wh in ['S only', 'S + one other', 'P + I + S']:
    s = df[df['where'] == wh].copy()
    ticked = np.zeros(len(s), bool)
    succ = np.zeros(len(s), bool)
    answered = np.zeros(len(s), bool)
    for name, tick, col in ITEMS[:3]:
        t = (s[tick] == 1).values
        ticked |= t
        answered |= t & s[col].isin([0, 1]).values
        succ |= t & (s[col] == 1).values
    a = s[answered]
    print(f'{wh:14s} firms ticking one of these: {ticked.sum():3d}, answered: {answered.sum():3d}, '
          f'any success (weighted): {np.average(succ[answered], weights=a.weight):.2f}, '
          f'unweighted successes {succ[answered].sum()}')
