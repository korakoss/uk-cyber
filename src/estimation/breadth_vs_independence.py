"""Fresh check (2026-10-05): does the extra cost of broad firms (above what independent incidents would
give) apply to breached firms only, or also to firms without a breach?

Plain simulation, no fitted model:
- 3 type groups: P = phishing, I = impersonation, S = everything else (incl. ransomware; ransomware-only
  firms are too few to stand alone).
- 'Independent' prediction for a firm hit by a set of groups: for each group, draw a firm at random
  (by survey weight) from the firms hit by ONLY that group; take its worst cost band and its breach flag.
  The simulated firm's worst cost = the largest of the draws; it counts as breached if any draw was breached.
- Compare observed vs predicted for multi-group firms, overall and split by breach status
  (for the split, the simulated firms are split by their simulated breach flag).
Breach marker as in breach_cost.py. Attacked businesses with a cost answer. Pooled over sizes.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
df['c'] = df['damage_bands'].astype(int)
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
OTHER = ['type1', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
df['P'] = (df['type6'] == 1).astype(int)
df['I'] = (df['type5'] == 1).astype(int)
df['S'] = (df[OTHER] == 1).any(axis=1).astype(int)
df['set'] = df.apply(lambda r: ''.join(g for g in 'PIS' if r[g] == 1), axis=1)
df = df[df['set'] != '']

single = {g: df[df['set'] == g] for g in 'PIS'}
for g in 'PIS':
    s = single[g]
    print(f'single-group firms {g}: n={len(s)}, breached {s.breach.sum()}')


def draw(g, k):
    s = single[g]
    idx = rng.choice(len(s), size=k, p=(s.weight / s.weight.sum()).values)
    return s.c.values[idx], s.breach.values[idx]


def stats(c, w):
    w = w / w.sum()
    return (w[c == 1].sum(), w[c >= 4].sum(), w[c >= 6].sum())


SIMS = 20000
print('\nshares: no cost / >= £500 / >= £5k ; obs = observed, ind = independent prediction')
print(f"{'set':5s} {'who':13s} {'n':>4s} {'obs none':>9s} {'ind none':>9s} {'obs 500+':>9s} {'ind 500+':>9s} "
      f"{'obs 5k+':>8s} {'ind 5k+':>8s}")
for st in ['PI', 'PS', 'IS', 'PIS']:
    obs = df[df['set'] == st]
    cs, bs = [], []
    for g in st:
        c, b = draw(g, SIMS)
        cs.append(c)
        bs.append(b)
    cmax = np.max(cs, axis=0)
    bany = np.max(bs, axis=0)
    for who, om, sm in [('all', np.ones(len(obs), bool), np.ones(SIMS, bool)),
                        ('not breached', obs.breach.values == 0, bany == 0),
                        ('breached', obs.breach.values == 1, bany == 1)]:
        o = obs[om]
        if len(o) == 0:
            continue
        so = stats(o.c.values, o.weight.values)
        ss = stats(cmax[sm], np.ones(sm.sum()))
        print(f'{st:5s} {who:13s} {len(o):4d} {so[0]:9.2f} {ss[0]:9.2f} {so[1]:9.2f} {ss[1]:9.2f} '
              f'{so[2]:8.2f} {ss[2]:8.2f}')
    print(f'{"":5s} breached share: obs {np.average(obs.breach, weights=obs.weight):.2f}, ind {bany.mean():.2f}')
