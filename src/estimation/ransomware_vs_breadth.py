"""Is the 'breached broad firms cost more than independence' excess just ransomware? (2026-10-05)
Plain cross-tab, breached firms only (breach marker as in breach_cost.py):
  rows    = ransomware ticked or not
  columns = breadth measured WITHOUT the ransomware tick (number of other types ticked)
Cells: n, share with worst >= £500, share >= £5k (weighted).
If breadth matters with ransomware held fixed (both rows rise left to right), there is a breadth fact beyond
ransomware. If only the ransomware row is high and the other row is flat, it is mostly ransomware.
Also the same table for NOT breached firms, and a list of the ransomware firms with narrow breadth.
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
df['c'] = df['damage_bands'].astype(int)
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
NON_R = ['type6', 'type5', 'type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
df['R'] = (df['type1'] == 1).astype(int)
df['k'] = (df[NON_R] == 1).sum(axis=1)          # types ticked other than ransomware
df['kb'] = pd.cut(df['k'], [-1, 1, 2, 3, 99], labels=['0-1', '2', '3', '4+'])
COLS = ['0-1', '2', '3', '4+']


def cell(s):
    if len(s) == 0:
        return f"{'-':>22s}"
    w = s.weight / s.weight.sum()
    return f'{len(s):3d}  {w[s.c >= 4].sum():.2f} / {w[s.c >= 6].sum():.2f}    '


for lab, d in [('BREACHED firms', df[df.breach == 1]), ('NOT breached firms', df[df.breach == 0])]:
    print(f'\n=== {lab}: n   share >= £500 / >= £5k, by number of types ticked other than ransomware ===')
    print(f"{'':16s}" + ''.join(f'{c:>22s}' for c in COLS))
    for r, rl in [(0, 'no ransomware'), (1, 'ransomware')]:
        print(f'{rl:16s}' + ''.join(f'{cell(d[(d.R == r) & (d.kb == c)]):>22s}' for c in COLS))

BL = {1: 'none', 2: '<100', 3: '100-500', 4: '500-1k', 5: '1k-5k', 6: '5k-10k', 7: '10k-20k', 8: '20k-50k',
      9: '50k-100k', 10: '100k-500k'}
print('\n=== Ransomware firms (all), worst cost band by breadth without ransomware (counts) ===')
r = df[df.R == 1]
print(pd.crosstab(r['c'].map(BL), r['kb']).reindex(list(BL.values())).fillna(0).astype(int))
print('breached share of ransomware firms by breadth:',
      {c: f'{r[r.kb == c].breach.mean():.2f} (n={len(r[r.kb == c])})' for c in COLS})

# --- 2026-10-05: independence comparison by breadth band, firms WITHOUT ransomware ---
# Pools: firms hit by only one group (P phishing / I impersonation / S other types), no ransomware tick.
# Prediction for a band: pick an observed firm of that band (by weight), draw its worst cost and breach flag
# independently per group, with one S draw per S type it ticked; worst = largest draw; breached if any draw breached.
# Keep breached simulated firms; compare with observed breached firms of the band. Bootstrap 90% interval on the gap
# (resample pools and band firms).
rng = np.random.default_rng(2)
OTHER_S = ['type2', 'type3', 'type4', 'type16', 'type8', 'type7', 'type15', 'type9']
nr = df[df.R == 0].copy()
nr['P'] = (nr['type6'] == 1).astype(int)
nr['I'] = (nr['type5'] == 1).astype(int)
nr['nS'] = (nr[OTHER_S] == 1).sum(axis=1)
nr['S'] = (nr['nS'] >= 1).astype(int)
nr['set'] = nr.apply(lambda r: ''.join(g for g in 'PIS' if r[g] == 1), axis=1)
nr = nr[nr['set'] != '']
pools = {g: nr[nr['set'] == g] for g in 'PIS'}
print('\n=== Independence by breadth band, no ransomware ===')
print('pools (single-group firms, no ransomware): ' +
      ', '.join(f'{g} n={len(p)} breached {int(p.breach.sum())}' for g, p in pools.items()))


def predict(band, pl, sims):
    ow = (band.weight / band.weight.sum()).values
    pick = rng.choice(len(band), size=sims, p=ow)
    cmax = np.zeros(sims, int)
    bany = np.zeros(sims, bool)
    for g in 'PIS':
        has = band[g].values[pick] == 1
        k = band['nS'].values[pick] if g == 'S' else np.ones(sims, int)
        for j in range(int(k.max()) if has.any() else 0):
            m = has & (k > j)
            s = pl[g]
            idx = rng.choice(len(s), size=m.sum(), p=(s.weight / s.weight.sum()).values)
            cmax[m] = np.maximum(cmax[m], s.c.values[idx])
            bany[m] |= s.breach.values[idx] == 1
    c = cmax[bany]
    return (c >= 4).mean(), (c >= 6).mean(), bany.mean()


def observed(band):
    b = band[band.breach == 1]
    w = b.weight / b.weight.sum()
    return w[b.c >= 4].sum(), w[b.c >= 6].sum(), np.average(band.breach, weights=band.weight)


print(f"{'band':5s} {'n':>4s} {'n br':>5s} {'>=£500 obs':>11s} {'ind':>5s} {'gap 90%':>16s} "
      f"{'>=£5k obs':>10s} {'ind':>5s} {'gap 90%':>16s} {'breached obs':>13s} {'ind':>5s}")
for lab, ks in [('2', [2]), ('3', [3]), ('4+', [4, 5, 6, 7, 8, 9, 10])]:
    band = nr[nr.k.isin(ks)]
    o = observed(band)
    p = predict(band, pools, 40000)
    g500, g5k = [], []
    for _ in range(300):
        pb = {g: pools[g].iloc[rng.integers(0, len(pools[g]), len(pools[g]))] for g in 'PIS'}
        bb = band.iloc[rng.integers(0, len(band), len(band))]
        ob, pr = observed(bb), predict(bb, pb, 4000)
        g500.append(ob[0] - pr[0])
        g5k.append(ob[1] - pr[1])
    a, b = np.percentile(g500, [5, 95])
    c_, d = np.percentile(g5k, [5, 95])
    print(f'{lab:5s} {len(band):4d} {int(band.breach.sum()):5d} {o[0]:11.2f} {p[0]:5.2f} [{a:+.2f}, {b:+.2f}] '
          f'{o[1]:10.2f} {p[1]:5.2f} [{c_:+.2f}, {d:+.2f}] {o[2]:13.2f} {p[2]:5.2f}')
