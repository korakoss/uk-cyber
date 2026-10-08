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

# --- 2026-10-05: robustness of 'broad firms are breached less often than independence predicts' ---
# (a) bootstrap: resample firms (single-group pools and the observed group) and recompute the gap;
# (b) within size; (c) stricter marker (outcome item only).
print('\n=== Breached share, observed minus independent prediction (bootstrap 90% interval) ===')


def gap(data, marker, st, B=500):
    def pred(d):
        p_not = 1.0
        for g in st:
            s = d[d['set'] == g]
            p_not *= 1 - np.average(s[marker], weights=s.weight)
        return 1 - p_not

    def obs(d):
        s = d[d['set'] == st]
        return np.average(s[marker], weights=s.weight)

    point = obs(data) - pred(data)
    boots = []
    parts = [data[data['set'] == k] for k in list(st) + [st]]
    for _ in range(B):
        d = pd.concat([p.iloc[rng.integers(0, len(p), len(p))] for p in parts])
        boots.append(obs(d) - pred(d))
    return obs(data), pred(data), point, np.percentile(boots, [5, 95])


df['outcome_only'] = (num('outcome_any') == 1).astype(int)
for marker, mlab in [('breach', 'breach marker'), ('outcome_only', 'outcome item only')]:
    for szlab, m in [('all sizes', df.index == df.index), ('Micro', df.sizeb == 1), ('Small+', df.sizeb >= 2)]:
        for st in ['PI', 'PS', 'PIS']:
            d = df[m]
            if (d['set'] == st).sum() < 10 or min((d['set'] == g).sum() for g in st) < 10:
                continue
            o, p, g, ci = gap(d, marker, st)
            print(f'{mlab:18s} {szlab:9s} {st:4s} n={(d["set"] == st).sum():3d}  obs {o:.2f}  ind {p:.2f}  '
                  f'gap {g:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]')
    print()
print('single-group breach rates (weighted):',
      {g: round(np.average(df[df.set == g].breach, weights=df[df.set == g].weight), 2) for g in 'PIS'})

# --- 2026-10-05: is 'breached broad firms cost much more than independent breaches' solid? ---
# Checks on PIS breached firms, share >= £5k and >= £500, observed vs independent prediction:
#  (a) bootstrap interval (resample pools and observed firms);
#  (b) one S draw per S TYPE ticked (S lumps types; PIS firms tick more of them);
#  (c) PIS firms without ransomware (ransomware is nearly absent from the S-only pool: 2 firms);
#  (d) Micro vs Small+.
print('\n=== Robustness of the cost excess in breached broad firms ===')
df['nS'] = (df[OTHER] == 1).sum(axis=1)
df['R'] = (df['type1'] == 1).astype(int)
single = {g: df[df['set'] == g] for g in 'PIS'}  # rebuild with the new columns
print('S-only pool: firms', len(single['S']), ' with ransomware ticked', int(single['S'].R.sum()))


def sim_pred(obs_firms, pools, per_type, sims=20000):
    """Independent prediction for the observed firms' own group sets: for each simulated firm pick an observed firm
    (by weight), draw per group (or per S type) from the single-group pools; keep simulated firms with any breach."""
    ow = (obs_firms.weight / obs_firms.weight.sum()).values
    pick = rng.choice(len(obs_firms), size=sims, p=ow)
    cmax = np.zeros(sims, int)
    bany = np.zeros(sims, bool)
    for g in 'PIS':
        has = obs_firms[g].values[pick] == 1
        k = obs_firms['nS'].values[pick] if (g == 'S' and per_type) else np.ones(sims, int)
        for j in range(int(k.max())):
            m = has & (k > j)
            s = pools[g]
            idx = rng.choice(len(s), size=m.sum(), p=(s.weight / s.weight.sum()).values)
            cmax[m] = np.maximum(cmax[m], s.c.values[idx])
            bany[m] |= s.breach.values[idx] == 1
    c = cmax[bany]
    return (c >= 4).mean(), (c >= 6).mean()


def obs_stats(o):
    w_ = o.weight / o.weight.sum()
    return w_[o.c >= 4].sum(), w_[o.c >= 6].sum()


def check(lab, obs_all, per_type, boot=200):
    ob = obs_all[obs_all.breach == 1]
    o5, o5k = obs_stats(ob)
    p5, p5k = sim_pred(obs_all, single, per_type)
    gaps = []
    for _ in range(boot):
        pools_b = {g: single[g].iloc[rng.integers(0, len(single[g]), len(single[g]))] for g in 'PIS'}
        ob_all_b = obs_all.iloc[rng.integers(0, len(obs_all), len(obs_all))]
        obb = ob_all_b[ob_all_b.breach == 1]
        gaps.append(obs_stats(obb)[1] - sim_pred(ob_all_b, pools_b, per_type, sims=4000)[1])
    lo, hi = np.percentile(gaps, [5, 95])
    print(f'{lab:44s} n breached {len(ob):3d}  >=£500 obs {o5:.2f} ind {p5:.2f}   >=£5k obs {o5k:.2f} ind {p5k:.2f}  '
          f'gap [{lo:+.2f}, {hi:+.2f}]')


pis = df[df['set'] == 'PIS']
check('PIS, one S draw per group', pis, False)
check('PIS, one S draw per S type', pis, True)
check('PIS without ransomware, per S type', pis[pis.R == 0], True)
check('PIS with ransomware, per S type', pis[pis.R == 1], True)
check('PIS Micro, per S type', pis[pis.sizeb == 1], True)
check('PIS Small+, per S type', pis[pis.sizeb >= 2], True)
ps = df[df['set'] == 'PS']
check('PS, per S type', ps, True)
