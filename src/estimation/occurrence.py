"""Fresh look (2026-10-05) at which attack types occur together (the type tick-boxes, Q53A).
Attacked businesses only (firms with no attack are all-zero and would inflate any association).
Shares weighted; n unweighted.
Questions:
 1. Can one attack tick several boxes? Look at firms attacked only once.
 2. Are types independent? Compare how often pairs co-occur vs what independence predicts.
 3. Is the dependence roughly one-dimensional? (a) number of types ticked vs independence;
    (b) each type's rate as the number of OTHER types ticked rises; (c) correlation eigenvalues.
"""
import itertools
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7))].copy()

TYPES = {'type6': 'phish', 'type5': 'imperson', 'type2': 'malware', 'type1': 'ransom', 'type3': 'DoS',
         'type4': 'bankhack', 'type16': 'takeover', 'type8': 'outsider', 'type7': 'staff',
         'type15': 'eavesdrop', 'type9': 'other'}
X = (df[list(TYPES)] == 1).astype(int)
X.columns = list(TYPES.values())
w = df['weight'].values
df['ntypes'] = X.sum(axis=1).values
small_plus = (df['sizeb'] >= 2).values


def ws(mask):
    return w[mask].sum() / w.sum()


print('=== 1. Number of types ticked, by attack frequency ===')
FREQ = {1: 'once', 2: '<monthly', 3: 'monthly', 4: 'weekly', 5: 'daily', 6: 'several/day'}
print(f"{'freq':12s} {'n':>4s} " + ' '.join(f'{k:>6s}' for k in ['1', '2', '3', '4+']))
for f, lab in FREQ.items():
    s = df[df['freq'] == f]
    sw = s['weight']
    print(f'{lab:12s} {len(s):4d} ' + ' '.join(f'{sw[s.ntypes == k].sum() / sw.sum():6.2f}' for k in [1, 2, 3])
          + f' {sw[s.ntypes >= 4].sum() / sw.sum():6.2f}')

once = df['freq'] == 1
print('\n-- firms attacked ONCE that ticked 2+ types: most common combinations (unweighted n)')
combo = X[once & (df['ntypes'] >= 2)].apply(lambda r: '+'.join(r.index[r == 1]), axis=1)
print(combo.value_counts().head(10).to_string())

print('\n=== 2. Pairs: observed joint share / share expected if independent (attacked firms) ===')
print('(1.0 = independent; >1 = occur together more often than chance)')
names = list(X.columns)
main = ['phish', 'imperson', 'malware', 'ransom', 'DoS', 'bankhack', 'takeover', 'outsider', 'staff']
p = {c: np.average(X[c], weights=w) for c in names}
print(f"{'':10s}" + ''.join(f'{c:>9s}' for c in main))
for a in main:
    row = []
    for b in main:
        if a == b:
            row.append(f"{'-':>9s}")
        else:
            joint = np.average(X[a] * X[b], weights=w)
            row.append(f'{joint / (p[a] * p[b]):9.2f}')
    print(f'{a:10s}' + ''.join(row))
print('pair counts (unweighted) for the rare types are small; e.g. ransom+DoS n =', int((X.ransom * X.DoS).sum()))

print('\n-- same ratio within size groups, for a few pairs (is it just size?)')
for a, b in [('phish', 'imperson'), ('malware', 'ransom'), ('ransom', 'DoS'), ('bankhack', 'takeover'),
             ('imperson', 'takeover'), ('malware', 'outsider')]:
    out = []
    for lab, m in [('Micro', ~small_plus), ('Small+', small_plus)]:
        ww = w[m]
        pa, pb = np.average(X[a][m], weights=ww), np.average(X[b][m], weights=ww)
        out.append(f'{lab} {np.average(X[a][m] * X[b][m], weights=ww) / (pa * pb):5.2f}')
    print(f'{a}+{b}: ' + '   '.join(out))

print('\n=== 3a. Number of types ticked: observed vs if types were independent ===')
# exact distribution of a sum of independent yes/no items with the observed rates
probs = [p[c] for c in names]
dist = np.array([1.0])
for q in probs:
    dist = np.convolve(dist, [1 - q, q])
print(f"{'#types':>7s} {'observed':>9s} {'independent':>12s}")
for k in range(1, 7):
    obs = ws(df['ntypes'].values == k) if k < 6 else ws(df['ntypes'].values >= 6)
    ind = dist[k] if k < 6 else dist[6:].sum()
    print(f'{k:7d} {obs:9.3f} {ind / (1 - dist[0]):12.3f}')
print('(independent column is conditioned on >= 1 type, since all these firms were attacked)')

print('\n=== 3b. Rate of each type by number of OTHER types ticked ===')
print(f"{'type':10s}" + ''.join(f'{k:>7s}' for k in ['0', '1', '2', '3+']) + '   n(others=0,1,2,3+)')
for c in main:
    others = df['ntypes'].values - X[c].values
    rates, ns = [], []
    for k in [0, 1, 2, 3]:
        m = (others == k) if k < 3 else (others >= 3)
        rates.append(np.average(X[c].values[m], weights=w[m]))
        ns.append(m.sum())
    print(f'{c:10s}' + ''.join(f'{r:7.2f}' for r in rates) + '   ' + ','.join(str(n) for n in ns))

print('\n=== 3c. Weighted correlation matrix of the 9 main flags: eigenvalues ===')
print('(if one common factor drives co-occurrence, the first eigenvalue stands out and the rest are near 1 or below)')
Z = X[main].values.astype(float)
mu = np.average(Z, axis=0, weights=w)
C = np.cov((Z - mu).T, aweights=w)
sd = np.sqrt(np.diag(C))
R = C / np.outer(sd, sd)
ev, evec = np.linalg.eigh(R)
order = np.argsort(ev)[::-1]
print('eigenvalues:', np.round(ev[order], 2))
v = evec[:, order[0]]
v = v * np.sign(v.sum())
print('first direction loadings:', dict(zip(main, np.round(v, 2))))
for i in [1, 2]:
    v = evec[:, order[i]]
    print(f'direction {i + 1} loadings:', dict(zip(main, np.round(v, 2))))
print('note: in 3b the "0 others" column is always 1.00 by construction (attacked firms tick >= 1 type); ignore it.')

# --- 2026-10-05, after user question: redo the pair table on ALL businesses (incl. not attacked).
# Restricting to attacked firms hides the "attacked at all" part of the shared factor and forces
# phishing/impersonation to look independent or negative.
alldf = pd.read_csv('data/proc/data.csv', low_memory=False)
alldf = alldf[(alldf['questtype'] == 1) & (alldf['type11'].isin([0, 1]))].copy()
XA = (alldf[list(TYPES)] == 1).astype(int)
XA.columns = list(TYPES.values())
wa = alldf['weight'].values
pa_ = {c: np.average(XA[c], weights=wa) for c in main}
print(f'\n=== 2b. Pair ratios on ALL businesses (n={len(alldf)}) ===')
print(f"{'':10s}" + ''.join(f'{c:>9s}' for c in main))
for a in main:
    print(f'{a:10s}' + ''.join(f"{'-':>9s}" if a == b else
                               f'{np.average(XA[a] * XA[b], weights=wa) / (pa_[a] * pa_[b]):9.2f}' for b in main))
