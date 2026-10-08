"""Re-run of the old factor analysis of the attack-type flags (type_cooccurrence_structure.py, 2026-09-26),
on the current data loading (2026-10-08). The old script depended on modules from another repo.
All surveyed businesses (questtype 1), including those that ticked no type; survey weights. 11 types (incl. 'other').
For each flag pair: tetrachoric correlation (each yes/no flag treated as an underlying continuous scale cut at a
threshold; the correlation of those scales). Then:
  - eigenvalues of the 11 x 11 correlation matrix and the leading eigenvectors (one general factor vs clusters)
  - a one-factor fit (each type's correlation with one common factor = its loading; pair correlation predicted as
    the product of the two loadings) and the leftover correlation per pair (observed minus predicted)
  - average-linkage clustering on 1 - r
Pooled, Micro, Small+Medium+Large.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, multivariate_normal
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

TYPES = {'type6': 'Phish', 'type5': 'Imper', 'type2': 'Malwr', 'type16': 'Takov', 'type3': 'DoS', 'type1': 'Ransm',
         'type4': 'BankH', 'type8': 'AcOut', 'type7': 'AcStf', 'type15': 'Eavsd', 'type9': 'Other'}
df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df.questtype == 1) & df[list(TYPES)].notna().all(axis=1)]
X = (df[list(TYPES)] == 1).values.astype(int)
W = df.weight.values
SZ = df.sizeb.values
SHORT = list(TYPES.values())


def tetrachoric(a, b, w):
    tab = np.array([[w[(a == i) & (b == k)].sum() for k in (0, 1)] for i in (0, 1)])
    tab = tab / tab.sum() * len(a) + 0.5
    pa, pb = tab[1].sum() / tab.sum(), tab[:, 1].sum() / tab.sum()
    ta, tb = norm.ppf(1 - pa), norm.ppf(1 - pb)

    def nll(r):
        p11 = 1 - norm.cdf(ta) - norm.cdf(tb) + multivariate_normal.cdf([ta, tb], cov=[[1, r], [r, 1]])
        p10, p01 = (1 - norm.cdf(ta)) - p11, (1 - norm.cdf(tb)) - p11
        P = np.clip(np.array([[1 - p11 - p10 - p01, p01], [p10, p11]]), 1e-12, 1)
        return -(tab * np.log(P)).sum()
    return minimize_scalar(nll, bounds=(-0.98, 0.98), method='bounded').x


def analyse(X, w, label):
    n, k = X.shape
    print('\n' + '=' * 100 + f'\n{label} (n={n})\n' + '=' * 100)
    prev = np.average(X, axis=0, weights=w)
    print('  prevalence (weighted) / firms: ' + '  '.join(f'{s} {p:.3f}/{int(X[:, i].sum())}' for i, (s, p) in
                                                       enumerate(zip(SHORT, prev))))
    R, J = np.eye(k), np.zeros((k, k), int)
    for a in range(k):
        for b in range(a + 1, k):
            R[a, b] = R[b, a] = tetrachoric(X[:, a], X[:, b], w)
            J[a, b] = J[b, a] = int((X[:, a] & X[:, b]).sum())
    print('\n  tetrachoric r (upper) / firms with both (lower):')
    print('         ' + ' '.join(f'{s:>6s}' for s in SHORT))
    for a in range(k):
        print(f'  {SHORT[a]:>6s} ' + ' '.join(f'{R[a, b]:6.2f}' if b > a else (f'{J[a, b]:6d}' if b < a else '     -')
                                           for b in range(k)))
    ev, V = np.linalg.eigh(R)
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    print('\n  eigenvalues: ' + ' '.join(f'{e:.2f}' for e in ev) + f'   (share of first {ev[0] / ev.clip(0).sum():.2f})')
    for i in range(3):
        v = V[:, i] * (np.sign(V[:, i].sum()) if i == 0 else 1)
        print(f'    ev{i + 1}: ' + ' '.join(f'{s} {x:5.2f}' for s, x in zip(SHORT, v)))

    # one-factor fit: minimise squared differences between off-diagonal r and loading products
    iu = np.triu_indices(k, 1)
    r = minimize(lambda l: ((R[iu] - np.outer(l, l)[iu]) ** 2).sum(), np.full(k, 0.6), method='L-BFGS-B',
                 bounds=[(-0.99, 0.99)] * k)
    L = r.x
    res = R - np.outer(L, L)
    print('\n  one-factor loadings: ' + ' '.join(f'{s} {x:.2f}' for s, x in zip(SHORT, L)))
    print(f'  leftover correlation (observed r - loading product): mean |leftover| {np.abs(res[iu]).mean():.3f}; '
          f'largest:')
    order = np.argsort(-np.abs(res[iu]))
    for m in order[:10]:
        a, b = iu[0][m], iu[1][m]
        print(f'     {SHORT[a]}+{SHORT[b]:6s} r {R[a, b]:5.2f}  predicted {L[a] * L[b]:5.2f}  leftover {res[a, b]:+5.2f}  '
              f'(firms with both {J[a, b]})')

    D = 1 - R
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(np.clip(D, 0, None), checks=False), method='average')
    members = {i: [SHORT[i]] for i in range(k)}
    print('\n  average-linkage merges (mean r between the merged groups):')
    for step, (a, b, dist, _) in enumerate(Z):
        a, b = int(a), int(b)
        members[k + step] = members[a] + members[b]
        print(f'    r={1 - dist:5.2f}: {"+".join(members[a])}  <->  {"+".join(members[b])}')


analyse(X, W, 'POOLED, all businesses')
analyse(X[SZ == 1], W[SZ == 1], 'MICRO')
analyse(X[SZ >= 2], W[SZ >= 2], 'SMALL + MEDIUM + LARGE')
