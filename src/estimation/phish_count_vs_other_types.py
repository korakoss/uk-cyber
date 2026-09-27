"""Pearson correlation: phishing count vs number of OTHER survey types hit (phishing-flagged firms with a count).

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/phish_count_vs_other_types.py
"""

import numpy as np

import joint_five_channel as f
from type_cooccurrence_structure import load

X, w, size = load()
d = f.load_firms()
N = d["N"].values
m0 = (X[:, 0] == 1) & ~np.isnan(N)
n_other = X[:, 1:].sum(1)


def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    c = np.average((x - mx) * (y - my), weights=w)
    return c / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


for lab, sizes in (("Micro", [1]), ("Small+Medium+Large", [2, 3, 4]), ("All", [1, 2, 3, 4])):
    m = m0 & np.isin(size, sizes)
    print(f"{lab:20s} n={m.sum():4d}  Pearson r (weighted) = {wcorr(N[m], n_other[m], w[m]):.3f}"
          f"   (unweighted) = {np.corrcoef(N[m], n_other[m])[0, 1]:.3f}")


from scipy.stats import rankdata

print("\nSpearman (Pearson on ranks):")
for lab, sizes in (("Micro", [1]), ("Small+Medium+Large", [2, 3, 4]), ("All", [1, 2, 3, 4])):
    m = m0 & np.isin(size, sizes)
    rx, ry = rankdata(N[m]), rankdata(n_other[m])
    print(f"{lab:20s} n={m.sum():4d}  Spearman rho (weighted) = {wcorr(rx, ry, w[m]):.3f}"
          f"   (unweighted) = {np.corrcoef(rx, ry)[0, 1]:.3f}")


print("\nlog phishing count by # other types hit (0 / 1 / 2+): weighted mean and SD of ln N")
print("(multiplicative: mean rises, SD roughly constant; additive: SD shrinks with exposure)")
g_other = np.minimum(n_other, 2)
rng = np.random.default_rng(0)
for lab, sizes in (("Micro", [1]), ("Small+Medium+Large", [2, 3, 4]), ("All", [1, 2, 3, 4])):
    m = m0 & np.isin(size, sizes)
    y, gg, ww = np.log(N[m]), g_other[m], w[m]
    row = []
    for v in (0, 1, 2):
        s = gg == v
        mu = np.average(y[s], weights=ww[s])
        sd = np.sqrt(np.average((y[s] - mu) ** 2, weights=ww[s]))
        bs = []
        idx_s = np.where(s)[0]
        for _ in range(1000):
            b = rng.choice(idx_s, len(idx_s))
            mb = np.average(y[b], weights=ww[b])
            bs.append(np.sqrt(np.average((y[b] - mb) ** 2, weights=ww[b])))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        row.append(f"{['0', '1', '2+'][v]}: n={s.sum():3d} mean {mu:.2f} SD {sd:.2f} [{lo:.2f}, {hi:.2f}]")
    print(f"{lab:20s} " + "   ".join(row))
