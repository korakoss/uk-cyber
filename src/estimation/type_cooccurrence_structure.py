"""Shape of the co-occurrence of the survey's attack-type flags (Q53A), before imposing channels.

All firms (non-attacked included -> no conditioning-on-attacked selection), survey-weighted.
Types: 10 genuine Q53A types with any firms (students-access has none).
  - prevalence and raw joint counts per pair (to see which cells rest on few firms)
  - tetrachoric correlation matrix (latent-propensity correlation; not prevalence-driven like phi)
  - eigenvalue spectrum + leading eigenvectors: one general factor vs several clusters
  - average-linkage hierarchical clustering on 1 - tetrachoric r
Repeated within Micro and within Small+Medium+Large as a check (rare types get noisy).

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/type_cooccurrence_structure.py
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm, multivariate_normal
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

import joint_four_channel as j
import joint_five_channel as f
from data import load_raw

TYPES = {"type6": "Phishing", "type5": "Impersonation", "type1": "Ransomware", "type2": "Malware",
         "type3": "DoS", "type4": "Bank hacking", "type16": "Takeover", "type8": "Access:outsider",
         "type7": "Access:staff", "type15": "Eavesdropping"}
SHORT = ["Phish", "Imper", "Ransm", "Malwr", "DoS", "BankH", "Takov", "AcOut", "AcStf", "Eavsd"]


def load():
    d = f.load_firms()
    raw = load_raw()
    keep = j.banded(raw["type_comb1"]).isin([0, 1])
    t = {c: (j.banded(raw[c]) == 1).astype(int)[keep] for c in TYPES}
    anyflag = sum(t[c] for c in j.GENUINE_TYPE_COLS if c in t) if False else None
    flags = np.column_stack([t[c].values for c in TYPES])
    all_genuine = np.column_stack([(j.banded(raw[c]) == 1).astype(int)[keep].values for c in j.GENUINE_TYPE_COLS])
    att = j.banded(raw["type_comb1"])[keep].values
    sel = (att == 0) | (all_genuine.sum(1) > 0)
    X = flags[sel]
    X[att[sel] == 0] = 0
    assert len(X) == len(d)
    return X, d["weight"].fillna(d["weight"].median()).values, d["sizeb"].values


def tetrachoric(a, b, w):
    """Weighted ML tetrachoric correlation from the 2x2 table (0.5 added to cells)."""
    tab = np.array([[w[(a == i) & (b == k)].sum() for k in (0, 1)] for i in (0, 1)])
    tab = tab / tab.sum() * len(a) + 0.5
    pa = tab[1].sum() / tab.sum()
    pb = tab[:, 1].sum() / tab.sum()
    ta, tb = norm.ppf(1 - pa), norm.ppf(1 - pb)

    def nll(r):
        p11 = 1 - norm.cdf(ta) - norm.cdf(tb) + multivariate_normal.cdf([ta, tb], cov=[[1, r], [r, 1]])
        p10 = (1 - norm.cdf(ta)) - p11
        p01 = (1 - norm.cdf(tb)) - p11
        p00 = 1 - p11 - p10 - p01
        P = np.clip(np.array([[p00, p01], [p10, p11]]), 1e-12, 1)
        return -(tab * np.log(P)).sum()

    return minimize_scalar(nll, bounds=(-0.98, 0.98), method="bounded").x


def analyse(X, w, label):
    n, k = X.shape
    w = w / w.mean()
    print("\n" + "=" * 96)
    print(f"{label}  (n={n})")
    print("=" * 96)
    prev = np.average(X, axis=0, weights=w)
    print("  prevalence (weighted) / raw n:")
    print("    " + "  ".join(f"{s} {p:.3f}/{int(X[:, i].sum())}" for i, (s, p) in enumerate(zip(SHORT, prev))))

    R = np.eye(k)
    J = np.zeros((k, k), int)
    for a in range(k):
        for b in range(a + 1, k):
            R[a, b] = R[b, a] = tetrachoric(X[:, a], X[:, b], w)
            J[a, b] = J[b, a] = int((X[:, a] & X[:, b]).sum())
    print("\n  tetrachoric r (upper) / raw joint count (lower):")
    print("         " + " ".join(f"{s:>6s}" for s in SHORT))
    for a in range(k):
        cells = [f"{R[a, b]:6.2f}" if b > a else (f"{J[a, b]:6d}" if b < a else "     -") for b in range(k)]
        print(f"  {SHORT[a]:>6s} " + " ".join(cells))

    # nearest PSD for the spectrum
    ev, V = np.linalg.eigh(R)
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    print("\n  eigenvalues: " + " ".join(f"{e:.2f}" for e in ev) +
          f"   (share of first {ev[0] / ev.clip(0).sum():.2f})")
    print("  leading eigenvectors (sign-fixed so first is positive):")
    print("         " + " ".join(f"{s:>6s}" for s in SHORT))
    for i in range(3):
        v = V[:, i] * (np.sign(V[:, i].sum()) if i == 0 else 1)
        print(f"    ev{i + 1}: " + " ".join(f"{x:6.2f}" for x in v))

    D = 1 - R
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(np.clip(D, 0, None), checks=False), method="average")
    members = {i: [SHORT[i]] for i in range(k)}
    print("\n  average-linkage merges (similarity = 1 - distance = mean tetrachoric r between clusters):")
    for step, (a, b, dist, _) in enumerate(Z):
        a, b = int(a), int(b)
        members[k + step] = members[a] + members[b]
        print(f"    r={1 - dist:5.2f}: {'+'.join(members[a])}  <->  {'+'.join(members[b])}")


def main():
    X, w, size = load()
    analyse(X, w, "POOLED, all firms, survey-weighted")
    analyse(X[size == 1], w[size == 1], "MICRO")
    m = np.isin(size, [2, 3, 4])
    analyse(X[m], w[m], "SMALL+MEDIUM+LARGE")


if __name__ == "__main__":
    main()
