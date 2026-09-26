"""SVD (correspondence analysis) of frequency-answer distributions across firm groups.

Idea (mixture rank): if firms come in K exposure tiers, each with a fixed distribution over the
frequency answer, and groups differ only in their tier mix, then every group's answer
distribution is a blend of K profiles. The centred, chi-square-scaled group x answer table
(correspondence analysis) then has at most K-1 non-noise dimensions:
  2 tiers -> 1 dimension (groups on a line); 3 tiers -> 2 (groups fill a triangle);
  smooth exposure -> a curved arc / several small dimensions.
Answer categories: not attacked, once, <monthly, monthly, weekly, daily, several/day.
Groups: sector (SIC combined) within Micro; size x sector cells for Small+Medium+Large and pooled
(size also shifts rates, so tier profiles differ by size there - Micro is the clean test).
Raw counts (size x sector are the sampling strata). Groups with < 20 firms dropped; attacked
firms with no frequency answer dropped. Noise floor: singular values after permuting group
labels across firms (keeps group sizes and overall answer mix), 500 permutations.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/frailty_freq_svd.py
Writes build/frailty_freq_svd.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joint_four_channel as j
import joint_five_channel as f
from data import load_raw

HERE = os.path.dirname(os.path.abspath(__file__))
CATS = ["none", "once", "<monthly", "monthly", "weekly", "daily", "several/day"]
SIZE_NAME = {1: "Micro", 2: "Small", 3: "Medium", 4: "Large"}
SIZE_COL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
INK, INK2, GRID, SURF = "#1f1f1e", "#6b6a64", "#e4e3dd", "#fcfcfb"
MIN_N = 20


def load():
    d = f.load_firms()
    raw = load_raw()
    keep = j.banded(raw["type_comb1"]).isin([0, 1])
    allg = np.column_stack([(j.banded(raw[c]) == 1).astype(int)[keep].values for c in j.GENUINE_TYPE_COLS])
    a0 = j.banded(raw["type_comb1"])[keep].values
    sel = (a0 == 0) | (allg.sum(1) > 0)
    d["sector"] = pd.to_numeric(raw.loc[keep, "sector_comb2"], errors="coerce")[sel].reset_index(drop=True).values
    d["ans"] = np.where(d["att"] == 0, 0, d["freq"])
    return d[d["ans"].notna() & d["sector"].notna()].copy()


def ca(N):
    P = N / N.sum()
    r, c = P.sum(1), P.sum(0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    F = (U * s) / np.sqrt(r)[:, None]          # principal row coordinates
    G = (Vt.T * s) / np.sqrt(c)[:, None]       # principal column coordinates
    return s, F, G


def table(sub, key):
    g = sub.groupby(key)
    N = np.array([[np.sum(gg["ans"].values == k) for k in range(7)] for _, gg in g], float)
    keys = list(g.groups.keys())
    ok = N.sum(1) >= MIN_N
    return N[ok], [k for k, o in zip(keys, ok) if o], sub[sub.set_index(key).index.isin([k for k, o in zip(keys, ok) if o])]


def gid(sub, key):
    cols = [key] if isinstance(key, str) else key
    return np.array(["|".join(str(int(v)) for v in row) for row in sub[cols].values])


def gid_key(k):
    return "|".join(str(int(v)) for v in (k if isinstance(k, tuple) else (k,)))


def perm_null(sub, key, keys, rng, B=500):
    lab = gid(sub, key)
    ans = sub["ans"].values.astype(int)
    ukeys = [gid_key(k) for k in keys]
    out = []
    for _ in range(B):
        a = rng.permutation(ans)
        N = np.array([np.bincount(a[lab == k], minlength=7) for k in ukeys], float)
        out.append(ca(N)[0])
    return np.array(out)


def analyse(sub, key, title, rng):
    N, keys, sub = table(sub, key)
    s, F, G = ca(N)
    null = perm_null(sub, key, keys, rng)
    inertia = s ** 2
    print(f"\n{title}: {len(keys)} groups, {int(N.sum())} firms")
    print("  singular values: " + " ".join(f"{v:.3f}" for v in s[:6]))
    print("  perm-null 95%:   " + " ".join(f"{v:.3f}" for v in np.percentile(null, 95, axis=0)[:6]))
    print(f"  share of inertia dim1 {inertia[0] / inertia.sum():.2f}, dim2 {inertia[1] / inertia.sum():.2f}")
    print("  answer-category coordinates (dim1, dim2): " +
          ", ".join(f"{c} ({g[0]:+.2f},{g[1]:+.2f})" for c, g in zip(CATS, G)))
    return dict(s=s, null=null, F=F, G=G, keys=keys, n=N.sum(1), title=title)


def plot(results, path):
    fig, axes = plt.subplots(2, len(results), figsize=(5.2 * len(results), 9.4), facecolor=SURF)
    for col, R in enumerate(results):
        ax = axes[0, col]
        k = min(6, len(R["s"]))
        x = np.arange(1, k + 1)
        lo, hi = np.percentile(R["null"], [5, 95], axis=0)
        ax.fill_between(x, lo[:k], hi[:k], color=GRID, lw=0, label="no-structure range (shuffled groups, 5-95%)")
        ax.plot(x, R["s"][:k], color=INK, lw=2, marker="o", ms=8, mec=SURF, mew=2, label="observed")
        ax.set_xticks(x)
        ax.set_xlabel("dimension", color=INK2)
        ax.set_ylabel("singular value", color=INK2)
        ax.set_title(R["title"], color=INK, fontsize=12, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)

        ax = axes[1, col]
        F, G = R["F"], R["G"]
        for (key, xy, n) in zip(R["keys"], F, R["n"]):
            size = key[0] if isinstance(key, tuple) else 1
            ax.scatter(xy[0], xy[1], s=8 + n * 0.9, color=SIZE_COL[int(size)], alpha=0.85,
                       edgecolor=SURF, linewidth=2, zorder=3)
        for c, g in zip(CATS, G):
            ax.scatter(g[0], g[1], marker="x", s=40, color=INK2, zorder=2)
            ax.annotate(c, g[:2], xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK2)
        ax.axhline(0, color=GRID, lw=1, zorder=1)
        ax.axvline(0, color=GRID, lw=1, zorder=1)
        ax.set_xlabel("dimension 1", color=INK2)
        ax.set_ylabel("dimension 2", color=INK2)
        sizes = sorted({int(k[0]) if isinstance(k, tuple) else 1 for k in R["keys"]})
        for sz in sizes:
            ax.scatter([], [], s=60, color=SIZE_COL[sz], label=f"{SIZE_NAME[sz]} group")
        ax.scatter([], [], marker="x", color=INK2, label="answer category")
        ax.legend(frameon=False, fontsize=8, loc="best", labelcolor=INK2)
        ax.set_title("groups (dot area ∝ firms) and answer categories", color=INK2, fontsize=9, loc="left")
    for ax in axes.ravel():
        ax.set_facecolor(SURF)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle("Frequency-answer distributions across firm groups: how many dimensions?\n"
                 "K exposure tiers → at most K−1 dimensions above the shuffled-noise band",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, facecolor=SURF)


def main():
    d = load()
    rng = np.random.default_rng(0)
    res = [analyse(d[d["sizeb"] == 1], "sector", "Micro: sector groups", rng),
           analyse(d[d["sizeb"].isin([2, 3, 4])], ["sizeb", "sector"], "Small+Medium+Large: size×sector", rng),
           analyse(d, ["sizeb", "sector"], "All sizes: size×sector", rng)]
    path = os.path.join(HERE, "build", "frailty_freq_svd.png")
    plot(res, path)
    print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
