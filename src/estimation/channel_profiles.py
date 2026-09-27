"""Channel statistics for the channel split: serious homogeneous-ish; ransomware and impersonation separate.

Per-attack-ish cost profiles per survey attack type, two slices:
  CLEAN  - single-channel firms: hit by ONLY this type (of the 10 genuine types). Their worst incident is
           this type's; these types are Poisson at low rates (Step 4), so it is ~one attack's cost.
           Phishing uses phishing-only firms with exact count 1 (phishing firms have many attacks).
  DISR   - firms where this type was the most disruptive attack (larger n; contaminated by max over
           all attacks and by 'most disruptive' selection).
Profile: n, weighted shares no cost / £1-500 / £500-5k / £5k+, share of costly incidents >= £10k.
Tests on 3 bins (no cost / £1-500 / £500+), weighted X2 / Kish deff and permutation p (raw counts):
  T1 other-serious subtypes homogeneous (ransomware excluded)
  T2 ransomware vs other serious pooled
  T3 impersonation vs other serious pooled
  T4 impersonation vs phishing

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/channel_profiles.py
"""

import numpy as np
from scipy.stats import chi2

import joint_five_channel as f
from type_cooccurrence_structure import load

TYPES = ["Phishing", "Impersonation", "Ransomware", "Malware", "DoS", "Bank hacking", "Takeover",
         "Access:outsider", "Access:staff", "Eavesdropping"]
DISR = {"Phishing": 6, "Impersonation": 5, "Ransomware": 1, "Malware": 2, "DoS": 3, "Bank hacking": 4,
        "Takeover": 11, "Access:outsider": 9, "Access:staff": 7, "Eavesdropping": 10}
OTHER_SERIOUS = ["Malware", "DoS", "Bank hacking", "Takeover", "Access:outsider", "Access:staff", "Eavesdropping"]


def profile(band, w):
    if len(band) == 0:
        return None
    s = [np.average(band == 1, weights=w), np.average((band >= 2) & (band <= 3), weights=w),
         np.average((band >= 4) & (band <= 5), weights=w), np.average(band >= 6, weights=w)]
    costly = band >= 2
    big = np.average(band[costly] >= 7, weights=w[costly]) if costly.any() else np.nan
    return s, big


def x2(T):
    E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
    return ((T - E) ** 2 / np.where(E > 0, E, 1)).sum()


def test(groups, rng, label):
    """groups: list of (band array, weight array)."""
    cb = np.concatenate([np.digitize(b, [2, 4]) for b, _ in groups])
    g = np.concatenate([np.full(len(b), i) for i, (b, _) in enumerate(groups)])
    w = np.concatenate([ww for _, ww in groups])
    w = w / w.mean()
    G = len(groups)
    tab = lambda gg, ww: np.array([[ww[(gg == i) & (cb == k)].sum() for k in range(3)] for i in range(G)])
    Tw, Tr = tab(g, w), tab(g, np.ones(len(g)))
    dof = (G - 1) * 2
    stat = x2(Tr)
    perm = np.array([x2(tab(rng.permutation(g), np.ones(len(g)))) for _ in range(5000)])
    pw = chi2.sf(x2(Tw) / (w ** 2).mean(), dof)
    print(f"    {label:52s} n={len(g):4d}  weighted p={pw:.3f}   permutation p={np.mean(perm >= stat):.3f}")


def main():
    X, w, size = load()
    d = f.load_firms()
    band, N, dis = d["band"].values, d["N"].values, d["disrupta"].values
    ok = ~np.isnan(band)
    rng = np.random.default_rng(0)
    n_types = X.sum(1)
    sl = {}
    for i, t in enumerate(TYPES):
        clean = ok & (n_types == 1) & (X[:, i] == 1)
        if t == "Phishing":
            clean = clean & (N == 1)
        sl[("CLEAN", t)] = clean
        sl[("DISR", t)] = ok & (dis == DISR[t])
    for sname in ("CLEAN", "DISR"):
        print("\n" + "=" * 104)
        print("SINGLE-CHANNEL FIRMS (per-attack cost)" if sname == "CLEAN" else "MOST-DISRUPTIVE FIRMS (larger n; max + selection contaminated)")
        print("=" * 104)
        print(f"  {'type':>16s} {'n':>4s}   {'no cost':>7s} {'£1-500':>7s} {'£500-5k':>7s} {'£5k+':>7s}   {'costly: >=£10k':>14s}")
        for t in TYPES:
            m = sl[(sname, t)]
            p = profile(band[m], w[m])
            if p is None:
                print(f"  {t:>16s} {0:4d}")
                continue
            s, big = p
            print(f"  {t:>16s} {m.sum():4d}   " + " ".join(f"{v:7.2f}" for v in s) + f"   {big:14.2f}")
        m_os = np.zeros(len(d), bool)
        for t in OTHER_SERIOUS:
            m_os |= sl[(sname, t)]
        s, big = profile(band[m_os], w[m_os])
        print(f"  {'OTHER SERIOUS':>16s} {m_os.sum():4d}   " + " ".join(f"{v:7.2f}" for v in s) + f"   {big:14.2f}")
        print("  tests (3 bins: no cost / £1-500 / £500+):")
        populated = [t for t in ["Malware", "DoS", "Bank hacking", "Takeover"] if sl[(sname, t)].sum() >= 5]
        test([(band[sl[(sname, t)]], w[sl[(sname, t)]]) for t in populated], rng,
             f"T1 other serious homogeneous ({', '.join(populated)})")
        mr = sl[(sname, "Ransomware")]
        test([(band[mr], w[mr]), (band[m_os], w[m_os])], rng, "T2 ransomware vs other serious")
        mi = sl[(sname, "Impersonation")]
        test([(band[mi], w[mi]), (band[m_os], w[m_os])], rng, "T3 impersonation vs other serious")
        mp = sl[(sname, "Phishing")]
        test([(band[mi], w[mi]), (band[mp], w[mp])], rng, "T4 impersonation vs phishing")


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def count_behaviour():
    """Count side of the channel definition, per survey attack type:
      prevalence by size band (and large/micro ratio); hit probability per exposure tier from the
      3-class flag model (within Micro / within Small+Medium+Large); frequency answer of firms hit
      by ONLY that type (share 'once', share 'monthly or more'); co-occurrence with phishing and with
      impersonation.
    Mix tests for pairs (A, B): among firms hit by A or B, is the weighted share hit by A the same
      across size bands, and across exposure tiers? X2/deff and permutation.
    Merging A and B into one channel is safe if per-attack costs match OR the mix is constant."""
    from counts_within_tier import _tiers
    X, w, size = load()
    d = f.load_firms()
    freq = d["freq"].values
    rng = np.random.default_rng(0)
    wn = w / w.mean()
    n_types = X.sum(1)

    tiers = np.full(len(X), -1)
    th_by = {}
    for lab, sizes in (("Micro", [1]), ("Rest", [2, 3, 4])):
        m = np.isin(size, sizes)
        t, _, pi, th = _tiers(X[m], wn[m] / wn[m].mean(), rng)
        tiers[m] = t
        th_by[lab] = (pi, th)

    print("\n" + "=" * 110)
    print("COUNT BEHAVIOUR PER TYPE")
    print("=" * 110)
    print(f"  {'type':>16s}  {'prevalence micro/small/med/large':>34s} {'L/M':>5s}   "
          f"{'hit prob by tier (Rest) low/mid/high':>37s}   {'alone: once':>11s} {'monthly+':>8s}   {'+phish':>6s} {'+imp':>6s}")
    for i, t in enumerate(TYPES):
        prev = [np.average(X[size == s, i], weights=w[size == s]) for s in (1, 2, 3, 4)]
        th = th_by["Rest"][1][:, i]
        alone = (n_types == 1) & (X[:, i] == 1) & ~np.isnan(freq)
        once = np.average(freq[alone] == 1, weights=w[alone]) if alone.sum() >= 3 else np.nan
        mplus = np.average(freq[alone] >= 3, weights=w[alone]) if alone.sum() >= 3 else np.nan
        hit = X[:, i] == 1
        withP = np.average(X[hit, 0], weights=w[hit]) if i != 0 else np.nan
        withI = np.average(X[hit, 1], weights=w[hit]) if i != 1 else np.nan
        print(f"  {t:>16s}  " + " ".join(f"{v:7.3f}" for v in prev) + f" {prev[3] / max(prev[0], 1e-9):5.1f}   " +
              " ".join(f"{v:11.3f}" for v in th) + f"   {once:11.2f} {mplus:8.2f}   {withP:6.2f} {withI:6.2f}"
              + f"   (alone n={alone.sum()})")
    print("  tier shares (Rest): " + " ".join(f"{v:.2f}" for v in th_by["Rest"][0]) +
          "   (Micro): " + " ".join(f"{v:.2f}" for v in th_by["Micro"][0]))

    def mix_test(ia, ib, label):
        hitA, hitB = X[:, ia] == 1, X[:, ib] == 1
        m = hitA | hitB
        # share of 'A-type' in each firm: A only = 1, B only = 0, both = 0.5 (split)
        a = np.where(hitA & hitB, 0.5, hitA.astype(float))[m]
        ww = wn[m]
        out = []
        for gname, g in (("size", size[m].astype(int)), ("tier", np.where(size[m] == 1, 0, 3) + tiers[m])):
            levels = [v for v in np.unique(g) if (g == v).sum() >= 5]
            keep = np.isin(g, levels)
            gg = np.searchsorted(levels, g[keep])
            T = np.array([[(ww[keep] * (gg == j) * a[keep]).sum(), (ww[keep] * (gg == j) * (1 - a[keep])).sum()]
                          for j in range(len(levels))])
            Tr = np.array([[((gg == j) * a[keep]).sum(), ((gg == j) * (1 - a[keep])).sum()] for j in range(len(levels))])
            st = x2(Tr)
            perm = [x2(np.array([[((pg == j) * a[keep]).sum(), ((pg == j) * (1 - a[keep])).sum()]
                                 for j in range(len(levels))])) for pg in (rng.permutation(gg) for _ in range(3000))]
            shares = T[:, 0] / T.sum(1)
            pw = chi2.sf(x2(T) / (ww[keep] ** 2).mean(), len(levels) - 1)
            out.append(f"{gname}: A-share " + "/".join(f"{s:.2f}" for s in shares) + f" (p={pw:.3f}, perm {np.mean(np.array(perm) >= st):.3f})")
        print(f"    {label:32s} " + "   ".join(out))

    print("\n  MIX TESTS: is the relative frequency of A vs B the same across sizes (micro/small/medium/large)"
          "\n  and across exposure tiers (micro low/mid/high, rest low/mid/high)?")
    idx = {t: i for i, t in enumerate(TYPES)}
    Xs = X.copy()
    other = [idx[t] for t in OTHER_SERIOUS]
    serious_any = X[:, other].max(1)
    X_ext = np.c_[X, serious_any]
    TYPES_ext = TYPES + ["OTHER SERIOUS"]
    Xsave = X
    X = X_ext
    mix_test(idx["Impersonation"], len(TYPES), "A=Impersonation, B=other serious")
    mix_test(idx["Ransomware"], len(TYPES), "A=Ransomware, B=other serious")
    mix_test(idx["Impersonation"], idx["Phishing"], "A=Impersonation, B=Phishing")
    mix_test(idx["Malware"], idx["DoS"], "A=Malware, B=DoS")
    mix_test(idx["Malware"], idx["Takeover"], "A=Malware, B=Takeover")
    mix_test(idx["Malware"], idx["Bank hacking"], "A=Malware, B=Bank hacking")
    X = Xsave


if __name__ == "__main__" and "counts" in __import__("sys").argv:
    count_behaviour()


def mix_by_exposure():
    """Non-circular exposure mix test: for pair (A, B), among firms hit by A or B, share of A-type
    by number of OTHER types hit (excluding A, B and, for serious pairs, the rest of their group is
    kept as exposure read-out): 0 / 1 / 2+. Pooled over sizes and within Micro / Rest."""
    X, w, size = load()
    rng = np.random.default_rng(0)
    idx = {t: i for i, t in enumerate(TYPES)}
    other = [idx[t] for t in OTHER_SERIOUS]
    S = X[:, other].max(1)

    def run(a_vec, b_vec, excl_cols, label):
        m = (a_vec | b_vec).astype(bool)
        a = np.where(a_vec & b_vec, 0.5, a_vec.astype(float))[m]
        keep_cols = [c for c in range(X.shape[1]) if c not in excl_cols]
        k = np.minimum(X[:, keep_cols].sum(1), 2)[m]
        out = []
        for slab, sm in (("all", np.ones(m.sum(), bool)), ("Micro", size[m] == 1), ("Rest", size[m] >= 2)):
            ww = (w[m] / w[m].mean())[sm]
            aa, kk = a[sm], k[sm]
            T = np.array([[(ww * (kk == j) * aa).sum(), (ww * (kk == j) * (1 - aa)).sum()] for j in range(3)])
            Tr = np.array([[((kk == j) * aa).sum(), ((kk == j) * (1 - aa)).sum()] for j in range(3)])
            st = x2(Tr)
            perm = [x2(np.array([[((pk == j) * aa).sum(), ((pk == j) * (1 - aa)).sum()] for j in range(3)]))
                    for pk in (rng.permutation(kk) for _ in range(3000))]
            pw = chi2.sf(x2(T) / (ww ** 2).mean(), 2)
            out.append(f"{slab}: " + "/".join(f"{v:.2f}" for v in T[:, 0] / np.maximum(T.sum(1), 1e-9)) +
                       f" n={'/'.join(str(int((kk == j).sum())) for j in range(3))} (p={pw:.3f}, perm {np.mean(np.array(perm) >= st):.3f})")
        print(f"    {label:32s} " + "   ".join(out))

    print("\n  MIX BY EXPOSURE (share of A among A-or-B firms, by # OTHER types hit 0 / 1 / 2+):")
    I, R, P = X[:, idx["Impersonation"]] == 1, X[:, idx["Ransomware"]] == 1, X[:, idx["Phishing"]] == 1
    run(I, S == 1, [idx["Impersonation"]] + other, "A=Impersonation, B=other serious")
    run(R, S == 1, [idx["Ransomware"]] + other, "A=Ransomware, B=other serious")
    run(I, P, [idx["Impersonation"], idx["Phishing"]], "A=Impersonation, B=Phishing")
    for t in ("DoS", "Takeover", "Bank hacking"):
        run(X[:, idx["Malware"]] == 1, X[:, idx[t]] == 1, [idx["Malware"], idx[t]], f"A=Malware, B={t}")


if __name__ == "__main__" and "mixexp" in __import__("sys").argv:
    mix_by_exposure()
