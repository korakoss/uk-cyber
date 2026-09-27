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


if __name__ == "__main__":
    main()
