"""Is the worst incident (damage_bands) close to the annual total? Compare with crimecost_bands.

Both on the same 13-band cost scale (1 none ... 10 £100k-500k). Per firm with both valid: band difference
(crimecost - damage) and midpoint ratio. Tabulated overall, by frequency answer, and for single-channel groups.
Also: SPSS label of crimecost_bands and its coverage among attacked firms. Survey-weighted shares.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/total_vs_worst.py
"""

import glob

import numpy as np
import pandas as pd
import pyreadstat

import joint_five_channel as f
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load, SHORT

MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000, 9: 75000, 10: 300000,
       11: 750000, 12: 3000000, 13: 5000000}
FLAB = {1: "once", 2: "<monthly", 3: "monthly", 4: "weekly", 5: "daily", 6: "sev/day"}


def summary(label, cc, db, w, m):
    m = m & ~np.isnan(cc) & ~np.isnan(db)
    if m.sum() == 0:
        return
    diff = cc[m] - db[m]
    ww = w[m]
    lower = np.average(diff < 0, weights=ww)
    same = np.average(diff == 0, weights=ww)
    higher = np.average(diff > 0, weights=ww)
    both = m & (db > 1)
    ratio = np.array([MID[int(a)] / MID[int(b)] for a, b in zip(cc[both], db[both])]) if both.sum() else np.array([np.nan])
    tot_ratio = np.average([MID[int(a)] for a in cc[m]], weights=ww) / max(np.average([MID[int(b)] for b in db[m]], weights=ww), 1e-9)
    print(f"  {label:34s} n {m.sum():4d}  total<worst {lower:4.2f}  same band {same:4.2f}  total>worst {higher:4.2f}   "
          f"median ratio (worst>0) {np.median(ratio):5.2f}   ratio of weighted mean midpoints {tot_ratio:5.2f}")


def main():
    sav = glob.glob("/home/user/uk-cyber/data/raw/*.sav")[0]
    _, meta = pyreadstat.read_sav(sav, metadataonly=True)
    for c in ("crimecost_bands", "damage_bands"):
        print(f"{c}: {meta.column_names_to_labels.get(c)}")
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    cc = pd.to_numeric(r["crimecost_bands"], errors="coerce").values
    cc = np.where((cc >= 1) & (cc <= 13), cc, np.nan)
    db, fq = d["band"].values, d["freq"].values
    w = d["weight"].fillna(d["weight"].median()).values
    att = d["att"].values == 1
    print(f"\nattacked firms {att.sum()}: damage_bands valid {int((att & ~np.isnan(db)).sum())}, crimecost valid "
          f"{int((att & ~np.isnan(cc)).sum())}, both {int((att & ~np.isnan(db) & ~np.isnan(cc)).sum())}")
    print("crimecost band counts (both valid): " + str(pd.Series(cc[att & ~np.isnan(db) & ~np.isnan(cc)]).value_counts().sort_index().to_dict()))
    print("\nper firm, crimecost (total) vs damage (worst incident):")
    summary("all attacked", cc, db, w, att)
    for k in range(1, 7):
        summary(f"freq {FLAB[k]}", cc, db, w, att & (fq == k))
    only = X.sum(1) == 1
    summary("phishing-only", cc, db, w, att & only & (X[:, 0] == 1))
    summary("impersonation-only", cc, db, w, att & only & (X[:, SHORT.index("Imper")] == 1))
    summary("2+ types", cc, db, w, att & (X.sum(1) >= 2))
    summary("worst incident > 0 only", cc, db, w, att & (db > 1))


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


# --- #1 worst vs annual total, two gaps (2026-10-02) ---------------------------------------------------------------
# (a) across channels: under the fitted narrow cost model (narrow_cost_model.py baseline params), annual = sum of per-channel
#     handling/material costs, observed = max. Simulate E[sum] vs E[max] for the 806 narrow firms (weighted).
# (b) repeats within a channel: success counts (P phisheng, R ranssoft, S tkvrsuc+dossoft+virussoft). Observed worst for
#     firms with 1 vs 2+ successes in a channel, against 'each success an independent cost draw' (max of n draws from the
#     1-success firms' worst costs, bootstrap). Upper-bound add-on to the annual total: (n-1) extra draws per firm.
# Run: ... total_vs_worst.py gaps

def gaps():
    import narrow_cost_model as ncm
    from breach_measures import num
    rng = np.random.default_rng(1)
    P = ncm.unpack(np.load("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/narrow_params.npy"))
    idx, hits, lo, hi, oa, sm, w = ncm.setup()
    R = 4000
    S_, M_, sets = np.zeros(len(idx)), np.zeros(len(idx)), []
    for k, (i, hit) in enumerate(zip(idx, hits)):
        cost = np.zeros((R, len(hit)))
        for j, c in enumerate(hit):
            br = rng.random(R) < P["q"][c]
            mu = np.where(br, P["mu_m"][c], P["mu_h"][c]) + P["delta"] * sm[i]
            sg = np.where(br, P["s_m"], P["s_h"])
            z = rng.random(R) < np.where(br, P["m0"], P["h0"][c])
            cost[:, j] = np.where(z, 0.0, np.exp(mu + sg * rng.standard_normal(R)))
        S_[k], M_[k] = cost.sum(1).mean(), cost.max(1).mean()
        sets.append("".join(ncm.CH[c] for c in hit))
    ww = w[idx]
    print("(a) model-implied annual sum vs worst (narrow firms, weighted means)")
    print(f"  all {len(idx)}: E[sum] £{np.average(S_, weights=ww):,.0f}  E[max] £{np.average(M_, weights=ww):,.0f}  ratio {np.average(S_, weights=ww) / np.average(M_, weights=ww):.3f}")
    sets = np.array(sets)
    for st in sorted(set(sets), key=lambda x: -(sets == x).sum()):
        m = sets == st
        if m.sum() >= 15:
            print(f"  {st:3s} n {m.sum():4d}  E[sum] £{np.average(S_[m], weights=ww[m]):,.0f}  E[max] £{np.average(M_[m], weights=ww[m]):,.0f}  ratio {np.average(S_[m], weights=ww[m]) / np.average(M_[m], weights=ww[m]):.3f}")

    print("\n(b) repeats within a channel (all firms hit by the channel; tightened worst-incident midpoints)")
    X, size, band, D, oa2, w2 = ncm.data()
    from broad_interaction import channels, tightened
    C = channels(X)
    lo2, hi2 = tightened(band)
    mid = np.where(hi2 == 0, 0.0, (lo2 + hi2) / 2)
    r = aligned_raw()
    cnt = {"P": num(r, "phisheng"), "R": num(r, "ranssoft"),
           "S": np.nansum(np.column_stack([num(r, c) for c in ("tkvrsuc", "dossoft", "virussoft")]), 1)}
    cnt["S"] = np.where(np.isnan(np.column_stack([num(r, c) for c in ("tkvrsuc", "dossoft", "virussoft")])).all(1), np.nan, cnt["S"])
    for ch, col in (("P", 0), ("R", 2), ("S", 3)):
        n = cnt[ch]
        base = (C[:, col] == 1) & ~np.isnan(n) & ~np.isnan(mid)
        one, two = base & (n == 1), base & (n >= 2)
        if one.sum() < 5 or two.sum() < 3:
            print(f"  {ch}: too few (1: {one.sum()}, 2+: {two.sum()})")
            continue
        pool, pw = mid[one], w2[one] / w2[one].sum()
        ns = n[two].astype(int)
        print(f"  {ch}: 1 success n {one.sum()}, 2+ n {two.sum()} (counts median {np.median(ns):.0f}, max {ns.max()}, share >=5 {np.mean(ns >= 5):.2f})")
        for capn in (5, 1000):
            sim = np.array([[rng.choice(pool, min(k, capn), p=pw).max() for k in ns] for _ in range(300)])
            print(f"     cap {capn:4d}: P(worst>=£500) obs 1: {np.average(mid[one] >= 500, weights=w2[one]):.2f}  2+: {np.average(mid[two] >= 500, weights=w2[two]):.2f}"
                  f"  pred 2+ {np.mean(sim >= 500):.2f} | P(>=£5k) obs 1: {np.average(mid[one] >= 5000, weights=w2[one]):.2f}"
                  f"  2+: {np.average(mid[two] >= 5000, weights=w2[two]):.2f}  pred 2+ {np.mean(sim >= 5000):.2f}"
                  f" | mean worst obs 2+ £{np.average(mid[two], weights=w2[two]):,.0f} pred £{sim.mean():,.0f}")
        hitm = (C[:, col] == 1) & ~np.isnan(mid)
        e1 = np.average(pool, weights=pw)
        for capn in (5, 1000):
            extra = np.where(base & (n >= 2), (np.minimum(n, capn) - 1) * e1, 0.0)
            print(f"     upper-bound add-on (each extra success = fresh draw, mean £{e1:,.0f}), cap {capn}: "
                  f"£{np.average(extra[hitm], weights=w2[hitm]):,.0f} per hit firm vs mean worst £{np.average(mid[hitm], weights=w2[hitm]):,.0f}"
                  f" (+{np.average(extra[hitm], weights=w2[hitm]) / np.average(mid[hitm], weights=w2[hitm]):.0%})")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "gaps":
    gaps()
