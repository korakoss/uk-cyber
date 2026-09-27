"""Multiplicativity of the latent on the survey's own phishing streams.

Streams with counts: targeted (phishcon, Q89E exact; phishcondk band midpoint where 'don't know') and
engaged (phisheng, Q89C exact). No count exists for untargeted/mass phishing.
Latent read-out: number of OTHER survey types hit (phishing excluded). Phishing-flagged firms,
survey-weighted, within Micro / Small+Medium+Large. Per stream and # other types:
  P(stream count >= 1); among firms with >= 1: mean and SD of ln(count), with bootstrap interval on SD.
Multiplicative: mean ln rises, SD roughly constant; additive: SD shrinks.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/latent_on_streams.py
"""

import numpy as np
import pandas as pd

import joint_four_channel as j
import joint_five_channel as f
from data import load_raw
from type_cooccurrence_structure import load

DK_MID = {1: 0, 2: 1, 3: 2.5, 4: 4.5, 5: 8, 6: 15.5, 7: 35.5, 8: 75.5, 9: 150}


def aligned_raw():
    raw = load_raw()
    keep = j.banded(raw["type_comb1"]).isin([0, 1])
    allg = np.column_stack([(j.banded(raw[c]) == 1).astype(int)[keep].values for c in j.GENUINE_TYPE_COLS])
    a0 = j.banded(raw["type_comb1"])[keep].values
    sel = (a0 == 0) | (allg.sum(1) > 0)
    return raw.loc[keep][sel].reset_index(drop=True)


def main():
    X, w, size = load()
    d = f.load_firms()
    r = aligned_raw()
    assert len(r) == len(d)
    num = lambda c: pd.to_numeric(r[c], errors="coerce")
    con = num("phishcon").where(lambda s: (s >= 0) & (s < 997))
    band = num("phishcon_bands").where(lambda s: s.between(1, 9))
    con = con.fillna(band.map(DK_MID)).values                   # exact where given, else band midpoint
    eng = num("phisheng").where(lambda s: (s >= 0) & (s < 997)).values
    phish = X[:, 0] == 1
    k_other = X[:, 1:].sum(1)
    rng = np.random.default_rng(0)
    print("valid counts among phishing firms: targeted", int((phish & ~np.isnan(con)).sum()),
          " engaged", int((phish & ~np.isnan(eng)).sum()))
    print("max: targeted", np.nanmax(con[phish]), " engaged", np.nanmax(eng[phish]))

    for sname, cnt in (("TARGETED (phishcon)", con), ("ENGAGED (phisheng)", eng)):
        print("\n" + "=" * 100)
        print(sname)
        print("=" * 100)
        for glab, gm in (("Micro", size == 1), ("Small+Medium+Large", size >= 2)):
            base = phish & gm & ~np.isnan(cnt)
            print(f"  {glab}:  {'other':>5s} {'n':>4s} {'P(>=1)':>7s} {'n>=1':>5s} {'mean ln':>8s} {'SD ln':>6s} {'SD 95%':>14s}")
            for k in range(0, k_other[base].max() + 1):
                m = base & (k_other == k)
                if m.sum() == 0:
                    continue
                pos = m & (cnt >= 1)
                p1 = np.average(cnt[m] >= 1, weights=w[m])
                if pos.sum() >= 2:
                    y, ww = np.log(cnt[pos]), w[pos]
                    mu = np.average(y, weights=ww)
                    sd = np.sqrt(np.average((y - mu) ** 2, weights=ww))
                    idx = np.where(pos)[0]
                    bs = []
                    for _ in range(500):
                        b = rng.choice(idx, len(idx))
                        mb = np.average(np.log(cnt[b]), weights=w[b])
                        bs.append(np.sqrt(np.average((np.log(cnt[b]) - mb) ** 2, weights=w[b])))
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    tail = f"{mu:8.2f} {sd:6.2f} [{lo:4.2f}, {hi:4.2f}]"
                else:
                    tail = "       -      -"
                print(f"  {'':{len(glab) + 2}s}{k:5d} {m.sum():4d} {p1:7.2f} {pos.sum():5d} {tail}")


if __name__ == "__main__":
    main()
