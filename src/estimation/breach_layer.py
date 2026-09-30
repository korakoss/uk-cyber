"""Breach layer, channel by channel: is the survey's 'success' count the event that carries cost?

Per channel with a success count: among flagged firms,
  - coverage of the success question and its distribution (P(>=1), values);
  - the channel's own annual cost question (where it exists) by success count 0 / 1 / 2+: answered n and bands;
  - the worst-incident band by success count 0 / 1 / 2+ (firms hit only by that channel's group where possible).
Channels: phishing (phisheng, engaged), takeover (tkvrsuc), DoS (dossoft), malware (virussoft),
ransomware (ranssoft = ransom demanded). Unweighted.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/breach_layer.py
"""

import numpy as np
import pandas as pd

import joint_five_channel as f
from latent_on_streams import aligned_raw
from type_cooccurrence_structure import load, SHORT

CH = [("phishing", "Phish", "phisheng", None), ("takeover", "Takov", "tkvrsuc", "tkvrcost_bands"),
      ("DoS", "DoS", "dossoft", "doscost_bands"), ("malware", "Malwr", "virussoft", "viruscost_bands"),
      ("ransomware", "Ransm", "ranssoft", "ranscost_bands")]
WG = [(1, 1), (2, 3), (4, 5), (6, 13)]   # worst band: none / <£500 / £500-5k / £5k+
TG = [(1, 3), (4, 6), (7, 12)]           # type cost (12-band, 1 = <£100): <£500 / £500-5k / £5k+


def main():
    X, _, size = load()
    d = f.load_firms()
    r = aligned_raw()
    band = d["band"].values
    num = lambda c: pd.to_numeric(r[c], errors="coerce").values
    for lab, flag, sv, cv in CH:
        hit = X[:, SHORT.index(flag)] == 1
        only = hit & (X.sum(1) == 1)
        s = num(sv)
        s = np.where((s >= 0) & (s < 997), s, np.nan)
        ans = hit & ~np.isnan(s)
        pos = s[ans & (s >= 1)]
        print(f"\n{lab.upper()}: hit {hit.sum()}, success question answered {ans.sum()}, P(success>=1) "
              f"{np.mean(s[ans] >= 1):.3f}; values " + "  ".join(f"{int(x)}:{int((pos == x).sum())}" for x in np.unique(pos)))
        levels = [("succ 0", s == 0), ("succ 1", s == 1), ("succ 2+", s >= 2)]
        if cv:
            c = num(cv)
            c = np.where((c >= 1) & (c <= 12), c, np.nan)
            print(f"  own annual cost ({cv}) answered by hit firms: {int((hit & ~np.isnan(c)).sum())}")
            for ll, lm in levels:
                m = ans & lm
                cc = c[m & ~np.isnan(c)]
                print(f"    {ll:8s} n {m.sum():4d}  cost answered {len(cc):3d}  <£500 / £500-5k / £5k+  "
                      + " ".join(f"{np.mean((cc >= a) & (cc <= b)) if len(cc) else float('nan'):.2f}" for a, b in TG))
        for nm, base in (("all hit", ans), (f"{lab}-only", ans & only)):
            print(f"  worst-incident band by success, {nm}:  none / <£500 / £500-5k / £5k+")
            for ll, lm in levels:
                m = base & lm & ~np.isnan(band)
                b = band[m]
                print(f"    {ll:8s} n {m.sum():4d}  " + " ".join(f"{np.mean((b >= a) & (b <= c_)) if len(b) else float('nan'):.2f}" for a, c_ in WG))


if __name__ == "__main__":
    main()
