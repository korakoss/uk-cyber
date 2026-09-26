"""Part 1 of the iid-cost assumption: does per-attack cost behave the same however many attacks a
firm gets? Posterior-predictive check against the fitted size model ('rates', joint_size_model.py).

Under iid costs within a channel, the worst incident of a firm with k attacks follows G(b)^k with
one per-attack G. The model encodes this, so the check is whether it reproduces, as a function
of attack count:
  A. phishing-only firms, by exact phishing count N (targeted + mass mixture handled by the model):
     P(no cost), worst-band mix.  Firms with costly incidents report N more often, so simulated
     firms are 'reported' with the observed reporting rate by worst band (MAR given band).
  B. impersonation-only / other-serious-only / ransomware-only firms, by frequency answer
     (simulated through the calibrated count->answer mapping).
Simulation is population-representative: firms per size proportional to ONS counts; observed
side uses survey weights.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/iid_cost_check.py
"""

import json
import os
import numpy as np

import joint_four_channel as j
import joint_five_channel as f
import joint_size_model as sm

HERE = os.path.dirname(os.path.abspath(__file__))
NB_BUCKETS = [(1, 1), (2, 3), (4, 10), (11, 50), (51, 10**6)]
FREQ_LAB = ["once", "<monthly", "monthly", "weekly", "daily", "sev/day"]


def simulate(q, n, F, rng):
    exposed = rng.random(n) < q["pi"]
    K, Y = {}, {}
    for ch in f.CH:
        rate = q[f.RATE[ch]] * np.where(exposed, np.exp(q["d" + ch]), 1.0)
        if ch in ("T", "I"):
            k = rng.poisson(rate)
        else:
            r = q["r" + ch]
            k = rng.negative_binomial(r, r / (r + rate))
        k = np.minimum(k, j.KMAX)
        G = j.band_cdf(q["p" + ch], q["mu" + ch], q["s" + ch])
        y = (G[None, :] ** k[:, None] < rng.random(n)[:, None]).sum(1)
        K[ch], Y[ch] = k, np.where(k == 0, 0, y)
    Ymat = np.stack([Y[ch] for ch in f.CH], 1)
    ktot = sum(K.values())
    b = j.BUCKET_OF_K[np.minimum(ktot, j.KMAX)]
    freq = np.where(ktot > 0, (F[b].cumsum(1) < rng.random(n)[:, None]).sum(1) + 1, 0)
    return dict(N=K["T"] + K["M"], KI=K["I"], KR=K["R"], KS=K["S"], band=Ymat.max(1),
                freq=np.minimum(freq, 6))


def summarise(band, w):
    out = []
    for lo, hi in ((1, 1), (2, 3), (4, 13)):
        out.append(np.average((band >= lo) & (band <= hi), weights=w) if w.sum() else np.nan)
    return out


def main():
    d = f.load_firms()
    F, _, _ = j.calibrate_freq(d)
    r = json.load(open(os.path.join(HERE, "build", "size_model_rates.json")))
    q, sh = sm.unpack(np.array(r["x"]), r["blocks"])
    rng = np.random.default_rng(7)
    tot = sum(sm.N_BY_SIZE.values())
    sims = [simulate(sm.size_q(q, sh, g), int(1_500_000 * sm.N_BY_SIZE[s] / tot) + 20_000, F, rng)
            for g, s in enumerate(sm.SIZES)]
    S = {k: np.concatenate([x[k] for x in sims]) for k in sims[0]}
    sw = np.concatenate([np.full(len(x["N"]), sm.N_BY_SIZE[s] / len(x["N"])) for x, s in zip(sims, sm.SIZES)])

    w = d["weight"].fillna(d["weight"].median()).values
    band, N = d["band"].values, d["N"].values
    fP, fI, fR, fS = (d[c].values == 1 for c in ("fP", "fI", "fR", "fS"))

    # --- A: phishing-only by exact N ---
    ph_only = fP & ~fI & ~fR & ~fS & ~np.isnan(band)
    rep_rate = {}
    for b_ in range(1, 14):
        m = ph_only & (band == b_)
        if m.sum():
            rep_rate[b_] = np.average(~np.isnan(N[m]), weights=w[m])
    sim_ph = (S["N"] > 0) & (S["KI"] == 0) & (S["KR"] == 0) & (S["KS"] == 0)
    rr = np.array([rep_rate.get(int(b_), np.mean(list(rep_rate.values()))) for b_ in S["band"]])
    sim_rep = sim_ph & (rng.random(len(rr)) < rr)
    print("=" * 84)
    print("A. PHISHING-ONLY FIRMS BY EXACT PHISHING COUNT N  (worst band: no cost | £1-500 | £500+)")
    print("=" * 84)
    print("  reporting rate of N by worst band (obs): " +
          " ".join(f"b{k}:{v:.2f}" for k, v in sorted(rep_rate.items())))
    print(f"  {'N':>8s} {'obs n':>6s}   {'observed':>22s}   {'model':>22s}")
    for lo, hi in NB_BUCKETS:
        mo = ph_only & (N >= lo) & (N <= hi)
        ms = sim_rep & (S["N"] >= lo) & (S["N"] <= hi)
        o, s = summarise(band[mo], w[mo]), summarise(S["band"][ms], sw[ms])
        lab = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f"{lo}+")
        print(f"  {lab:>8s} {mo.sum():6d}   " + " ".join(f"{v:6.2f}" for v in o) + "   " +
              " ".join(f"{v:6.2f}" for v in s))

    # --- B: single-channel firms by frequency answer ---
    freq = d["freq"].values
    for lab_, obs_m, sim_m in (
            ("IMPERSONATION-ONLY", fI & ~fP & ~fR & ~fS, (S["KI"] > 0) & (S["N"] == 0) & (S["KR"] == 0) & (S["KS"] == 0)),
            ("OTHER-SERIOUS-ONLY", fS & ~fP & ~fI & ~fR, (S["KS"] > 0) & (S["N"] == 0) & (S["KI"] == 0) & (S["KR"] == 0))):
        print("\n" + "=" * 84)
        print(f"B. {lab_} FIRMS BY FREQUENCY ANSWER  (worst band: no cost | £1-500 | £500+)")
        print("=" * 84)
        print(f"  {'freq':>9s} {'obs n':>6s}   {'observed':>22s}   {'model':>22s}   share of firms obs/model")
        mo_all = obs_m & ~np.isnan(band) & ~np.isnan(freq)
        for groups in ([1], [2, 3], [4, 5, 6]):
            mo = mo_all & np.isin(freq, groups)
            ms = sim_m & np.isin(S["freq"], groups)
            o, s = summarise(band[mo], w[mo]), summarise(S["band"][ms], sw[ms])
            lab = "/".join(FREQ_LAB[g - 1] for g in groups)
            sh_o = w[mo].sum() / w[mo_all].sum()
            sh_s = sw[ms].sum() / sw[sim_m].sum()
            print(f"  {lab[:9]:>9s} {mo.sum():6d}   " + " ".join(f"{v:6.2f}" for v in o) + "   " +
                  " ".join(f"{v:6.2f}" for v in s) + f"   {sh_o:.2f}/{sh_s:.2f}")


if __name__ == "__main__":
    main()
