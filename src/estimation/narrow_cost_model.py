"""Narrow-firm cost model (firms hit by 1-2 channels), fitted as one coherent model.

Per hit channel c (P phishing, I impersonation, R ransomware, S other serious):
  material breach with prob q_c (independent across channels);
  if breached: incident cost M_c = 0 with prob m0, else lognormal(mu_m[c] + delta*Small+, sigma_m);
  if not:      handling cost H_c = 0 with prob h0[c], else lognormal(mu_h[c] + delta*Small+, sigma_h).
Observed per firm: outcome flag (Q56A any outcome) = at least one breach; worst-incident cost = max over hit channels,
on TIGHTENED intervals (total band intersected with summed components).
Likelihood: P(outcome) * P(worst in interval | outcome), enumerating breach subsets (<= 2 channels). Survey-weighted.
Checks: observed vs predicted cost-group shares by outcome x channel set; implied mean handling / material cost.

Run: PYTHONPATH=/home/user/md-clean/src:src/estimation python3 src/estimation/narrow_cost_model.py
"""

import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from broad_interaction import channels, tightened
from material_breach import data

CH = "PIRS"
GROUPS = [(0, 0), (0, 500), (500, 5000), (5000, 20000), (20000, 1e9)]


def unpack(p):
    sig = lambda z: 1 / (1 + np.exp(-z))
    return dict(q=sig(p[0:4]), h0=sig(p[4:8]), mu_h=p[8:12], s_h=np.exp(p[12]), m0=sig(p[13]),
                mu_m=p[14:18], s_m=np.exp(p[18]), delta=p[19])


def cdfs(x, P, sm):
    """Per-channel CDFs at x (scalar >= 0) for handling and material costs."""
    if x <= 0:
        return P["h0"].copy(), np.full(4, P["m0"])
    lx = np.log(x)
    Fh = P["h0"] + (1 - P["h0"]) * norm.cdf((lx - P["mu_h"] - P["delta"] * sm) / P["s_h"])
    Fm = P["m0"] + (1 - P["m0"]) * norm.cdf((lx - P["mu_m"] - P["delta"] * sm) / P["s_m"])
    return Fh, Fm


def worst_cdf(x, hit, P, sm, outcome):
    """P(worst <= x, outcome) for a firm with hit channels `hit` (list of indices)."""
    Fh, Fm = cdfs(x, P, sm)
    tot = 0.0
    for k in range(len(hit) + 1):
        for B in itertools.combinations(hit, k):
            if (len(B) > 0) != outcome:
                continue
            pr = np.prod([P["q"][c] if c in B else 1 - P["q"][c] for c in hit])
            g = np.prod([Fm[c] if c in B else Fh[c] for c in hit])
            tot += pr * g
    return tot


def firm_lik(i, hit, lo, hi, outcome, P, sm):
    if hi == 0:
        return worst_cdf(0, hit, P, sm, outcome)
    return worst_cdf(hi, hit, P, sm, outcome) - (worst_cdf(lo, hit, P, sm, outcome) if lo > 0 else worst_cdf(0, hit, P, sm, outcome))


def setup():
    X, size, band, D, oa, w = data()
    C = channels(X)
    nch = C.sum(1)
    lo, hi = tightened(band)
    m = ~np.isnan(band) & ~np.isnan(oa) & (nch >= 1) & (nch <= 2)
    idx = np.where(m)[0]
    hits = [list(np.where(C[i] == 1)[0]) for i in idx]
    return idx, hits, lo, hi, oa, (size >= 2).astype(float), w / w[m].mean()


def fit():
    idx, hits, lo, hi, oa, sm, w = setup()

    def nll(p):
        P = unpack(p)
        s = 0.0
        for i, hit in zip(idx, hits):
            s -= w[i] * np.log(max(firm_lik(i, hit, lo[i], hi[i], bool(oa[i]), P, sm[i]), 1e-300))
        return s
    p0 = np.r_[np.full(4, -2.5), np.full(4, 0.0), np.full(4, 5.0), 0.3, -1.0, np.full(4, 7.0), 0.3, 0.3]
    o = minimize(nll, p0, method="L-BFGS-B", options={"maxiter": 3000})
    o = minimize(nll, o.x, method="Nelder-Mead", options={"maxiter": 8000, "xatol": 1e-4, "fatol": 1e-6})
    return o.x, -o.fun, (idx, hits, lo, hi, oa, sm, w)


def report(p, ll, S):
    idx, hits, lo, hi, oa, sm, w = S
    P = unpack(p)
    print(f"firms {len(idx)}  loglik {ll:.1f}")
    print("material-breach prob q:   " + ", ".join(f"{c} {v:.3f}" for c, v in zip(CH, P["q"])))
    print("handling: P(zero)         " + ", ".join(f"{c} {v:.2f}" for c, v in zip(CH, P["h0"])) +
          f";  median if >0 " + ", ".join(f"{c} £{np.exp(v):,.0f}" for c, v in zip(CH, P["mu_h"])) + f";  sigma {P['s_h']:.2f}")
    print(f"material: P(zero) {P['m0']:.2f};  median if >0 " + ", ".join(f"{c} £{np.exp(v):,.0f}" for c, v in zip(CH, P["mu_m"])) +
          f";  sigma {P['s_m']:.2f};  Small+ x{np.exp(P['delta']):.2f}")
    print("implied means (Micro): handling " + ", ".join(
        f"{c} £{(1 - P['h0'][k]) * np.exp(P['mu_h'][k] + P['s_h'] ** 2 / 2):,.0f}" for k, c in enumerate(CH)) +
          "  | material event " + ", ".join(f"{c} £{(1 - P['m0']) * np.exp(P['mu_m'][k] + P['s_m'] ** 2 / 2):,.0f}" for k, c in enumerate(CH)))
    print("\nobserved vs predicted worst-cost shares (none / <£500 / £500-5k / £5k-20k / £20k+), by outcome x channel set")
    sets = {}
    for i, hit in zip(idx, hits):
        sets.setdefault(("".join(CH[c] for c in hit), int(oa[i])), []).append((i, hit))
    for key in sorted(sets, key=lambda k: -len(sets[k])):
        rows = sets[key]
        if len(rows) < 8:
            continue
        obs, pred, wt = np.zeros(5), np.zeros(5), 0.0
        for i, hit in rows:
            mid = (lo[i] + hi[i]) / 2
            g = 0 if hi[i] == 0 else next(j for j in range(1, 5) if GROUPS[j][0] <= mid < GROUPS[j][1])
            obs[g] += w[i]
            norm_ = worst_cdf(1e12, hit, P, sm[i], bool(oa[i]))
            cs = [worst_cdf(0, hit, P, sm[i], bool(oa[i]))] + [worst_cdf(b, hit, P, sm[i], bool(oa[i])) for _, b in GROUPS[1:4]] + [norm_]
            pred += w[i] * np.diff(np.r_[0.0, cs]) / norm_
            wt += w[i]
        print(f"  {key[0]:3s} outcome {key[1]}  n {len(rows):4d}  obs " + " ".join(f"{v:.2f}" for v in obs / wt) +
              "   pred " + " ".join(f"{v:.2f}" for v in pred / wt))


def main():
    p, ll, S = fit()
    np.save("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/narrow_params.npy", p)
    report(p, ll, S)


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


# --- Size-factor refit -------------------------------------------------------------------------------------------
# The pooled fit gave Small+ x0.75 on cost amounts. Suspects: one shift shared by handling and material (handling bulk
# dominates); size unable to act through zero shares / breach rates; weighting. Variants:
#   A  one shift (as above)                               B  separate shifts for handling and material amounts
#   C  B + Small+ logit shifts on h0 (shared over channels) and on q (shared over channels)
#   D  C unweighted
# Run: ... narrow_cost_model.py size

def unpack_v(p, v):
    P = unpack(p[:20])
    P["dh"] = p[19]
    P["dm"] = p[20] if v in "BCD" else p[19]
    P["dh0"] = p[21] if v in "CD" else 0.0
    P["dq"] = p[22] if v in "CD" else 0.0
    return P


def at_size(P, sm):
    sig = lambda z: 1 / (1 + np.exp(-z))
    lg = lambda x: np.log(x / (1 - x))
    Q = dict(P)
    Q["h0"] = sig(lg(P["h0"]) + P["dh0"] * sm)
    Q["q"] = sig(lg(P["q"]) + P["dq"] * sm)
    Q["mu_h"] = P["mu_h"] + P["dh"] * sm
    Q["mu_m"] = P["mu_m"] + P["dm"] * sm
    Q["delta"] = 0.0
    return Q


def fit_variant(v, S, p_start):
    idx, hits, lo, hi, oa, sm, w = S
    ww = np.ones_like(w) if v == "D" else w
    k = {"A": 20, "B": 21, "C": 23, "D": 23}[v]

    def nll(p):
        P = unpack_v(p, v)
        Ps = (at_size(P, 0.0), at_size(P, 1.0))
        s = 0.0
        for i, hit in zip(idx, hits):
            s -= ww[i] * np.log(max(firm_lik(i, hit, lo[i], hi[i], bool(oa[i]), Ps[int(sm[i])], 0.0), 1e-300))
        return s
    p0 = np.r_[p_start, np.zeros(23 - len(p_start))][:k]
    o = minimize(nll, p0, method="L-BFGS-B", options={"maxiter": 3000})
    o = minimize(nll, o.x, method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-4, "fatol": 1e-6})
    return o.x, -o.fun, k


def size_refit():
    S = setup()
    idx, hits, lo, hi, oa, sm, w = S
    print(f"firms {len(idx)}, Small+ {int(sm[idx].sum())}, Small+ weight share {w[idx][sm[idx] == 1].sum() / w[idx].sum():.3f}")
    base = np.load("/tmp/claude-0/-home-user-uk-cyber/6e8dfdbe-60fd-596b-8b20-907319eda82c/scratchpad/narrow_params.npy")
    start = np.r_[base, base[19]]
    for v in "ABCD":
        p, ll, k = fit_variant(v, S, start if v != "A" else base)
        if v in "BC":
            start = p
        P = unpack_v(p, v)
        print(f"\n{v}: loglik {ll:.1f}  params {k}")
        print(f"   handling amount x{np.exp(P['dh']):.2f}   material amount x{np.exp(P['dm']):.2f}"
              f"   h0 logit shift {P['dh0']:+.2f}   q logit shift {P['dq']:+.2f}")
        for lab, s_ in (("Micro", 0.0), ("Small+", 1.0)):
            Q = at_size(P, s_)
            print(f"   {lab:6s} q " + " ".join(f"{c}{x:.3f}" for c, x in zip(CH, Q["q"])) +
                  "  h0 " + " ".join(f"{c}{x:.2f}" for c, x in zip(CH, Q["h0"])) +
                  "  mean handling " + " ".join(f"{c}£{(1 - Q['h0'][j]) * np.exp(Q['mu_h'][j] + Q['s_h'] ** 2 / 2):,.0f}" for j, c in enumerate(CH)) +
                  "  mean material " + " ".join(f"{c}£{(1 - Q['m0']) * np.exp(Q['mu_m'][j] + Q['s_m'] ** 2 / 2):,.0f}" for j, c in enumerate(CH)))


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "size":
    size_refit()
