"""
Step 6 — national aggregation simulation. Pulls together every prior piece
(N(size) from ons_business_counts.py, empirical prevalence, empirical freq
distribution, the per-(size,freq) / per-size lognormal cost fits, and the
bridge multipliers) into a Monte Carlo estimate of the TOTAL annual UK
business cybercrime cost, expressed as a DISTRIBUTION, not a point.

Design decisions (agreed 2026-07-15):
- TWO decompositions run in parallel as a cross-check:
  (A) TYPE-BASED  — per-firm cost x bridge(disrupta), using this session's
      per-attack-type multipliers (Ransomware 2.21x, Impersonation drawn
      from its fragile 3.11-6.25x range each replicate, Malware 1.93x,
      DoS 0.36x, Hacking 1.01x, Takeover 5.50x, Phishing 1.02x; freq==1
      firms get bridge=1 exactly; unresolved/minor types fall back to 1.0).
  (B) FREQ-BASED  — per-firm cost x m(freq), where the cost draw uses the
      per-(size,freq) lognormal cell and m(freq)=1 for freq==1, else 1.02
      (the mixture-model-validated phishing best-estimate bridge, which the
      project treats as its representative freq-based best estimate — see
      NOTES "Estimation Pipeline" Step 5). This is deliberately the SIMPLE
      bridge; comparing (B) to (A) shows how much the per-type refinement
      (dominated by the big Ransomware/Impersonation multipliers) moves the
      national total versus a flat phishing-style bridge.

- COST DRAW: lognormal within-band interpolation. For a firm observed in
  damage band b (interval [L,U)), we draw a CONTINUOUS pound value from the
  fitted zero-inflated lognormal for its cell, conditioned on landing in
  [L,U). The top band (10, "£100k+") is treated as [100k, inf) with NO upper
  cap, so the draw follows the fitted lognormal tail — this is what lets rare
  very-large losses drive the total, rather than a fixed midpoint cap.

- WHY SIMULATE (not just multiply means): at N ~ 1.4M businesses, drawing
  1.4M i.i.d. firms per replicate would collapse the total to its mean by the
  CLT and understate uncertainty. The uncertainty that actually matters is in
  the *inputs*: prevalence, the band mix (esp. how many firms fall in the open
  top band — only 8 in the sample), the within-band/tail draw, and
  Impersonation's bridge. So we BOOTSTRAP the survey (resample firms within
  each size band, weighted by survey weight) B times; each replicate yields a
  cost-per-business per size band, which Squiggle then scales by the fixed
  N(size) and sums. The fitted lognormal SHAPES (mu,sigma) are held at their
  point estimates across replicates (refitting sparse cells per replicate is
  unstable); the dominant tail uncertainty is still captured because band-10
  membership — the count of very-large-loss firms — varies across resamples.

- ENGINE: the per-size cost-per-business bootstrap samples are emitted into
  Squiggle model files (build/national_typebased.squiggle and
  national_freqbased.squiggle), and run headless via the squiggle-lang CLI.
  Python computes the same aggregation directly as a cross-check.

Outputs (per decomposition): mean, median, 5th/95th percentile of the national
annual total (£), plus the per-size-band contribution breakdown. Written to
stdout and the generated .squiggle files are left in src/estimation/build/.
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

sys.path.insert(0, '.')
from proc import get_business_data, SPECIAL_CODE_THRESHOLD

RNG = np.random.default_rng(20260715)
B = 500  # bootstrap replicates

# --- N(size): ONS BPE 2025 employer frame (see ons_business_counts.py) ------
N_BY_SIZE = {1: 1_150_875, 2: 220_085, 3: 38_435, 4: 8_335}
SIZE_LABELS = {1: 'Micro', 2: 'Small', 3: 'Medium', 4: 'Large'}

# --- cost band structure (damage_bands 1..10; band 10 = £100k+, open tail) --
BAND_BOUNDS = {
    1: (0, 0), 2: (1, 100), 3: (100, 500), 4: (500, 1000), 5: (1000, 5000),
    6: (5000, 10000), 7: (10000, 20000), 8: (20000, 50000),
    9: (50000, 100000), 10: (100000, 500000),
}
# Band 10 is BOUNDED at £100k-£500k per the CSBS codebook. The scale explicitly
# continues (11=£500k-1M, 12=£1M-5M, 13=£5M+) and NO firm selected any band above
# 10 — so the data contains no evidence of any >£500k most-disruptive incident.
# Treating band 10 as open would manufacture tail cost the survey did not find.

# --- bridges ----------------------------------------------------------------
# Type-based (disrupta -> multiplier), from bridge_specification.py.
BEST_ESTIMATE_BRIDGE = {
    1: 2.21,   # Ransomware
    2: 1.93,   # Other malware
    3: 0.36,   # Denial of service
    4: 1.01, 7: 1.01, 9: 1.01,  # Hacking (broad)
    6: 1.02,   # Phishing
    11: 5.50,  # Website/social/email takeover
}
IMPERSONATION_LOW, IMPERSONATION_HIGH = 3.11, 6.25  # disrupta==5, drawn per replicate
FREQ_BRIDGE_BEST = 1.02  # freq-based m(freq) for freq>=2 (phishing-validated)

VALID_FREQ = {1, 2, 3, 4, 5, 6}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = get_business_data()
df = df[df['sizeb'].notna()].copy()
df['sizeb'] = df['sizeb'].astype(int)
df['attacked'] = df['freq'].isin(VALID_FREQ)
# clean damage band (only meaningful for attacked firms)
db = df['damage_bands']
df['band'] = np.where(df['attacked'] & db.notna() & (db < SPECIAL_CODE_THRESHOLD) & (db >= 1),
                      db, np.nan)


# ---------------------------------------------------------------------------
# Lognormal fitting (zero-inflated; p0 = P(band==1); fit mu,sigma on bands 2-10)
# ---------------------------------------------------------------------------
def fit_lognormal(bands):
    """bands: 1-D array of integer band codes (1..10). Returns (p0, mu, sigma)."""
    bands = bands[~np.isnan(bands)].astype(int)
    n = len(bands)
    if n == 0:
        return None
    counts = np.array([np.sum(bands == b) for b in range(1, 11)], dtype=float)
    p0 = counts[0] / n
    cond = counts[1:]  # bands 2..10
    if cond.sum() < 5:  # too few non-zero-cost obs to identify a shape
        return None

    def nll(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        pred = np.zeros(9)
        for i, b in enumerate(range(2, 11)):
            L, U = BAND_BOUNDS[b]
            lo = norm.cdf(np.log(max(L, 1)), mu, sigma)
            hi = 1.0 if np.isinf(U) else norm.cdf(np.log(U), mu, sigma)
            pred[i] = hi - lo
        s = pred.sum()
        if s < 1e-9:
            return 1e10
        pred = np.clip(pred / s, 1e-12, 1)
        return -(cond * np.log(pred)).sum()

    best = None
    for mu0 in (5, 7, 9, 11):
        for ls0 in (0.5, 1.0, 1.5):
            r = minimize(nll, [mu0, ls0], method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
            if best is None or r.fun < best.fun:
                best = r
    return p0, best.x[0], float(np.exp(best.x[1]))


def band_conditional_mean(mu, sigma, b):
    """Analytic E[X | L <= X < U] for X ~ Lognormal(mu,sigma), i.e. the
    lognormal within-band interpolation done exactly rather than by a single
    noisy point draw. Band 1 -> 0. Band 10 -> E[X | X >= 100k] (finite,
    captures the fitted tail without single-draw explosions)."""
    if b == 1:
        return 0.0
    L, U = BAND_BOUNDS[b]
    a = (np.log(max(L, 1)) - mu) / sigma
    bU = np.inf if np.isinf(U) else (np.log(U) - mu) / sigma
    denom = (1.0 if np.isinf(U) else norm.cdf(bU)) - norm.cdf(a)
    if denom < 1e-12:
        return np.exp(mu + sigma ** 2 / 2)
    num = (1.0 if np.isinf(U) else norm.cdf(bU - sigma)) - norm.cdf(a - sigma)
    return np.exp(mu + sigma ** 2 / 2) * num / denom


def band_cost_vector(mu, sigma, bands):
    """Map each observed band code -> its analytic conditional-mean £ cost."""
    lut = {b: band_conditional_mean(mu, sigma, b) for b in range(1, 11)}
    out = np.zeros(len(bands))
    for b in range(1, 11):
        m = bands == b
        if m.any():
            out[m] = lut[b]
    return out


# ---------------------------------------------------------------------------
# Fit shapes once: per-size (for type-based) and per-(size,freq) (for freq-based)
# ---------------------------------------------------------------------------
print("Fitting lognormal shapes (held fixed across bootstrap)...")
size_fit = {}
for sz in (1, 2, 3, 4):
    bands = df.loc[df['sizeb'] == sz, 'band'].values
    fit = fit_lognormal(bands)
    size_fit[sz] = fit
    if fit:
        print(f"  size {SIZE_LABELS[sz]:<7} p0={fit[0]:.3f} mu={fit[1]:.2f} sigma={fit[2]:.2f} "
              f"E[cost|>0]=£{np.exp(fit[1]+fit[2]**2/2):,.0f}")

# A per-(size,freq) shape is only trusted when the cell has enough non-zero-cost
# observations AND its fitted sigma is plausible (sparse cells otherwise produce
# garbage sigma, and the open-top-band conditional mean is hyper-sensitive to it —
# e.g. Large cells are all too thin). Otherwise fall back to the well-populated
# per-size pooled shape. Consistent with the earlier "sparse (size,freq) cells
# unsafe to use directly" finding (freq_distribution.py).
SF_MIN_NONZERO = 20
SF_SIGMA_CAP = 3.5
sf_fit = {}
sf_source = {}
for sz in (1, 2, 3, 4):
    for f in VALID_FREQ:
        bands = df.loc[(df['sizeb'] == sz) & (df['freq'] == f), 'band'].values
        n_nonzero = int(np.sum((~np.isnan(bands)) & (bands >= 2)))
        fit = fit_lognormal(bands)
        if fit is not None and n_nonzero >= SF_MIN_NONZERO and fit[2] <= SF_SIGMA_CAP:
            sf_fit[(sz, f)] = fit
            sf_source[(sz, f)] = 'cell'
        else:
            sf_fit[(sz, f)] = size_fit[sz]
            sf_source[(sz, f)] = 'size-pooled'
n_cell = sum(v == 'cell' for v in sf_source.values())
print(f"  freq-based cost cells: {n_cell}/{len(sf_source)} use own (size,freq) shape, "
      f"rest fall back to per-size pooled")

# ---------------------------------------------------------------------------
# Bootstrap: per-size cost-per-business under each decomposition
# ---------------------------------------------------------------------------
print(f"\nBootstrapping ({B} replicates)...")


def bridge_type(disrupta, freq, imp_bridge):
    if freq == 1:
        return 1.0
    if disrupta == 5:
        return imp_bridge
    return BEST_ESTIMATE_BRIDGE.get(int(disrupta), 1.0) if not pd.isna(disrupta) else 1.0


# precompute per-band, per-size firm arrays
size_dfs = {sz: df[df['sizeb'] == sz].reset_index(drop=True) for sz in (1, 2, 3, 4)}

# vectorised type-bridge lookup per firm (depends on freq==1 and disrupta;
# Impersonation handled separately as it varies per replicate)
def static_type_bridge(sub):
    """bridge that does NOT depend on the per-replicate Impersonation draw.
    Returns (bridge_array, is_impersonation_mask)."""
    freq = sub['freq'].values
    dis = sub['disrupta'].values
    br = np.ones(len(sub))
    is_imp = np.zeros(len(sub), dtype=bool)
    for i in range(len(sub)):
        if freq[i] == 1:
            br[i] = 1.0
        elif dis[i] == 5:
            is_imp[i] = True  # filled per replicate
            br[i] = 1.0
        else:
            br[i] = BEST_ESTIMATE_BRIDGE.get(int(dis[i]), 1.0) if not pd.isna(dis[i]) else 1.0
    return br, is_imp


type_static = {sz: static_type_bridge(size_dfs[sz]) for sz in (1, 2, 3, 4)}

boot_type = {sz: np.zeros(B) for sz in (1, 2, 3, 4)}
boot_freq = {sz: np.zeros(B) for sz in (1, 2, 3, 4)}

for rep in range(B):
    imp_bridge = RNG.uniform(IMPERSONATION_LOW, IMPERSONATION_HIGH)
    for sz in (1, 2, 3, 4):
        sub = size_dfs[sz]
        n = len(sub)
        w = sub['weight'].values
        idx = RNG.choice(n, size=n, replace=True, p=w / w.sum())  # weighted bootstrap
        rs = sub.iloc[idx]
        att = rs['attacked'].values
        bands = rs['band'].values
        freq = rs['freq'].values

        safe_bands = np.where(att & ~np.isnan(bands), bands, 1).astype(int)

        # --- cost (type-based uses per-size shape) ---
        mu, sigma = size_fit[sz][1], size_fit[sz][2]
        cost_t = band_cost_vector(mu, sigma, safe_bands)
        # type bridge
        br, is_imp = type_static[sz]
        br_rs = br[idx].copy()
        br_rs[is_imp[idx]] = imp_bridge
        total_t = np.where(att, cost_t * br_rs, 0.0)
        boot_type[sz][rep] = total_t.mean()

        # --- freq-based: per-(size,freq) shape + m(freq) ---
        cost_f = np.zeros(n)
        for f in VALID_FREQ:
            m = att & (freq == f)
            if not m.any():
                continue
            fit = sf_fit[(sz, f)]
            cost_f[m] = band_cost_vector(fit[1], fit[2], safe_bands[m])
        mfreq = np.where(freq == 1, 1.0, FREQ_BRIDGE_BEST)
        total_f = np.where(att, cost_f * mfreq, 0.0)
        boot_freq[sz][rep] = total_f.mean()

# ---------------------------------------------------------------------------
# Emit Squiggle models and run them
# ---------------------------------------------------------------------------
BUILD_DIR = 'src/estimation/build'
os.makedirs(BUILD_DIR, exist_ok=True)


def sq_list(arr):
    return "[" + ", ".join(f"{x:.6g}" for x in arr) + "]"


def write_squiggle(path, per_size_samples, label):
    lines = [
        f"// {label} — national annual UK business cybercrime cost",
        "// per-business bootstrap samples (£/business) x fixed ONS N(size)",
    ]
    for sz in (1, 2, 3, 4):
        s = SIZE_LABELS[sz].lower()
        lines.append(f"cpb_{s} = SampleSet.fromList({sq_list(per_size_samples[sz])})")
        lines.append(f"n_{s} = {N_BY_SIZE[sz]}")
    contrib = " + ".join(f"n_{SIZE_LABELS[sz].lower()} * cpb_{SIZE_LABELS[sz].lower()}" for sz in (1, 2, 3, 4))
    lines.append(f"total = {contrib}")
    lines.append("// national annual total, summarised in £ billions")
    lines.append("{")
    lines.append("  mean_bn: mean(total) / 1e9,")
    lines.append("  p5_bn: quantile(total, 0.05) / 1e9,")
    lines.append("  median_bn: quantile(total, 0.5) / 1e9,")
    lines.append("  p95_bn: quantile(total, 0.95) / 1e9,")
    lines.append("}")
    with open(path, 'w') as fh:
        fh.write("\n".join(lines) + "\n")


write_squiggle(f"{BUILD_DIR}/national_typebased.squiggle", boot_type, "TYPE-BASED")
write_squiggle(f"{BUILD_DIR}/national_freqbased.squiggle", boot_freq, "FREQ-BASED")

# locate the squiggle CLI — prefer the permanent repo install (tools/squiggle),
# fall back to any scratchpad/npx copy.
import glob
SQ_CLI = None
_repo_cli = 'tools/squiggle/node_modules/@quri/squiggle-lang/dist/cli/index.js'
if os.path.exists(_repo_cli):
    SQ_CLI = _repo_cli
else:
    for c in glob.glob('/private/tmp/**/sqtool/node_modules/@quri/squiggle-lang/dist/cli/index.js', recursive=True):
        SQ_CLI = c
        break


def run_squiggle(path):
    if not SQ_CLI:
        return None
    try:
        out = subprocess.run(['node', SQ_CLI, 'run', path], capture_output=True, text=True, timeout=300)
        return out.stdout.strip() + (("\n[stderr] " + out.stderr.strip()) if out.stderr.strip() else "")
    except Exception as e:
        return f"(squiggle run failed: {e})"


# ---------------------------------------------------------------------------
# Python-side aggregation (cross-check) and report
# ---------------------------------------------------------------------------
def national_samples(per_size):
    tot = np.zeros(B)
    for sz in (1, 2, 3, 4):
        tot += N_BY_SIZE[sz] * per_size[sz]
    return tot


def summarise(name, per_size):
    tot = national_samples(per_size)
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"  National annual total (£): mean=£{tot.mean()/1e9:,.2f}bn  "
          f"median=£{np.median(tot)/1e9:,.2f}bn")
    print(f"  90% interval: £{np.percentile(tot,5)/1e9:,.2f}bn  –  £{np.percentile(tot,95)/1e9:,.2f}bn")
    print(f"  per-size contribution (mean £bn):")
    for sz in (1, 2, 3, 4):
        c = (N_BY_SIZE[sz] * per_size[sz]).mean()
        print(f"    {SIZE_LABELS[sz]:<7} £{c/1e9:,.2f}bn  (£{per_size[sz].mean():,.0f}/business x {N_BY_SIZE[sz]:,})")
    return tot


tot_t = summarise("(A) TYPE-BASED decomposition", boot_type)
tot_f = summarise("(B) FREQ-BASED decomposition", boot_freq)

print(f"\n{'=' * 78}\nCROSS-CHECK: type-based vs freq-based\n{'=' * 78}")
print(f"  type-based mean £{tot_t.mean()/1e9:,.2f}bn   vs   freq-based mean £{tot_f.mean()/1e9:,.2f}bn")
print(f"  ratio (type/freq): {tot_t.mean()/tot_f.mean():.2f}x")

print(f"\n{'=' * 78}\nSQUIGGLE ENGINE (headless CLI) — official aggregation\n{'=' * 78}")
if SQ_CLI:
    print(f"  CLI: {SQ_CLI}")
    print("  [type-based]", run_squiggle(f"{BUILD_DIR}/national_typebased.squiggle"))
    print("  [freq-based]", run_squiggle(f"{BUILD_DIR}/national_freqbased.squiggle"))
else:
    print("  squiggle CLI not found; .squiggle files written to", BUILD_DIR)

print("\nDone.")
