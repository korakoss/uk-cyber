# Project Notes

## Goal

Produce a synthetic estimate of **total annual cybercrime costs across all UK businesses**. Charities are not the primary target but the charity dataset may be used as supporting evidence for modeling assumptions shared with the business side.

---

## Core Findings (high-level summary)

These are the central empirical and modeling conclusions. Details and scripts are in the sections below.

**1. i.i.d. bridge model is rejected.**
The survey records cost of the single most disruptive attack, not total annual cost. The natural bridge (recover per-attack distribution from observed maxima via order statistics) requires E[cost] to increase with attack frequency. It does not — the relationship is reversed in all 4 size bands, and within every individual attack type. The negative freq–cost correlation is a genuine feature of the data, not a compositional artefact.

**2. For phishing, a 2-component mixture model is well-supported.**
Phishing is the most common attack type (~57% of incidents by disrupta). Within each size band, the (freq group × cost band) distribution matrix has rank ≈ 2 (variance explained by 2 SVD components: Micro 99.6%, Small 99.0%, Medium 94.2%). The two extracted components are interpretable: F_T (targeted/spear-phishing — near-zero no-cost, extends to high bands) and F_M (mass-market spam — ~60% no-cost, concentrated in cheap bands). For impersonation, rank-2 holds but the two components are nearly identical — no real mixture signal. Other attack types untested.

**3. freq=1 firms need no bridge.**
~20% of attacked firms reported exactly one incident. For these, the reported damage_bands cost IS the total annual cost. They are a clean anchor with no modeling ambiguity.

**4. Lognormal fits cost well within cells, but sigma is not stable.**
A zero-inflated lognormal fits the banded cost distribution reasonably (chi-sq 1–11, df≈6) within individual (size, freq) cells. Both mu and sigma vary across cells (sigma range 1.3–3.8); a shared-shape model won't work. P(cost > £500k) is negligible in most cells (< 0.2%) — the open-ended top band is not a major source of uncertainty except for Large firms where data is too sparse.

**5. Prevalence follows a logistic-in-log-size curve.**
Fraction of businesses attacked: 40% (Micro) → 51% (Small) → 66% (Medium) → 69% (Large). A logistic regression in log(employees) fits the four points with RMSE 2.2%. Weighted and unweighted estimates are nearly identical. Within the four known strata, empirical proportions are sufficient; the parametric model adds nothing in practice.

**6. Data quality note.**
damage_bands contains a non-integer SPSS special code (999.62004) that must be filtered with `>= 100`, not exact value matching. Fixed in proc.py. Also: freq encodes periodicity bands (once/monthly/weekly/daily), not literal attack counts — analyses treating freq values as counts are invalid.

---

## Estimation Pipeline — Current Status

**Formula:** Total annual UK cybercrime cost = Σ_size N(size) × P(attacked|size) × E[total annual cost | attacked, size]

| Step | What | Status |
|------|------|--------|
| 1 | N(size): UK business count per size band | Not started. Source: ONS UK Business Population Estimates |
| 2 | P(attacked\|size): prevalence | **Done.** Micro 40%, Small 51%, Medium 66%, Large 69%. Use empirical values directly. |
| 3 | P(freq\|size, attacked): freq distribution by size | **Not done.** Used freq as conditioning variable throughout but never examined how its distribution shifts by size band. Needed to sample freq in the simulation. Open: parametric ordinal model vs. four empirical distributions. |
| 4 | P(damage_bands\|freq, size): cost distribution | **Analysis done.** Lognormal fits well within cells; per-cell (mu, sigma). Handles within-band interpolation and top-band tail. |
| 5 | Bridge: damage_bands → total annual cost | **Done for phishing.** Bridge ≈ 1.02 for type M (~95% of firms); close to 1 for type T at low freq. Conservative lower bound (total = max) quantitatively supported. |
| 6 | Squiggle simulation and aggregation | Not started. Blocked on Steps 1 and 3. |

---

## Estimation Approach (high-level, agreed)

### Decomposition
Estimate per stratum (business size band):
1. **Prevalence** — fraction of businesses attacked (empirical weighted proportions per stratum)
2. **Frequency × Cost (joint)** — conditional on being attacked, sample (freq, cost_band) pairs from the *joint* empirical distribution per stratum (not independently, since freq and cost are correlated — confirmed by prior analysis)

### Multiple estimates in parallel
- **Conservative (lower bound):** minimal assumptions, mostly empirical; count only the largest attack per entity (ignoring freq−1 other attacks); use band lower bounds or midpoints for within-band cost
- **Pessimistic (upper bound):** treat the observed cost (which is actually the maximum attack) as if it were the typical attack cost, then multiply by freq — this overestimates by construction
- **Best estimate:** sample from the empirical joint (freq, damage_bands) per size stratum; convert damage_bands to £ via lognormal within-band interpolation; apply bridge multiplier ≈ 1.02 (validated by mixture model for phishing). The order-statistics bridge was explored and rejected — see Bridge Formalisation section.

### Weighted vs. unweighted
Every analysis should be run in two variants: one using the survey's statistical adjustment weights, one on raw unweighted rates. Divergence between variants is itself informative.

### Monte Carlo / Squiggle
The final simulation stage will use **Squiggle** (natively Monte Carlo). Python handles data analysis and fitting; Squiggle handles the simulation and produces distributional estimates (not point estimates).

---

## Data variables (key findings from codebook)

### damage_bands — primary cost variable (89% coverage among attacked)
"Estimate of the total cost for the **most disruptive** cyber security incident in the last 12 months." Band labels:
- 1 = No cost incurred
- 2 = <£100
- 3 = £100–£500
- 4 = £500–£1,000
- 5 = £1,000–£5,000
- 6 = £5,000–£10,000
- 7 = £10,000–£20,000
- 8 = £20,000–£50,000
- 9 = £50,000–£100,000
- 10 = £100,000–£500,000
- (11 = £500,000+, presumed)

Special codes: 997=Don't know, 999=Refused, plus a non-integer variant (999.62004) from SPSS — filter with `damage_bands >= 100`, not exact isin.

### crimecost_bands / notfraudcost_bands — total annual cost (very sparse)
- `crimecost_bands`: "Total cost of all crimes (including fraud)" — total annual cost per firm. Only 14.8% coverage (163/1101 attacked businesses). Too sparse to use as primary variable.
- `notfraudcost_bands`: "Total cost of all cyber crimes other than fraud." 10.3% coverage.
- These ~163 firms with valid `crimecost_bands` can be used to **calibrate / validate the bridge** by comparing total cost vs. single-incident cost directly.

### Attack type variables
- `type1`–`type16` (binary, Q53A): which attack types occurred (ransomware, malware, DoS, hacking, phishing, impersonation, unauthorised access, etc.)
- Type-specific total cost variables in COST_COLNAMES: `ranscost_bands`, `hackcost_bands`, `doscost_bands`, `viruscost_bands`, `fraudcost_bands` — these are total annual costs per attack type (not just largest attack)
- `freq` (Q54): attack frequency band — 1=once only, 2=>once but <monthly, 3=~monthly, 4=~weekly, 5=~daily, 6=several times/day

---

## Key Modeling Challenges

### Cost band problem
The cost variable is categorical (bands). Two sub-issues:
- **Interior bands:** within-band distribution assumption is unavoidable. Candidates: uniform (simple baseline), lognormal (theory-motivated, must be validated by checking predicted vs. observed band proportions). Run both as sensitivity variants.
- **Top open-ended band:** requires a tail assumption (Pareto or lognormal extrapolation). High stakes because large events dominate total cost estimates.

### Bridge problem (most important)
The survey records the cost of the **single largest attack**, not all attacks. To get total annual cost we need to recover the underlying attack cost distribution F from observed maxima.

**What `freq` actually encodes (from codebook, Q54):**
- 1 = Once only
- 2 = More than once but less than once a month (~2–11 attacks/year)
- 3 = Roughly once a month (~12/year)
- 4 = Roughly once a week (~52/year)
- 5 = Roughly once a day (~365/year)
- 6 = Several times a day (~1000+/year)

These are periodicity bands, not counts. This has two implications:
1. For freq=1 ("once only"): no bridge needed — one attack occurred, its cost is the total. This is a large clean subgroup (215 firms) where reported cost = total annual cost directly.
2. For freq≥2: the implied attack count k is itself uncertain (especially band 2, which spans 2–11). Converting band to count is a modeling step with its own uncertainty.

**Consistency checks run (src/estimation/consistency_checks.py):**
- *Monotonicity:* computed E[D | K=k] per size stratum and checked non-decrease in k. Violated in all four size bands — freq=1 firms consistently report higher mean cost bands than freq=2 firms, opposite of what i.i.d. predicts.
- *Chi-square GOF:* DISCARDED — was incorrectly treating freq band values as literal attack counts k and computing F(b)^k. Invalid given what freq actually encodes.

**Substantive interpretation of monotonicity failure:** Firms attacked more frequently tend to face cheaper attacks — consistent with a "many nuisance attacks vs. single targeted attack" pattern (e.g., phishing vs. ransomware). This means the i.i.d. assumption likely fails: the underlying cost distribution F depends on frequency, not just size. The bridge approach needs rethinking.

**Data note:** `damage_bands` has a non-integer special code 999.62004 that was not caught by the original integer filter (997, 999). Fixed in proc.py to filter `damage_bands >= 100`. All prior scripts using isin([997.0, 999.0]) are affected.

### Lognormal as candidate
Theoretically motivated for financial losses. Must be validated against data before use. Charity dataset can provide additional validation evidence if it fits there too.

---

## Planned Next Analyses

### A. damage_bands vs crimecost_bands comparison (bridge calibration) — DEAD END
Investigated and closed. Key findings:
- crimecost subsample (154 firms) is highly non-representative on damage_bands: "No cost" firms are 49% of full sample but only 11% of subsample — they don't bother reporting total costs.
- freq=1 consistency check: only 43% of once-only firms have matching bands; 50% have crimecost_bands LOWER than damage_bands by up to 9 bands.
- Root cause: damage_bands = broad all-in cost for single most disruptive incident (external payments + staff time + damage/disruption). crimecost_bands = direct financial losses only across all incidents. Different cost concepts entirely — not comparable for bridge calibration.

### B. Per-attack-type breakdown — COMPLETED (see also analyses 4 & 5 below)
Hypothesis tested: the negative freq–cost correlation is a compositional artefact (high-freq firms dominated by cheap phishing). Result: hypothesis rejected.

Key findings (src/estimation/attack_type_breakdown.py):
- Phishing does dominate high-freq firms (65% at once-only → 95%+ at weekly/daily), so composition effect exists.
- But monotonicity failure persists in every individual attack type — ransomware, phishing, DoS, impersonation, all of them.
- For phishing-only firms (cleanest subgroup, n=384): mean damage_band decreases monotonically once → weekly (1.71 → 1.38) before recovering at several/day. Clear violation.
- The negative freq–cost relationship is a genuine feature of the data, not a compositional artefact.

Conclusion: i.i.d. order-statistics bridge model is robustly rejected. See analyses 4 & 5 for further investigation.

**Analysis 4 (src/estimation/attack_type_breakdown.py) — full distribution by freq group within type:**
- Phishing (n=847): once-only group has heavier upper tail (P(damage ≥ £100k) = 2.3% vs 0.4–0.8% for higher-freq groups). "No cost" fraction lower for once-only (39% vs 50–51%). Consistent with once-only = targeted spear-phishing, high-freq = commodity spam.
- Impersonation: non-monotone — "weekly or more" group has heavier tail than medium-freq. Undermines simple composition story.
- Ransomware: too small (n=67) to conclude reliably; once-only does show 20% at £100k+ vs ~3% for higher-freq.
- Key pattern: high-freq groups are concentrated at "No cost" / low-cost bands. The extra k−1 attacks beyond the maximum are mostly cheap or free.

**Analysis 5 — type-specific total cost vs damage_bands for single-type firms:**
Cell sizes too small (most types n<3 with both variables valid). Approach closed.

**Key implication for bridge:** The bridge correction (total − maximum) may be small precisely for the groups where freq is high, because most of their additional attacks cost nothing. Formalising this argument is the current focus — see Bridge Formalisation section.

## Bridge Formalisation

The i.i.d. order-statistics bridge is rejected. The alternative argument for using damage_bands as a proxy:

**The key decomposition:**
Total annual cost = cost of maximum attack + sum of costs of remaining k−1 attacks.

- For freq=1: the second term is zero. damage_bands = total cost exactly.
- For freq>1: the second term is positive but potentially small — because analysis 4 shows high-freq groups are concentrated at "No cost" and low-cost bands, meaning most additional attacks contribute little.

**Three estimate tiers:**
1. **Conservative (lower bound):** total = damage_bands for all firms. Exact for freq=1; underestimates for freq>1 by ignoring additional attacks. Defensible because additional attacks are mostly cheap.
2. **Pessimistic (upper bound):** total = k_implied × damage_bands, treating the maximum cost as typical. Overestimates because the maximum is higher than typical by construction.
3. **Best estimate:** total = damage_bands + E[sum of k−1 additional attacks]. Needs a model for the per-attack cost distribution below the maximum.

**Formalising the best estimate — proposed approach:**
For each freq group, the "No cost" fraction in damage_bands distribution approximates the per-attack probability of zero cost (treating damage_bands as drawn from the per-attack distribution — a rough approximation). From this, we can bound or estimate the expected cost of additional attacks. The implied multiplier m(freq) = E[total] / E[maximum] stays close to 1 when the "No cost" fraction is high.

Alternatively: run sensitivity analysis with m ∈ {1, 1.5, 2, 5} applied per freq group, and present the range as the uncertainty band around the conservative estimate.

**Current status:** Approach agreed in principle. Not yet implemented.

## Mixture Model Test (src/estimation/mixture_model.py)

### Hypothesis
Under a 2-component mixture model (targeted F_T + mass F_M), all freq-conditional cost distributions p_f lie on a 1D affine subspace of R^10. Structural implication: the (freq group × band) matrix P has rank ≤ 2. Tested via SVD.

### Results

**Phishing (disrupta=6), all sizes pooled (n=579):**
- 2 components explain **99.0%** of variance; 3rd adds only 0.6%
- **Verdict: strongly consistent with rank-2**
- Extracted F_T (targeted): 0% no-cost, mean band 4.0, meaningful probability mass at £20k–50k (8.1%) and £100k–500k (3.9%) — genuine spear-phishing pattern
- Extracted F_M (mass): 58.5% no-cost, mean band 2.0 — commodity spam, mostly zero-cost to targeted firm
- Mixing weights π_f: once-only 12.9% targeted → >once<monthly 5.7% → monthly/weekly/daily 0% → several/day 22.3% (last value suspicious given n=47)

**Impersonation (disrupta=5), all sizes pooled (n=228):**
- 2 components explain **99.5%** of variance
- **Verdict: consistent with rank-2, but F_T ≈ F_M** (mean bands 2.55 vs 2.42, no-cost 38% vs 41%)
- The structure is essentially rank-1: no evidence of two distinct cost regimes for impersonation
- Mixing weights are poorly identified (jump from 1.0 to 0.0 between freq groups)

### Interpretation
- The mixture model is **supported for phishing** with interpretable components
- For impersonation, rank-2 holds but is degenerate — no real mixture signal
- The freq→π_f pattern for phishing is partially sensible (higher-freq = more mass-market), but the several/day outlier needs investigation (n=47, possibly sampling noise)

### Per-size-band results (phishing)

| Band | n_freq_grps | Var(2 comp) | Verdict | F_T mean band | F_M mean band | F_T no-cost | F_M no-cost |
|------|-------------|-------------|---------|---------------|---------------|-------------|-------------|
| Micro (1-9) | 6 | 99.6% | CONSISTENT | 3.34 | 1.68 | 0% | 66.4% |
| Small (10-49) | 5 | 99.0% | CONSISTENT | 2.54 | 1.90 | 40.7% | 59.4% |
| Medium (50-249) | 5 | 94.2% | MARGINAL | 3.15 | 2.11 | 14.9% | 54.8% |
| Large (250+) | — | — | too sparse | — | — | — | — |

π_f patterns: once-only consistently has the highest targeted fraction (micro 10.5%, small 23.6%, medium 64.7%). Monthly/weekly consistently near 0. "Several/day" shows an anomalous uptick (micro 29.3%, small 18.1%, medium 36.6%) — possibly noise at small n (n=21/12/8 respectively), or a genuine phenomenon.

Key observation: F_T and F_M are clearly separated for Micro (F_T has 0% no-cost) but less so for Small (F_T has 41% no-cost). This may reflect that targeted attacks on very small firms reliably incur some cost, while "targeted" attacks on slightly larger firms still often fail.

### Next steps (not yet run)
- Weighted variant (current script uses raw proportions only)
- Investigate "several/day" anomaly further if it matters for downstream modeling
- Decide whether to adopt mixture model as the bridge framework or continue exploring alternatives

---

## Prevalence Analysis (src/estimation/prevalence.py)

Fraction of businesses attacked, by size band. "Attacked" = freq ∈ {1–6}.

| Band | n | Unweighted | Weighted |
|------|---|-----------|---------|
| Micro (1-9) | 1014 | 40.6% | 39.7% |
| Small (10-49) | 565 | 51.0% | 49.4% |
| Medium (50-249) | 413 | 65.9% | 64.7% |
| Large (250+) | 188 | 68.6% | 67.8% |

Weighted and unweighted are nearly identical — survey adjustments barely affect prevalence.

Monotone in size (both variants). Overall prevalence ≈ 50%; the 49.5% of firms with missing/DK freq are essentially the non-attacked population (freq is only asked of attacked firms).

**Logistic-in-log-size fit:** P(attacked) = sigmoid(a + b·log(midpoint)), a≈−0.67, b≈0.25. RMSE 2.2pp, max residual 3pp — essentially perfect fit for a 2-parameter model on 4 points. Curve flattens at top (Large only marginally higher than Medium), consistent with sigmoid saturation.

**Implication:** For a stratified simulation using the 4 empirical strata, empirical proportions are fully sufficient and more defensible. The logistic fit is well-supported but adds no practical value within the data range. It would only matter for interpolation to continuous size, which is not planned.

---

## Cost Distribution: Lognormal Fit (src/estimation/cost_distribution.py)

Model: zero-inflated lognormal. P(cost=0) = p0 (fixed at observed no-cost fraction). Conditional on nonzero cost: Lognormal(mu, sigma), right-censored at £500k for fitting.

**Per (size, freq) cell chi-square values (df≈6):**
Most cells have chi-sq in the range 1–11 — acceptable. Lognormal is a legitimate parametric model for cost within individual cells.

**Pooled across freq (per size band):**

| Band | n | chi-sq | mu | sigma | p0 | P(>£500k) |
|------|---|--------|----|-------|----|-----------|
| Micro | 376 | 12.0 | 5.29 | 2.21 | 0.58 | 0.01% |
| Small | 265 | 3.2 | 5.06 | 2.99 | 0.47 | 0.18% |
| Medium | 232 | 7.0 | 5.67 | 2.67 | 0.43 | 0.15% |
| Large | 109 | 13.6 | 6.79 | 2.85 | 0.34 | 0.87% |

Pooled fit is good for Small/Medium, marginal for Micro and Large (expected: pooling across freq mixes different lognormals).

**freq=1 only (cleanest subgroup — no bridge needed):**

| Band | n | chi-sq | mu | sigma |
|------|---|--------|----|-------|
| Micro | 77 | 3.0 | 5.76 | 1.80 |
| Small | 59 | 7.5 | 5.27 | 3.50 |
| Medium | 50 | 5.2 | 4.62 | 3.12 |
| Large | 14 | 4.5 | 90.8 | 47.1 — degenerate (n too small) |

**Key findings:**
- **sigma is not stable** across cells (range 1.31–3.79, median 2.68). Cannot assume common shape — both mu and sigma must vary per cell.
- **P(cost > £500k)** is negligible (< 0.2%) for most cells. Only exception: Large | freq=1, but n=14 makes that estimate useless.
- **mu increases with firm size** (roughly), consistent with larger firms facing higher-cost incidents.
- Large | freq=1 (n=14): completely unreliable; must be pooled or treated empirically.

**Implication for estimation:**
Lognormal is a viable parametric model per cell. It provides: (i) principled within-band interpolation; (ii) a tail extrapolation for the top open-ended band (small effect except for Large firms). Cannot assume shared sigma — use per-cell (mu, sigma). For sparse cells (Large | freq=1), use empirical proportions or partial pooling.

---

## Mixture Bridge: Analytical Computation (src/estimation/mixture_bridge.py)

### Setup
Extends the mixture model to derive a principled bridge multiplier. Assumes i.i.d. attacks *within* each type (a much weaker assumption than i.i.d. across all firms, which was rejected).

Key insight: F_T and F_M from the SVD are distributions of the **maximum** attack cost, not individual attacks. The per-attack distribution is G = F^(1/k) (order-statistics inversion). The bridge is:

- E[max] = E[X | F] (mean of the observed max distribution)
- E[total] = k × E[X | G] = k × Σ_b (F_CDF(b)^(1/k) − F_CDF(b−1)^(1/k)) × midpoint(b)
- Bridge = E[total] / E[max]

### Results

**Type M (mass-market, ~95% of phishing firms):**
Bridge ≈ 1.02 for all k from 5 to 3000. Essentially flat. Reason: G_M = F_M^(1/k) concentrates near zero for large k — the k-1 additional attacks beyond the maximum cost almost nothing. **Conservative lower bound (total = max) is well-supported for type M.**

**Type T (targeted, ~5% of phishing firms):**
Bridge grows sharply with k: k=1 → 1.00, k=6 → 1.07, k=100 → 1.45, k=1000 → 5.0. Weighted bridge_T = 3.09, but this is driven almost entirely by the freq=6 anomaly (P(freq=6|T)=0.346 × bridge(1500)=7.0). Excluding freq=6: weighted bridge_T ≈ 1.03.

**Overall mixture bridge = 1.70**, but if the freq=6 anomaly is noise (likely at n=47): overall bridge ≈ 1.02.

**Bayes-derived P(freq | type):**
- Type T: 33.6% once-only, 31.9% >once<monthly, 0% monthly/weekly/daily, 34.6% several/day (anomalous)
- Type M: mostly monthly/weekly/daily with ~12.5% once-only
- pi_overall ≈ 5.2% of phishing firms are type T

**Poisson fit for k | type:**
Type T: λ_T = 1.9 — bad fit (predicts 0 mass at freq=6, but 34.6% is there). Bimodal structure inconsistent with unimodal Poisson. Type M: λ_M = 8.8 — very poor fit (predicts 82% in freq=2, observed 29%).

### Key conclusion
The conservative lower bound is quantitatively validated for type M (95% of firms). The remaining uncertainty is whether type T firms with freq=6 (several/day targeted attacks) are real — if yes, that subgroup has a meaningful bridge correction (~7×). If the freq=6 type-T assignment is noise, the overall correction is ~2%.

---

## Open Questions

These are unresolved issues that a future session should be aware of before diving in.

**1. Freq-as-conditioning-variable vs. freq-as-distribution (source of past confusion)**
Throughout the analyses, `freq` has been used in two distinct roles that are easy to conflate:
- *Conditioning variable for cost:* P(damage_bands | freq, size) — done, lognormal fits well.
- *Distribution to be modelled:* P(freq | size, attacked) — NOT done. This is what the Squiggle simulation needs to sample from: first draw a freq band, then draw a cost. These are separate questions. The parametric vs. empirical decision for freq refers to this second role only.

**2. Freq=6 type-T anomaly**
The mixture model assigns 34.6% of "targeted" (type T) mass to the several/day frequency group (n=47 phishing firms). If real, this group has a bridge multiplier of ~7×. If noise (probable given small n), the overall bridge is ~1.02. Not confirmed either way; the cell is too small to test cleanly.

**3. Mixture model scope**
The 2-component (targeted/mass-market) structure is validated for phishing (dominant type) and degenerate for impersonation. Other attack types (ransomware, malware, DoS, hacking) have not been tested. May matter if non-phishing firms dominate the cost tail.

---

## Planned Next Steps

1. **Freq distribution by size** — examine P(freq | size) empirically. Does it shift meaningfully across size bands? Then decide: parametric ordinal model vs. four separate empirical distributions over the 6 categories.
2. **Settle the joint sampling model** — once freq-by-size is understood, nail down the (freq → cost) sampling procedure for the Squiggle simulation.
3. **Fetch ONS business counts** by size band (external data, needed for Step 1 of the pipeline).
4. **Build the Squiggle model** with three tiers: conservative (bridge=1, empirical distributions), best estimate (lognormal within-band + bridge≈1.02), upper bound (bridge=k_implied).
5. **Optional:** run mixture model for other attack types (ransomware, malware) to check whether the 2-component structure generalises beyond phishing.
6. **Optional:** acquire an additional year of data as a robustness check; use charity data to validate shared modeling assumptions.

---

## Script Index

All scripts run from the project root: `source .venv/bin/activate && python3 <path>`. Data loaded via `proc.get_business_data()` / `proc.get_charity_data()`.

| File | Topic | Key outputs |
|------|-------|-------------|
| `proc.py` | Data loading | `get_business_data()`, `get_charity_data()`, `SPECIAL_CODE_THRESHOLD=100` |
| `src/estimation/consistency_checks.py` | i.i.d. bridge test | Monotonicity test (violated everywhere); discovery of 999.62004 special code |
| `src/estimation/bridge_calibration.py` | crimecost vs damage_bands | Avenue closed — different cost concepts, non-representative subsample |
| `src/estimation/attack_type_breakdown.py` | Cost by attack type and freq | Monotonicity failure persists within every type; disrupta analysis; full distribution by freq group |
| `src/estimation/bridge_multiplier.py` | Empirical bridge upper bound | MC multipliers per (size, freq) cell; freq=4+ gives implausible results (known upper bound only) |
| `src/estimation/mixture_model.py` | SVD rank-2 test | F_T/F_M extraction; π_f weights; per-size-band results for phishing and impersonation |
| `src/estimation/mixture_bridge.py` | Analytical bridge via mixture | Corrected bridge formula (G=F^(1/k)); bridge≈1.02 for type M; Poisson fit for k\|type (failed) |
| `src/estimation/prevalence.py` | Prevalence by size | Empirical proportions; logistic-in-log-size fit (RMSE 2.2%) |
| `src/estimation/cost_distribution.py` | Lognormal fit to cost bands | Per-cell (mu, sigma); chi-sq GOF; P(cost>£500k) per cell |

---

## Status
Pipeline structure agreed. Core empirical analyses complete (prevalence, lognormal cost fit, mixture model, bridge). Main open items before simulation: Step 3 (freq by size) and Step 1 (ONS counts).
