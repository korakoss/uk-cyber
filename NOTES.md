# Project Notes

## Handoff (as of 2026-07-07, updated same day)

**What the last session did:** Two threads. (1) Step 3 exploratory pass — examined P(freq|size), found Large firms have a genuinely different frequency profile than Micro/Small/Medium. (2) A deep validation pass on the phishing mixture model (F_T/F_M), using real per-attack count variables (`phishcon_bands`, `phisheng_bands`) discovered mid-session and now loaded via `proc.py`.

**This session added:** a further dig into the freq=6 anomaly (see item 2 below and the dated follow-up under "Freq=6 anomaly investigated"). Also produced `HANDOFF.md` — a comprehensive, non-code-oriented synthesis of all findings for handing the analytical narrative to a paper-writing collaborator or another instance without needing to re-derive from scripts/NOTES.

**Results, most to least important:**
1. **The phishing mixture model survived independent validation.** Regrouping the SVD rank-2 test by real engagement counts instead of freq band reproduces almost the same targeted (F_T) component and gives clean, monotonic mixing weights — strong evidence the targeted/mass-market structure is real, not an SVD artifact. Not yet wired into `mixture_bridge.py` (still uses freq-based weights).
2. **The freq=6 bridge anomaly: confirmed real, then its mechanism dug into further and found to be mixed, not resolved.** It's confirmed real (survives raw-data, weight, and outlier checks; replicates across 3 analyses). First engagement-based check suggested the targeted-attack mechanism didn't apply (engagement not elevated). Follow-up found the *targeted-rate* variable (`phishcon_bands`) IS elevated at freq=6 and a strong cost predictor there — but 2 of the 3 extreme-cost outliers driving the tail are multi-vector incidents (phishing plus other attack types), not pure phishing. So the anomaly is real but reflects a blend of mechanisms, not one clean story — still the single highest-leverage open item determining whether the phishing bridge is ~1.02 or higher. See `src/estimation/freq6_anomaly.py`.
3. **Step 3 pooling decision is still open**: Micro/Small/Medium share one freq distribution (statistically indistinguishable); Large is distinct (p=0.014). Not yet decided how to structure this for sampling.

**Recommended next step:** the freq=6 subgroup (item 2) is very unlikely to resolve further with more analysis of this same n=47 (n=3 for the outliers specifically) — treat it pragmatically (a bounded sensitivity range around the bridge multiplier) rather than continuing to chase it, and switch to a more tractable item: the Step 3 pooling decision, or starting the ONS business-count fetch (Step 1), both of which are pure forward progress toward the Squiggle simulation. If instead the priority is deepening the phishing bridge work, the natural next move is promoting the engagement-based mixing weights into `mixture_bridge.py`'s actual multiplier calculation.

**Full list of open decisions:** see "Open Decisions (quick reference)" below. **Full technical detail:** see the dated subsections under Mixture Model Test and Freq Distribution by Size.

**Orientation:** all scripts run via `source .venv/bin/activate && python3 <path>` from project root; data via `proc.get_business_data()`. Read this whole file before diving in — it's the authoritative record, not a summary of it.

---

## Goal

Produce a synthetic estimate of **total annual cybercrime costs across all UK businesses**. Charities are not the primary target but the charity dataset may be used as supporting evidence for modeling assumptions shared with the business side.

---

## Core Findings (high-level summary)

These are the central empirical and modeling conclusions. Details and scripts are in the sections below.

**1. i.i.d. bridge model is rejected.**
The survey records cost of the single most disruptive attack, not total annual cost. The natural bridge (recover per-attack distribution from observed maxima via order statistics) requires E[cost] to increase with attack frequency. It does not — the relationship is reversed in all 4 size bands, and within every individual attack type. The negative freq–cost correlation is a genuine feature of the data, not a compositional artefact.

**Correction/refinement (2026-07-09):** the original "mean damage_band declines with freq" and "P(damage≥high band) declines with freq" statistics (`attack_type_breakdown.py`) are computed *unconditionally* — the denominator includes "no cost" (damage_bands=1) firms, so a rising no-cost share mechanically drags those statistics down on its own. Decomposed this in `consistency_checks.py` (new section, "DECOMPOSITION"): separated the no-cost share (which does climb cleanly with freq: 38.5%→58.7% pooled) from the mean cost *conditional on having any cost at all* (which does NOT show a clean decline — it's noisy/flat, e.g. 3.68→3.47→3.76→3.36→4.22 pooled; similar for phishing alone). So "costly incidents get less costly as frequency rises" is not well-supported once you strip out the no-cost-share effect — that part of the original claim was overstated. What remains solid: (a) a majority of frequently-attacked firms have literally zero cost across everything that happened that year (near-tautological, since a zero maximum requires every incident to be zero), and (b) among the minority with a nonzero year, the severity of that worst incident doesn't get systematically cheaper with frequency (roughly flat, not declining). This is actually a cleaner basis for "extra attacks are mostly free" than the original framing — a uniform whole-distribution downshift would have predicted the conditional mean to *also* decline with frequency, and it doesn't. Still cannot directly verify what the non-maximal incidents look like for a firm that DID have a costly year — that remains an assumption, not a measured fact, especially for non-phishing types where no independent validation (like phishing's engagement/targeting data) exists. Also confirmed via re-run: the freq-cost anticorrelation holds in all 4 size bands individually (not just Large) — Large firms differ only in their freq *distribution shape* (Step 3 finding, a separate question), not in whether this anticorrelation applies to them specifically.

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

**7. Phishing's mixture model survives an independent validation attempt, using real (not inferred) attack data.**
Phishing is both the dominant attack type (~57% of incidents) and the *only* type with high-coverage per-attack count data: `phishcon_bands` (# targeted/personalised attacks, 76% coverage) and `phisheng_bands` (# attacks someone engaged with, 83% coverage) — other types have equivalent count variables but only 6-20% coverage, too sparse to use. Regrouping the SVD rank-2 test by real engagement counts (instead of freq band, which was the only grouping used originally) reproduces essentially the same F_T (targeted) component and gives a clean, monotonic mixing-weight curve (0% → 20% → 65% targeted as engagement rises) — freq-based mixing weights were noisy and non-monotonic. This is real evidence the 2-component structure isn't an SVD artifact. It does *not* resolve the single biggest open question driving bridge uncertainty — the freq=6 ("several times a day") targeted anomaly (n=47, see Open Questions #2) — which still swings the overall phishing bridge multiplier between ~1.02 and ~1.7. See the "Real per-attack count data" subsection under Mixture Model Test for full detail.

---

## Open Decisions (quick reference)

Consolidated list of things "on the table" — genuine open choices, not yet settled. Each links to fuller detail elsewhere in this file.

1. **Freq=6 targeted anomaly (n=47)** — investigated and confirmed real (not noise/artefact), but the mixture model's targeted-attack mechanism doesn't match it (engagement isn't elevated there). Still highest-leverage open question: what's the correct bridge multiplier for this subgroup, given the ~7x type-T formula may not actually apply? See Open Questions #2, `freq6_anomaly.py`.
2. **Step 3 pooling decision — DECIDED (2026-07-08):** pool Micro/Small/Medium into one empirical freq distribution (they're statistically indistinguishable, p=0.014 only Large differs); Large stays separate. Applies to the freq marginal only — cost distributions remain per (size, freq) cell, unpooled. See Freq Distribution by Size section.
3. **Promote engagement-based grouping (`phisheng_bands`) into the bridge calculation** — `mixture_bridge.py` still uses freq-based π_f; the engagement-based mixing weights are cleaner (monotonic) but not yet integrated. See Mixture Model Test → "Real per-attack count data".
4. **Scope of the mixture model — CLOSED (2026-07-08, re-confirmed 2026-07-09):** validated for phishing only (57% of incidents); attempted for ransomware/DoS/hacking/malware and found infeasible (each type has only 21-33 attacked firms, too few once split across 6 freq bands to run the rank-2 test at all — not a power issue fixable by threshold tuning). A later full inventory of the 12-category `disrupta` variable found one more testable type — website/social takeover (n=31) — that had been skipped by oversight; tested 2026-07-09 and found degenerate (technically passes rank-2 at 98.6% but F_T≈F_M, same failure mode as Impersonation), so the phishing-specific conclusion stands. Non-phishing bridge treatment will need a different approach (e.g. `bridge_multiplier.py`'s empirical multiplier, or conservative/pessimistic bounds only). See Open Questions #3.
5. **ONS business counts by size** — external data fetch, not started; blocks Step 6 (simulation) entirely.
6. **Minor/optional:** weighted variant of the mixture model (currently unweighted only); per-size-band mixture rerun using engagement instead of freq.

---

## Estimation Pipeline — Current Status

**Formula:** Total annual UK cybercrime cost = Σ_size N(size) × P(attacked|size) × E[total annual cost | attacked, size]

| Step | What | Status |
|------|------|--------|
| 1 | N(size): UK business count per size band | Not started. Source: ONS UK Business Population Estimates |
| 2 | P(attacked\|size): prevalence | **Done.** Micro 40%, Small 51%, Medium 66%, Large 69%. Use empirical values directly. |
| 3 | P(freq\|size, attacked): freq distribution by size | **Decided.** Pool Micro/Small/Medium into one shared empirical freq distribution; Large kept separate. Cost distribution (Step 4) stays per (size, freq) cell, not affected by this pooling. Large's own thin ~daily cell (n=6) still unresolved. |
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

### Scope extension attempt: other attack types (2026-07-08) — infeasible, not just untested

Tried to repeat the rank-2 test (`run_mixture_test`, pooled across sizes, same as the phishing/impersonation runs) for the four remaining major attack types: ransomware (disrupta=1), other malware (disrupta=2), denial of service (disrupta=3), hacking (disrupta=4). None of the real per-attack count variables that validated phishing's structure exist at usable coverage for these types (6-20%, per earlier notes), so this would only have tested the structural rank-2 hypothesis itself, with no independent check available even if it ran.

**Result: could not run at all.** Total attacked-firm counts per type are small (ransomware n=28, other malware n=33, DoS n=27, hacking n=21) and spread across all 5-6 freq bands, e.g. ransomware: {once=4, >once<monthly=8, monthly=7, weekly=2, daily=5, several/day=2}. The rank-2 test needs at least 3 freq groups with a reasonable minimum n (8-10) each to be meaningful; every one of the four types has at most 1 band clearing that bar. This isn't a marginal-power issue that a lower threshold would fix — the total sample per type (21-33 firms) is simply too small once split six ways.

**Implication:** the mixture-model scope question (Open Questions #3) is now closed, not open — the structure is validated for phishing only, tested-and-rejected-as-infeasible (not merely "not yet tried") for the other four major types, and untested for the remaining minor types (unauthorized access variants, video-conf eavesdropping, website takeover — all smaller still). Any bridge treatment for non-phishing attack types will have to rely on something other than this mixture framework (e.g. the cruder empirical multiplier approach in `bridge_multiplier.py`, or treating non-phishing types with the conservative/pessimistic bounds only, without a best-estimate bridge). See `src/estimation/mixture_model.py`, appended section "SCOPE EXTENSION."

**Full `disrupta` inventory check (2026-07-09):** confirmed via `df['disrupta'].value_counts()` that the codebook (Q64A) actually defines 12 categories, not 6. Most of the untested "remaining minor types" are genuinely too small to ever be worth testing: unauthorised access by staff n=8, by outsiders n=3, by students n=0, video-conf/IM eavesdropping n=0, "other" catch-all n=12 (also not a coherent single attack type). But **website/social media/email account takeover (disrupta=11, n=31)** was comparable in size to ransomware/malware/DoS/hacking (23-39) and had a label already defined in the script's `DISRUPTA_LABELS` dict, yet was never actually passed into `run_mixture_test` — a real gap, not a deliberate exclusion. Added to both the full 6-group loop and the coarse 3-bucket loop in `mixture_model.py`.

**Result for website/social takeover:** full 6-group test still infeasible (only 1 freq group, once-only n=17, clears the n≥8 threshold — same failure mode as the other four). Coarse 3-bucket test *did* run cleanly this time (buckets n=17/6/8, all clearing n≥5) — and passed the rank-2 threshold better than any other non-phishing type tested: 98.6% variance explained by 2 components (vs. malware's 88.3% rejected, DoS's 95.1% marginal), nearly matching phishing's own 99%+ numbers.

**But it's degenerate, not a second phishing-like finding.** The extracted F_T/F_M components are almost identical (mean band 3.05 vs 2.95, no-cost fraction 21% vs 28%) — the same failure mode seen with Impersonation earlier: the test passes because there isn't much cost variation across the 3 groups to begin with (any two nearby components will approximate a low-variance matrix well), not because a genuine two-type structure was found. Confirmed by the mixing weights, which swing wildly and non-monotonically (π_f: 5% once → 0% occasional → 96% frequent, on buckets of n=6 and n=8) — unlike phishing's engagement-validated monotonic 0%→20%→65% curve, this looks like noise amplification from having two nearly-identical components rather than a real signal.

**Updated conclusion:** closing this gap didn't change the answer. Of 6 types with adequate-ish sample size to test at all (phishing, impersonation, ransomware, malware, DoS, hacking, website takeover — 7 actually, counting website takeover), only phishing shows a real, independently-validated two-type structure. Impersonation and website/social takeover both technically clear a rank-2 statistical bar but are substantively degenerate (components too similar to represent distinct attack types). Ransomware and hacking can't be tested even at coarse resolution; malware fails outright; DoS is marginal-and-untrustworthy. The phishing-specific read stands, now on a more complete accounting of the `disrupta` variable.

**Follow-up: coarser 3-bucket retry (2026-07-08) — doesn't rescue it.** Collapsed the 6 freq bands into 3 (once / occasional [2-11x/yr] / frequent [weekly+]) to see if a much weaker version of the test could run at all. Note: 3 groups is the minimum for a non-trivial test — 2 groups would trivially have rank ≤2 regardless of the data, so a "once vs. more-than-once" split (as originally floated) wouldn't test anything.

Result: still infeasible for ransomware (once-only bucket n=4, below even a loosened n≥5 threshold) and hacking (frequent bucket n=1 — hacking has no observations at all past freq=4 in this sample). Malware and DoS could be tested:
- Malware (buckets n=6/15/12): 2 components explain only 88.3% of variance (3rd adds 11.7%) — **REJECTED**, does not fit even the loosened rank-2 structure.
- DoS (buckets n=5/13/9): 95.1% (3rd adds 4.9%) — marginal, but the once-only cell is n=5 and shouldn't be trusted.

Also, with only 3 groups the maximum possible rank is 3, so "explained by top 2 components" is a far weaker test than the original 6-group phishing version (max rank 6, where 99% was a genuinely stringent result) — DoS's "marginal" pass is much less informative than it looks.

**Conclusion:** bucketing does not rescue the mixture approach for non-phishing types. Two of four types can't be tested at all even at this coarse resolution; of the two that can, one fails outright and the other only marginally passes on data too thin to trust. This is now good evidence that the targeted/mass-market mixture structure is a **phishing-specific finding**, not a general pattern that other attack types also have but that we lack the data to detect. Non-phishing bridge treatment should not assume an analogous mixture structure exists — use a different method (empirical multiplier upper bound, or conservative/pessimistic bounds only).

### Real per-attack count data: validating F_T/F_M against ground truth (src/estimation/phishing_count_validation.py, mixture_model.py)

Discovered that the survey has real per-attack count-band variables for phishing specifically (not literal totals, but high-coverage subsets): `phishcon_bands` ("number of specifically targeted [personalised] phishing attacks," 76% coverage) and `phisheng_bands` ("number of times someone engaged with a phishing attack," 83% coverage). Now loaded via `proc.py` (`PHISH_COUNT_COLNAMES`). No equivalent count variables exist with usable coverage for other attack types (hacking/DoS/ransomware/malware counts exist but only 6-20% coverage).

**Does "targeted" (personalised) predict cost?** Weakly. Spearman corr(phishcon_bands, damage_bands) = 0.29. `phishcon_bands` measures whether an attack *looked* targeted, which turns out to be common (49-69% of phishing firms report ≥1 targeted attack across all freq bands) and only loosely tied to cost outcome. This directly contradicts the SVD-inferred π_f (which was 0% for 3 of 6 freq bands) — the two "targeted" constructs are not the same thing and phishcon_bands should not be used as a literal ground-truth label for SVD type membership.

**Does engagement predict cost?** Yes, much better. Spearman corr(phisheng_bands, damage_bands) = 0.37. Firms with **zero** engagement: mean damage band 2.02, 57.1% no-cost — matches the SVD's F_M (mean 2.0, 58.5% no-cost) almost exactly, despite `phisheng_bands` never being used in the SVD fit. This is real, independent validation that F_M ("mass-market/commodity phishing, cheap or free") is a genuine feature of the data, not an SVD artifact.

**Rerunning the rank-2 test grouped by `phisheng_bands` instead of `freq`** (only 3 of 9 engagement levels had n≥8: None n=465, "1" n=62, "2-3" n=19 — higher levels too sparse):
- 2 components explain 99.0% of variance (identical to the freq-based pooled result) — rank-2 holds again with a completely different grouping variable.
- Extracted F_T: mean band 4.19, 0% no-cost — nearly identical to the freq-based F_T (4.0, 0%). **Two independent groupings converge on the same targeted component.**
- Extracted F_M: mean band 2.55, 42.2% no-cost — same direction as freq-based F_M (2.0, 58.5%) but not an exact match; likely reflects the thinner data (3 groups vs 6).
- **π_f is monotonic in engagement level: 0% → 19.6% → 65.2%** — a clean dose-response relationship, unlike the freq-based π_f which was noisy and non-monotonic (12.9% → 5.7% → 0% → 0% → 0% → 22.3% with an unexplained freq=6 spike).

**Implication:** engagement level looks like a mechanistically better (and now partially validated) organizing variable for the targeted/mass mixture than freq band. This strengthens confidence that F_T/F_M is real structure, and gives a lead for a monotonic mixing-weight model that the freq-based version couldn't offer. Not yet integrated into the bridge multiplier calculation (`mixture_bridge.py` still uses freq-based π_f) — that's the natural next step if this direction is pursued further. Also not yet extended to per-size-band breakdowns using engagement instead of freq.

### Freq=6 anomaly investigated (src/estimation/freq6_anomaly.py)

Direct test of whether the freq=6 targeted anomaly (Open Questions #2) is real or an SVD/small-n artefact, using raw data rather than the mixture reconstruction.

**The anomaly is visible in the raw data, not just the SVD extraction.** Unweighted mean damage_band by freq group: once-only 2.09 → >once<monthly 2.02 → ~monthly 1.76 → ~weekly 1.56 → ~daily 1.93 → **several/day 2.53**. The last group breaks the otherwise-declining trend and has the highest mean of any freq group, plus the only meaningful high-cost tail (6.4% at band≥8, vs. 0-1.2% elsewhere, n=47). This is now the third independent observation of the same pattern — also seen in `attack_type_breakdown.py`'s original phishing-only analysis (1.71→1.38 then recovery) and in the SVD-derived π_f.

**Not a weight or outlier artefact.** The 3 high-cost observations (bands 8, 8, 10) driving the tail have unremarkable survey weights (0.05-0.53, below the group's own max of 2.49) — real, low-weight rows, not a few over-weighted observations distorting the picture. Size composition of the freq=6 group (Micro 21, Small 12, Medium 8, Large 6) roughly matches the overall attacked-firm mix; 2 of the 3 outliers happen to be Large firms, but that's a weak signal at n=3.

**But real engagement data complicates the "targeted" interpretation.** If freq=6's elevated cost were the same phenomenon as once-only's targeted attacks, engagement should be elevated there too — it isn't. P(≥1 real engagement) by freq group: 30.7% (once-only) → 13.3% → 15.1% → 10.3% → 21.4% (daily) → **20.0%** (several/day) — lower than once-only, not higher. Within freq=6 itself, engagement still predicts cost strongly (mean band 3.67 for the 9 engaged firms vs. 2.25 for the 36 non-engaged) — consistent with the general finding — but the group as a whole isn't unusually engagement-heavy.

**Read:** the freq=6 cost anomaly is real — it survives raw-data, weight, and outlier checks, and replicates across three independent analyses — but the mixture model's assumed *mechanism* (that this is the same spear-phishing/targeted phenomenon seen at once-only, just at high implied k) is only partly supported. Engagement, otherwise the best real proxy for "targeted," isn't elevated at freq=6. More likely explanation: freq=6 firms are under chronic/sustained attack (consistent with the Step 3 finding that Large firms skew toward frequent attack), and a subset experience genuinely high-cost incidents through a different mechanism than one-off spear-phishing — not necessarily via more successful or more personalised individual attempts. At n=47 (9 engaged, 3 in the cost tail) this can't be pinned down further with this dataset.

**Implication for the bridge:** this reframes rather than resolves the open question. It's not "is freq=6 real or noise" (it's real) — it's "does the type-T bridge formula (which assumes freq=6 firms are extreme-k targeted attacks, giving up to a ~7x multiplier) correctly describe what's actually happening in this subgroup." Given the engagement mismatch, applying that formula here is on shakier ground than previously assumed; the true bridge correction for this subgroup is unresolved, not simply "noise, use 1.0" or "real, use 7x."

**2026-07-07 follow-up: `phishcon_bands` (targeted-count) resolves part of the puzzle; multi-vector attacks explain the rest.** Two further checks, appended to `freq6_anomaly.py` (sections 4-5):

1. **"Targeted" (phishcon_bands) is NOT depressed at freq=6, unlike engagement.** P(≥1 targeted attack) by freq group: 39.0% (once-only) → 50.3% → 43.4% → 48.9% → 42.9% → **52.5%** (several/day) — freq=6 has the *highest* targeted rate of any group, essentially the opposite pattern from engagement (which was 20.0% at freq=6 vs. 30.7% at once-only). And within freq=6, targeted status is a strong cost discriminator: no targeted attack → mean band 1.79 (68.4% no-cost); ≥1 targeted attack → mean band 3.57 (19.0% no-cost) — a bigger split than the engagement-based one (2.25 vs 3.67). So the "targeted" construct, as opposed to the "engaged" construct, *does* support the spear-phishing story at freq=6. `phishcon_bands` and `phisheng_bands` are measuring different things (an attack can look personalised without anyone falling for it), and they disagree specifically at freq=6.

2. **2 of the 3 high-cost (band≥8) outliers are multi-vector attacks, not pure phishing.** Row-level detail: (a) sizeb=4, band=8 — phishing + DoS + impersonation flagged, cost also recorded in `doscost_bands`; (b) sizeb=4, band=10 — phishing + malware + impersonation + unauth. access (outsiders) + video-conf eavesdropping + website/social-media takeover, cost also recorded in `viruscost_bands` and `tkvrcost_bands` (this firm has 6 concurrent attack types); (c) sizeb=2, band=8 — phishing only, no other type flagged, no other cost variable populated — the one clean single-vector case. So `disrupta=6` ("phishing was the most disruptive") doesn't mean phishing was the *only* thing happening — for 2 of 3 outliers driving the tail, the recorded cost likely reflects a broader multi-vector compromise, with the phishing engagement/targeting variables only describing one piece of what happened. Attack-type breadth (mean # types flagged) doesn't differ much across freq groups overall (1.51–1.79), so this isn't a group-wide pattern — but it is a red flag specifically for the small number of observations that drive the cost tail.

**Updated read:** the freq=6 anomaly looks like a blend of two things: (i) a genuine elevated rate of targeted/personalised phishing (now corroborated by `phishcon_bands`, reversing the earlier engagement-based doubt), and (ii) at least some of the extreme-cost tail being multi-vector incidents where phishing is one component, not the sole cause. This means the ~7x type-T bridge multiplier is probably too clean a mechanism either way — even the "it's real, it's targeted" reading doesn't cleanly justify treating it as a pure single-attack-type bridge, given 2/3 of the outliers implicate other attack types. Still unresolved at n=47 (and n=3 for the outliers specifically), but the practical recommendation strengthens: treat this as a bounded sensitivity range rather than a single point estimate, since the underlying mechanism is evidently mixed.

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

## Freq Distribution by Size (Step 3) (src/estimation/freq_distribution.py)

Exploratory pass at P(freq | size, attacked) — the piece of Step 3 that was previously unexamined.

**Cell counts (size x freq, attacked firms):** mostly healthy (n=16-113). Two thin cells: Small|~daily (n=9), Large|~daily (n=6) — unsafe to resample directly in the joint-empirical-sampling plan.

**Row proportions:** Micro, Small, Medium look similar to each other across all 6 freq bands (once-only ~20%, mean freq band 2.68-2.85). **Large is distinct**: fewer once-only incidents (12% vs ~20%), heavier tail toward frequent/chronic attack (daily+several/day ~20% vs ~9-13% elsewhere), mean freq band 3.03.

**Chi-square tests:**
- Full 4x6 table: chi2=24.9, dof=15, p=0.051 — borderline, diluted by pooling 3 similar rows with 1 different one.
- Large vs. (Micro+Small+Medium pooled), 2x6: chi2=14.3, dof=5, **p=0.014 — significant.** Confirms Large genuinely has a different freq distribution; Micro/Small/Medium are statistically indistinguishable from each other.

**Implication:** points toward a 2-group structure for Step 3 — Micro/Small/Medium sharing (or close to sharing) one freq distribution, Large modeled separately — rather than 4 fully independent empirical distributions or one smooth parametric curve across all four bands. Also explains why sparsity concentrates in Large: its already-small sample splits across a genuinely heavier-tailed shape.

**Decided (2026-07-08):** pool Micro/Small/Medium into one shared empirical freq distribution (increases effective n for resampling freq bands, resolves the two thin cells since Small|~daily folds into the pooled M/S/M sample); keep Large separate. This pooling applies **only to the freq marginal** (P(freq | size, attacked), the Step 3 sampling distribution) — it does not extend to the cost distribution. Cost (Step 4, `cost_distribution.py`) is already modeled per (size, freq) cell individually and stays that way, since sigma/mu vary by size band regardless of the freq-pooling question. So the joint sampling procedure becomes: draw freq from the pooled M/S/M (or separate Large) distribution, then draw cost conditional on (actual size stratum, freq) as already established. Large's own freq|~daily cell (n=6) remains thin and unresolved — not addressed by this decision, since Large isn't pooled with anything.

---

## Open Questions

These are unresolved issues that a future session should be aware of before diving in.

**1. Freq-as-conditioning-variable vs. freq-as-distribution (source of past confusion)**
Throughout the analyses, `freq` has been used in two distinct roles that are easy to conflate:
- *Conditioning variable for cost:* P(damage_bands | freq, size) — done, lognormal fits well.
- *Distribution to be modelled:* P(freq | size, attacked) — NOT done. This is what the Squiggle simulation needs to sample from: first draw a freq band, then draw a cost. These are separate questions. The parametric vs. empirical decision for freq refers to this second role only.

**2. Freq=6 type-T anomaly — investigated, reframed twice, still not resolved**
The mixture model assigns 34.6% of "targeted" (type T) mass to the several/day frequency group (n=47 phishing firms). `src/estimation/freq6_anomaly.py` confirms the elevated cost is **real** (visible in raw unweighted data, not a weight or outlier artefact, replicated across 3 independent analyses). First pass: real engagement rates at freq=6 (20.0%) are *not* elevated relative to once-only (30.7%), undermining the assumption that this is the same targeted/spear-phishing mechanism the bridge formula was built around. Follow-up pass (2026-07-07): the *targeted* rate (`phishcon_bands`, distinct from engagement) IS elevated at freq=6 (52.5%, highest of any freq group) and is a strong within-group cost discriminator — so the spear-phishing story isn't dead after all. But 2 of the 3 extreme-cost outliers driving the tail are multi-vector incidents (phishing plus DoS/malware/impersonation/etc., with cost also recorded under other attack types' cost variables), meaning at least part of the tail isn't a pure-phishing phenomenon the bridge formula could apply to cleanly. Net: real, partly targeted-attack-driven, partly multi-vector-driven — no single clean mechanism, still unresolved at n=47. Recommendation: treat as a bounded sensitivity range rather than resolving to one multiplier.

**3. Mixture model scope — CLOSED, infeasible for other types (2026-07-08)**
The 2-component (targeted/mass-market) structure is validated for phishing (dominant type) and degenerate for impersonation. Extension to ransomware/malware/DoS/hacking was attempted and found infeasible — each type has only 21-33 attacked firms total, too few once split across 6 freq bands to run the rank-2 test (at most 1 freq group per type clears even a lenient n≥8 threshold). Not resolvable with more analysis of this dataset; a non-phishing bridge treatment will need a different method entirely (empirical multiplier or conservative/pessimistic bounds).

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
| `proc.py` | Data loading | `get_business_data()`, `get_charity_data()`, `SPECIAL_CODE_THRESHOLD=100`, `PHISH_COUNT_COLNAMES` (phishcon_bands, phisheng_bands) |
| `src/estimation/consistency_checks.py` | i.i.d. bridge test | Monotonicity test (violated everywhere); discovery of 999.62004 special code |
| `src/estimation/bridge_calibration.py` | crimecost vs damage_bands | Avenue closed — different cost concepts, non-representative subsample |
| `src/estimation/attack_type_breakdown.py` | Cost by attack type and freq | Monotonicity failure persists within every type; disrupta analysis; full distribution by freq group |
| `src/estimation/bridge_multiplier.py` | Empirical bridge upper bound | MC multipliers per (size, freq) cell; freq=4+ gives implausible results (known upper bound only) |
| `src/estimation/mixture_model.py` | SVD rank-2 test | F_T/F_M extraction; π_f weights; per-size-band results for phishing and impersonation; scope-extension attempt to ransomware/malware/DoS/hacking found infeasible (too few firms per type) |
| `src/estimation/mixture_bridge.py` | Analytical bridge via mixture | Corrected bridge formula (G=F^(1/k)); bridge≈1.02 for type M; Poisson fit for k\|type (failed) |
| `src/estimation/prevalence.py` | Prevalence by size | Empirical proportions; logistic-in-log-size fit (RMSE 2.2%) |
| `src/estimation/cost_distribution.py` | Lognormal fit to cost bands | Per-cell (mu, sigma); chi-sq GOF; P(cost>£500k) per cell |
| `src/estimation/freq_distribution.py` | Freq distribution by size (Step 3) | (size x freq) cross-tab; Large differs from Micro/Small/Medium (p=0.014) |
| `src/estimation/phishing_count_validation.py` | Validate F_T/F_M against real phishing counts | phishcon_bands (targeted) weak cost predictor (ρ=0.29); phisheng_bands (engaged) strong (ρ=0.37) and matches SVD F_M almost exactly |
| `src/estimation/freq6_anomaly.py` | Investigate freq=6 targeted anomaly | Anomaly real (not weight/outlier artefact); engagement not elevated there but targeted-rate (phishcon_bands) is; 2/3 extreme-cost outliers are multi-vector incidents, not pure phishing |
| `src/estimation/preparedness_analysis.py` | Does cyber preparedness explain the freq→no-cost pattern? | Motivating hypothesis rejected; found a different, counter-intuitive, size-independent finding instead — see "Preparedness vs. cost" section below |

---

## Preparedness vs. cost (src/estimation/preparedness_analysis.py, 2026-07-09/10)

**Motivating question**: why does the no-cost share among attacked firms climb with attack frequency (see consistency_checks.py DECOMPOSITION, Core Finding #1 correction)? Candidate hypothesis: better-prepared firms deflect more incidents at zero cost, and preparedness happens to correlate with frequency.

**Data**: Q31 `rules1-20` (15 real technical controls after excluding "don't know"/"none of these" meta-responses, full coverage among attacked firms) summed into `num_controls` (0-15); Q33B `trained` (staff cyber training, full coverage) as `trained_yes`. Added `PREPAREDNESS_COLNAMES` to `proc.py` (also includes policy/review/strategy/corprisk/update/insurex, noted but not used here — substantial routing-related missingness, not checked yet).

**Hypothesis rejected**: preparedness does not vary with attack frequency (mean num_controls flat at ~10.4-10.9 across all freq groups 1 through 5+; trained% noisy, no trend). So preparedness cannot mechanically explain the freq→no-cost climb.

**Unexpected finding instead**: preparedness has a real, backwards-from-naive-expectation relationship with cost outcome — more-prepared firms are *less* likely to have a zero-cost year, not more. True of every one of the 15 individual controls (all show lower "yes" rate among no-cost firms than nonzero-cost firms, -4pp to -13pp). Initially looked like it might be explained by size (preparedness scales steeply with size: mean controls Micro 9.0→Large 13.2; no-cost rate falls steeply with size: Micro 58%→Large 34%) — but checked properly and **it survives controlling for size**:
- Joint logistic regression (no_cost ~ freq_group + sizeb + num_controls + trained_yes): num_controls OR=0.93/control (p=0.004), independent of size (OR=0.85/band, p=0.02). freq_group's own effect weakens to marginal (p=0.055) once size+prep are controlled — part but not all of the raw freq→no-cost pattern was riding on these confounds. Pseudo-R²=0.034 — real but small effects, most of the variance in outcome is unexplained by this model.
- Stratified by size band (no pooling): High-prep firms have a lower no-cost rate than low-prep firms in **all four size bands** (Micro 53.6% vs 59.5%, Small 44.5% vs 50.0%, Medium 36.8% vs 55.0%, Large 33.3% vs 38.5%). Per-band regressions all point the same direction (Micro/Medium significant, Small/Large not, but same sign — Large has only n=13 low-prep firms, treat as weak evidence).

**Key clarifying split — presence vs. severity of cost**: re-ran the same size-stratified check on cost *severity conditional on having any cost* (mean damage_band among nonzero-cost firms only), and this relationship **disappears / becomes inconsistent**: no clean direction across size bands (Micro: high-prep lower severity; Small/Medium: high-prep higher severity; Large: tied, n too small). Pooled OLS (severity ~ num_controls + sizeb, nonzero-cost firms only): num_controls coefficient not significant (p=0.156); size itself is significant (p=0.002, bigger firms have costlier incidents when they have one). So preparedness relates to **whether any cost gets recorded at all**, not to **how bad it is once it exists** — a real and fairly clean split.

**Interpretation (candidate explanations, not adjudicated by this single-wave cross-sectional data)**:
1. Detection/attribution effect — better-monitored, better-trained firms notice and correctly attribute costs that less-prepared firms simply don't register. Best fit to the presence/severity split above (would predict preparedness → presence but not severity, which is what's observed).
2. Reverse causality — firms hurt by past incidents subsequently invest in controls/training; "high preparedness" partly reflects past harm, not future prevention.
3. Confound via digital complexity/exposure (distinct from raw employee-count size) — weaker fit, since this would more plausibly also predict severity, which isn't observed.

**Not yet done**: separating "digital exposure" from raw size; using the lower-coverage `review`/`strategy` variables (about *when* policies were adopted) to get at the reverse-causality question with some time-ordering; checking whether the residual freq effect (p=0.055 after controls) fully vanishes with a richer preparedness battery.

**Follow-up: full distributional breakdown (Step 5), not just means (2026-07-10).** Checked the full nonzero-cost `damage_bands` distribution (all bands + tail probabilities), not just the mean, broken out separately by freq_group, sizeb, and prep_level.

- **By frequency**: mostly flat/noisy (means 3.36-3.76 across freq groups 1-4), except the **5+ group (daily/several-times-a-day, n=69) has a visibly fatter tail** — mean 4.22, P(≥£20k)=10.1%, P(≥£100k)=2.9%, roughly double the other groups. This is the same freq=6 anomaly investigated for phishing specifically in `freq6_anomaly.py` — showing up here in the *pooled, all-attack-types* view too, independent corroboration that it isn't phishing-specific.
- **By size**: clean and monotonic, as expected — mean severity climbs steadily (Micro 3.34 → Small 3.53 → Medium 3.74 → Large 4.50) and so does the tail (P(≥£20k): 1.3%→6.4%→6.1%→9.7%; P(≥£100k): 1.3%→1.4%→0.8%→4.2%).
- **By preparedness, pooled (not size-controlled)**: looked alone, High-prep firms appear to have *worse* conditional severity (mean 3.88 vs 3.39, P(≥£20k) 6.8% vs 3.2%) — apparently contradicting the earlier size-controlled regression finding (num_controls not significant for severity, p=0.156).
- **Resolved by splitting jointly on size × prep_level**: the pooled "High prep = worse severity" pattern does NOT hold within strata — it *reverses* in Micro (High-prep mean 3.13 < Low-prep 3.42), and the gaps in Small/Medium aren't a clean uniform story either (Large ties, n too small). Confirms the pooled Step 5c table was picking up size composition (High-prep skews toward bigger firms, bigger firms have worse severity) rather than a genuine preparedness→severity effect — consistent with, and a good concrete illustration of, the OLS non-result from Step 4d. Reinforces: the presence/severity split (preparedness predicts *whether* there's a cost, not *how bad* it is) holds up under a full distributional look, not just at the level of means.

**Follow-up: does frequency correlate with preparedness at all? (Step 6, 2026-07-10)** Direct test, not just eyeballing flat-looking group means: Spearman/Pearson correlations between freq (raw and grouped) and both num_controls and trained_yes, plus a partial correlation controlling for size. All near zero and non-significant (|ρ| ≤ 0.03, p ranging 0.34-0.69 across all variants, including the size-controlled partial correlation). **Cleanly rules out "frequently-attacked firms are more/less prepared"** as an explanation for anything downstream — preparedness and frequency are genuinely independent dimensions in this data.

**Conceptual discussion, logged for future reference**: raised the question of why higher frequency correlates with a *higher* no-cost share at all, given that if "real"/costly attacks arrived at some baseline rate independent of a separate high-volume "low-effort junk" stream, more total attempts should mechanically increase the chance of at least one costly hit (more draws = more chances), predicting no-cost share should *fall* with frequency, not rise. Resolution (tentative, not further tested): this is inconsistent with an *additive independent-processes* model, but consistent with the phishing mixture model's own π_f (targeted-component mixing weight) curve — 12.9% (once-only) → 5.7% → 0% → 0% → 0% → 22.3% (several/day) — which shows targeted/costly attacks essentially *absent*, not just diluted, across the middle frequency bands. This looks less like "more draws from a shared pool with a small constant chance of a real hit" and more like **firms sorting into one regime or the other** (mostly-harmless-automated-noise vs. occasional-real-attempt), with unusually little overlap in the middle frequencies. What causes that sorting (target selection by attackers, industry, visibility, infrastructure) is not something this dataset can adjudicate — flagged as an open interpretive question, not a resolved one.

**Follow-up: frequency-stratified preparedness check (Step 7, 2026-07-10)** — mirrors the earlier size-stratified check (Step 4a/4c), but stratifying by freq_group instead of size, to test whether the preparedness→no-cost relationship is itself robust to conditioning on frequency.
- **Step 7a** (no-cost % by prep level, within each freq group): High-prep firms have a *lower* no-cost rate in **all 5 frequency groups**, no reversals — gaps range from −7.3pp (once-only) to −22.1pp (5+, the largest gap of any group).
- **Step 7b** (per-freq-group logistic regressions): all 5 slopes negative (OR 0.86-0.94); 3 of 5 individually significant (freq=2,3,5+; p<0.05), the other two (freq=1, freq=4) directionally consistent but weaker (p=0.095, p=0.29) — likely a power issue given similar effect sizes, not a real reversal.
- **Step 7c** (interaction model, num_controls × freq_group, controlling for size): no significant interaction terms (all p>0.35) — no evidence the preparedness effect size itself varies by frequency group.
- **Contrast with the severity finding**: this is the opposite outcome from the Step 5d severity check, which *did* fall apart/reverse under stratification (revealing a size confound). Here the no-cost effect survives stratification cleanly in both size (Step 4) and frequency (Step 7) — combined with Step 6's null freq×prep correlation, the working picture is that **preparedness and frequency act as two independent, non-interacting predictors of the no-cost outcome** (preparedness lowers it regardless of attack frequency; frequency raises it regardless of preparedness level), rather than either one mediating or confounding the other.

---

## Status
Pipeline structure agreed. Core empirical analyses complete (prevalence, lognormal cost fit, mixture model, bridge). Main open items before simulation: Step 3 (freq by size) and Step 1 (ONS counts).

---

## New data source found: ic3.zip (2026-07-08) — explored, not yet used

`ic3.zip` (unexplained/untouched in repo root since 28 May) turned out to be 14 FBI **IC3 (Internet Crime Complaint Center) Annual Reports**, PDF, years 2011–2025 (2014 missing). Extracted to `data/raw/ic3/` (gitignored candidate — not yet added to `.gitignore`, ~39MB of PDFs). Installed `poppler` (via brew) to enable `pdftotext`/page reading, since no PDF library was in the venv.

**What's in them:** US national cybercrime *complaint* statistics (not a business survey) — annual complaint counts and total $ losses (2025: 1,008,597 complaints, $20.877bn losses), broken down by crime type (phishing, ransomware, BEC, extortion, investment fraud, etc.), by victim age group, by US state, plus narrative sections on ransomware, elder fraud, cryptocurrency, AI-related crime. Business Email Compromise and Ransomware get dedicated detail sections with year-over-year trend data.

**Relevance assessment (not acted on):** This is US, complaint-based (self-reported to FBI, not a stratified business survey like the UK CSBS), and not stratified by business size — structurally very different from the primary dataset (`data/raw/data.sav`, the UK Cyber Security Breaches Survey). Not usable as a direct input to the UK business estimate. Potential uses if wanted later: (a) rough cross-national sanity check / order-of-magnitude comparison for total losses, (b) crime-type mix comparison (e.g. is phishing similarly dominant in US complaint data — yes, 191,561 complaints, largest category by count, consistent with phishing's ~57% share of UK incidents), (c) time-series trend context (complaints/losses have grown ~20x since 2001). None of this has been analyzed quantitatively yet — this was purely an unzip-and-inspect pass, no numbers pulled into any script or estimate.

**Update (2026-07-08, same day):** read all 14 reports in full (parallelized across 6 research agents, one per group of years) and extracted every quantitative table — crime-type breakdowns by count and by loss, state/country breakdowns, age demographics, ransomware/BEC/crypto/tech-support deep dives, elder-fraud figures, historical trend series. Synthesized into `data/raw/ic3/IC3_SUMMARY.md` (headline year-over-year tables, dominant crime types by year, data-quality caveats). Full plain-text extractions of all 14 PDFs also persisted at `data/raw/ic3/text/*.txt` (via `pdftotext -layout`) so future sessions can grep for a specific figure without re-running extraction or spawning agents again.

**Headline takeaways:** complaints roughly tripled 2011→2025 (314k→1.01M) while reported losses grew ~43x ($485M→$20.9B) — average loss per complaint rose sharply, driven by BEC and later investment/crypto fraud growth, not just complaint volume. Phishing has been the #1 crime type by complaint count every year since 2019 (consistent with this project's UK finding that phishing dominates business incidents). BEC was the #1 loss category 2015–2021; investment fraud (crypto-driven) overtook it from 2022 onward. Important caveat for any comparison work: **no business-size or business-vs-individual breakdown exists anywhere in these reports** — this is the structural reason IC3 data can't feed directly into the UK per-business cost estimate; it remains cross-national context/color only, per the assessment above. Category definitions also shift across years (e.g. Phishing/Spoofing merged in 2023), so raw year-over-year comparisons need care — see caveats in `IC3_SUMMARY.md` §4.

**Not yet decided:** whether/how to use this dataset at all in the actual estimate. Flag for discussion before investing further analysis time here, per project norm of not going deep without consulting first.

**Update (2026-07-09): crime-type tables built as DataFrames.** `src/ic3_crime_types.py` hand-transcribes the "crime type by count" and "crime type by loss" tables for 2015-2025 (2011-2013 lack a unified table in the source reports; 2014 report missing) into pandas DataFrames — long format (year, category_raw, category, value) and wide/pivoted format (year x canonical category). Includes a `canonicalize()` mapping so categories that got renamed/merged across years (Phishing/Vishing/Smishing/Pharming → Phishing → Phishing/Spoofing after the 2023 merge with Spoofing; BEC/EAC → BEC → Business Email Compromise) line up under a stable name, while keeping the raw per-year label too so merge/split points stay visible rather than being silently blended. Outputs written to `data/raw/ic3/crime_type_{counts,losses}_{long,wide}.csv`.

**Update (2026-07-09), same day: growth decomposition analysis, written up in full in `data/raw/ic3/IC3_SUMMARY.md` §7.** Extended `src/ic3_crime_types.py` with several follow-on analyses on the same crime-type data. Headline findings: (1) IC3's total loss growth is highly concentrated — 5 "leader" categories (Investment, Tech Support, BEC/EAC, Personal Data Breach, Government Impersonation) explain ~80% of loss growth in both 2020→2025 and 2022→2025 windows, with Investment alone ~54-57%; complaint-*count* growth is not similarly concentrated (it's several categories rising while others, notably Phishing, actively shrink in volume). (2) A "flat total loss" category can hide two opposite mechanisms — count falling with severity rising (Overpayment/419, Credit Card Fraud) vs. count rising with severity falling (Ransomware, Botnet, Data Breach) — always decompose before interpreting a flat aggregate. (3) The pooled "rest" (all categories excluding the 5 leaders) is not stagnant — grew 5.3x nominal / 3.9x real (CPI-adjusted) 2015→2025 — but its share of total dollar losses fell from 66% to 21% as the leaders grew far faster (37.9x nominal / 27.9x real); inflation (35.8% cumulative 2015-2025) explains only a small slice of any of this. (4) Investment fraud's growth is overwhelmingly crypto-driven: crypto-specific investment fraud accounts for 87.9% of Investment's loss growth 2021→2025, making crypto-investment fraud the single largest identifiable driver of total IC3 loss growth in that period.
