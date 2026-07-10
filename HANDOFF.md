# UK Cybercrime Cost Estimation — Comprehensive Handoff Document

*Last updated: 2026-07-07*

## Purpose of this document

This is a self-contained synthesis of everything the analysis has found so far, written for
someone picking up the analytical/interpretive thread of this project — not the code. It
consolidates findings currently scattered across `NOTES.md` (the running technical log) and
individual analysis scripts into one structured narrative. Read this to understand *what we
know, what we don't, and what's still genuinely undecided* — without needing to run or read any
code. (`NOTES.md` remains the authoritative record for anyone continuing the coding work — this
document is a derived synthesis of it, current as of the date above, and may drift out of date
if `NOTES.md` is updated afterward without a corresponding update here.)

---

## 1. Project goal

Produce a synthetic estimate of **total annual cybercrime costs across all UK businesses**, for
a GovAI paper. The estimate needs to be a distribution (with uncertainty), not a single point
figure — the final simulation stage will use Squiggle, a Monte Carlo modeling language, to
produce that distributional output. A parallel charity dataset exists and is not the primary
target, but may be used as supporting/validating evidence for modeling assumptions shared with
the business side.

## 2. Data source

The underlying data is the UK Cyber Security Breaches Survey (a government-run annual survey of
UK businesses and charities about cyber attacks and their costs), loaded from an SPSS export into
`data/proc/data.csv`. Businesses are split into four size strata by employee count: Micro (1–9),
Small (10–49), Medium (50–249), Large (250+). Roughly half of surveyed businesses report having
experienced at least one cyber attack in the past 12 months; only attacked businesses answer the
detailed frequency/cost/attack-type questions this analysis is built on.

## 3. The core problem, in plain terms

The survey asks each attacked business about **the single most disruptive incident** it
experienced in the last 12 months — how much it cost (in a cost band, not an exact figure) and
how frequently attacks of that kind occurred (a periodicity band: once only, more than once but
less than monthly, roughly monthly, roughly weekly, roughly daily, or several times a day).

What we actually want is **total annual cost per business** — the sum of every attack's cost,
not just the worst one. The gap between "cost of the worst attack" and "total cost of all
attacks" is what we call **the bridge problem**, and it's the central methodological challenge of
this project. Everything past the basic prevalence numbers is, in one way or another, about how
to bridge that gap credibly.

A second, related subtlety: the frequency variable is a *periodicity band*, not a literal attack
count. "Several times a day" could mean 1,000+ attacks a year; "more than once but less than
monthly" could mean anywhere from 2 to 11. Converting a periodicity band into an implied attack
count is itself a modeling choice with real uncertainty, especially for the higher bands.

## 4. The estimation framework

The overall formula being built toward:

> Total annual UK cybercrime cost = Σ over size strata of **N(size) × P(attacked | size) × E[total annual cost | attacked, size]**

Each stratum's contribution requires three ingredients:

1. **N(size)** — how many UK businesses exist in that size band (external data, not survey-derived).
2. **P(attacked | size)** — what fraction of businesses in that stratum were attacked at all.
3. **E[total annual cost | attacked, size]** — the average total cost, conditional on having been attacked, which itself requires solving the bridge problem above.

Three estimate "tiers" are being carried in parallel throughout, rather than committing to one
number:

- **Conservative (lower bound):** assume total cost = cost of the single worst attack (i.e., no
  bridge correction at all). This is exact for businesses attacked only once, and an
  underestimate otherwise, but a defensible one — see Section 6.4.
- **Pessimistic (upper bound):** assume the worst attack's cost is *typical*, and multiply by the
  implied number of attacks. This mechanically overestimates, because the "worst" incident is by
  definition atypically bad.
- **Best estimate:** model the distribution of costs among the non-worst attacks directly (via
  the mixture model described in Section 6.5) rather than assuming either extreme.

## 5. Data quality notes (things that will bite you if unaware)

- The cost variable (`damage_bands`) contains a non-integer SPSS special code (999.62004,
  alongside the more obvious 997/999 "don't know / refused" codes). Filtering must use "cost
  band value below 100" rather than checking against an exact list of excluded codes — an earlier
  pass missed this and had to be corrected.
- The frequency variable's bands are periodicity categories, not attack counts, as noted above.
  Any analysis that silently treated frequency-band values as literal counts (this happened once,
  in an early chi-square goodness-of-fit attempt) is invalid and was discarded.
- Two "total annual cost" variables exist directly in the survey (`crimecost_bands`,
  `notfraudcost_bands`) but have very low coverage (10–15% of attacked firms) and were found to
  measure a narrower cost concept (direct financial loss only) than the primary cost variable
  (which includes staff time and broader disruption) — see Section 6.4 for why this avenue was
  closed rather than used to directly calibrate the bridge.

## 6. Findings, by topic

### 6.1 Prevalence — what fraction of businesses are attacked

Clean and essentially finished. The fraction of businesses attacked rises monotonically with
size: Micro 40%, Small 51%, Medium 66%, Large 69%. This pattern holds almost identically whether
or not survey statistical weights are applied, meaning the raw sample composition and the
weighted population estimate agree closely here. A logistic curve fit against log(firm size)
matches these four points almost perfectly, but since the simulation will use exactly these four
size strata (not a continuous size variable), the extra step of fitting a smooth curve adds
nothing practical — the four empirical percentages can be used directly.

### 6.2 Attack frequency, by firm size

Less settled. Looking at how the six frequency bands are distributed within each size stratum:
Micro, Small, and Medium firms all show statistically indistinguishable frequency profiles from
each other (roughly 20% "once only," average frequency band around 2.7–2.9 out of 6). **Large
firms are different** — significantly fewer one-off incidents (12% vs. ~20%), and a
meaningfully heavier tail toward chronic/frequent attack (about 20% of large firms report daily
or several-times-daily attacks, vs. 9–13% in the other three strata). A formal statistical test
confirms Large is distinct from the pooled other three strata (p=0.014), while a test among just
Micro/Small/Medium shows no such difference.

**Open decision:** whether to pool Micro/Small/Medium into a single shared empirical frequency
distribution (which increases the effective sample size available for the simulation to draw
from) or keep all four strata fully separate despite two of the resulting cells being thin
(Small-daily has only 9 observations, Large-daily only 6). This hasn't been decided yet.

### 6.3 Cost distribution, within a given (size, frequency) group

A "zero-inflated lognormal" model — some fixed probability of literally zero cost, and a
lognormal distribution over the cost bands otherwise — fits the data well within individual
(size × frequency) cells. This gives a principled way to convert a cost *band* (e.g., "£1,000 to
£5,000") into an actual point estimate or distribution, and to handle the open-ended top band
("£500,000+") with a reasonable tail assumption.

The important caveat: neither the "zero-cost probability" nor the shape parameters of the
lognormal are stable across cells — they must be fit per (size, frequency) cell rather than
assumed to share a common shape. The probability of a cost exceeding £500,000 is negligible
(well under 1%) in nearly every cell, meaning the open-ended top band isn't a major driver of
uncertainty — except for Large firms, where the relevant subsample is small enough that this
number is unreliable rather than reassuring.

### 6.4 The bridge problem — why the obvious approach fails

The most natural way to recover "total cost from all attacks" out of "cost of the worst attack"
is a statistical technique based on order statistics: if you assume every attack's cost is drawn
independently from the same distribution, the worst-of-k-draws should on average get *more*
extreme as k (attack frequency) increases. So businesses hit more often should report
higher worst-case costs.

**This is not what the data shows.** In every one of the four size strata, and within every
individual attack type examined (ransomware, phishing, denial-of-service, impersonation), the
relationship runs the opposite way: firms attacked more frequently report *lower* worst-case
costs on average. This was checked carefully against the possibility that it's an artifact of
attack-type mix (i.e., that frequently-attacked firms are just dominated by cheap, common attacks
like phishing, dragging the average down) — that composition effect is real and present, but the
reversed relationship persists even within single attack types, so it isn't the whole
explanation. The straightforward independent-draws bridge model is rejected as a description of
this data.

A separate, more direct attempt to calibrate the bridge — comparing the primary cost variable
against the survey's other "total annual cost" variables — was tried and abandoned. Those
total-cost variables have very low response coverage, and the subsample that does answer them is
badly non-representative (firms reporting "no cost" on the primary variable are 49% of the full
sample but only 11% of the subsample that answers the total-cost question — people who had no
cost mostly don't bother filling in a follow-up total-cost field). Worse, the two variables
appear to measure different things: the primary variable is a broad "all-in" cost concept
(payments, staff time, disruption) for one incident, while the alternative "total cost" variable
captures direct financial losses only, across all incidents. They aren't comparable enough to use
for calibration.

**The reframing that unlocked progress:** total annual cost = (cost of the worst attack) + (sum
of costs of all the *other* attacks that year). For firms attacked only once (about 20% of
attacked firms), that second term is exactly zero — the reported cost already *is* the total, no
modeling needed. For firms attacked more than once, the question becomes: how much do those
additional, non-worst attacks typically add? Examining the cost-band distribution *within* each
frequency group shows that firms attacked frequently are heavily concentrated in the "no cost" /
cheap bands — consistent with a picture where the extra attacks beyond the worst one are mostly
cheap or free. This directly supports the idea that the bridge correction may be small in
practice for the majority of firms, i.e. that using the worst-attack cost as a stand-in for the
total (the "conservative" tier from Section 4) may not be dramatically wrong.

### 6.5 The mixture model — a validated structural finding for phishing

Phishing is the single most common attack type (roughly 57% of all reported worst-incidents), and
turns out to have unusually rich supporting structure in the data, so it has received the deepest
investigation. Other attack types have not yet been tested to the same depth (see Section 7).

**The core finding:** within each size stratum, phishing incidents decompose cleanly into two
underlying groups:

- **A "targeted" group (spear-phishing-like):** essentially never costs literally nothing, and
  has real probability mass in the expensive cost bands (tens to hundreds of thousands of
  pounds).
- **A "mass-market" group (commodity spam-like):** roughly 60% of the time costs literally
  nothing, and what cost does occur is concentrated in the cheap bands.

This isn't a hypothesis — it was tested via a technique (an SVD/rank test on the matrix of
frequency-group × cost-band proportions) that would have failed the test if the data hadn't
actually had this two-group structure, and it passed cleanly in three of the four size strata
(over 94–99.6% of the variation is explained by exactly two components; a third component adds
almost nothing). It did not hold up the same way for the second-most-common attack type,
impersonation — there, the same test technically passes, but the two extracted groups turn out to
look nearly identical to each other, meaning there's no real two-group structure for
impersonation specifically.

**Independent validation of the same structure using real data (not just the statistical
extraction):** the survey happens to include two extra questions specific to phishing — how many
of the attacks looked personalised/targeted, and how many times someone actually engaged with
(fell for) an attack — both with good response coverage (76% and 83% respectively; equivalent
questions exist for other attack types but with coverage too low, 6–20%, to be usable). Regrouping
the same statistical test using real engagement-count data instead of the frequency band
reproduces essentially the same "targeted" cost profile, and — unlike the frequency-based
grouping, which gave a noisy, non-monotonic pattern — gives a clean, monotonic result: 0% of
firms with zero engagement look "targeted," rising to 20% and then 65% as engagement increases.
This is strong independent confirmation that the two-group structure is a genuine feature of the
real world, not a statistical artifact of the test used to find it.

**Where this leaves the phishing bridge multiplier:** for the mass-market group (roughly 95% of
phishing-affected firms), the analysis shows the "extra" attacks beyond the worst one contribute
almost nothing to the total, regardless of how frequently the firm is attacked — the appropriate
bridge multiplier stays close to 1.0 (i.e., worst-attack cost ≈ total cost) across the whole
plausible range of attack frequency. For the targeted group (roughly 5% of phishing-affected
firms), the multiplier is not flat — it grows the more frequently the firm is attacked, reaching
as high as roughly 7× in the most extreme frequency category. **Whether that most-extreme
category should be trusted is the single largest unresolved question in the whole bridge
analysis**, covered next.

### 6.6 The "several times a day" anomaly — the most important open puzzle

The statistical extraction in Section 6.5 assigns an unusually large share (about 35%) of the
"targeted" group's mass to the rarest, most extreme frequency category: firms attacked "several
times a day." This is suspicious on its face — it's a small subgroup (47 phishing-affected
firms), it interrupts an otherwise clean declining trend (average cost band falls steadily from
"once only" through "roughly weekly," then jumps back up sharply at "several times a day"), and
if it's real, it's the single biggest lever on the overall phishing bridge multiplier: roughly
1.02 (almost no correction) if this subgroup is noise, versus roughly 1.7 if it's a genuine,
extreme-multiplier targeted-attack subgroup.

**The investigation into this subgroup has gone through two rounds so far, and the honest
current state is: it's real, but it doesn't have one clean explanation.**

*Round one* checked whether this was a statistical artifact — a fluke of the extraction method, a
few outlier data points, or a survey-weighting quirk. None of those held up: the elevated cost is
visible directly in the raw, unprocessed data (not just in the statistical reconstruction); the
handful of highest-cost observations driving it have unremarkable survey weights, not unusually
high ones that would distort the picture; and the same declining-then-jumping pattern has now
shown up independently in three separate analyses. So the anomaly itself — elevated cost among
firms attacked several times a day — is not noise.

However, round one also checked whether this subgroup shows the signature we'd expect if it
really were extreme-frequency spear-phishing: elevated *engagement* (people actually falling for
the attacks). It doesn't — engagement in this subgroup (20%) is actually *lower* than in the
"once only" group (31%), which undercuts the assumption that this is the same targeted-attack
mechanism the bridge formula was built to describe.

*Round two* (this session) followed up in two ways. First, it checked a different real-data
proxy — how many attacks *looked* targeted/personalised, as distinct from whether anyone actually
engaged with them. That measure tells a different story than engagement did: firms attacked
several times a day have the *highest* rate of looking-targeted attacks of any frequency group
(52.5%, versus 39% for "once only"), and within this subgroup specifically, whether a firm had
any targeted-looking attacks is a strong predictor of cost (average cost roughly doubles for
firms that did). So the "looks targeted" story is actually well-supported here — it was
specifically the "engagement" proxy that looked wrong, and the two proxies disagree with each
other at exactly this subgroup.

Second, round two looked directly at the handful of individual firms driving the extreme-cost
tail (the 3 firms, out of 47, with the highest recorded costs). Two of the three turn out to have
reported *multiple concurrent attack types*, not phishing alone — one also flagged
denial-of-service and impersonation attacks (with cost separately recorded under the DoS cost
variable too); another flagged five additional attack types alongside phishing (malware,
impersonation, unauthorized access, video-conferencing eavesdropping, and website/social-media
takeover), with cost also recorded under two other attack-type-specific cost variables. Only the
third of the three outlier firms reported phishing as its only attack type. In other words, a
majority of the extreme-cost cases in this subgroup may not be "an extreme case of phishing" so
much as "a firm undergoing a broad multi-vector compromise, of which phishing was one recorded
component."

**Bottom line:** the "several times a day" cost anomaly is real, not a data artifact. Part of it
is genuinely explained by elevated targeted-attack rates (contrary to the initial engagement-based
doubt). But part of it — specifically the small number of most-extreme-cost cases — looks more
like general multi-vector compromise than a pure, escalated version of the spear-phishing pattern
seen elsewhere. There is no single clean mechanism to point to, and at a sample size of 47 (and
just 3 for the outlier cases specifically), this is unlikely to resolve further through more
analysis of this same dataset. **The practical recommendation is to stop trying to force this
into a single point estimate and instead carry it as a bounded sensitivity range** (something
like: report the overall estimate under both the "treat this subgroup as noise, multiplier ≈1.02"
and "treat it as fully real and targeted, multiplier ≈1.7" assumptions, rather than picking one).

## 7. What hasn't been tested yet

- The two-group (targeted/mass-market) mixture structure has only been validated for phishing.
  Ransomware, other malware, denial-of-service, and hacking/unauthorized-access have not been put
  through the same test — mainly because phishing is the only attack type with a high-coverage
  real per-attack count variable to validate against, and it's unclear how much would generalize.
  This matters more if non-phishing attack types turn out to dominate the cost tail overall, which
  hasn't been checked.
- External business-population counts (how many UK businesses exist in each size band) haven't
  been sourced yet. This is a hard blocker for the final step of the pipeline (turning
  per-business estimates into a national total) but is independent of everything else in this
  document — it's a data-fetching task, not an open analytical question.
- The final Monte Carlo simulation (Squiggle) hasn't been built yet — it depends on the Section
  6.2 pooling decision and the external business counts above.

## 8. Pipeline status at a glance

| Step | What it produces | Status |
|---|---|---|
| 1. Business counts by size | External population totals, needed to scale per-business estimates to a national total | Not started |
| 2. Prevalence | Fraction of businesses attacked, by size | **Done** — see Section 6.1 |
| 3. Frequency distribution by size | How often attacked firms are hit, by size | Explored, pooling decision pending — see Section 6.2 |
| 4. Cost distribution | Converts a cost band into a cost distribution, by (size, frequency) cell | **Done** — see Section 6.3 |
| 5. Bridge (worst-attack cost → total annual cost) | The core modeling correction | **Done for phishing** (~95% of phishing firms have bridge ≈ 1.02); the "several times a day" subgroup remains an open sensitivity range — see Sections 6.4–6.6 |
| 6. Full simulation | National distributional estimate | Not started; blocked on Steps 1 and 3 |

## 9. Open decisions that need a call made (not purely analytical — judgment calls)

1. **How to treat the "several times a day" phishing subgroup in the final bridge multiplier** —
   recommend a bounded sensitivity range (≈1.02 to ≈1.7) rather than resolving to one number; see
   Section 6.6.
2. **Whether to pool Micro/Small/Medium into one shared frequency distribution**, or keep all
   four size strata fully separate despite two resulting thin data cells; see Section 6.2.
3. **Whether to extend the mixture-model analysis to other attack types** (ransomware, malware,
   DoS, hacking) before finalizing the bridge approach, given it's currently validated only for
   phishing (57% of incidents); see Section 7.
4. **Whether/when to promote the real-engagement-based mixing weights (Section 6.5) into the
   actual bridge multiplier calculation** — currently the bridge still uses frequency-band-based
   weights, even though the engagement-based weights are demonstrably cleaner.

## 10. Recommended next step

Per the current running notes, the most productive next move is **not** further forensic analysis
of the freq=6 subgroup (Section 6.6) — that avenue is judged close to exhausted at this sample
size. Instead, forward progress should come from either: (a) making the Section 6.2 pooling
decision and formalizing the frequency-sampling procedure for the simulation, or (b) starting the
external business-population data fetch (Section 7 / Step 1), since both are pure prerequisites
for building the actual Squiggle simulation and don't depend on resolving any of the remaining
open analytical questions above.

---

*For anyone returning to the code: see `NOTES.md` for the full technical log (including script
names, statistical test values, and exact figures behind every claim above), and the `Script
Index` table within it for where each analysis lives under `src/estimation/`.*
