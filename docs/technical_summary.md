# UK Business Cybercrime Cost — Technical Executive Summary

*As of 2026-07-15. Companion to `NOTES.md` (the detailed running record). This document is meant to be read on its own: after it, you should understand how the headline number was produced, which components came from where, what was tried and abandoned along the way, and how much to trust the result.*

---

## 1. The result

**Estimated total annual cost of cybercrime to UK businesses ≈ £3.0–3.2 billion/year** (mean £3.1–3.2bn, median ~£3.0bn), with a **90% interval of roughly £1–6 billion**.

This is deliberately a *distribution*, not a point estimate — the whole pipeline is built to propagate uncertainty rather than collapse it to a single figure.

Two independent methods (see §4) produce this number and agree to within 1%. That agreement is one of the main reasons to trust the central estimate. The caveats — and there are important ones — are in §6.

Per-firm-size breakdown (type-based method):

| Size band | Firms (N) | £/business | Contribution |
|---|---|---|---|
| Micro (1–9) | 1,150,875 | ~£1,700 | £1.96bn |
| Small (10–49) | 220,085 | ~£3,900 | £0.86bn |
| Medium (50–249) | 38,435 | ~£4,200 | £0.16bn |
| Large (250+) | 8,335 | ~£13,500 | £0.11bn |

Micro businesses dominate the total by sheer count, despite the lowest per-firm cost.

---

## 2. What we're estimating, and the core formula

The estimate decomposes by firm size band:

> **Total = Σ_size  N(size) × P(attacked | size) × E[total annual cost | attacked, size]**

The data source is the **UK Cyber Security Breaches Survey (CSBS)** — a government survey of UK businesses. Its central limitation, which shapes the entire project, is that it records the cost of a firm's **single most disruptive incident** in the year, *not* the firm's total annual cost. Converting one to the other is "the bridge problem" (§5) and is where most of the intellectual work went.

Rather than multiply the means out (which would throw away all uncertainty), we run a **Monte Carlo simulation**: resample the survey, propagate each component's uncertainty, and read percentiles off the resulting distribution of the national total.

---

## 3. The components — where each number comes from

### N(size): number of UK businesses per size band
- **Source:** ONS/DBT *Business Population Estimates 2025* (start-of-2025 figures).
- **Frame choice (important):** we use the **employer** breakdown (businesses with 1+ employees): Micro 1,150,875 / Small 220,085 / Medium 38,435 / Large 8,335 (= 1,417,730 employers). We **exclude the ~4.27 million zero-employee businesses** (sole traders). Reason: the survey's Micro band starts at 1 employee, so sole traders are out of the survey frame, and our prevalence/cost figures — estimated on 1–9-employee firms — can't be extended to them without an unsupported assumption. This is a scope choice with a large effect on the headline (see §6).
- File: `src/estimation/ons_business_counts.py`.

### P(attacked | size): prevalence
- **Source:** empirical, directly from the survey (weighted proportions). Micro ~40%, Small ~51%, Medium ~66%, Large ~69% — prevalence rises steeply with size.
- We use these empirically rather than fitting a smooth curve; the four points are well-measured and there's no need to model them.
- File: `src/estimation/prevalence.py`.

### P(freq | size, attacked): attack-frequency distribution
- **Source:** empirical from the survey. `freq` is a *periodicity band* (once / >once-but-<monthly / ~monthly / ~weekly / ~daily / several-a-day), not a literal count.
- **Choice:** Micro/Small/Medium were found statistically indistinguishable in their frequency profiles and are **pooled**; Large is kept separate. (This pooling applies only to the frequency marginal — the cost model below stays per (size, freq).)
- File: `src/estimation/freq_distribution.py`.

### E[cost | size, freq]: the per-incident cost model
- The survey's cost variable (`damage_bands`) is **banded** (interval-censored): we know a cost fell in "£1k–£5k", not the exact figure, and the top band is open-ended (£100k+).
- **Model:** a zero-inflated lognormal per cell — a point mass at "no cost" plus a lognormal for positive costs, fit by maximum likelihood on the band intervals (interval-censored MLE). The lognormal was validated by checking that its predicted band proportions match the observed ones; it fits well.
- **Why lognormal:** standard for financial-loss magnitudes, and it fit the data. Not imposed blindly — checked.
- File: `src/estimation/cost_distribution.py`.

### The bridge: single-incident cost → total annual cost
This is the hard component and has its own section (§5). In brief, it is a per-firm multiplier applied to the incident cost, and we derive it two different ways as a cross-check.

---

## 4. How the simulation works

File: `src/estimation/national_simulation.py`. Engine: the **Squiggle** probabilistic-programming CLI, run headless (no browser) via `tools/squiggle/run.sh`; generated models live in `src/estimation/build/*.squiggle`. Python reproduces the same aggregation independently as a check.

The mechanism, per replicate:
1. For each size band, **bootstrap-resample** the surveyed firms (weighted by survey weight).
2. For each resampled firm, compute its **total annual cost** — zero if not attacked; otherwise its incident cost (drawn from the fitted lognormal, see §5.6) times its bridge multiplier.
3. Average to a **cost-per-business** for that size band.
4. Squiggle multiplies each band's cost-per-business by the fixed ONS N(size) and sums → one draw of the **national total**.

Repeat 500 times → a distribution over the national total. Prevalence enters automatically because non-attacked firms contribute zero, so a band's average already blends "how often attacked" with "how much it costs when attacked."

**Why simulate rather than multiply?** At ~1.4M businesses, literally drawing 1.4M independent firms per replicate would collapse the total to its mean (law of large numbers) and *understate* uncertainty. The uncertainty that actually matters lives in the *inputs* — prevalence, the mix of cost bands (especially how many firms land in the open top band), and the shakier bridge multipliers. Bootstrapping the survey is what surfaces that uncertainty.

---

## 5. The bridge problem — the intellectual core, including the dead ends

The survey gives the cost of a firm's worst single incident; we need its total annual cost. Everything below is about recovering that. This is also where most of the project's learning (and failed attempts) accumulated — worth reading before re-opening the topic.

### 5.1 The tempting approach that fails: i.i.d. order statistics
The natural idea: treat the reported cost as the *maximum* of `k` independent attacks and back out the underlying per-attack distribution. **Rejected.** The data shows the opposite of what this model predicts: firms attacked *more* frequently report *lower* worst-incident costs, not higher. Interpreted substantively, this is a "many cheap nuisance attacks vs. one expensive targeted attack" pattern — the per-attack cost distribution depends on frequency, breaking the independence assumption the model needs. This held up within every individual attack type, so it isn't a composition artifact.

### 5.2 The calibration dead end: total-cost variables
The survey has sparse "total annual cost" variables (`crimecost_bands`) for ~15% of firms — tempting for directly calibrating the bridge. **Dead end.** That subsample is badly non-representative (firms reporting zero cost almost never fill it in), and the variable turns out to measure a *different cost concept* (direct financial losses only) than `damage_bands` (broad all-in incident cost). Not comparable.

### 5.3 What worked for phishing: a mixture model
Phishing (~half of all incidents) was cracked with a two-component mixture (a mass of cheap commodity attacks + a thin tail of targeted ones). It yields a bridge multiplier of **~1.02** — i.e. for phishing, the worst incident is essentially the whole annual cost, because the other attacks are nearly free. Validated and robust.

### 5.4 What worked for four more types: per-type censored fits
For Other malware, Denial-of-service, Hacking, and Website/social-media takeover, fitting the interval-censored cost model *per attack type* produced usable bridge multipliers directly (Malware ~1.93×, DoS ~0.36×, Hacking ~1.01×, Takeover ~5.50×). DoS below 1 reflects that its worst incidents are relatively expensive but its total is dominated by them.

### 5.5 The two hard types, and the technique that unstuck them
**Ransomware and Impersonation** initially resisted every parametric model (fits were degenerate or propped up by a single outlier). The breakthrough was a **clean-reference-subsample + Monte Carlo** technique:
- Find a subgroup where the reported cost is *unambiguously* the firm's total — for Ransomware, firms with a newly-discovered attack-count variable indicating a single attack; for Impersonation, firms where impersonation was the *only* attack type all year.
- Use that clean subgroup's cost distribution as the per-incident distribution, and Monte-Carlo-simulate totals for the messier multi-incident firms.
- **Ransomware** resolved to a solid **~2.21×** multiplier, and passed a check that its large outlier is consistent with the fitted heavy tail.
- **Impersonation** made real progress but stayed **fragile**: removing a single firm from its small reference sample swings the estimate by ~50%. It is therefore carried as a **range (3.11–6.25×)**, not a point, and that range is drawn fresh on every simulation replicate so its uncertainty flows into the final interval. Diagnostic work confirmed this is a genuine *data-coverage* gap (Impersonation has zero exact cost observations and most of its evidence is unresolved), not something more modeling can fix.

All multipliers are consolidated in `src/estimation/bridge_specification.py`, which raised bridge coverage from ~0% (for non-phishing repeat-attacked firms) to **99.4%** of attacked-firm weight.

### 5.6 Two modeling choices inside the cost draw (both forced by heavy tails)
- **Within-band cost: analytic conditional mean, not a random point draw.** We first drew a random continuous value inside each firm's cost band from the fitted lognormal. With an open £100k+ top band, this was catastrophically unstable — a single firm drawing far into the tail could dominate an entire replicate, producing nonsense totals (hundreds to tens of thousands of £bn) while the median stayed sane. The fix: use the lognormal's *analytic* expected cost within each band (including a finite expected value for the £100k+ tail). This still honors "lognormal within-band interpolation" and still lets the fitted tail set the top band's weight, but removes the single-draw lottery. A hard tail *cap* or a Pareto tail was considered and not adopted — the lognormal has a finite mean, so no cap is strictly needed — but remains an available alternative.
- **Sparse cost cells fall back to a pooled shape.** Many (size, freq) cells — all Large ones especially — are too thin to fit the tail parameter reliably, and the top-band expected cost is hyper-sensitive to it (a garbage fit briefly produced a £7.7bn-per-business Large figure). So a cell uses its own shape only if it has ≥20 non-zero-cost observations and a plausible spread; otherwise it borrows the well-populated per-size shape.

### 5.7 The two decompositions (the cross-check)
Because the bridge can be sliced either by **attack type** or by **frequency**, we run both:
- **Type-based:** incident cost × per-type multiplier (§5.3–5.5).
- **Frequency-based:** per-(size,freq) cost × a flat 1.02 bridge for repeat-attacked firms (the phishing-validated figure as a representative default).

They stratify differently and use different bridge logic, yet agree to **0.99×** on the national total. This is strong evidence the central estimate is not an artifact of how we chose to slice the bridge.

---

## 6. How much to trust it — robustness

The honest summary: **the center of the distribution is robust; the upper tail and the scope are not.** The levers sort into four kinds.

### A. Scope/framing levers — large, deterministic, mostly pushing the number *down*
These sit *outside* the £1–6bn statistical interval; they're choices that move the whole thing, and they largely explain why £3bn feels low:
- **The employer-only frame is probably the single biggest lever.** Excluding 4.27M sole traders removes a huge population. Micro firms already contribute ~£2bn from 1.15M businesses; sole traders, even at a fraction of that per-firm cost, could add a comparable amount and plausibly *double* the total. Legitimate scope choice, but the first thing to flag to anyone who finds the number low.
- **Consumer/individual cybercrime is entirely excluded** — this is business-side only. Most alarming headline UK cybercrime figures are consumer-fraud-dominated and not comparable.
- **Self-reported single-incident costs almost certainly undercount** diffuse and long-tail harms (reputational damage, lost future business, unbilled staff time). This is a systematic downward bias that sits *outside* the modeled uncertainty entirely.

So "interestingly low" is substantially by construction: business-only, employer-only, self-reported, single-incident-anchored — each choice trims.

### B. Tail fragility — the real modeling weak point
Only **8 firms** fall in the open £100k+ band, and a cost total is inherently tail-driven. The top band's contribution rides on how those 8 firms resample and on the fitted lognormal spread. Swapping the lognormal tail for a heavier Pareto, or perturbing the spread, could move the top-band contribution substantially. Classic "a handful of observations drive the tail which drives the total." Robustness here is **low** and this deserves a dedicated sensitivity analysis.

### C. Impersonation
Already partly reflected in the interval (its 3.11–6.25× bridge is redrawn each replicate) and known to be the shakiest type. It's a meaningful contributor to the *width* of the interval, which is appropriate.

### D. What is genuinely robust (reassuring)
- **Type-based and frequency-based methods agree to 0.99×** — how we slice the bridge barely matters.
- **Prevalence** is empirical and well-measured — not a lever.
- **Phishing** (~half of incidents) has a bridge of ~1.02, so the largest single chunk of the estimate is essentially "sum the reported phishing costs" — very well pinned.
- The **median has been stable** throughout development, even when a bug was blowing up the mean. The center of mass is solid.

**Overall read:** ~£3bn is a defensible *lower-central* estimate of self-reported employer-business incident costs — robust in its center, fragile in its upper tail, and nested inside a wider band of legitimate scope choices (sole-trader inclusion above all).

---

## 7. Status and where to go next

**Done:** all pipeline components (N(size), prevalence, frequency, per-cell cost model, bridges for all attack types), the national simulation, and the Squiggle engine running headless.

**Recommended next work, in priority order of impact on the headline:**
1. **Scope sensitivity — sole-trader inclusion.** The largest lever; worth an explicit alternative-frame estimate.
2. **Top-band tail sensitivity.** Lognormal vs. Pareto vs. capped tail, plus leave-one-out on the 8 top-band firms.
3. **Propagate cost-shape uncertainty.** Currently the fitted lognormal shapes are held fixed across the bootstrap; letting them vary would widen the interval, mostly via the tail.
4. **Charities.** The survey covers charities too; entirely untouched — the whole analysis so far is business-side only.

**Key files:** `national_simulation.py` (simulation), `bridge_specification.py` (bridge multipliers), `ons_business_counts.py` (N), `prevalence.py`, `freq_distribution.py`, `cost_distribution.py`, `type_specific_montecarlo_bridge.py` (the Ransomware/Impersonation technique). `tools/squiggle/run.sh` runs any Squiggle model headless. `NOTES.md` is the detailed running record.
