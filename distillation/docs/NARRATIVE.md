# NARRATIVE

Companion write-up of the cybercrime-cost estimate: methodology and findings.
Technical but focused on reasoning and results rather than implementation detail.

*(This document is being built section by section. First section below.)*

---

## Valuing banded costs: the parametric ("B") cost model and what it reveals

### The problem

The survey never records a firm's cost as a number — only as a **band**
(e.g. "£5,000–£10,000"), and the top populated band is £100k–£500k. To sum costs
across the country, every band must be turned into a single pound figure. How you
do that turns out to be one of the two decisive choices in the whole estimate
(the other being the bridge from worst-incident to annual total).

There are two coherent ways to value the bands, and they disagree by a factor of
two on the national total. This section documents the fully parametric one
("approach B") and uses it to expose where the headline's uncertainty actually
lives.

### The two approaches

Both fit the same object — a **zero-inflated lognormal** cost distribution, by
**censored maximum likelihood** (each firm's cost is known only to lie in its
band's interval, so the likelihood uses interval probabilities, not point
values). They differ in what they then *trust* the fitted model to do.

| | band frequencies (how many firms per band) | within-band value | internally consistent? |
|---|---|---|---|
| **A — semiparametric** (headline) | **empirical** — the observed firm counts | lognormal | no (hybrid) |
| **B — fully parametric** (this section) | **lognormal** — the fitted shape | lognormal | yes |

- **Approach A** keeps each firm's *own observed band* and uses the lognormal
  only to place a value *within* that band. The band mix — including how many
  firms sit in the top band — comes straight from the data. It is a semiparametric
  hybrid: an empirical band histogram with a parametric ruler laid over each bin.
  It is internally *inconsistent* (the lognormal it uses for placement predicts
  fewer top-band firms than are empirically observed), but that inconsistency is
  deliberate: it refuses to let the model override observed frequencies.

- **Approach B** stratifies the fit by **(size × type)**, collapses each cell to a
  single analytic mean cost, and discards each firm's individual band — replacing
  it with the cell mean. The fitted shape sets *both* the band mix and the value.
  It is the clean, self-consistent statistical object, and it lets attack **type**
  move the cost distribution itself, not merely the bridge multiplier.

In both, the top band is **bounded** at £100k–£500k. The survey offered higher
bands (£500k–£1M, £1M–£5M, £5M+) and no firm chose any of them, so no unbounded
lognormal tail is used; the possible >£500k catastrophic tail is handled as a
separate, explicit sensitivity, not manufactured here.

### The result

Approach B gives a national total of **£0.99bn (Impersonation low) to £1.42bn
(Impersonation high)** — roughly **half** the ~£2.26bn semiparametric baseline.
(Ten of the 40 size×type cells have enough data to fit their own shape; the rest
fall back to the well-populated per-size shape.)

### Why the two halve each other — and why it matters

The gap is **almost entirely Micro, and almost entirely the tail.** Comparing the
per-firm cost each approach assigns, by size band:

| size | empirical mean (A) | fitted mean (B) | A/B | observed band-10 rate | fitted band-10 rate |
|---|---|---|---|---|---|
| **Micro** | £1,291 | £872 | **1.48×** | **0.46%** | **0.09%** |
| Small | £2,969 | £2,932 | 1.01× | 0.66% | 0.66% |
| Medium | £2,449 | £3,247 | 0.75× | 0.39% | 0.67% |
| Large | £8,546 | £9,243 | 0.92× | 2.06% | 2.41% |

Small matches; Medium and Large actually go the *other* way (B slightly higher).
The whole divergence is **Micro**, where the fitted lognormal predicts a 0.09%
top-band rate but the data shows **0.46% — a 5× gap**. Because Micro has 1.15M
businesses, its shortfall drags down the national total.

And "0.46%" rests on **two firms**. The Micro tail histogram is:

```
band 8 (£20–50k):   0 firms
band 9 (£50–100k):  0 firms
band 10 (£100–500k): 2 firms   ← an isolated pair above an empty gap
```

The lognormal, fit to Micro's whole shape, *expects* 0.35 firms in band 10 and
sees 2 — a ~5% event under its own model (a mild upward fluctuation, borderline).
A smooth decaying curve physically cannot produce "nothing at £20k–£100k, then a
cluster at £100k–£500k," so it reads the gap as the signal and shrinks the 2
firms toward its curve. The empirical approach reads the 2 firms as the signal.

**Neither is refutable from two data points.** A confidence interval on a count of
2 spans roughly 0 to 5; the lognormal's 0.35 is barely inside it. The data simply
cannot distinguish "Micro has a genuinely heavy tail" from "two firms fluctuated
up above an empty stretch."

A key corollary: a goodness-of-fit test does **not** resolve this. The per-size
chi-square GOF *passes* for Micro (p=0.70) — but only because it must pool the
sparse upper bins to run, netting the 2 excess band-10 firms against the two
empty bands below them. The GOF validates the *body* and is structurally blind to
the *tail*. "The cost model fits" is true and largely irrelevant to the headline's
main uncertainty, which lives entirely in the untestable tail.

### What B is, and isn't

Approach B is a **shrinkage estimator for the tail**: it borrows strength from the
well-estimated body to regularise the sparse top band. That is the right instinct
for describing a typical firm — but the national total is a *sum*, and heavy-tailed
sums are dominated by their rare large events. Shrinking observed large events
because they are rare biases a total downward, and the bias has a known direction:
financial-loss distributions are empirically heavier-tailed than lognormal, so B's
lognormal is if anything too light in exactly the region that matters. B's £1.0bn
is therefore best read as a **floor**, not a truer central estimate.

Its value is not that it is more accurate — approach A, by leaning on the model
less (empirical frequencies, model only for within-band placement), is the more
defensible default for a tail-driven total. B's value is that it **isolates and
quantifies the central fragility of the entire estimate**: the headline is not
"£2.2bn ± a bit"; it is a number whose lower half turns on the interpretation of
**two Micro firms sitting above an empty gap.** Reassuringly, B (£1.0bn) lands
close to the independent freq-based lower variant (~£1.0bn), so two unrelated
routes agree on where the floor sits.

The genuinely principled resolution is neither A nor B but a single model whose
band frequencies match the data *and* carry an honest tail — a heavier-tailed
family (e.g. a generalised-Pareto tail spliced above £100k) rather than a lognormal
body extrapolated into the tail. Such a model would land at or above A, confirming
B as a floor; that is the main methodological upside still on the table.

*Scripts: `runs/estimate_size_type_lognormal.py` (approach B) using
`src/fitting.py`; baseline A is the national simulation. Diagnostic evidence for
the per-size / tail claims: `runs/diag_global_vs_stratified_fit.py`.*
