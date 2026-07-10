# FBI IC3 Annual Reports — Quantitative Data Summary (2011–2025)

Reference document only — **this is US complaint data, not the UK CSBS survey**. Kept for potential cross-national sanity checks / crime-type-mix comparison, not as an input to the business cost estimate. See NOTES.md "New data source found: ic3.zip" for context.

Source PDFs: `data/raw/ic3/*.pdf`. Full plain-text extractions (via `pdftotext -layout`) persisted at `data/raw/ic3/text/*.txt` — grep these directly rather than re-extracting or re-spawning research agents. 2014 report is missing from the archive (gap in the series). Full per-year detail (state-by-state tables, age×gender breakdowns, case narratives) is in the `.txt` files; this document holds only the headline series and figures useful for comparison.

---

## 1. Headline totals by year

| Year | Total complaints | Total reported losses | Avg. loss/complaint |
|---|---|---|---|
| 2011 | 314,246 | $485,253,871 | $1,544 (all) / $4,187 (reporting loss) |
| 2012 | 289,874 | $525,441,110 | $1,813 / $4,573 |
| 2013 | 262,813 | $781,841,611 | — |
| 2014 | *(report missing)* | — | — |
| 2015 | 288,012 | $1,070,711,522 | $3,718 / $8,421 |
| 2016 | 298,728 | ~$1.33B (contemporaneous) / **$1.5B** (per 2019 report's retrospective table — discrepancy, see §4 note) | — |
| 2017 | 301,580 | ~$1.42B (contemporaneous) / $1.4B (retrospective) | — |
| 2018 | 351,937 | ~$2.71B (contemporaneous) / $2.7B (retrospective) | — |
| 2019 | 467,361 | $3.5B | — |
| 2020 | 791,790 | $4.1–4.2B (both figures printed in-report) | — |
| 2021 | 847,376 | $6.9B | — |
| 2022 | 800,944 | $10.2–10.3B (both figures printed in-report) | — |
| 2023 | 880,418 | $12.5B | — |
| 2024 | 859,532 | $16.6B | $19,372 |
| 2025 | 1,008,597 | $20.877B | $20,699 |

**Pattern:** complaint count roughly tripled 2011→2025 (314k → 1.01M); reported losses grew ~43x over the same span ($485M → $20.9B), i.e. average loss per complaint rose sharply — driven mostly by the growth of high-value categories (BEC, investment/crypto fraud), not just more complaints. Losses grew much faster than complaint counts every year from 2019 onward.

Cumulative complaints since IC3's founding (May 2000): ~3.76M (as of 2016) → ~4.88M (2019) → ~5.68M (2020) → ~6.5M (2021) → ~7.3M (2022) → ~8M (2023) → ~9M (2024) → ~10M (as of 2025).

## 2. Dominant crime types by year (top 3 by complaint count)

| Year | #1 | #2 | #3 |
|---|---|---|---|
| 2011–2013 | No unified ranked table in these reports (individual scam deep-dives only: FBI-related scams, identity theft, advance fee fraud) | | |
| 2015 | Non-Payment/Non-Delivery (67,375) | 419/Overpayment (30,855) | Identity Theft (21,949) |
| 2016 | Non-Payment/Non-Delivery (81,029) | Personal Data Breach (27,573) | 419/Overpayment (25,716) |
| 2017 | Non-Payment/Non-Delivery (84,079) | Personal Data Breach (30,904) | Phishing (25,344) |
| 2018 | Non-Payment/Non-Delivery (65,116) | Extortion (51,146) | Personal Data Breach (50,642) |
| 2019 | **Phishing (114,702)** — first year phishing takes #1 | Non-Payment/Non-Delivery (61,832) | Extortion (43,101) |
| 2020 | Phishing (241,342) | Non-Payment/Non-Delivery (108,869) | Extortion (76,741) |
| 2021 | Phishing (323,972) | Non-Payment/Non-Delivery (82,478) | Personal Data Breach (51,829) |
| 2022 | Phishing (300,497) | Personal Data Breach (58,859) | Non-Payment/Non-Delivery (51,679) |
| 2023 | Phishing/Spoofing (298,878, merged category) | Personal Data Breach (55,851) | Non-Payment/Non-Delivery (50,523) |
| 2024 | Phishing/Spoofing (193,407) | Extortion (86,415) | Personal Data Breach (64,882) |
| 2025 | Phishing/Spoofing (191,561) | Extortion (89,129) | Investment (72,984) |

**Phishing has been the #1 crime type by complaint volume every year since 2019**, consistent with this project's UK finding that phishing dominates (~57% of business incidents). Note category-definition churn: "Spoofing" was tracked separately through 2022, merged into "Phishing/Spoofing" from 2023 onward (2023 report retroactively restates 2021/2022 combined figures — see §4).

## 3. Dominant crime types by year (top 3 by dollar loss)

| Year | #1 | #2 | #3 |
|---|---|---|---|
| 2015 | BEC ($246.2M) | Confidence Fraud/Romance ($203.4M) | Non-Payment/Non-Delivery ($121.3M) |
| 2016 | BEC/EAC ($360.5M) | Confidence Fraud/Romance ($219.8M) | Non-Payment/Non-Delivery ($138.2M) |
| 2017 | BEC/EAC ($676.2M) | Confidence Fraud/Romance ($211.4M) | Non-Payment/Non-Delivery ($141.1M) |
| 2018 | BEC/EAC ($1.30B) | Confidence Fraud/Romance ($362.5M) | Investment ($253.0M) |
| 2019 | BEC/EAC ($1.78B) | Confidence Fraud/Romance ($475.0M) | Spoofing ($300.5M) |
| 2020 | BEC/EAC ($1.87B) | Confidence Fraud/Romance ($600.2M) | Investment ($336.5M) |
| 2021 | BEC/EAC ($2.40B) | Investment ($1.46B) | Confidence Fraud/Romance ($956.0M) |
| 2022 | **Investment ($3.31B)** — first year investment fraud overtakes BEC | BEC ($2.74B) | Tech Support ($806.6M) |
| 2023 | Investment ($4.57B) | BEC ($2.95B) | Tech Support ($924.5M) |
| 2024 | Investment ($6.57B) | BEC ($2.77B) | Tech Support ($1.46B) |
| 2025 | Investment ($8.65B) | BEC ($3.05B) | Tech Support ($2.13B) |

**BEC was the #1 loss category every year 2015–2021, then investment fraud (driven heavily by cryptocurrency investment scams / "pig butchering") overtook it from 2022 onward** and the gap has widened every year since.

## 4. Data-quality notes relevant to any comparison work

- **Category redefinitions break strict year-over-year comparability**: "Phishing" and "Spoofing" were separate categories through 2022, merged into "Phishing/Spoofing" starting with the 2023 report (which retroactively restates 2021/2022 figures under the merged label — e.g. 2022 restated as 321,136 complaints, up from the original 300,497 "Phishing"-only figure). "Harassment/Threats of Violence" and "Terrorism" were similarly merged into "Threats of Violence" around 2021. New categories appear periodically (Botnet, SIM Swap, Data Breach split from Personal/Corporate Data Breach, AI-related descriptor in 2025, Charity as a standalone line in 2025).
- **Contemporaneous vs. retrospective totals sometimes disagree** (not just rounding): 2016 total losses are given as ~$1.33B in the 2016 report itself, but the 2019 report's 5-year retrospective table restates 2016 as $1.5B. No explanation given in either report. Treat any single year's figure as report-vintage-dependent; prefer the contemporaneous (same-year) report's own figure unless doing a same-report multi-year comparison.
- **Internal inconsistencies exist within single reports** (flagged by extraction agents, not resolved): e.g. 2020 report states losses as "$4.1B" in the intro letter vs. "$4.2B" on its own infographic page; 2022 intro says "$10.2B" vs. infographic "$10.3B"; 2021 RAT recovery figures show "$443.48M" in one place and "$433.48M" in that same graphic's own footnote text. These are source-document artifacts, not extraction errors.
- **Ransomware loss figures are explicitly flagged by IC3 itself as undercounts** every year they appear — the adjusted-loss figure excludes business downtime, wages, equipment, and remediation costs, and not all victims report a dollar figure. Ransomware complaint counts (~1,000–3,700/year) and reported losses (~$0.5M–$60M/year) are consistently tiny relative to its real-world impact — a useful caution if this project ever wants to lean on IC3 ransomware figures for calibration.
- **Multi-year historical trend charts are frequently image-only** in these PDFs — bar values don't reliably survive text extraction with correct year alignment. The headline totals table above (§1) was built from each year's own clean "total complaints"/"total losses" summary figure, not from reading chart bars, and should be trusted over any chart-derived series in the raw text files.
- **No business-size or business-vs-individual-victim breakdown exists anywhere in these reports.** All demographic breakdowns are by age, gender (2011–2013 only), state, and country — there is no way to isolate "business victims" from this dataset, which is the main structural reason it can't feed directly into the UK per-business cost estimate.

## 5. Elder fraud (60+) trend, where reported separately

| Year | 60+ complaints | 60+ losses |
|---|---|---|
| 2019 | 68,013 | $835M |
| 2020 | 105,301 | $966M |
| 2021 | 92,371 | $1.68B |
| 2022 | 88,262 (part of general age table; no dedicated section) | $3.1B |
| 2023 | 101,068 (age table) | $3.4B |
| 2024 | 147,127 | $4.885B |
| 2025 | 201,266 | $7.748B |

Elder fraud losses have grown faster than the overall total in recent years (2024→2025: overall complaints +17%, overall losses +26%; 60+ complaints +37%, 60+ losses +59%).

## 6. Full per-year detail

Everything the extraction passes captured — full crime-type tables (both count and loss, all ~26-40 categories per year depending on year), state-by-state breakdowns (all 50 states + DC + territories), age×gender tables, top international countries, ransomware/BEC/crypto/tech-support deep dives, RAT (Recovery Asset Team) recovery figures, and every flagged ambiguity — is preserved in the six research-agent transcripts from this session (not reproduced here for length) and independently recoverable from `data/raw/ic3/text/*.txt`. If a specific figure is needed later (e.g. a particular state's loss total for a particular year), grep the relevant year's `.txt` file directly rather than re-deriving.

---

## 7. Growth decomposition analysis (`src/ic3_crime_types.py`)

Follow-up quantitative analysis built on the crime-type-by-count/by-loss DataFrames (2015–2025; §2/§3 above are the source tables). Investigates which categories actually drive the loss/count totals' growth, rather than looking at the totals alone. All figures below are computed directly by the script (`python3 src/ic3_crime_types.py`), which also writes the underlying long/wide CSVs to this directory.

### 7.1 Loss growth is highly concentrated in ~5 categories; count growth is not

Decomposing category-level deltas against the total change in two windows (2020→2025 and 2022→2025):

| Window | Total loss growth (categories present both years) | Top contributor | Categories needed for ~80% of growth |
|---|---|---|---|
| 2020→2025 | +$15.3B | Investment: 54.3% alone | 4 (Investment, Tech Support, BEC/EAC, Personal Data Breach) |
| 2022→2025 | +$9.36B | Investment: 57.0% alone | 4 (Investment, Tech Support, Personal Data Breach, Government Impersonation) |

**"Growth leader" set** (union of consistent top contributors across both windows): **Investment, Tech Support, BEC/EAC, Personal Data Breach, Government Impersonation.** BEC is included despite slowing growth (only 3.3% of the 2022→2025 increase, down from 7.7% in the 2020→2025 window) because it remains one of the largest absolute-dollar categories throughout — it was the #1 loss category every year 2015–2021 before Investment overtook it (see §3).

**Complaint-count growth is a different, messier story.** It's not "a few grew, the rest stagnated" — several categories are actively *shrinking* in count while others surge, roughly offsetting: Extortion, Investment, Government Impersonation, and Tech Support all grew 1.5–2.8x (2022→2025), while Phishing/Spoofing complaint volume actually *fell* (0.60–0.71x of its earlier value across the two windows — down from a 2021 peak of 323,972 to 191,561 in 2025), alongside declines in Non-Payment/Non-Delivery, Overpayment/419, Identity Theft, Advanced Fee, and Credit Card Fraud.

### 7.2 "Flat total loss" hides two opposite mechanisms

For categories whose total loss barely moved (growth factor <1.2x) in a given window, decomposing into count-growth × avg-loss-per-complaint growth shows two distinct patterns, not one:

**Pattern A — count falling, severity rising per complaint** (your "count down, dollars held" intuition — confirmed, but only for a subset):
| Category | Window | Count growth | Avg loss/complaint |
|---|---|---|---|
| Overpayment/419 | 2020→2025 | 0.20x | $4,645 → $10,437 (+125%) |
| Overpayment/419 | 2022→2025 | 0.35x | $6,200 → $10,437 (+68%) |
| Credit Card Fraud | 2022→2025 | 0.82x | $11,492 → $15,056 (+31%) |
| Identity Theft | 2020→2025 | 0.73x | $5,065 → $5,867 (+16%) |
| Phishing/Spoofing | 2020→2025 | 0.71x | $1,004 → $1,127 (+12%) |

Overpayment/419 is the standout case: complaint volume collapsed to a fifth of its 2020 level, but total loss only fell by half — average loss per complaint more than doubled.

**Pattern B — count rising, severity falling per complaint** (opposite mechanism, same flat-total appearance):
| Category | Window | Count growth | Avg loss/complaint |
|---|---|---|---|
| Ransomware | 2020→2025 | 1.46x | $11,786 → $8,950 (−24%) |
| Ransomware | 2022→2025 | 1.51x | $14,404 → $8,950 (−38%) |
| Botnet | 2022→2025 | 1.26x | $30,105 → $19,383 (−36%) |
| Data Breach (Corporate) | 2022→2025 | 1.42x | $164,337 → $109,826 (−33%) |
| Real Estate/Rental | 2022→2025 | ~flat | $33,848 → $22,244 (−34%) |

Takeaway: a flat aggregate dollar figure is not informative on its own about a category's underlying dynamics — always decompose into count × severity before interpreting "no change."

### 7.3 Pooling everything except the 5 growth leaders

Excluding Investment, Tech Support, BEC/EAC, Personal Data Breach, and Government Impersonation, and summing the ~20 remaining categories ("pooled rest"), 2015→2025:

| | 2015 | 2025 | Nominal growth | Share of total loss |
|---|---|---|---|---|
| Leaders (loss) | $421M | $15.94B | 37.9x | 34% → 79% |
| Pooled rest (loss) | $819M | $4.35B | 5.3x | 66% → 21% |
| Leaders (count) | 41,107 | 245,426 | 6.0x | 11% → 32% |
| Pooled rest (count) | 327,655 | 523,026 | 1.6x | 89% → 68% |

The pooled rest is **not stagnant** — it grew substantially in both dollars and volume — but is being outpaced roughly 7x (loss) and 3.7x (count) by the leader categories, and its share of total dollar losses fell from two-thirds to about a fifth of the total over the decade. Even within the pooled rest, average loss per complaint rose 3.3x ($2,500 → $8,322, 2015→2025) — a real, if much shallower, severity trend running through the "rest" bucket too.

**Inflation-adjusted (CPI-U, deflated to constant 2025 USD; cumulative US inflation 2015→2025 = 35.8%, source: BLS-derived historical index via usinflationcalculator.com, fetched 2026-07-09):**

| | Nominal growth | Real growth | Constant-2025-$ trajectory |
|---|---|---|---|
| Pooled rest | 5.31x | **3.91x** | $1.11B → $4.35B |
| Leaders | 37.87x | **27.88x** | $572M → $15.94B |
| Total | 16.37x | **12.05x** | $1.68B → $20.30B |

Inflation explains only a small slice of the pooled rest's nominal growth — it remains genuine, substantial real growth (real dollars quadrupled), just dwarfed by the leaders' real growth, which stays enormous (~28x) even after stripping out price effects entirely. This was never an inflation story. Note also that the pooled rest's *real* trajectory is not monotonic: it roughly tripled to a 2021 peak (~$3.51B in constant 2025 dollars), fell back to ~$2.92B by 2023, then resumed growth to $4.35B by 2025 — a genuine multi-year dip that the leaders' more consistently upward path does not show.

### 7.4 Investment fraud's growth is overwhelmingly crypto-driven

Crypto-specific investment fraud loss figures (narrative call-outs in each report's dedicated Investment/Crypto section, not the main crime-type table; rounded to 2-3 sig figs in the source text, cross-validated via each report's own YoY % claim against the prior year's figure):

| Year | Total Investment loss | Crypto-investment loss | Crypto share |
|---|---|---|---|
| 2021 | $1.46B | $907M | 62.3% |
| 2022 | $3.31B | $2.57B | 77.6% |
| 2023 | $4.57B | $3.96B | 86.6% |
| 2024 | $6.57B | $5.80B | 88.3% |
| 2025 | $8.65B | $7.23B | 83.6% |

2021→2025: total Investment loss grew 5.94x (+$7.19B). **Crypto-specific investment fraud accounts for 87.9% of that increase** (+$6.32B, 7.97x growth on its own); non-crypto investment fraud (traditional securities, real-estate schemes, etc.) grew a comparatively modest 2.59x, contributing only 12.1% of the increase. Crypto's *share* of Investment peaked around 2024 (88.3%) and dipped slightly in 2025 (83.6%) — non-crypto investment fraud grew faster than crypto between 2024 and 2025 (+84% vs. +25%), too recent a shift to call a reversal, but worth watching if this analysis is revisited.

Combined with §7.1 (Investment ≈ 55-57% of all IC3 loss growth in recent years), this means **crypto-investment fraud specifically is the single largest identifiable driver of total IC3 loss growth since 2021** — roughly half of all incremental dollars reported to IC3 in that period trace back to it.
