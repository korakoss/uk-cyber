# Project Notes

Started fresh on 2026-10-05. The old notes are in `archive/NOTES_old_2026-07_to_2026-10.md`. They are history only, not a working document. Only look in them for a specific detail, and check anything taken from them before relying on it.

This file is being filled in by going through the project together with the user, one topic at a time.

## Goal (agreed 2026-10-05)
- Estimate the total yearly cost of cybercrime incidents to UK businesses. Sole traders are left out because the survey doesn't include them.
- "Cybercrime" and "cost" mean whatever the survey asks about. We don't redefine them.
- The survey is the main input. The job is to scale it up to all UK businesses, bringing in supporting data only where needed (for example, how many businesses there are).
- We may reason informally from outside evidence or well-known cases (for example, to judge how big the largest losses can get), but we don't plan to bring in any major new data source.

## Data (agreed 2026-10-05)
- UK Cyber Security Breaches Survey 2025: 3,835 organisations, including charities and schools; we use the businesses. Survey weights scale the sample up to the business population. Files: `data/raw/data.sav`, `data/proc/data.csv`; codebook in `data/proc/csbs.txt`. UK business counts by size come from ONS (`data/external/`).
- Main questions used, all about the last 12 months:
  - which kinds of attack the firm had (a tick-list);
  - how often it was attacked (a band from "once" to "several times a day");
  - the cost of its single worst incident, as a band split into parts (staff time, payments, disruption, and so on);
  - for some attack types, counts of attempts and successes, what happened, and how long recovery took. Some questions went to only half the sample.
  - A total-cost-for-the-year question exists, but very few firms answered it.
- Main limits:
  - Only the worst incident is priced, not the year's total.
  - Costs come as bands, not exact figures.
  - There are few large firms and few big losses, and the top cost bands are nearly empty.
  - Answers are self-reported, and some firms skip the cost questions.
- Nobody has gone through all 527 columns by hand, so other useful columns may exist.
- User's view: the highest attack-frequency answers ("daily", "several times a day") are less trustworthy. This probably matters little, because those attacks are likely to cost almost nothing.

## Facts (being collected one at a time with the user)

### F1. Size, attack frequency and worst-incident cost (`src/estimation/core_vars.py`, 2026-10-05)
Note: `disrupta` is the TYPE of the worst breach, not its cost. The worst-incident cost is `damage_bands`.
Businesses only (questtype = 1). Shares below are weighted.
- **Being attacked rises with size.** The share attacked is 40% for Micro, 49% Small, 65% Medium and 68% Large.
- **Among attacked firms, how often they're attacked barely changes with size.** About 20% were attacked once, about 28% less than monthly, and about 5–8% several times a day, for every size band. Large firms have slightly more "several times a day" (14%, n = 129).
- **Worst-incident cost rises clearly with size.** "No cost" is 57% for Micro, 47% Small, 42% Medium and 35% Large. A cost of £5k or more is 2%, 6%, 8% and 21%.
- **Being attacked more often doesn't mean a costlier worst incident.** There's no upward trend across frequency bands, within Micro or within Small+. If anything, firms attacked only once are slightly less often at "no cost".
- **Reverse view:** Small+ firms with a worst incident of £5k–50k were more often attacked "once" or "less than monthly" (65%) than firms at "no cost" (42%). The samples are small.
- **Top of the range:** no business is above £500k. Eight are at £100k–500k (2 Micro, 2 Small, 1 Medium, 3 Large).

### F2. Crime types (`src/estimation/crime_types.py`, 2026-10-05)
Attacked businesses (n = 1,101). Shares are weighted. "Targeted" phishing = at least one phishing email containing the recipient's personal details (`phishcon_bands`). This is the survey's own measure of "specifically targeted".
- **How common:** phishing 85% of attacked firms, impersonation 35%, malware 18%. Ransomware, DoS, bank hacking and takeover are about 6–7% each. Outsider access, staff access, eavesdropping and other are each 4% or less. About half of phishing firms ticked nothing else.
- **Phishing counts:**
  - About half of phishing firms had at least one targeted email (answered by 822 of 957 phishing firms).
  - 14% had someone engage with a phishing email at least once. Engagement more than 3 times is almost never reported.
- **High attack frequency vs mass phishing (user's recollection not confirmed).**
  - Firms attacked daily or several times a day are no more often phishing-only than firms attacked monthly or weekly (about half in each).
  - They report MORE targeted phishing. 30–38% of them had 21 or more targeted emails, against 2–3% of firms attacked less than monthly.
  - Caveat: "contains personal details" may include mass emails that just insert the recipient's name, so the survey's "targeted" may partly be bulk mail.
  - Firms attacked only once are the least phishing-heavy (59% phishing, 49% impersonation).
- **Cost by the type of the worst incident** (`disrupta`; the share costing £5k or more is given in brackets):
  - mass phishing: 74% no cost (0%)
  - targeted phishing: 52% no cost (2%)
  - impersonation: 41% no cost (3%)
  - takeover: 35% no cost (3%)
  - DoS: 32% no cost (5%)
  - bank hacking: 27% no cost (10%)
  - malware: 38% no cost (12%)
  - ransomware: 3% no cost (41%)
  
  This matches the user's ordering: mass phishing < targeted phishing ≈ impersonation < the rarer types < ransomware.
- **Means by type are unreliable.** For impersonation, ransomware, DoS and bank hacking, a single firm (a £100k–500k incident) makes up 70–82% of the weighted mean. Without that firm, the means are about £1.3k, £8.8k, £1.4k and £2.1k.
- **Firms that ticked only one type are cheaper** than firms whose worst incident is that type. For example, impersonation-only firms are 62% no cost and 1% at £5k or more. Big costs come with several types ticked; to be looked at separately.

### F3. Which types occur together (`src/estimation/occurrence.py`, 2026-10-05)
Attacked businesses only. Shares are weighted.
- **One attack can tick several boxes (confirmed).** 40% of firms attacked "once" ticked 2 or more types, mostly phishing + impersonation (43 firms). Next come phishing + malware (8) and phishing + impersonation + bank hacking (5). The share ticking 2 or more is the same for "once" as for "less than monthly" or "monthly". It only rises for weekly or daily firms.
- **Every type goes with every other type.** On ALL businesses (including those not attacked), every pair of types co-occurs more often than chance would predict: 2–2.5 times for pairs involving phishing, 3–4 times for impersonation, and 4–9 times among the rarer types. [Correction 2026-10-05: an earlier version of this note, using attacked firms only, said phishing and impersonation were about independent of the rest. That was an artefact. Looking only at attacked firms hides the "attacked at all" part of the shared pattern.]
- **The rarer types go together.** Malware, ransomware, DoS, bank hacking, takeover, outsider access and staff access each co-occur with one another at about 2–4 times the rate chance predicts. This holds within Micro and within Small+ separately, so it isn't just a size effect. Outsider and staff access are especially tied (about 15 times chance, small n).
- **Firms are either narrow or broad.** Compared with independent types (same rates):
  - one type: 58% observed vs 41% expected
  - two or three types: 35% vs 56%
  - four or more: 7% vs 3%
- **What "one-dimensional" means here.** There is a single hidden firm-level quantity (call it exposure). Given a firm's exposure, the types occur independently; firms with high exposure get more types of everything. Under this picture, all the co-occurrence comes from that one quantity, and every type rises with it by roughly the same amount.
- **Evidence (old work, 2026-09-26; not yet re-run):**
  - `type_cooccurrence_structure.py`: on all firms, the correlations between yes/no flags (estimated as if each flag were an underlying continuous scale cut at a threshold) are mostly 0.4–0.7 for every pair. One leading factor explains 56% of the variation (50% Micro, 63% Small+), with near-equal loadings of 0.27–0.36. The second factor is about noise level (eigenvalue around 1) and points in different directions for Micro and Small+.
  - `latent_on_counts.py`: the same quantity also raises attack COUNTS. Firms hit by 2 or more other types have 2–3 times the phishing count at every quantile.
  - Caveat from `frailty_flag_structure.py`: grouping into 4 type-groups, a formal check for a single factor partly failed. Phishing + impersonation and ransomware + other serious types pair up beyond the one factor. This is plausibly because one incident ticks related boxes (a spoofed email is ticked as both phishing and impersonation; ransomware is also ticked as malware), not a second kind of exposure.
  - My attacked-only check (2026-10-05) is consistent: each rare type climbs steadily with the number of other types ticked (ransomware 2%, 7%, 20% at 1, 2, 3+ others).
- **Shape of the hidden exposure from the FLAGS ALONE (`frailty_shape_flags.py`, 2026-09-26; not re-run).**
  - The flags can't tell separate levels from a smooth scale: two or three levels and a continuous scale explain about the same share of the co-occurrence.
  - The best-fitting levels are ordered: every type's hit rate rises from one level to the next, so they are rungs of one scale.
  - Micro: 2 levels are enough.
  - Small+: 3 levels, 67% / 30% / 4%. The top 4% are hit by most types.
  - The big "quiet" group needs no model to see: about half of all businesses (60% of Micro) report no attack at all.
- **Shape with attack COUNTS added (joint model: flags + frequency bands + phishing counts + worst cost; old fits, 2026-09-24/26; not re-run).** Attack rates per year come from the counts and frequency bands, not from the flags.
  - **Across firms it is lumpy and skewed, not a smooth bell curve.** A free fit, with no shape assumed (`frailty_free_mixture.py`), settled on three levels: quiet 55% (about 0.03 attacks a year per kind, i.e. essentially never attacked), middle 37%, and hot 9%. Two levels fit clearly worse. A smooth bell-shaped (lognormal) spread also fit worse than three levels.
  - **It doesn't scale every type equally.** Going from middle to hot: targeted phishing x2.5, impersonation x4, mass phishing x7, serious types (malware, ransomware, DoS, hacking and so on) x9. Serious types for hot vs quiet firms: about x120. So the top level is mostly about serious attacks and phishing volume. (The "near-equal loadings" result is on a correlation scale; the two results are not in conflict.)
  - **The exposed share rises with size.** In the two-level version it is 37% Micro, 42% Small, 57% Medium and 66% Large.
  - **Caveats:**
    - A free fit like this always lands on a few separate levels even when the truth is smooth. So "three levels" really means a big quiet mass plus a long upper tail. It does not prove that firms come in three kinds.
    - The fits depended on turning the frequency bands into attack counts, and that conversion fit badly (too few "once" answers).
    - In that old model, the national cost per firm barely changed across shapes (£729–732). The shape mattered for which firms carry the cost, not for the average.
- Open: whether the "broad" firms are firms exposed to more kinds of attack, or one big incident ticking several boxes (old notes leaned towards the latter for costly firms). This is to be looked at with cost.

### F4. More phishing goes with more other types (`src/estimation/phishing_volume_vs_breadth.py`, 2026-10-05)
Phishing-ticking attacked businesses. Plain cross-tabs, weighted. "Serious" = any type other than phishing and impersonation.
- **Yes, gradually and in both size groups.** By exact phishing count (`Cybercrime_phishsum`, 566 answers), the share with any serious type is:
  - 1 attack: 32%
  - 2–5: 38%
  - 6–20: 42%
  - 21–100: 52%
  - over 100: about 70%
  
  The number of serious types and the impersonation share rise the same way. Micro and Small+ are nearly identical.
- **The same holds by number of targeted phishing emails.** Any serious type: 21% for none, rising to about 65% for over 100.
- **Overall attack frequency (`freq`) shows it only weakly and unevenly**, presumably because it is a coarse answer covering all types.
- **Reverse view:** firms with 2 or more serious types have higher phishing counts, mostly in the upper part of the range. 75th percentile: Micro 41 vs 12 for phishing-only firms; Small+ 100 vs 15. Medians differ less: 6 vs 4, and 10 vs 3.
- **Caveats:**
  - Who answered the exact count is uneven: 39% of phishing-only firms vs 79% of firms with 2 or more serious types.
  - Part of the link may be direct rather than shared exposure: phishing is a common way in for other attacks, so more phishing can itself lead to more of the other types.

## Cost

### F5. Worst-incident cost rises steeply with the number of types ticked (`src/estimation/cost_breadth.py`, 2026-10-05)
Attacked businesses with a cost answer (n = 982). Shares are weighted.
- **Share with no cost, and with £5k or more, by number of types ticked:**

  | Types ticked | No cost | £5k or more |
  |---|---|---|
  | 1 | 66% | 1% |
  | 2 | 44% | 3% |
  | 3 | 25% | 7% |
  | 4+ | 8% | 28% |

  Same shape within Micro and within Small+.
- **The same holds counting only serious types.** Ignoring phishing and impersonation, 3+ serious types gives 41–42% at £5k or more in both size groups.
- **Concentration:** firms with 3 or more types are 15% of attacked firms but carry about 77% of the weighted cost (band midpoints). Firms with 1 type are 63% of firms and 7% of the cost.
- **Within a given number of types, attack frequency doesn't matter.** For 3+ types, the share at £5k or more is 17% for "once", 15% for "less than monthly to monthly" and 15% for "weekly or more". For 1 type it is about 1% at every frequency. So what predicts a big loss is how broad the firm's attacks were, not how many attacks it had.
- Means per group are again driven by single firms (25–90% from the top firm), so read the shares, not the means.
- Open next: is breadth a cause (more exposed firms) or a result (one big incident ticking several boxes)?

### Earlier claims on breaches and big breaches (from the old notes, reviewed with the user 2026-10-05; NOT yet re-checked)
- **What "breach" means.** It is a firm-level marker on the WORST incident. It is set by the survey's outcome items (Q56A: money stolen, data lost, systems corrupted, and so on), or by consequence items: restore took a day or more, staff stopped work, recovery costs, revenue loss. Old name "W2"; user decision 2026-10-02. It is not observed per type. Per-type breach rates were inferred from firms hit by only 1–2 type groups.
- **Breaches are much costlier (agreed).** Narrow firms with an outcome marker: 24% no cost, 8% at £5k or more. Firms with no marker at all: 72% no cost, about 0% at £5k or more.
- **Breaches are rare for phishing and impersonation, not for the rarer types (correction to the user's recollection).** Estimated chance that a hit becomes a breach: phishing about 5%, impersonation about 5%, ransomware about 24%, other serious about 37%. The rarer types are rare to be hit by, but a hit often becomes a breach.
- **"Steeper than max of independent breaches" (agreed), with one nuance.** The breadth used was the number of type GROUPS (phishing / impersonation / ransomware / other serious), not raw types. Independent per-group breaches, with the worst taken as the maximum, predict about 6% at £5k or more for 3+ groups. Observed: 34%. Fine for 1–2 groups (8% observed vs 6% predicted).
- **Big vs ordinary breaches is a modelling choice, not a split the data forced.** In the fitted model, each breach is "big" with a chance of 1% / 6% / 29% / 71% for 1 / 2 / 3 / 4 groups hit (big: median about £4.5k, wide spread). A simpler alternative, where breach cost just scales about x2 per extra group, fit equally well and was slightly preferred on AIC. Both forms fit the data. They differ about 1.6x in average cost for broad firms (tail extrapolation).
- **Big is tied to breadth (agreed).** This persists within exposure levels: exposure adds nothing once breadth is known.
- **Homogeneous apart from bank hacking (roughly agreed).**
  - Among broad, costly breached firms: no detectable effect of ransomware (x0.8–1.2), 4 vs 3 groups or size.
  - Bank hacking: x9 [3–26] among broad breached firms (n = 23), but not among narrow firms (x0.6; there it is often money stolen but cheap, median £350).
  - Later checks suggested the bank-hacking tick in these firms is mostly part of the incident itself, not separate bank attacks: 10 of 16 had no separate bank-attack attempts. So it reads as "the intrusion reached the money", not a separate attack type.
  - Weaker side note: broad firms with ransomware or malware but without phishing + impersonation looked bimodal (handled vs disaster), n = 12.

### F6. A breach makes the worst incident much costlier, for every type with data (`src/estimation/breach_cost.py`, 2026-10-05)
Breach marker (user decision 2026-10-02): any Q56A outcome (`outcome_any`), OR restore took a day or more, OR staff stopped from working / revenue loss / recovery costs. 244 of 982 attacked firms with cost answers (22% weighted).
Note: Q56A and Q57 ask about ALL breaches in the year, not just the worst one. So the marker belongs to the firm's year, not strictly to the priced incident.
- **By type of the worst incident** (no cost → no cost; share at £5k or more in brackets), not breached vs breached:

  | Type | Not breached | Breached |
  |---|---|---|
  | phishing | 68% (0%) | 31% (8%) |
  | impersonation | 52% (0%) | 7% (13%) |
  | malware | 84% (0%) | 9% (19%) |
  | takeover | 62% (0%) | 12% (6%) |
  | bank hacking | 99% (0%) | 15% (12%) |

  - DoS is weaker: 45% (9%) not breached vs 26% (3%) breached, n = 7 and 20.
  - Ransomware worst incidents are nearly all breached (23 of 28): 0% no cost, 53% at £5k or more.
  - The access types and "other" have under 10 firms each.
- **Single-group firms show the same:**
  - phishing only: 71% → 34% no cost;
  - impersonation only: 67% → 28%;
  - other serious only: 71% → 36%.
- **The stricter marker (outcome item only) gives the same picture.**
- **Caveat:** part of the marker is itself about cost (recovery costs, revenue loss, time to restore), so "breach → costlier" is partly built in. The outcome-only version avoids most of that and still shows it.
- Not breached firms almost never reach £5k, whatever their breadth: 5 of 738 in total.

### F7. No visible "big vs ordinary" split among breaches: broader breaches shift the whole cost distribution up (`breach_cost.py`, 2026-10-05)
- The fact the old big/ordinary model was built to explain: among breached firms, the high-cost share rises with breadth much faster than independent breaches would give (old check: 34% vs 6% at £5k or more for 3+ groups). A simpler model (cost scale about x2 per extra group) explained it equally well.
- **Breached firms, weighted quantiles of worst cost:**

  | | q10 | q25 | median | q75 | q90 |
  |---|---|---|---|---|---|
  | narrow (1–2 groups), Micro | none | none | £100–500 | £1–5k | £1–5k |
  | broad (3–4 groups), Micro | <£100 | £100–500 | £500–1k | £5–10k | £10–20k |
  | narrow (1–2 groups), Small+ | none | <£100 | £100–500 | £1–5k | £5–10k |
  | broad (3–4 groups), Small+ | <£100 | £500–1k | £1–5k | £10–20k | £20–50k |

  Every quantile moves up by about one or two bands. "No cost" falls from 34% (1 group) to 5% (3+ groups).
- **This looks like a whole-distribution shift, not a new high bump on top of an unchanged low part.** A 2-cluster story would leave broad firms' lower quantiles where narrow firms' are. They aren't. The band histograms show no clear second peak; the dip at £500–1k between £100–500 and £1–5k is most likely band width (the £1–5k band is 5 times wide).
- **Caveats:** coarse bands; few broad breached firms (Micro 21, Small+ 88); 4-group firms only 26.
- Reading: the data support "broader breaches are costlier across the board". Nothing here requires a separate class of big breaches.

### F8. The breadth excess over independence: breached and non-breached firms (`src/estimation/breadth_vs_independence.py`, 2026-10-05)
- **Method (plain simulation, no fitted model).** Three groups: P = phishing, I = impersonation, S = all other types incl. ransomware. For a firm hit by a set of groups, draw for each group a random firm hit by ONLY that group (by weight), and take its worst cost and breach flag. The simulated worst = the largest draw; the simulated firm is breached if any draw was. Pooled over sizes. Single-group pools: P 384, I 76, S 44 firms.
- **Results** (observed vs independent):

  | Groups hit | Firms | n | £500 or more | £5k or more |
  |---|---|---|---|---|
  | all three (PIS) | breached | 100 | 73% vs 39% | 29% vs 7% |
  | all three (PIS) | not breached | 79 | 19% vs 6% | 1% vs 0% |
  | two (PI / PS) | breached | 32 / 39 | | 10–12% vs 5–7% |
  | two (PI / PS) | not breached | | 12–14% vs 3–6% | about 0 |

  IS has only 15 firms.
- **Reading:**
  - The excess over independence is NOT limited to breaches. Non-breached broad firms also cost more than independent incidents would give, but only in the £500–5k range; they essentially never reach £5k.
  - The excess at £5k or more, which is where the money is, is in breached firms. It is strongest with all three groups (about 4 times), and present but mild with two groups (about 2 times).
- **Side observation:** broad firms are breached LESS often than independent groups predict (PIS 54% vs 68%; PS 45% vs 64%). The S-only pool is small and maybe unusual (61% of S-only firms are breached), so this is weak.
- **Caveat:** the non-breached excess could partly be breaches the marker misses (outcome questions cover the whole year; soft items may be skipped).

### F9. "Broad firms are breached less often than independence predicts": real, but it comes from the 'other types' group (`breadth_vs_independence.py`, 2026-10-05)
User flagged this as a fact to pay attention to: it bears on WHY breadth matters. Breach counts are unknown, so the competing stories are (a) several independent breaches adding up and (b) one breach spreading across types. (a) predicts breached shares at or above independence; (b) allows below.
- **Breached share, observed vs independent** (union of single-group rates); gap with bootstrap 90% interval:

  | Set | Size | Observed | Independent | Gap |
  |---|---|---|---|---|
  | PI | all | 16% | 17% | -0.01 [-0.11, +0.07] (no gap) |
  | PS | all | 45% | 63% | -0.19 [-0.37, -0.01] |
  | PIS | all | 54% | 68% | -0.14 [-0.29, +0.01] |
  | PS | Small+ | 37% | 80% | -0.42 [-0.64, -0.19] |
  | PIS | Small+ | 54% | 83% | -0.29 [-0.43, -0.12] |
  | PS / PIS | Micro | | | -0.10 to -0.14, intervals include 0 |

  The outcome-item-only marker gives the same pattern.
- **The gap appears ONLY in sets that include S (other types), and comes from S-only firms' very high breach rate.** Single-group breach rates: P 7%, I 12%, S 61% (S-only n = 44). Phishing + impersonation combine just as independence predicts.
- **Two readings, not yet separated:**
  1. Spreading / one incident ticking several boxes: in broad firms one breach accounts for several ticks, so the breach rate per tick is lower.
  2. The S tick means different things in narrow and broad firms. A firm that ticks ONLY malware / takeover / etc. may tick it because it was harmed (blocked attempts go unnoticed or unreported), while heavily attacked broad firms also tick S for attempts. Then S-only firms are the wrong baseline.
- Open: find a way to tell 1 from 2, e.g. through what S-only firms look like (attempt counts, frequency, outcomes) compared with S in broad firms.
- **Logic correction (2026-10-05).** Spreading on its own does NOT predict below-independence breach rates. A breach that starts in one group and spreads moves a breached firm into a broader set, which RAISES broad sets' breached share. So the below-independence fact points more towards reading 2 (S ticks in broad firms are often unsuccessful attempts; S-only firms are tick-because-harmed), or towards broad firms being harder to breach per attack. It is not positive evidence for spreading.
- **First test (`src/estimation/s_tick_meaning.py`).** Per-type success items exist for malware (`virussoft_comb`), takeover (`tkvrsuc_comb`), DoS (`dossoft_comb`) and ransomware (`ranssoft_comb` = ransom demanded); most ticking firms answered them.
  - Share of ticks that were successful, S-only vs P+I+S: malware 21% vs 9%, takeover 35% vs 17%, DoS 41% vs 22%. Ransomware goes the other way: 22% vs 39%.
  - Pooled over malware / takeover / DoS, the firm had any success among its ticked types: S-only 30% (6 of 22), S + one other 10% (7 of 98), P+I+S 17% (28 of 151).
  - Direction consistent with reading 2, but the S-only group is tiny (22 firms, 6 successes).
  - Also: per-type success is far below the firm-level breach marker everywhere (S-only 30% vs 61%). The marker is mostly set by other items (outcomes / soft consequences), not by these per-type successes.

### F10. With all firms as the baseline, breach prevalence fits independence; the odd group is S-only firms (`src/estimation/breach_independence_fit.py`, 2026-10-05)
- **Method (user's suggestion: don't use single-group firms as the baseline).** Under independence, P(no breach | set) = product of per-group escape chances. The three escape chances were fitted to all 7 sets at once (weighted binomial likelihood), then observed vs fitted breached share was compared per set. Bootstrap 90% intervals come from refitting on resampled firms. All attacked businesses (n = 1,101).
- **Fitted per-channel breach chance:** phishing 6%, impersonation 9%, other types 43%. About the same in Micro and Small+.
- **Observed vs fitted:**
  - P, I, PI, IS and PIS all fit. PIS: observed 52% vs fitted 51%.
  - S-only firms are breached MORE than fitted: 61% vs 43%, gap +0.18 [+0.04, +0.32]; Small+ 76% vs 43%.
  - PS is somewhat less breached: 39% vs 46%, gap -0.08 [-0.14, -0.01]; Small+ -0.16.
- **Reading:** the F9 "broad firms breached below independence" mainly reflects the S-only baseline. Firms whose only ticks are "other" types are unusually often breached, consistent with "ticked because it hurt". With a baseline drawn from all firms, breach prevalence in broad firms (including all three groups) is what independent per-channel breach chances predict. The PS shortfall is the one remaining departure (modest; strongest in Small+, n = 56).
- **For the A vs B question:** prevalence is consistent with A (independent chances per channel). Prevalence alone doesn't rule out B. Spreading would push broad sets' breach rate UP, and no such excess is seen in PIS, which mildly argues against spreading being common, or at least against it creating broad firms out of narrow ones.
- **How solid is F10? Not very (checked 2026-10-05, same script).**
  - "PIS fits" is close to built in. P's rate is pinned by the 390 phishing-only firms, so the S rate is set mostly by the 212 PIS firms themselves.
  - The informative check is whether one S rate fits all S-containing sets. S chance implied by each set, given the fitted P and I rates: S-only 61% (n = 47), PS 35% (n = 120), IS 46% (n = 19), PIS 43% (n = 212). They disagree.
  - S lumps several types. Mean number of S types ticked: S-only 1.16, PS 1.37, IS 1.28, PIS 1.94 (25% of PIS firms tick 3 or more). At the TYPE level, independence would give PIS firms a HIGHER S breach chance than S-only firms, yet theirs is lower (43% vs 61%). So at type level the "breached below independence" pattern partly comes back. It still leans on the small S-only group, but PS vs PIS (more S types, more breached: 35% vs 43%) is in the expected direction.
  - Net: breach prevalence data neither clearly fits nor clearly breaks independence. The grouping choice (lumping S) and the small S-only group decide the answer.
- **How solid is the F8 cost excess? Moderately (checked 2026-10-05, `breadth_vs_independence.py`).** Breached PIS firms; observed vs independent; bootstrap 90% interval on the £5k-or-more gap:

  | Variant | n breached | £500 or more | £5k or more | Gap |
  |---|---|---|---|---|
  | one draw per group | 100 | 73% vs 39% | 29% vs 7% | [+0.08, +0.34] |
  | one draw per S TYPE ticked (accounts for broad firms ticking more S types) | 100 | 73% vs 46% | 29% vs 10% | [+0.04, +0.33] |
  | without ransomware | 72 | 70% vs 44% | 21% vs 9% | [-0.02, +0.27] |
  | with ransomware | 28 | 81% vs 54% | 51% vs 13% | [+0.09, +0.63] |
  | Micro | 16 | | 24% vs 9% | interval includes 0 |
  | Small+ | 84 | | 35% vs 10% | [+0.09, +0.40] |

  PS (two groups): 12% vs 8%, no clear excess.
- **Reading:**
  - The direction holds through every check. The size is uncertain: somewhere between about 1.5x and 4x at £5k or more.
  - It is clearest in Small+ and in firms that ticked ransomware. Ransomware hardly appears outside broad firms (5 of 44 S-only firms tick it), so "ransomware is costly" and "breadth makes breaches costly" can't be cleanly separated.
  - Without ransomware the excess at £500 or more is still clear (70% vs 44%), while at £5k or more it is borderline.
- **Baseline bias:** the comparison still uses single-group firms (breached pools: P 28, I 10, S 27). [Corrected 2026-10-05 by F12: an earlier claim that this bias makes the true excess larger is wrong. S-only breached firms are mostly CHEAP: often single events with an outcome but little cost. So the S pool may push the independence prediction DOWN and overstate the excess. Direction unclear.]

### F11. Breadth matters without ransomware; ransomware adds on top (`src/estimation/ransomware_vs_breadth.py`, 2026-10-05)
Breached firms. Breadth = number of types ticked OTHER than ransomware. Cells give n, then £500 or more / £5k or more (weighted):

| | 0–1 types | 2 types | 3 types | 4+ types |
|---|---|---|---|---|
| no ransomware | 59, 31% / 4% | 56, 52% / 8% | 50, 69% / 11% | 34, 63% / 34% |
| ransomware | 7, 48% / 13% | 7, 17% / 17% | 10, 70% / 59% | 21, 92% / 55% |

- **Without ransomware, cost still rises with breadth.** The £5k-or-more share jumps at 4+ types (34%), and £500 or more rises steadily. So there is a breadth fact beyond ransomware.
- **At the same breadth, ransomware firms are costlier** (3 types: 59% vs 11%; 4+: 55% vs 34%). Within ransomware firms, breadth also matters: narrow ransomware firms are mostly cheap, though n is tiny. All ransomware firms (n = 67) are breached 54% / 47% / 71% / 84% across the breadth bands.
- **Not-breached firms:** £500 or more rises with breadth (3% → 30%), but £5k or more stays at about 1% everywhere.
- **Caveat:** this table shows that cost rises with breadth, not that it rises faster than independence would give. Independence also predicts a rise (the maximum of more draws). The earlier per-type independence check without ransomware was borderline at £5k (21% vs 9%, interval [-0.02, +0.27]). The steep jump at 4+ non-ransomware types (8–11% → 34%, n = 34) is the part most likely to exceed independence. Not yet tested against an independence prediction by breadth band.
- **Independence test by breadth band, no ransomware (same script, 2026-10-05).** Pools: single-group firms without ransomware (P 384 / 28 breached, I 76 / 10, S 39 / 23); one S draw per S type ticked. Breached firms, observed vs independent, bootstrap 90% interval on the gap:

  | Types ticked | n (breached) | £500 or more | £5k or more |
  |---|---|---|---|
  | 2 | 256 (56) | 52% vs 35% [-0.01, +0.40] | 8% vs 5% [-0.04, +0.14] |
  | 3 | 109 (50) | 69% vs 36% [+0.08, +0.57] | 11% vs 5% [-0.06, +0.19] |
  | 4+ | 53 (34) | 63% vs 44% [-0.14, +0.50] | 34% vs 7% [+0.04, +0.48] |

  - **Without ransomware there is still an excess over independence.** In the middle range (£500 or more) it is clear at 3 types. In the big-cost range (£5k or more) it is clear only at 4+ types, resting on 34 breached firms.
  - Breached prevalence is again far below this independence prediction (28/35, 50/70, 55/89%). That is the S-only baseline problem from F9/F10, made worse by giving every S type the S-only breach rate. The cost prediction uses the same small pools, so treat the sizes as rough.

### F12. What is odd about firms whose only ticks are "other" types (`src/estimation/s_only_firms.py`, 2026-10-05)
47 firms. Firm-by-firm listing in the script output.
- **Mostly a single event.** 45% were attacked "once", against 17% for firms that tick S along with phishing or impersonation. Most tick a single S type (mean 1.16).
- **They don't report phishing**, which 85% of attacked firms do. They look like firms that report "the thing that happened", not their background of attempts.
- **Often breached, but cheaply.** 61% breached (48% via an outcome item). The outcomes are things like temporary loss of access, services down, a third-party loss, or money stolen from a bank account. Of the 28 breached, only 3 cost £5k or more; most are under £500.
- **Per type, breached share S-only vs S with phishing / impersonation:** malware 67% (n = 14) vs 39% (n = 211); ransomware 98% (5) vs 58% (75); bank hacking 79% (14) vs 68% (63); DoS 74% (7) vs 65% (68); takeover 35% (6) vs 61% (69).
- **Reading:** nothing is wrong with the S types themselves. The S-only firms are a peculiar, small group whose single tick usually describes one actual incident. In broad firms, S ticks more often include attempts that did nothing. So S-only firms are a poor baseline for "what an S hit is like" in other firms:
  - their breach RATE is too high (F9/F10);
  - their breach COST is low, so cost-excess estimates built on them (F8, F11) may be overstated.
- Implication: the independence comparisons so far all lean on this baseline. A baseline that doesn't depend on S-only firms is needed before the size of the breadth excess can be trusted.

### F13. Per-type breach chances from all firms: breach prevalence is consistent with independence at type level (`src/estimation/type_breach_rates.py`, 2026-10-05)
- **Method.** P(no breach | ticked types) = product of per-type escape chances. 11 escape chances were fitted on all 1,101 attacked businesses (weighted), with bootstrap 90% intervals.
- **Breach chance per tick:**
  - phishing 6% [4–9]
  - impersonation 9% [5–13]
  - malware 19% [12–28]
  - other 25% [7–44]
  - ransomware 35% [18–53]
  - takeover 36% [15–53]
  - DoS 51% [28–68]
  - outsider access 52% [0–82]
  - bank hacking 59% [40–73]
  - staff access and eavesdropping: undetermined
- **Fit by number of types ticked (observed vs fitted):**

  | Types | n | Observed | Fitted |
  |---|---|---|---|
  | 1 | 510 | 11% | 9% |
  | 2 | 338 | 25% | 24% |
  | 3 | 138 | 48% | 48% |
  | 4 | 56 | 45% | 66% |
  | 5+ | 59 | 79% | 83% |

  Only the 4-type group departs (gap -0.21, n = 56). 5+ fits again, so this is plausibly noise.
- **Fitting on firms with 2+ types only gives nearly the same rates** and predicts single-type firms well for phishing, takeover and bank hacking. Single-type malware firms are more often breached than predicted (53% vs 14%, n = 12), the same S-only oddity as F12, now small.
- **Reading:** with a type-level, all-firm baseline, broad firms are breached about as often as independent per-type chances predict. The "breached below independence" result (F9) was the lumped-group, S-only-baseline artefact. On breach prevalence, nothing argues against story A (independent breach chances per type), and there is no sign of an excess that spreading would create.

### F14. Breached-firm cost with per-type breach chances and per-type costs: a breadth effect remains, about x1.6 per extra type (`src/estimation/breach_cost_independence.py`, 2026-10-05)
- **Method.** 244 breached firms with a cost answer. Which ticked types were breached is unknown; under independence each possible breached subset gets probability from the F13 per-type chances (held fixed). Each breach costs zero (probability z) or a lognormal draw with a type-specific median (the 4 thin access/other types share one) and one shared spread. Worst = largest breach cost. Non-breach clean-up costs are ignored. A breadth term multiplies every breach cost by a factor per extra type ticked. Fitted by weighted likelihood on the cost bands, with and without the breadth term.
- **Result.** Adding the breadth term improves the fit by 9.5 log-likelihood points for 1 parameter (strong). Factor x1.62 per extra type, bootstrap 90% interval x1.25–x2.08 (60 resamples). Without the term, per-type medians are pushed up for types common in broad firms (e.g. ransomware £1,494 → £350 once breadth is allowed).
- **Observed vs predicted shares** (independence = no breadth term):

  | Types ticked | n | £500 or more: obs / independence / with term | £5k or more: obs / independence / with term |
  |---|---|---|---|
  | 1 | 60 | 32 / 42 / 34% | 5 / 12 / 6% |
  | 2 | 62 | 51 / 49 / 47% | 8 / 16 / 12% |
  | 3 | 57 | 57 / 50 / 51% | 13 / 16 / 15% |
  | 4 | 26 | 55 / 54 / 61% | 38 / 18 / 23% |
  | 5+ | 39 | 90 / 64 / 79% | 51 / 25 / 47% |
  | 3+ no ransomware | 84 | 67 / 51 / 56% | 20 / 16 / 21% |
  | 3+ with ransomware | 38 | 56 / 60 / 65% | 41 / 22 / 30% |

  Independence over-predicts narrow firms and under-predicts broad ones, which is the signature of a breadth effect on cost. With the term the fit is good, except 4 types (n = 26).
- **Reading:** even with each type's own breach chance and own breach cost, and the worst taken as the maximum, breached firms' cost rises with breadth beyond independence. The big-cost part of the excess is concentrated in 4+ types and in ransomware firms. Without ransomware, the excess at £5k or more is small (20% vs 16%), and at £500 or more it is clearer (67% vs 51%).
- **Caveats:**
  - Lognormal + zero shape assumed.
  - Breach chances fixed (their uncertainty is not carried through).
  - Non-breach costs ignored.
  - Breadth enters only as a simple scale per type.
  - Only the worst incident is seen.
  - Bootstrap is short.

### Current reading of the breadth effect (2026-10-05, summary of F5–F14; for discussion)
- **What the data support:**
  1. **Breach chance:** each ticked type carries roughly its own independent chance of leading to a breach. Broad firms are breached about as often as that predicts (F13). Breadth doesn't make a breach more likely beyond having more types.
  2. **Breach cost:** given a breach, broad firms' worst incident is bigger than "the largest of independent breaches, each priced by its own type" (F14). The excess is small up to 3 types and large at 4+. It is better described as a jump than a smooth x1.6 per type.
  3. The whole cost distribution moves up, not just a few extra big cases (F7). It doesn't depend on how often the firm was attacked (F5). It is strongest with ransomware, but not created by ransomware alone (F11, F14).
  4. In one sentence: broad firms aren't breached more than expected, but when they are, the breach is bigger, as if it were one bigger event rather than more events.
- **What the data don't settle:** the direction.
  - (a) Facing more kinds of attack makes a breach worse (more routes for an intrusion to spread, a more capable attacker).
  - (b) A worse breach makes the firm tick more types (the incident touched accounts, bank details, systems; or recall of a bad year).
  
  Both fit the facts above.

### F15. The big-cost breadth uplift needs breadth AND an "intruder inside" type (`src/estimation/breadth_composition.py`, 2026-10-05)
Breached firms with a cost answer (244). Costly = worst incident £5k or more.
- **Combinations are diverse.** The 35 costly breached firms with 4+ types show 33 different type combinations. The 30 non-costly ones are equally varied. No single recipe.
- **Size and attack frequency don't differ** between costly and non-costly broad breached firms.
- **Types over-represented in costly broad firms:** ransomware (60% vs 33%), bank hacking (40% vs 20%), outsider access (31% vs 17%), staff access (29% vs 20%). Takeover goes the other way (34% vs 53%). Phishing, impersonation, malware and DoS are about equal.
- **Splitting breached firms on whether they ticked ANY of ransomware / bank hacking / outsider access / staff access** (costly share by number of types ticked):

  | Types ticked | none of the four | at least one |
  |---|---|---|
  | 1 | 4% (n = 52) | 10% (n = 8) |
  | 2 | 10% (n = 51) | 1% (n = 11) |
  | 3 | 11% (n = 32) | 14% (n = 25) |
  | 4+ | 1% (n = 12) | 58% (n = 53) |

  - Without any of the four, there is no big-cost uplift with breadth at all. The share at £500 or more rises to 3 types (28% → 67%), then falls back at 4+ (28%, n = 12).
  - With one of the four but narrow (1–3 types), firms are not costly either.
  - The uplift sits where a firm is broad (4+) AND has one of these four.
- **What the four have in common:** an intruder actually inside the firm's systems or accounts (encrypting data, in the bank account, accessing files or networks). Phishing, impersonation, DoS, takeover and malware attempts can all happen from outside.
- **Reading:** costly broad breaches look like intrusions that got inside and reached many things. This fits reading (b), where the incident's footprint ticks many boxes, at least as well as (a).
- **Caveats:**
  - The four types were picked AFTER seeing which were over-represented (a risk of fitting noise). The 1% at 4+ without them rests on 12 firms.
  - Ransomware ticks include attempts.
  - Next check: whether the outcome items (systems corrupted, money stolen, files lost) line up with these four types in costly firms.
- **Are the four types present in narrow firms? Yes, often; there they are cheap (same script, section 6).** Of 158 firms ticking any of the four, 89 are narrow (1–3 types). Narrow vs broad (4+):

  | | Narrow | Broad |
  |---|---|---|
  | breached | 60% (44) | 72% (53) |
  | costly among breached | 8% | 58% |
  | £500 or more among breached | 47% | 87% |

  Per type, costly among breached, narrow vs broad:
  - bank hacking 5% (n = 21) vs 74% (n = 20)
  - ransomware 16% (14) vs 57% (31)
  - outsider access 0% (5) vs 64% (16)
  - staff access 2% (6) vs 68% (16)

  So the uplift is not "these types are absent from narrow firms". The same tick is common and often breached in narrow firms, but contained and cheap there. It is costly only together with breadth: an interaction, not either factor alone.

### F16. Outcomes line up with the "inside" types; costly incidents are multi-outcome events, narrow or broad (`src/estimation/outcomes_vs_types.py`, 2026-10-05)
Breached firms with a cost answer (244). Outcomes = Q56A, which covers the year's breaches, not only the worst.
- **The ticks match the consequences** (share with the outcome, firms ticking the type vs breached firms not ticking it):
  - ransomware (n = 45): systems corrupted 36% vs 10%; temporary loss of access 66% vs 26%; services down 49% vs 22%; third-party loss 35% vs 12%.
  - bank hacking (n = 41): money stolen 45% vs 4%; accounts misused 14% vs 5%.
  - outsider access (n = 21): systems corrupted 29% vs 13%; temporary loss of access 60% vs 31%; services down 44% vs 25%.
  - staff access (n = 22): money stolen 43% vs 10%; paid attackers 43% vs 5%; accounts misused 20% vs 6%. Possibly staff tricked into paying, rather than file access.
  
  So the type ticks are not noise: each goes with the consequences you'd expect.
- **Matching outcome, narrow vs broad:**
  - Ransomware matches throughout (57–98%).
  - Bank hacking: narrow cheap 54%, broad costly only 42%, broad cheap 0%. So in many broad firms the bank-hacking tick comes WITHOUT money stolen; "the intrusion reached the money" is not the general story.
- **Mean number of distinct outcomes:**

  | | Narrow, cheap | Narrow, costly | Broad, cheap | Broad, costly |
  |---|---|---|---|---|
  | with an inside type | 1.2 (n = 36) | 3.5 (n = 8) | 1.4 (n = 19) | 3.5 (n = 34) |
  | without | 1.15 (n = 121) | 0.9 (n = 14) | 1.4 (n = 11) | |

  Systems corrupted is about 46–48% in both costly inside groups, and money stolen about 30%.
  - The few costly NARROW firms with an inside type look just like the costly BROAD ones: about 3.5 outcomes, a similar mix.
  - The costly event seems to be the same kind of thing (a compromise with many consequences). It is simply much more common among broad firms.
  - Costly narrow firms without an inside type (n = 14) are a different kind: about 0.9 outcomes, presumably staff time or disruption without a compromise.
- **Reading:** number of ticks and number of outcomes both look like measures of how far one incident reached. This favours reading (b), the footprint of a big compromise, without proving it.
