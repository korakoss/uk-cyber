# Project Notes

Started fresh on 2026-10-05. The old notes are in `archive/NOTES_old_2026-07_to_2026-10.md`. They are history only, not a working document. Only look in them for a specific detail, and check anything taken from them before relying on it.

This file is being filled in by going through the project together with the user, one topic at a time.

## Summary of findings so far (2026-10-06; details in F1–F18 below)
- **Size and frequency (F1):** bigger firms are attacked more often (40% Micro → 68% Large). Among attacked firms, how often they're attacked barely changes with size. Worst-incident cost rises clearly with size. Being attacked more often does not mean a costlier worst incident.
- **Types (F2):** phishing 85% of attacked firms, impersonation 35%, malware 18%, others about 6% each. Cost order: mass phishing < targeted phishing ≈ impersonation < other types < ransomware. Average costs per type are mostly single firms.
- **Co-occurrence (F3, F4):** one attack can tick several boxes. All types co-occur more than chance, mostly through one shared exposure level: a quiet majority and a small heavily attacked group. More phishing goes with more other types.
- **Cost is concentrated in broad firms (F5):** the 15% of attacked firms with 3+ types carry about 77% of cost. Breadth predicts a big loss; attack frequency doesn't.
- **Breaches (F6, F7):** a breach makes the worst incident much costlier for every type. Without one, £5k or more is almost never reached. There is no separate "big breach" cluster; broader breaches shift the whole cost distribution up.
- **Breach chance (F9–F13):** with per-type chances fitted on all firms, broad firms are breached about as often as independent per-type chances predict. The earlier "breached below independence" result was an artefact of lumping types together and of a small, odd baseline of S-only firms (F12).
- **Cost beyond independence (F8, F11, F14):** given a breach, broad firms cost more than independent breaches priced by type would give: about x1.6 per extra type, mainly at 4+ types.
- **What carries it (F15, F16):** the big-cost uplift needs breadth AND an "inside" type (ransomware, bank hacking, outsider or staff access): 58% costly vs 1–14% otherwise. Inside types are common in narrow firms too, but cheap there. Outcomes match the ticked types, and costly incidents have about 3.5 different consequences, narrow or broad.
- **Exposure vs footprint (F18):** costly broad firms are NOT more attacked from outside (less phishing volume, DoS and spoofing). Cheap broad firms are the heavily attacked ones. So breadth in costly firms looks like the footprint of a spreading compromise, not heavy exposure. This is inferred: one multi-method attacker and recall effects aren't excluded.
- **Dead ends (F9 test, F17):** per-type "success" items measure something different from breaches. Per-type attack counts are about 1 everywhere, with ambiguous zeros, so they can't test spreading.
- **Open:** see "Open topics" at the end.

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

### F17. Per-type counts for the "inside" types: they exist but can't separate spreading from separate attacks (`src/estimation/type_counts.py`, 2026-10-05)
- **What exists:**
  - ransomware: `Cybercrime_ranssum`; `ranssoft` (ransom demanded)
  - malware: `Cybercrime_virussum`; `virussoft`
  - unauthorised access: `hackcount`, which EXCLUDES instances that led to fraud or ransomware; `Cybercrime_hacksum`
  - takeover attempts, including online bank accounts: `tkvrcount`, with the same exclusion; `tkvrsuc`
  - DoS: `doscount`, no exclusion
  
  Coverage among ticking breached firms is mostly good, except `Cybercrime_virussum` and partly `ranssum`.
- **Result** (breached firms; narrow/broad × cheap/costly): the median count is 1 in almost every cell, for every type. There is no "regular repeated attacks" baseline for these types to contrast with: one or two instances is the norm everywhere.
- Zeros are more common in broad costly firms: ransomware ticked but no ransom demanded 53% (vs 20–43% elsewhere); takeover attempts 0 in 50% (vs 0% elsewhere); outsider access 0 in 40% (vs 20%). This fits "the tick was part of the incident, with no separate attempts". But for `tkvrcount`/`hackcount` a zero is EXPECTED when the instance led to fraud or ransomware (the exclusion rule), so it cannot be read as evidence. Cells hold 5–20 firms.
- DoS (no exclusion, but an outside-entry type) has HIGHER counts in broad costly firms (median 2.5 vs 1).
- **Reading:** per-type counts do not give a usable test of spreading. They are low everywhere, the zeros are ambiguous because of the exclusion rule, and the cells are small.

### F18. Exposure vs footprint: costly broad firms are NOT more exposed; cheap broad firms are (`src/estimation/spread_vs_exposure.py`, 2026-10-05)
Breached firms with a cost answer. Groups: narrow cheap 157, narrow costly 22, broad (4+) cheap 30, broad costly 35. Shares weighted.
- **Part 2: exposure on things a breach cannot create, broad costly vs broad cheap:**
  - phishing count over 20: 10% vs 63% (31 / 23 answering); median count 12 vs 17
  - targeted phishing 6 or more: 40% vs 49%
  - DoS ticked: 27% vs 56%
  - impersonation that was outside spoofing only (no access / takeover): 61% vs 90%
  - attacked weekly or more: 50% vs 47%
  - attacked once: 18% vs 12%
  
  So costly broad firms are no more exposed on outside-attack types than cheap broad firms, and on phishing volume, DoS and spoofing they are LESS exposed. Cheap broad firms are the heavily-attacked ones: lots of phishing, DoS and spoofing, so their breadth comes from exposure, and it stays cheap. Costly broad firms get their breadth from "inside" types (F15) with ordinary exposure.
- **Part 1: direct linkage items (thin):**
  - "Impersonation involved account takeover": broad costly 29% (n = 30) vs broad cheap 5% (n = 28), narrow 0–10%.
  - "Impersonation involved unauthorised access": 14% vs 9%, a small difference.
  - "Impersonation using information from the breach": asked of too few firms (2–12 per group).
  - Any fraud: 35% vs 26%. Fraud attribution answered by 14 vs 4 firms. In costly broad firms frauds are attributed to varied sources (takeover 6, phishing 6, outsider access 5, bank hacking 4), but rarely to 2 or more types per firm (10%).
  
  Mild support for footprint via impersonation-with-takeover; otherwise too thin to read.
- **Reading:** this favours reading (b), or at least rules against the simple version of (a). The breadth of costly firms is not the breadth of heavy exposure; it is breadth from compromise-type ticks. There appear to be two kinds of broad firm:
  1. heavily attacked from outside (cheap);
  2. compromised, with the incident's footprint showing as several types (costly).
- **Caveats:** 30–35 firms per broad group; phishing count answered by 23–31; weighted shares on small cells.

### User synthesis (2026-10-05), with qualifications
- **User:** there is evidence of a spreading mechanism, which comes with a noticeable cost uplift. Firms where a breach spread are not especially attacked on the facets it spread to.
- **Qualifications:**
  - **"Spreading" is inferred, not seen directly.**
    - For: costly broad breadth comes from "inside" types, not outside exposure (F15, F18); costly incidents carry about 3.5 distinct outcomes (F16); impersonation-with-takeover is tied to costly broad firms (F18).
    - Not excluded: a capable attacker using several methods at once (the same in this data, and fine to treat together); a bad year making respondents tick more boxes (recall).
  - **"Not attacked on the spread facets":** what was measured is that costly broad firms are NOT more attacked on facets a breach cannot create (phishing volume, DoS, spoofing are lower or equal; overall frequency about equal). For the facets the breach spread to (the inside types), per-type attempt counts are about 1 everywhere and the zeros are ambiguous (F17). So the data can't say whether those facets were separately attacked.
  - **Cost uplift:** among breached firms, 58% are costly with 4+ types plus an inside type, against 1–14% elsewhere (F15). The core rests on 35 costly broad firms.
- **Possible implication for the model (not decided):** breadth in costly firms is an OUTCOME of the incident, so it should not be an input. Instead, a breach (from any entry type) may escalate into a multi-facet compromise with a higher cost. The chance of escalating doesn't appear tied to outside exposure.

## Open topics (as of 2026-10-05, not yet revisited in the fresh notes)
1. **The top of the cost range.** No business reported over £500k. A handful at £100k–500k carry much of the total (old estimate: about a third of the national total from single losses over £500k). Size likely matters most here (the largest real-world UK incidents hit very large firms).
2. **How cost scales with firm size, especially Medium/Large.** These are few in the sample, while Micro is about 80% of the weight.
3. **Small costs adding up.** Only the worst incident is priced, so many small clean-ups per year are invisible. The old range (£0.13–0.24bn) is driven by an assumed cost shape.
4. **More than one breach per firm per year.** The total exceeds the worst incident.
5. **Firms that skipped the cost questions** (about 11% of attacked firms).
6. **Spreading/escalation details for the model:** what drives the chance a breach escalates (entry type, size?), and the cost distribution of escalated compromises (this ties into 1).
- **Refinement (2026-10-06): is "costly broad firms get less phishing volume" a selection effect?**
  - Partly. Comparing within 4+ firms selects on breadth: a firm can reach 4+ through outside exposure or through inside (footprint) ticks, so within that group the two look negatively related even if they aren't overall.
  - Check without selecting on breadth (breached firms, by exact phishing count 1 / 2–5 / 6–20 / over 20):
    - share with an inside type is flat: 33% / 43% / 36% / 42%. So the chance of a compromise isn't tied to phishing volume.
    - costly share, with an inside type: 20% / 52% / 55% / 23% (n = 15 / 18 / 12 / 23).
    - costly share, without an inside type: 12% / 11% / 4% / 1%.
  - So, even without selecting on breadth, the most heavily phished breached firms are less often costly. The costliest sit at moderate volume.
  - Possible reason: for heavily attacked firms, the year-level breach marker may come from many small incidents. Not checked. Cells are about 20 firms.
  - Revised statement: the costliest firms are not the most heavily attacked. Very heavy phishing goes with cheaper breaches, not dearer ones.

### F19. Mass and breachy sources: no sign they avoid each other's firms (`src/estimation/sources_targeting.py`, 2026-10-06)
- **The question (user's dilemma):** do the volume source (mass phishing etc.) and the breachy source target firms independently given shared factors (H1), or prefer different firms (H2)?
- **Test:** per-firm shares among ALL attacked firms (no conditioning on the breach marker), by phishing volume. H2 predicts the per-firm chance of a costly compromise falls with volume; H1 predicts flat or rising.
- **By exact phishing count** (1 / 2–5 / 6–20 / 21–100 / over 100; n = 132 / 153 / 124 / 75 / 26):
  - costly compromise (costly + inside type): 2.0% / 5.5% / 4.8% / 3.9% / 4.1%
  - costly at all: 4.5% / 7.2% / 5.5% / 4.3% / 4.1%
  - inside type ticked: 16% / 15% / 18% / 20% / 33%
  - breached: 30% / 24% / 24% / 35% / 57%
  - Small+ alone: costly compromise 2.9% / 6.4% / 10.9% / 3.9% / 18% (n = 19 at the top).
- **By targeted phishing count** (1–5 / 6–20 / 21+): costly compromise 3.3% / 4.1% / 3.7%.
- **By overall frequency:** noisy, no consistent direction.
- **Reading:**
  - The per-firm chance of a costly compromise is roughly flat across phishing volume (about 4–5% after the lowest band), not falling. No sign of H2.
  - Breach markers and inside ticks DO rise with volume, but the extra breaches are cheap. So among BREACHED firms the costly share falls at high volume: dilution, as H1 allows (F18 refinement).
  - Current view: the sources target firms roughly independently given shared factors. Heavy-volume firms collect more cheap breach markers.
- **Caveats:**
  - Small cells at high volume.
  - The exact count is answered selectively: phishing firms without a count are very cheap, 0.8% costly.
  - "Costly compromise" uses the inside-type definition from F15.

## Breach counts

### F20. Evidence on repeat breaches: costly repeats look rare (`src/estimation/breach_counts_evidence.py`, 2026-10-06)
- **Per-type success counts** (breached firms ticking the type; share with 2+ among those with 1+):
  - phishing engaged with: 25% (n = 186). Costly firms 59%, but few with 1+. Engagement is not a breach.
  - successful takeover: 2% (n = 36)
  - successful malware: 2% (n = 90)
  - successful DoS: 39% (n = 33; only about 13 with 1+)
  - ransom demanded: 22% (n = 38). Costly ransomware firms: all exactly 1 (n = 22).
  
  So for the inside / costly types, 2+ successes are rare. Repeats show up mainly for DoS and phishing engagement, which are mostly cheap.
- **Yearly total (`crimecost_bands`, same 13-band scale) vs worst incident** (154 firms answered both; breached firms answer much more often, 46% vs 6%):
  - The total is LOWER than the worst in 55–65% of firms, so it is a narrower cost concept (likely direct costs only) and cannot serve as the yearly total.
  - Total HIGHER than worst: breached 3%, costly 1%, not breached 16%. Of 14 such firms, only 4 rise to £10–50k.
  - Because the concept is narrower, "higher" is a lower bound on extra cost, but it is rare.
- **Implication for the count model:**
  - Independent random arrivals at the fitted per-type breach chances (bank hacking 59%, DoS 51%, takeover 36%) would imply 25–35% of breached firms with 2+ breaches of that type. The success counts show about 2% for takeover and malware.
  - So per-type breach occurrence behaves more like yes/no per year than like repeated independent arrivals, at least for the costly types.
  - Simplest candidate: per type at most one breach per year. Incidents can bundle several types (spreading). The yearly cost of breaches is roughly the worst incident, plus a small allowance for repeats of the cheap kinds (DoS, phishing engagement).
- **Caveats:**
  - Success is not the same as breach (F9).
  - Count questions are answered by subsets.
  - The total-cost question has a narrower concept and is answered selectively.
  - Rare repeats in a sample of about 1,000 can still matter across 1.4M firms (user point from 2026-10-02): this says repeats are rare, not absent.

### Simplest breach-count model: concrete sketch (proposed 2026-10-07, not decided)
- **Per firm-year:**
  1. **Hits:** which attack types reach the firm (exposure; depends on size and the shared exposure level).
  2. **Breach per hit type:** yes/no with chance b_t (F13), at most one per type per year (F20).
  3. **Escalation:** each breach turns into a spreading compromise with chance e (one number, maybe by size). A compromise adds "inside" type ticks (its footprint) and takes its cost from a compromise cost distribution (costlier, many consequences, F16). Otherwise the breach takes an ordinary cost by type.
  4. **Yearly cost:** the breaches' costs (in practice usually one breach, so about the worst incident), plus clean-up for non-breach hits.
- **To pin from data:** b_t for the outside/entry types; e; ordinary and compromise cost distributions. Checks: per-firm costly-compromise share of about 4–5% of attacked firms (F19); breadth and outcome patterns of costly firms; rarity of repeats (F20).
- **Simpler alternative (descriptive):** take the firm's ticked types as given, per-type yes/no breaches with b_t, one cost draw for the firm's breach-year scaled x1.6 per extra type (F14). Easier to fit, but it treats breadth as a cause, although for costly firms breadth looks like an outcome.
- **Open choices:**
  - Does e depend on size or on how many types hit the firm? Not on outside volume (F18/F19).
  - Are several breached types in one year one incident or several? Repeats look rare (F20), so one is the simplest.
- **User decision (2026-10-07):** no hard "at most one breach" cap. That is an a priori limit with no justification. Use a Poisson count of breaches. Repeats being absent or rare in the sample is fine, but for the 1.4M-firm population this matters, so carry an interval on the rate through to the national results.
- **Correction to F20's implication (2026-10-07).** The "25–35% of breached firms would have 2+" figure used per-TICK breach chances (e.g. 59% for bank hacking among firms that ticked it). That is the wrong unit: the tick itself is partly selected by the breach. At the firm level:
  - 22% of attacked firms are breached → Poisson rate about 0.25 → about 12% of breached firms with 2+ breaches;
  - costly compromise about 4.5% of attacked firms → rate about 0.046 → about 2% of compromised firms with 2+.
  
  These are not clearly contradicted by the success counts (2+ in about 2% for takeover/malware, 22% ransom, 39% DoS). With Poisson, the rate is fixed by the breached share; repeats are not a free parameter. The real open question is firm-to-firm heterogeneity: some firms being repeatedly breached would give more repeats than Poisson.
- **What is known about repeat breaches and the worst cost (status 2026-10-07):**
  - "The worst cost is too stable for repeated iid draws" is established for ATTACKS: the worst cost doesn't rise with attack frequency (F1, F5).
  - For BREACHES the old work (2026-10-02) found the repeat rate weakly identified; nothing clearly contradicts Poisson repeats. Hints both ways: firms with several ransom demands are cheaper (handled attempts); several phishing successes look like one bigger event; no dose-response of cost with number of successes in broad firms.
- **User steer (2026-10-07):** don't lean on the per-type "success" counts for breach counts; the concept overlaps too little with breach (F9). Reason primarily from cost distributions instead.
- **Proposed way to use cost distributions (not run yet).**
  - The worst incident is the largest of the year's breach costs. If some firms have more breaches, their worst cost shifts up and their "no cost" share falls by a predictable amount (largest of N draws).
  - So, among breached firms, compare worst-cost distributions across groups whose expected breach count should differ, using measures a breach cannot create: phishing volume, DoS, overall frequency, size.
  - Compare with what largest-of-N predicts under Poisson, from no firm-to-firm variation up to strong variation.
  - The range of variation the band distributions allow gives the interval to carry to the national estimate.
  - Known so far: cost doesn't rise with attack frequency at fixed breadth (F5), and heavily phished firms are breached more but cheaper (F18/F19). Both hint that extra breaches in heavily exposed firms are cheap, or rare.
  - Limit: the largest of a few draws moves slowly, so this may only give a wide bound.

### F21. Repeats bounded from cost distributions: extra cost from repeat breaches roughly +1% to +15% (`src/estimation/breach_counts_from_costs.py`, 2026-10-07)
- **Method.**
  - Single-breach cost distribution G = worst-cost bands of breached firms attacked ONCE (n = 85; one attack, one breach). G: none 18%, <£100 11%, £100–500 24%, £500–1k 10%, £1–5k 27%, £5k+ 10%.
  - Breach counts are negative binomial with shape k (k = infinity is plain Poisson; small k = firms differ a lot, more repeats). The mean is set so that P(N ≥ 1) matches each group's breached share.
  - Predicted worst = largest of N draws from G. Compared with observed worst bands of breached firms in groups that should have more breaches. Weighted log-likelihood across k.
- **Results** (log-likelihood relative to the best k; a drop of about 2 or more counts against):

  | Group (breached share, n breached) | Poisson: mean N / extra cost | Lowest k not ruled out | Extra cost there |
  |---|---|---|---|
  | attacked less than monthly to monthly (18%, 109) | 1.10 / +1% | about 0.5 (k = 0.2: -5.5) | +4% |
  | attacked weekly or more (14%, 50) | 1.08 / +1% | about 0.1 | +17% |
  | phishing count 2–20 (24%, 73) | 1.14 / +2% | best at k = 0.2 (Poisson -6.2) | +17% |
  | phishing count over 20 (40%, 46) | 1.28 / +3% | about 0.5 (k = 0.2: -9.5) | +13% |

  k = 0.1 or below is ruled out in 3 of 4 groups (extra cost +25% to +70%).
- **Reading:**
  - Plain Poisson implies few repeats (1.1–1.3 breaches per breached firm) and adds about 1–3% to breach cost.
  - The cost distributions allow at most moderate firm-to-firm variation, which adds up to about +15%.
  - Working interval for the repeat add-on: about +1% to +15% of breach cost.
- **Caveats:**
  - The phishing 2–20 group "prefers" more repeats only because its worst costs are higher (26% vs 11% at £5k or more). That is more likely the costly-compromise concentration at moderate volume (F19) than repeats. The same confound could inflate the upper bound generally.
  - G is assumed the same everywhere. Heavy-volume firms' extra breaches seem cheaper (F19), so G-like repeats are the costly case.
  - G rests on 85 firms. Phishing-count groups can include once-attacked firms. Small groups. Sizes are pooled.
  - Bounds repeats that cost like G; much cheaper repeats are not bounded but add little.
- **Can the cost distribution tell 0/1 breaches from Poisson repeats? No (2026-10-07).**
  - At the observed breach rates (breached share 14–40% by group), Poisson gives only 1.08–1.28 breaches per breached firm.
  - Its predicted worst-cost distribution is almost the same as 0/1 (= G): "no cost" 14–17% vs 18%; £5k or more 11–13% vs 10%. A shift of 1–4 points, well inside the noise for groups of 46–109 firms.
  - There is no group with a high enough breach rate to separate them.
  - What the costs CAN do is rule out MANY repeats (strong firm-to-firm variation, F21).
  - Practical consequence: 0/1 vs Poisson is not decidable from this data, and at sample-level rates it changes cost by only 1–3%. The open lever for the population is firm-to-firm variation, bounded by F21 at about +15%.
- **Options for handling breach counts in the model (proposed 2026-10-07, user to decide):**
  1. Poisson breaches per firm, with the rate depending on what we can observe: size and the shared exposure level (F3). That already gives firms different breach-proneness, grounded in data.
  2. Leftover firm-to-firm variation beyond that, which the survey can't see, as an interval: plain Poisson up to the F21 bound. Simplest form: a multiplier of about 1.01–1.15 on breach costs.
  3. A rare repeat-prone tail (e.g. 1 in 1,000 firms with many breaches) as a stated sensitivity scenario. It is invisible in the sample, so it is assumed, not estimated.
- **DECISION (user, 2026-10-07):** breach counts are Poisson per firm, with the rate from size and exposure, as long as nothing in the data contradicts it, and nothing does (F21: Poisson fits within noise). No rare repeat-prone tail: there is no evidence for one, so it is dropped (options 2–3 above withdrawn as defaults). The interval comes from uncertainty in the rates themselves. The F21 bound on extra firm-to-firm variation (up to about +15%) is kept only as a note, not built in.

## Model status (2026-10-07)
Per firm-year:
1. **Which attack types reach the firm.** Depends on size and one shared exposure level (F3: a quiet majority, a middle group and a small heavily attacked group; or a smooth scale). The volume source and the breachy source target firms roughly independently given these (F19). STATUS: structure agreed; not refitted in the fresh work (old fits exist).
2. **Breaches.** Poisson count per firm, rate from size and exposure (DECIDED 2026-10-07). STATUS: open how the rate is built. Per-type chances (F13) exist, but "inside" ticks are partly the footprint of a breach, so breaches should be generated from the entry types.
3. **Escalation.** Each breach may spread: extra inside-type ticks, many consequences, a much higher cost. STATUS (2026-10-07, F22–F24):
   - Marker: breached, 4+ types, an inside type, worst >= £5k (agrees with the consequence count).
   - Spread chance: one chance for all entry types (entry type can't be recovered), rising with size: 6% / 9% / 18% of breached Micro / Small / Medium+Large firms. Caveat: bigger firms pass the breadth condition more easily, so part of this may come from the marker.
   - How far: inside types reached 1: 58%, 2: 35%, 3+: 8%; somewhat further in bigger firms.
   - Cost: one spread cost distribution, the same whatever the number or kind of inside types (DECISION 2026-10-07, replaces the earlier 'rises with inside count'; F25: the inside count raises the chance a broad breach is a spread one, not its cost). Typical spread breach about £14k; the mean (£15k–60k) depends on the top band.
   - Open: fitting the spread cost distribution (the £5k cutoff in the marker needs undoing, e.g. the mixture reading in F23), its top end, and whether the size effect is real.
4. **Cost per breach.** An ordinary breach cost by type, and a compromise cost. No separate "big breach" cluster beyond this (F7). STATUS: not yet fitted. Open: the top end above about £100k and how cost scales with firm size.
5. **Clean-up of attacks that didn't breach.** Small costs; the yearly total is unobserved (only the worst incident is priced). STATUS: open; the old range is assumption-driven.
6. **Yearly cost** = breach costs + clean-up; national = scaled by ONS business counts per size band. Firms that skipped the cost questions: old estimate about +5%.

### Escalation: what we currently claim (checked with the user, 2026-10-07)
- **Agreed working view:** a breach either stays in its lane, or spreads to involve other ("inside") types. A spreading breach shows up as extra type ticks AND a much higher cost (F15, F16, F18). This is inferred, not proven: one attacker using several methods, or recall, aren't ruled out.
- **"Two kinds" shows in footprint and consequences, not in the cost numbers alone.** There is no visible two-cluster split in breach costs (F7): broader breaches shift the whole cost distribution up.
- **"Uplift is about the same whatever the spreading profile, except bank hacking is costlier" is NOT established in the fresh work.** It comes from the old notes (bank hacking x9 [3–26], n = 23; ransomware no effect) and was not re-checked. Fresh results partly disagree:
  - Ransomware seemed to add cost at a given breadth (F11, F14), but those comparisons include broad firms with NO inside type. Among spread firms only, ransomware adds nothing (F22). So the old "no ransomware effect" stands.
  - Bank hacking: costly among breached 74% in broad firms (n = 20) vs 5% in narrow, but in costly broad firms the bank-hacking tick comes with money stolen / accounts misused only 42% of the time (F16). So "the intrusion reached the money" is not the general story.
  - Fresh check done: F22 below.

### F22. Cost of spread breaches by which inside types are involved (`src/estimation/escalation_profiles.py`, 2026-10-07)
Spread proxy: breached, 4+ types, at least one inside type (n = 53, 58% at £5k or more). Gaps are with minus without the type, bootstrap 90% intervals. Small n throughout.
- **Ransomware: no difference** (57% vs 61%, gap -0.05 [-0.37, +0.31]). Firms whose only inside type is ransomware are on the cheap side (32%, n = 13).
- **Bank hacking: costlier, but not firmly** (74% vs 40%, gap +0.34 [-0.03, +0.59]; n = 20 vs 33). Same direction as the old x9, weaker.
- **Outsider / staff access: small gaps, wide intervals** (+0.07, +0.11). Staff access firms have more at £20k or more (43% vs 13%, n = 16).
- **Number of inside types matters more than which one:** 1 inside type 48% (n = 30), 2 types 88% (n = 17), 3+ types 75%, with 63% at £20k or more (n = 6). Part of the bank-hacking gap may be this, since bank hacking often comes with other inside types (not separated; n too small).
- **Narrow breached firms with an inside type stay cheap whatever the type** (8% at £5k or more, n = 44). Small+ 27% vs Micro 4% there.
- **Reading:** the uplift is not clearly homogeneous. The clearest pattern is "more inside types, costlier", which fits spreading: a bigger footprint, a bigger cost. Bank hacking may add on top; it's unclear.
- **DECISION (user, 2026-10-07):** for now, model the cost of a spread breach as rising with the number of inside types involved, with no separate bank-hacking effect: nothing clearly sets bank hacking apart, so the simpler form wins.
- **Next (user, 2026-10-07):** sort out escalation questions: (1) what sets the chance a breach spreads (size, entry type), (2) how many inside types a spread breach reaches, (3) how to mark spread firms when fitting (footprint vs consequence count).

### F23. Marking spread breaches: cost plus breadth, checked against other indicators (`src/estimation/escalation_marker.py`, 2026-10-07)
User's suggestion: mark spread breaches mainly by the cost uplift, with breadth (4+ types and an inside type) as a precondition. Breached firms; broad costly n = 34, broad cheap 19, narrow with inside type 44, narrow without 135.
- **Consequences line up strongly.** Broad costly: 3.5 outcomes on average, 69% with 3+, restore a day or more 74%, 1.7 of 3 impact items. Broad cheap look like ordinary narrow breaches: 1.4 outcomes, 13% with 3+, restore 31%, 0.65 impact items (narrow with inside type: 1.37, 10%, 47%, 0.59).
  - Caveat: impact items (recovery costs, revenue loss) are partly cost by construction; the outcome items less so. Outcomes cover all the year's breaches.
- **Number of inside types:** broad costly 1.5, broad cheap 1.1, narrow 1.1. Cheap broad firms again look ordinary.
- **Weak or no signal:** impersonation involving access (40% vs 14%, small n); any fraud (35% vs 30%); inside ticks with count <= 1 (69% vs 61%): these don't separate the groups.
- **Cost below £5k:** broad cheap firms are a bit costlier than narrow inside-type breaches (42% vs 23% in £1k–5k), but no "none"-heavy pattern. Roughly ordinary.
- **Mixture reading:** if broad+inside breaches are a mix of ordinary breaches (8% costly, like narrow ones) and spread ones, the spread share is 55% if spread breaches are always costly, 61% at 90%, 70% at 80%.
- **Reading:** the cost-plus-breadth marker agrees with the consequence count, the cleanest independent indicator. Linkage items and count data add nothing either way.

### F24. What sets the chance a breach spreads, and how far it spreads (`src/estimation/escalation_drivers.py`, 2026-10-07)
Spread marker as F23 (breached, 4+ types, inside type, >= £5k). 244 breached firms, 34 spread. 90% bootstrap intervals.
- **Size matters.** Spread share of breached firms: Micro 6% [2–11], Small 9% [3–15], Medium+Large 18% [12–25].
  - It is the breadth step that rises with size (4+ types with an inside type: 10% / 18% / 28%); costly given that is flat (62% / 48% / 66%).
  - Caveat: bigger firms tick more types anyway (separate attacks), so they meet the breadth precondition more easily. Part of the size effect may be the marker, not spreading.
- **Entry type can't be recovered.** The worst-incident type (disrupta) names an inside type for 61% of spread firms vs 20% of broad cheap firms (ransomware alone: 11 of 34), so it mostly names where the incident ended up. The outside types ticked are nearly the same in spread and broad cheap firms (phishing 99% vs 94%, malware 77% vs 76%, takeover 56% vs 52%), except impersonation (74% vs 99%). No sign of a particular way in. Practical reading: one spread chance for all entry types, varying by size.
- **How far it spreads.** Inside types among spread firms: 1: 58%, 2: 35%, 3+: 8% (mean 1.5); broad cheap firms: 1: 90%, 2: 7%, 3+: 4% (mean 1.1). Medium+Large spread firms reach more (mean 1.9, n = 23) than Micro/Small (about 1.5, n = 11).
- **Spread firms do NOT tick more types in total** (mean 5.1 vs 5.0 for broad cheap). They tick more inside types and fewer others.
- **The extra inside type is mostly bank hacking:** spread 69% vs broad cheap 35%; ransomware 57% vs 61%, outsider 13% vs 10%, staff 12% vs 8%. So "more inside types" and "bank hacking" are tangled again here (see F22 decision): they can't be separated with this data.

### F25. Cost of spread breaches, fitted without the £5k cutoff (`src/estimation/spread_cost_fit.py`, 2026-10-07)
Mixture fit on all 53 broad breached firms with an inside type: a share e are spread breaches (lognormal cost), the rest ordinary (cost like a reference group of narrow breached firms). Small n; wide intervals.
- **The spread share depends on the reference group:** e = 0.49 [0.34–0.92] with narrow inside-type breaches as reference, 0.67 [0.41–1.00] with all narrow breaches.
- **The typical spread breach costs about £14k** (median £13.9k–14.2k in both fits).
- **The lognormal shape fits badly.** Marked spread firms pile up in £10k–20k (14 of 34 firms, 56% of weight), yet 4 firms sit at £100k–500k. A lognormal can't do both: one fit goes narrow (mean £15k, misses the top), the other goes wide (mean £28k).
- **Raw marked spread firms (n = 34), band midpoints:** weighted mean £57k; one Micro firm at £100k–500k carries 60% of it, and the mean is £25k without it. None above £500k.
- **Number of inside types:** with 2+ inside types, a much larger share is spread (e 0.82 vs 0.36 for 1 type). The cost given spread is similar (median £15k vs £13k). The £100k+ firms are mostly in the 1-type group. So in this fit, more inside types mainly raise the chance that a broad breach is a spread one, not the cost of a spread breach.
- **Reading:** the bulk of spread costs is well pinned (about £5k–20k). The mean (£15k–60k) depends on the top band, i.e. on the tail question, which this data can't settle.
- **DECISION (user, 2026-10-07):** drop the inside-count effect on spread cost. Spread breaches get one cost distribution. The number of inside types stays only as the footprint (how many extra ticks a spread breach adds, F24). Replaces the earlier decision after F22.

### F26. The top of the cost range: who the costly firms are (`src/estimation/top_band.py`, 2026-10-07)
Attacked firms with a cost answer (n = 982). Band 10 = £100k–500k; bands above £500k exist and no firm chose them.
- **Only 8 firms at £100k+** (Micro 2, Small 2, Medium 1, Large 3); 11 at £50k+; 26 at £20k+.
  - Weighted share at £100k+: Micro 0.5%, Small 0.7%, Medium 0.4%, Large 2.1%. Each rests on 1–3 firms.
- **They carry most of the mean** (band midpoints, £300k for band 10): 64% of the all-firm weighted mean (Micro 72%, Small 53%, Medium 38%, Large 58%). Without them the all-firm mean drops from £2,361 to £842.
  - The two Micro firms have ordinary weights (1.4, 1.5), so they're not weighting artefacts.
- **What they are:** 4 of 8 are spread breaches; 3 are breaches that didn't spread (Micro ransomware, Small bank hacking, Large ransomware); 1 is not breached (Small, DoS, attacked once).
- **Two of the 8 contradict themselves:** worst incident £100k–500k but yearly total "none" (the non-breached Small DoS firm, and the Medium spread firm). Another (Large) gives a total lower than its worst. These may be misreported.
- **Above £500k:** nothing observed. With 8 firms above £100k, that is weak evidence about how heavy the tail is (old notes, tail_ceiling_pareto.py: the empty bands can't bound it).
