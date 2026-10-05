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
