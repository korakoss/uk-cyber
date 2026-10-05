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
- **Shape of the hidden exposure (old model fits, 2026-09-24/26; not re-run).** These were model-based: attack rates per firm were fitted from the flags, the frequency bands and the phishing counts.
  - **Across firms it is lumpy and skewed, not a smooth bell curve.** A free fit, with no shape assumed (`frailty_free_mixture.py`), settled on three levels: quiet 55% (about 0.03 attacks a year per kind, i.e. essentially never attacked), middle 37%, and hot 9%. Two levels fit clearly worse. A smooth bell-shaped (lognormal) spread also fit worse than three levels.
  - **It doesn't scale every type equally.** Going from middle to hot: targeted phishing x2.5, impersonation x4, mass phishing x7, serious types (malware, ransomware, DoS, hacking and so on) x9. Serious types for hot vs quiet firms: about x120. So the top level is mostly about serious attacks and phishing volume. (The "near-equal loadings" result is on a correlation scale; the two results are not in conflict.)
  - **The exposed share rises with size.** In the two-level version it is 37% Micro, 42% Small, 57% Medium and 66% Large.
  - **Caveats:**
    - A free fit like this always lands on a few separate levels even when the truth is smooth. So "three levels" really means a big quiet mass plus a long upper tail. It does not prove that firms come in three kinds.
    - The fits depended on turning the frequency bands into attack counts, and that conversion fit badly (too few "once" answers).
    - In that old model, the national cost per firm barely changed across shapes (£729–732). The shape mattered for which firms carry the cost, not for the average.
- Open: whether the "broad" firms are firms exposed to more kinds of attack, or one big incident ticking several boxes (old notes leaned towards the latter for costly firms). This is to be looked at with cost.
