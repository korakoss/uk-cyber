# Evidence for the model's assumptions

Collected argument for the "Assumptions" section of the writeup. Each claim is stated, followed by the
evidence (with the script that produced it) and its caveats. Survey-weighted throughout unless noted.
Scripts are in `src/estimation/`.

## 0. The model being defended

Each firm has a **size band** *s* (micro / small / medium / large) and a latent **exposure class** *Z*.
Its attack log is generated per **channel** *c* ∈ {targeted phishing, mass phishing, impersonation,
ransomware, other serious}:

- **Counts.** Mean = base rate(*c*, *s*) × multiplier(*c*, *Z*); law Poisson (targeted, impersonation,
  other serious) or NegBin (mass, ransomware). Channels independent given (*s*, *Z*).
- **Costs.** Each attack independently: no cost with probability 1 − p(*c*), otherwise
  lognormal(μ(*c*), σ(*c*)).

| Piece | Depends on | Does not depend on |
|---|---|---|
| Base attack rate | channel, size | exposure |
| Exposure multiplier | channel, exposure | size |
| Exposure-class probabilities | size | channel |
| Count law and NegBin shape | channel | size, exposure |
| Success gate, lognormal μ, σ | channel | size, exposure, count |

Estimate: E[cost | *s*] = Σ_c E[K_c | *s*] · p(*c*) · E[cost | success, *c*]; national = Σ_s N_s · E[cost | *s*].

Pending changes: exposure with 3 levels instead of 2 (C3); other serious as Poisson (C5); possibly a
micro-vs-rest shift on other-serious costs (B2).

---

## A. Channels

**A1. Phishing is two processes: targeted (rare, often costly) and mass (frequent, almost never costly).**
- The signature: among phishing-only firms, P(no cost at all) is flat across the number of phishing
  attacks (0.49 / 0.63 / 0.57 / 0.61 / 0.43 for N = 1, 2–3, 4–10, 11–50, 51+). One iid channel predicts
  a steep fall (0.96 → 0.03); flat is only possible if most extra attacks are near-harmless and a
  separate minority causes the cost.
- A two-component (rank-2) SVD structure in phishing firms' frequency × cost patterns (99%+), reproduced
  by grouping on the survey's own "targeted" question (`phishcon_bands`).
- The two-channel model beats one channel decisively on counts and costs (ΔAIC 172, phishing-only
  firms) — `phishing_count_mixture.py`.
- Caveat: the two channels are latent; no attack is labelled targeted or mass.

**A2. Impersonation is its own channel, internally homogeneous.**
- Distinct cost profile when most disruptive: 41% no cost, typical costly incident ~£160 (phishing: 65%,
  ~£100).
- The same two-cluster test that finds targeted/mass phishing is degenerate for impersonation (one
  population).
- Weak counter-hint: impersonation-only firms answering "once" have *less* no-cost (0.50) than those
  attacked less than monthly/monthly (0.79) — a phishing-like signature (`iid_cost_check.py`); ~2 SE.

**A3. Ransomware is split out of serious.**
- When most disruptive: 3% no cost vs 27–44% for other serious subtypes; 31% of its costly incidents
  exceed £10k vs 3–14% — `serious_subtypes.py`.
- Homogeneity across serious subtypes rejected (p = 0.006), entirely due to ransomware.
- Has its own count variable (35 firms), which pins its frequency.

**A4. The rest of "serious" is lumped as one channel.**
- Homogeneity for the populated subtypes (malware, DoS, bank hacking, takeover; 21–36 most-disruptive
  firms each): p = 0.33 (weighted 0.996) — `serious_subtypes.py homog`. Takeover also degenerate in the
  two-cluster test.
- Sparsity for the rest: unauthorised access (staff / outsider) and eavesdropping have 0–8
  most-disruptive firms each; lumped by necessity.
- Caveat: low power ("no detectable difference", not "identical").

**A5. Caveat on all channels: categories overlap.** Respondents sometimes tick two types for one
incident, mainly phishing + impersonation, also ransomware + malware.
- Firms answering "breached once" flag 2+ channels 38% of the time, as often as firms attacked monthly.
- 37 firms answering "once" and reporting exactly 1 phishing attack still flag another channel, mostly
  impersonation — `frailty_flag_structure.py once|oncecounts`.

---

## B. Costs

**B1. Per-attack cost = point mass at zero + lognormal, per channel.**
- Worst-incident distribution among firms where the channel was most disruptive, observed vs model:
  phishing p = 0.73 (effective n 355), impersonation 0.58 (109), ransomware 0.25 (13), other serious 0.99
  (65) — `channel_cost_checks.py`.
- The zero mass is structural: most attacks cost literally nothing, which a lognormal cannot produce.
- Above £500k the lognormal is extrapolation; the only constraint is that no firm chose bands 11–13,
  which the likelihood uses.
- Caveat: ransomware's check has little power; its tail width (σ) is the least pinned number in the
  model (95% interval ≈ 2.2–4.2) — `ransomware_fit_check.py`.

**B2. Costs depend on channel, not size.**
- Obviously channel-dependent (e.g. success rate <1% for mass phishing vs ~30–40% for targeted,
  impersonation, ransomware).
- Model-based: letting success rates vary by size, p = 0.94; cost levels, p = 0.16; fitting every size
  separately, p = 0.12 — `joint_size_model.py`.
- Model-free, single-attack firms (worst incident = the one attack) — `cost_size_agnostic.py`:
  - phishing, exact count 1: micro 0.49 / 0.33 / 0.18 vs rest 0.51 / 0.38 / 0.11 (no cost / £1–500 /
    £500+), p = 0.90; matched on count across 5 buckets, p = 0.95;
  - impersonation, all single-channel firms (≈1 attack each in every size under B3 + C5): p = 0.59;
  - other serious, all single-channel firms: micro 0.64 / 0.24 / 0.12 vs rest **0.21 / 0.50 / 0.28**,
    p = 0.29.
- Caveat: other serious (and ransomware) consistently lean costlier for larger firms — in the model's
  free fit, the "once" slice and the all-single-channel slice. Not significant (~18 firms per group), but
  the shared costs may understate large-firm serious costs. Coarse bins cannot see tail differences.

**B3. Costs are iid across a firm's attacks within a channel.** Three parts:
1. *Independent of how many attacks* — supported for phishing: P(no cost) by exact count matches the
   model within noise (`iid_cost_check.py`). Untested elsewhere.
2. *No firm-level cost factor* (≈ frailty acting on costs) — untestable: one worst-incident cost per
   firm; per-type annual cost variables too sparse (5 firms report costs for 2+ types). A descriptive
   screen shows a hint for impersonation only — `channel_cost_checks.py`.
3. *Independent within a firm* — indistinguishable from 2 with this data.
- If 2–3 fail, the likely direction is underestimation.

---

## C. Counts

Built from data patterns, each step adding structure the data demands.

**C1. A shared firm-level latent drives attack occurrence across all types, within every size band.**
- Raw: firms hit by all four channel groups are 53× more common than independence predicts; single
  channels are under-represented — `frailty_flag_structure.py`.
- Across the 10 raw survey types (no channel grouping, no model): one dominant factor, 56% of variance
  (50% micro, 63% rest), near-equal loadings 0.27–0.36 — `type_cooccurrence_structure.py`.
- One-dimensional: the second factor is ~noise (eigenvalue ≈ 1) and changes direction between size
  groups. The residual pairings (phishing–impersonation, ransomware–malware) are small and partly A5
  co-labelling.
- Not just size: holds within each size band; size-level parameters cannot create differences among
  firms of the same size.

**C2. The same latent scales attack counts, roughly multiplicatively.**
- Phishing count (exact) by number of *other* channels hit: firms with 2+ others have ×2–3 the counts at
  every quantile (micro ×3.0 / 3.0 / 2.8 / 7.1 at the 25/50/75/90th; rest ×2.0 / 2.0 / 2.5 / 2.8);
  median shift 90% interval ×1.2–×4 — `latent_on_counts.py`.
- Robust to reweighting for count reporting (costlier firms report counts more often).
- A single extra impersonation flag gives no shift (co-labelling, A5); a single extra serious/ransomware
  flag a mild one.

**C3. The latent's distribution: about three levels.**
- From flags only (no count assumptions), share of the co-occurrence structure captured
  (0% = no latent, 100% = ceiling) — `frailty_shape_flags.py share`:

  | | 2 classes | 3 classes | continuous |
  |---|---|---|---|
  | Micro | 73–86% | 78–92% | 74–87% |
  | Small + medium + large | 69–83% | 85–96% | 84–96% |

- Continuous explains no more than the matching discrete model. Discrete classes are ordered (rungs of
  one scale). Micro: 2 levels suffice; larger firms need a third, very exposed level (~4% of firms).
- Consistent with the SVD of frequency answers across sectors, which shows one direction of variation
  — `frailty_freq_svd.py`.
- The estimate is insensitive to the shape (£729–731 per firm pooled across binary / 3-class /
  continuous) — `frailty_richness_test.py`, `frailty_free_mixture.py`.

**C4. On top of the latent, phishing and ransomware counts have large channel-private spread → NegBin.**
- With firms sorted into exposure tiers from their flags, Poisson-within-tier makes a sharp prediction
  with nothing fitted to counts: λ = −ln(1 − p_tier) — `counts_within_tier.py poisson`.
- Low tier (83% of micro, 67% of larger firms; p ≈ 0.22): Poisson predicts 88% single attacks and none
  above 10; observed 34–36% single, 29–31% above 10; mean 15–21 vs 1.1. Upper tiers the same.
- Bias works against this result (co-labelling pushes heavily phished firms out of the low tier). Not
  missed shared exposure: 3 tiers capture ~90% of cross-type structure.
- Ransomware: same direction on 12 firms.

**C5. Impersonation and other serious counts are Poisson given the latent.**
- Single-channel firms' frequency answers vs Poisson pushed through the calibrated count→answer map:
  impersonation p = 0.06 / 0.41, other serious 0.07 / 0.61 (micro / rest); phishing-only reference
  p < 0.001, so the test has power — `counts_within_tier.py tests24v2`.
- Model-based: other serious as Poisson not rejected in the size model (p = 0.08).
- Targeted phishing: Poisson by assumption (not separable inside the phishing total).

**C6. NegBin is adequate for mass phishing.**
- Phishing = targeted Poisson + mass NegBin, each tier's hit rate pinned to the flags, shape shared across
  tiers: fits the low tier (p = 0.59 micro, 0.08 rest) — `counts_within_tier.py tests24v2`.
- Upper tiers misfit consistently (too many 2–3, too few 4–10), likely partly heaping of reported counts
  at 5 / 10.
- In the joint model the NegBin shape stays small after latent and size (mass r ≈ 0.2, ransomware ≈ 0.04);
  dropping NegBin was strongly rejected. Ransomware's shape rests on two firms reporting 24 and 100.

**C7. Size acts on base rates and exposure-class probabilities only.**
- Letting attack rates and the exposed share vary by size: large gain (Δ log-likelihood 165, 18 params);
  exposed share 0.35 / 0.41 / 0.58 / 0.65 micro → large — `joint_size_model.py`.
- Multipliers and NegBin shapes are shared across sizes by assumption; fitting every size separately was
  not significantly better (p = 0.12) and produced absurd tails in thin groups.

---

## D. Known gaps and untestable assumptions

1. **Frequency → count mapping** (deferred): calibrated on phishing counts, applied to all channels; the
   model produces too few "once" answers.
2. **Co-labelling** (A5) is not modelled; it slightly inflates the shared latent.
3. **Costs not depending on exposure, and independent within a firm** (B3.2–3): untestable; likely
   direction of failure is underestimation.
4. **Other-serious (and ransomware) costs may rise with size** (B2): consistent lean, not significant.
5. **Upper-tier phishing count shape** (C6) misfits, plausibly heaping.
6. **Targeted phishing Poisson** (C5) and **latent multipliers / NegBin shapes shared across size** (C7):
   assumptions, not directly tested.
7. **Ransomware tail width** (B1): the dominant quantitative uncertainty in the total.
8. **Phishing-count reporting** is higher for costlier firms; treated as explained by the observed cost
   band.
