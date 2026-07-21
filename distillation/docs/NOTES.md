# Distillation — running record (my notes)

Purpose: a clean re-presentation of the finished cybercrime-cost estimate and its
key defenses, extracted from the ~28 exploratory scripts in `src/estimation/`.
The working tree is READ-ONLY; all new work lives under `distillation/`.

Layout:
- `docs/NOTES.md`   — this file (scratch, decisions, reproduction status)
- `docs/NARRATIVE.md` — companion write-up (methodology + findings; not started)
- `src/`   — reusable, side-effect-free utilities (load, fit, bridge, constants)
- `runs/`  — self-contained analyses; read data via src/, write to runs/results/
- `runs/results/` — emitted numbers/tables

Rule of admission: a piece earns a place here only if it feeds the ~£2.2bn
headline or defends it. Everything else stays cited-not-ported.

Reproduction contract: each distilled unit must reproduce the figure the
original recorded, checked by a `verify_*.py` in runs/.

## Scope (agreed)
Rebuild the spine (Steps 1-6 -> national estimate) + top robustness checks
(GDP-scaling refutation, body-vs-tail, upward sensitivity, censored floor) as a
real REFACTOR (shared machinery extracted to src/), not a literal port. Python
is the primary engine; Squiggle optional. Per-type bridge multipliers enter as
documented constants with provenance, not re-derived.

## Build log

### Unit 1 — src/data.py (data-access foundation)  ✅ DONE + VERIFIED
Distilled from proc.py (get_business_data) + the constants re-declared across
src/estimation (BAND_BOUNDS, N_BY_SIZE, etc.). One authoritative copy of:
- loader `load_business_data()` — business frame (typex==1), curated columns,
  plus centrally-defined derived cols `attacked` and clean `band` (special
  codes 997/999/999.62 stripped via SPECIAL_CODE_THRESHOLD).
- codebook constants: SIZE_LABELS, BAND_BOUNDS (band 10 BOUNDED 100k-500k),
  VALID_FREQ, N_BY_SIZE (ONS employer frame), PREVALENCE, DISRUPTA_LABELS.
Reads the raw CSV in place from the read-only working tree (path resolved
relative to the module, so runs work from any cwd).

Verification (`runs/verify_data.py`): reproduces proc.py exactly — 2180 firms,
all shared columns (sizeb/weight/freq/damage_bands/disrupta) MATCH, both derived
columns MATCH the inline logic in national_simulation.py. Self-test attack rates
41/51/66/69% align with known prevalence. ALL CHECKS PASS.

### Unit 2 — src/fitting.py + src/bridge.py + approach B  ✅ DONE + REPRODUCED
- `src/fitting.py`: censored-MLE zero-inflated lognormal fit (LognormalFit),
  `band_conditional_mean` (bounded within-band E[X|band], used by A) and
  `bounded_cell_mean` (fully-parametric cell mean, used by B). Band 10 bounded.
- `src/bridge.py`: per-type bridge multipliers as documented constants (freq==1
  ->1.0; Impersonation range; provenance to bridge_specification.py etc.).
- `runs/estimate_size_type_lognormal.py`: approach B — fit lognormal per
  (size,type), collapse to bounded cell mean, discard per-firm band. Result
  £0.99-1.42bn, REPRODUCES the original src/estimation/size_type_cellmean_estimate.py
  exactly. Output -> runs/results/estimate_size_type_lognormal.txt.

### Unit 3 — approach A national baseline (THE HEADLINE)  ✅ DONE + REPRODUCED
`runs/estimate_national_baseline.py`: the spine. Semiparametric approach A —
keep each firm's OWN observed band (empirical band mix), place a £ value within
it via the per-size fitted lognormal (`band_conditional_mean`, band 10 bounded),
multiply by the per-type bridge (freq==1 -> 1.0; Impersonation drawn per
replicate from 3.11-6.25x). Weighted bootstrap B=500 within each size band ->
per-size £/business -> x N(size) -> national-total distribution. Same seed
(20260715) and RNG draw order (one Impersonation uniform, then 4 weighted
resamples per replicate) as national_simulation.py's type-based branch, so it
reproduces the recorded headline to 2 s.f.:
- mean £2.23bn, median £2.06bn, 90% interval £0.79-4.35bn — all four within the
  0.15 £bn reproduction tolerance. ALL CHECKS PASS.
- per-size: Micro £1.50bn (£1,301/biz), Small £0.57bn, Medium £0.11bn, Large
  £0.05bn — Micro is ~67% of the total, consistent with the body-vs-tail finding.
Python-only engine (Squiggle emit skipped per scope; the original's Squiggle
matched this Python aggregation to 2 s.f. anyway). Output ->
runs/results/estimate_national_baseline.txt.

Spine status: A (this, £2.2bn headline) + B (£1.0bn parametric floor) both now
distilled and reproduced. The two bookend the honest range and both trace to the
same ~2 Micro top-band firms.

### NARRATIVE.md — first section written
"Valuing banded costs: the parametric (B) cost model and what it reveals."
Documents A (semiparametric, empirical band mix, £2.26bn) vs B (fully parametric
lognormal, £1.0bn); the A/B table; the tail diagnosis (whole gap is Micro, 2
firms above an empty gap, lognormal predicts 0.09% vs observed 0.46%, ~5% event);
GOF passes but is blind to the tail; B is a tail-shrunk FLOOR not a truer central;
the principled fix is a GPD-spliced heavy tail (would land at/above A).

## Key finding this session (for the eventual headline framing)
The estimate's central fragility is NOT abstract modeling choice — it is the read
on ~2 Micro top-band firms. Honest range: ~£1.0bn (B / freq-based, tail-shrunk)
to £2.26bn (A / baseline), with a genuine but UNquantifiable upside the survey
cannot exclude (tail-ceiling probe: with only 8 firms >£100k, the empty high bands
barely constrain a fat tail; ceiling is set by an external cap on max single loss,
not the data — see src/estimation/tail_ceiling_pareto.py, £6bn @ £5M cap ... £18bn
@ £500M cap, non-converging). GDP-scaling refutation still to be distilled.

## Exploratory scripts left in the ORIGINAL tree this session (cited-not-yet-distilled)
- src/estimation/size_type_cellmean_estimate.py (approach B origin + A-vs-B diagnosis)
- src/estimation/tail_ceiling_pareto.py (catastrophic-tail ceiling probe)
- distillation/runs/diag_global_vs_stratified_fit.py (global vs per-size fit GOF)

## Next unit (not started)
Approach A (the headline spine) is now DONE (Unit 3 above). Remaining robustness
checks slated for the distillation, in rough priority:
- (a) GDP-scaling refutation (sector + size cuts; the real-aGVA rerun is the
  strong version) — the biggest "bases covered" defense not yet distilled.
- (b) body-vs-tail decomposition (~64% of the total from 8 top-band firms).
- (c) upward-sensitivity / inflator scenarios (the honest £1.0-7bn range).
- (d) censored floor (model-free bounds) — smallest, quick.
A NARRATIVE section for the A/B spine pairing would also close out the headline.
