# Lab 08: wave 4-2 second wave

This lab implements [02 Second-wave protocol](../../docs/00_project_flow/wave4-2/02_second_wave_README.md) on the 12 formal test beds and their 24 primary library-variants (S9 raw Z, S9_2 marginally standardised Z). The modules are:

- W2-00: reproducibility gate.
- W2-01: R*-free blind search.
- W2-02: practical accuracy of the analytic global minima.
- W2-03: conditional Hessian energy.
- W2-04: improved inference of fixed metrics.
- W2-05: synthesis.

No NP was retrained.

All numbers below are generated from `results/` by `analyze.py` (complete tables in [results/report_tables.md](results/report_tables.md)). **They are development results.** Every toy and every candidate (B08, the A1 list, the controls) was examined during method development, so nothing here is independent confirmation.

## Main findings (2026-11-26)

### W2-00: gate passed

452/452 acceptance checks pass. They cover:

- The blind payload contains no truth.
- The blind S9_2 frame equals the saved S8_2 normaliser.
- The Lab 08 designs equal the Lab 07 designs for Q01–Q05.
- The Lab 08 decoder moments equal the Lab 07 tensors (relative difference 0).
- The Lab 08 objectives equal the saved Lab 07 losses at every S9 / S9_2 rotation for the 7 searched candidates.
- Class C output scaling and prediction invariance.
- 13 unit tests, including a truth-file firewall: the blind objective and search run under an audit hook that blocks the S1 dataset, every S8/S9 directory, the evaluation files and the Lab 07 tensors, and the hook demonstrably fires.

All 15,280 first-wave status flags (Q01 pilot all 415 metrics, plus B08 at every measure / fidelity / seed / mask) were recomputed from the saved losses and match `pass_summary.csv.gz`. The first-wave claims for `B08.M03.c0.01` and `c0.03` reproduce:

| Claim | Count |
|---|---|
| Inferred marginal | 24/24 |
| Analytic perfect | 8/24 |
| Analytic marginal | 12/24 |
| B-CR inferred marginal under MASK-COL | 4/6 |

The inferred-marginal seed-status changes at fidelity C (6 / 7 cells) all occur in MASK-COL cells; there are none under FULL / MASK-ALL. Lab 07's `metric_evidence.csv` counts 1 / 2 changes under a different (joint, panel-level) definition.

### W2-01: actual blind recovery, the principal new result

Setup:

- **Objective and selection.** Split-safe BS-B: TRAIN-subject objective, VALIDATION tie-breaker, TEST evaluation after locking. Q01, fidelity C.
- **Seeds.** 3 integration × 3 search seeds for E01/E02.
- **Search budget.** 400/2000 Haar starts with Givens refinement.
- **Pass criterion (evaluation only).** Lab 05: MCC ≥ supervised − .05 and leak max < .2 (C) / < .3 (CR).

| Candidate | Toys passing in all 9 seeds, S9 | same, S9_2 | 24-library coverage | Failing toys (S9_2) |
|---|---|---|---|---|
| E02 `B08.M03.c0.03` | 7/12 | **10/12** | 17/24 | CR3, CR5 (class B) |
| E01 `B08.M03.c0.01` | 6/12 | 9/12 | 15/24 | C3L, CR3, CR5 |
| C01 `UX.lq.1`, C02 `B01.M03.c0.01`, C04 `B09.w75.M03.c0.01` | 6/12 | 6/12 | 12/24 | all six CR toys |
| C03 `Fz.A+B+C` | 7/12 | 4/12 | 11/24 | mixed |
| N01 identity / N02 fixed Haar | 0 | 0 | 0 | — |

**E02 recovery quality.** For E02 with S9_2, all 12 C-family libraries pass with MCC within .001–.008 of the supervised ceiling. On CR3L, CR4, CR4L and CR5L, MCC is within .015–.025 of the ceiling and leak max is .19–.28.

**E02 failures.**

- CR3 and CR5 (class B) fail in both variants, with leak max .33–.44.
- The raw S9 variant also fails CR4, CR4L and CR5L.
- Every failing E01 / E02 search is diagnosed post-lock as **"objective / estimand issue"**: the blind loss at the found R is *lower* than the same frozen loss at R*. A larger search budget therefore cannot fix them.

**Search reliability.**

- Selected rotations agree across seeds to a median relative distance of .007 (E02).
- On an independent 1024 × 128 TRAIN cache (BS-D), the selected E02 finalist is still the best finalist in 216/216 cells; for E01 it is 203/216.
- The blind-diagnostic escalation (S1, 5× budget, plus a search on the D cache) triggered only for C3L. There E02 recovers in 7/8 escalated searches, E01 in 2/8.

**Sensitivity (BS-C).**

- Q02 and Q05 reproduce the pattern: E02 has 14/16 and 13/16 FULL libraries passing.
- Bank estimands stay weak:
  - MASK-ALL, Q01 / Q02 / Q04 / Q05: 2–3 of 8 libraries.
  - MASK-COL: 3 of 8.

**Secondary NP runs (BS-E, S9).**

- E01 and E02 pass C1#r2, C2#r2 and C3#r2.
- Both fail CR4#r2 (FULL) and CR3#b5e-5 (MASK-ALL and MASK-COL, both variants).

**Post-hoc shortlist pilot (`run_shortlist.py`).** This is development only: the metric and surface were chosen after the truth-referenced W2-04 flags. `SL0.UXUU.e0.01` on the E-RBF teacher surrogate passes 8/12 toys (S9). It recovers exactly the class-B toys E02 misses (CR3 S9, CR5 both variants, CR5L S9), but fails C3L, CR3L, CR4 and CR4L. On the frozen decoder (AD) the same metric passes 5/12.

**Conclusion.** A single fixed blind algorithm (E02 + S9_2) recovers the parameters on 10 of 12 toys. No fixed algorithm covers all 12, and the failures are failures of the objective, not of the optimiser. Following decision rule 24 ("B08 fails with lower loss than R*"), the next step is objective repair / conditional or extended structure for B-CR, then independent confirmation.

### W2-02: are the lower analytic minima scientifically consequential?

The Lab 07 W1-04 search was replayed bit-for-bit: 738 metric × target searches, max |ΔL| 7e-15, the persisted suspicious Q reproduced in 422/422. This time every refined finalist was saved, and ideal (U = Q T_s on TEST T) and transferred (U = Q R* V for each toy × variant) recovery were evaluated.

For displaced CR minima of the best analytic solution:

| Distance d_SP/√(2k) | Searches | Ideal recovery (MCC ≥ .95, leak < .2) | Transferred recovery (Lab 05) |
|---|---|---|---|
| ≤ .02 | 90 | 100% | 100% |
| .02–.05 | 224 | 100% | 100% |
| .05–.10 | 134 | 100% | 100% |
| .10–.15 | 10 | 20% | 100% |
| .15–.20 | 4 | 0% | 25% |
| > .20 | 124 | 0% | 0% |

**Most CR "counterexamples" are practically harmless.** 450 CR rows have a displaced minimum that is still practically adequate, against 136 genuine failures, which all lie beyond d ≈ .15. 162 CR rows are flat / tied: the objective cannot identify one axis system.

**Practically adequate metrics.** 51 of 93 priority metrics have a best analytic solution that recovers T in ideal and transferred space on all 8 logit targets × both toys × both variants. These include 31 of the 33 A1 metrics and the UX-only SL0.UX / SL0m.UX / UXcol / UXrms / B01 / B05 recipes; 44 of the 51 have no inconclusive target.

**Fine-angle diagnostics** (13 angles in every plane plus 8 random directions, paired re-evaluation on independent caches):

- I is not a local minimum in 202 of 465 CR metric × target cells; the median lower point is at 5°.
- In C families this holds for 16 of 273 cells.
- For `UX.lq.1`, refinement from I escapes to a lower loss on every CR target.
- Its perturbation restarts (2–20°) never return to I on C2, C3 or any CR target: the basin coverage is weak, as Lab 07 already flagged.

**Conclusion.** The exact Gap 1 failure is real (most CR minima are displaced), but is practically small for d ≤ .1. The remaining practical failures are the far minima and the flat / tied objectives.

### W2-03: conditional Hessian energy

Setup:

- The comparison uses the same product points throughout: 128 TRAIN subjects × 128 X rows, or the full bank × 32 subjects.
- 17 metrics, two arms, both variants.
- Configurations: P0, P1, P2, P4 (K = 4 / 8 / 16, frequency or equal weights), P5 (λ = .25 / .5 / .75) and N0 / N2.

**Findings.**

- **Stimulus conditioning (P1, P4) helps neither arm.** In the C families the analytic pass rate falls from 12/12 to 4/12 libraries for most UX recipes. CR inferred passes also fall: B08.M03 goes from 12/12 under P0 to 0/12 under P1.
- **Latent conditioning (P2) reproduces P0** within ±2 libraries.
- **P5 fusions and N0 / N2 normalisations** never exceed P0.

**Gate outcome.** No configuration improves CR analytic or inferred coverage by more than one library. The frozen escalation rule (CH-C seeds 73 / 101 and medium fidelity, CH-D global / blind tests) therefore did not fire. The pooled baseline is retained; per 02 section 15, the next step is richer derivatives / alternative structure.

### W2-04: improved inference of fixed metrics

All estimators were fitted in the base frame, selected only by validation predictive loss (S-min, or S-1pct within 1%), and evaluated against the analytic losses at the identical Q01 points.

| Estimator | A1 cells inferred marginal, C | same, CR | UX rel. error CR | UU rel. error CR |
|---|---|---|---|---|
| AD decoder (baseline) | .987 | .003 | .078 | .78 |
| E-SMOOTH h = .025–.2 SD | .987 | .003 | .078 | .78 |
| W1 polynomial | .71 | 0 | .14 | .85 |
| E-POLY mono / herm, D0 (S-min) | .75 | .03–.04 | .15 | .84 |
| E-POLY, D1 teacher | .93 | .003 | .072 | .77 |
| E-RBF block, D1 | 1.00 | .033 | .11 | 1.18 |
| IF-C cross-fit (PSD), D0 polynomial | .77 | .08 | .15 | .77 |

**Findings.**

- **Latent-only smoothing of the decoder at the predeclared bandwidths leaves its Hessians unchanged:** the matrix errors agree to 3 decimals.
- **No estimator repairs CR inference**, although every D1 surrogate predicts the teacher almost exactly (validation MSE ≤ 4e-2 in Y units, and the polynomial teacher fits ≤ 6e-3).
- **Cross-fitting** removes some variance energy: CR marginal rises from .03 to .07–.08. The cross-products have negative eigenvalues in 16% of CR J blocks.
- **One combination stands out:** `SL0.UXUU.e0.01` on the E-RBF block D1 surrogate reaches 8/12 CR inferred-marginal libraries, with analytic 24/24. This prompted the shortlist blind pilot above.

**Gate outcome.** This is decision rule 24's "all surrogates fail despite high prediction accuracy". Next: a disjoint-context diagnostic, gradient / contrast targets, or alternative structure.

### W2-05: decision summary (rules of 02 section 24)

| Observation | Meaning | Next action |
|---|---|---|
| E02 + S9_2 recovers 10/12 toys in every seed; the failures (CR3, CR5 class B) have a blind loss below L(R*) | the B08 objective, not the optimiser, fails on B-CR | objective repair / extended structure for B-CR, then independent confirmation (fresh data, NP seeds, families) |
| Displaced analytic CR minima keep ideal and transferred recovery up to d ≈ .1; 51 metrics are practically adequate everywhere | exact Gap 1 failure, practically small | keep these metrics as Gap 2 targets; report exact failure separately |
| Conditional aggregation improves neither arm | pooled HE kept | richer derivatives / alternative simplicity (Step 5) |
| Smoothing, polynomial, RBF and cross-fitting surrogates do not repair CR inference | derivative / representation structure limiting, not prediction | disjoint-context diagnostic; gradient, contrast or reduced-axis targets |
| Bank MASK-ALL / MASK-COL results stay weak in every module | identification / support limitation | separate estimands; observed-row contrast (Step 5C) |

The registry for the next wave is [results/next_wave_candidates.csv](results/next_wave_candidates.csv).

## Deviations from the specification and what was not run

- **Not run:**
  - W2-01: S2 (optional optimizer comparison); the Y-P control.
  - W2-02: the probability-channel control.
  - W2-03:
    - P3 (pointwise penalty: cost; P1/P4/P2 failed first);
    - CH-C seed 73 / 101 and medium-fidelity escalation, and CH-D (no configuration met the frozen escalation rule);
    - the optional bank-contrast pilot of 15.1.
  - W2-04:
    - joint X smoothing; E-MLP;
    - the bootstrap support score of 19.2; shrinkage;
    - IF-D / IF-E;
    - GP / GAM / local polynomial.
  - The disjoint-context encoder diagnostic.
- **Logistic degree-5 polynomials (class B, D0)** were skipped for cost. The cap was declared before fitting and is listed in `inference/IF-B/*.skipped.csv`.
- **Estimator states.** Most IF-B estimator states were not persisted; the fits are deterministic by seed and replayable. Only the shortlist RBF surrogate state is saved, in `outputs/.../inference/estimator_fits/`.
- **BS-E secondary s9_2 searches** of C1#r2, C2#r2, C3#r2 and CR4#r2 were run but not evaluated: those runs have no S8_2 reference ceiling.
- **IF-A / IF-B evaluation points** are the Lab 07 batch-A TEST points (Q01 pilot, seed 31), so the comparison with the first wave is paired.
- **IF-C** uses the D0 polynomial family with the lowest validation loss (blind) and Q01 medium TEST points.
- **Blind selection rule.** The numerical tie-break by validation loss was used in 94–96% of E01 / E02 searches, because several refined finalists converge to the same minimum (median relative distance .007).
- **The shortlist pilot (X01 / X02)** was added after W2-04 was evaluated. It is recorded in `configs/blind_candidates.json` as `post_hoc_shortlist`, outside the frozen W2-01 registry.

### Bugs found and fixed during the run

All affected outputs were deleted and recomputed.

- The job list carried a Windows carriage return.
- In W2-03, a mixed boolean / integer NumPy index transposed the per-latent J moments. A regression test was added.
- The W2-03 reconstruction check divided 0/0 for the C families, whose analytic UU block is identically zero.
- The W2-02 merge treated "supported" minima by their best non-identity finalist. It now uses I.

## How to run

```bash
cd metric_lab/08_wave4_2_second_wave
PY=/d/Programs/anaconda_new/envs/py11/python
for t in C1 C2 C3 CR3 CR4 CR5 C1L C2L C3L CR3L CR4L CR5L "C1#r2" "C2#r2" "C3#r2" "CR3#b5e-5" "CR4#r2"; do $PY prepare.py --toy "$t"; done   # blind payload + eval files
$PY -m pytest -q tests && $PY validate.py                                  # W2-00 gate
./run_w201.sh                                                              # BS-A, BS-B, BS-C, BS-E, BS-D (jobs are resumable)
$PY run_oracle_diagnostics.py --list | xargs -I{} $PY run_oracle_diagnostics.py --target {}; $PY run_oracle_diagnostics.py --merge   # W2-02
./run_w203.sh && $PY run_conditional.py --analyze                         # W2-03
./run_w204.sh && $PY run_inference.py --analyze                           # W2-04 (IF-A, IF-B, IF-C)
for t in C1 C2 C3 CR3 CR4 CR5 C1L C2L C3L CR3L CR4L CR5L; do $PY run_shortlist.py --toy $t; done   # post-hoc shortlist pilot
$PY run_blind.py --evaluate && $PY analyze.py && $PY figures.py         # post-lock evaluation, W2-05 tables, figures
```

## Design

- **Blind side.** These modules import no truth module, as the AST test enforces:
  - `wave2/blind_inputs.py`, `designs.py`, `objectives.py`, `optimizers.py`, `conditional.py`, `surrogates.py`;
  - `run_blind.py` (search part).
  They read only `outputs/.../blind_payload/<toy>.npz` (no T), the frozen checkpoint and their own caches. They reuse the validated Lab 07 numerical modules read-only.
- **Evaluation side.** These open T / R* only after the blind outputs are hashed into `results/blind_locks.csv`:
  - `wave2/evaluate_recovery.py`, `blind_eval.py`, `oracle_merge.py`;
  - `run_oracle_diagnostics.py`;
  - the analytic arms of `run_conditional.py` and `run_inference.py`.
- **Coordinates.** V = Z (S9) or (Z − mean)/sd over all subjects (S9_2); U = R V; d raw / d U = diag(sd) Rᵀ. Squared energies are never rotated. Fixed integration points are transported by R.
- **Search.** Haar starts (both determinant components) plus identity, then the best distinct starts (relative distance modulo signed permutations > .05), then Givens coordinate descent with spans 45 → .5° and capped evaluations. The search RNG is separate from the integration RNG. Selection takes the lowest blind loss, never MCC.

## Outputs

| results/ | Content |
|---|---|
| w2_manifest.json, config_snapshot.json, validation_checks.csv, reproduction_audit.csv | W2-00 snapshot, hashes, checks, recomputed first-wave flags |
| blind_locks.csv, blind_recovery.csv, blind_toy_variant.csv, blind_coverage.csv, blind_library_coverage.csv, blind_categories.csv, blind_seed_stability.csv, blind_reference_loss.csv, blind_bsd_reevaluation.csv, blind_search_trace.csv.gz, found_rotations.npz, recovery_correlations.npz | W2-01: every search, selected R, TEST T–U correlations, L(R*) diagnosis, finalists with their (evaluation-only) MCC |
| oracle_replay_audit.csv, oracle_solution_registry.csv, oracle_solutions.npz, oracle_recovery.csv, oracle_finalist_recovery.csv.gz, oracle_metric_practical.csv, oracle_fine_sweeps.csv.gz, oracle_fine_summary.csv, oracle_local_escape.csv | W2-02 |
| conditional_pass_summary.csv.gz, conditional_coverage.csv | W2-03 (analytic, inferred and joint flags per metric × configuration × library) |
| inference_IF-A/B/C_summary.csv.gz, inference_IF-A/B_fits.csv, inference_estimator_comparison.csv | W2-04 |
| next_wave_candidates.csv, report_tables.md | W2-05 |
| figures/ | T–U heatmaps (unrotated / supervised / blind, every toy × variant, failures included), blind loss vs MCC, oracle distance vs recovery, fine-angle curves, pooled vs conditional, surrogate derivative error vs rotation performance, BS-B pass matrix |

Large arrays are in `outputs/metric_lab/08_wave4_2_second_wave/W2r1/`: blind_payload, eval_only, moment_cache, blind, oracle_diagnostics, conditional, inference, logs.
