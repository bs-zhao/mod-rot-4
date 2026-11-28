# Metric Lab 10 — wave 4-2 fourth wave: Q05 replication, integration-measure diagnosis, three gaps, lock and CONF-A

Implements [04_fourth_wave_README.md](../../docs/00_project_flow/wave4-2/04_fourth_wave_README.md) (W4-00 … W4-06).
Code: this folder.  Arrays: `outputs/metric_lab/10_wave4_2_fourth_wave/W4r1/` (not committed).  Tables: `results/` (`report_tables.md`
is analyzer-generated).  Lab 01-09 code, outputs and the W3 lock are read-only (verified: `results/replay_audit.csv` check G).

## 1. One-page result

| Question (spec) | Answer | Evidence |
|---|---|---|
| W4-00: is the historical 36/36 Q05 observation reproducible? | **Yes, exactly.** Caches bit-identical (J max rel. diff 0), every archived finalist re-scores to < 1e-10, the re-executed S0 searches reproduce every selection (68/72 bit-identical, 4 at 1-4e-8 with identical loss), recovery 36/36 s9 (30/36 s9_2) re-derived from the archived R | `replay_audit.csv`, `replay_summary.json` |
| W4-01: does a fixed Q05 method pass 12/12 robustly? | **No.** RW-GL / Q05 / s9 passes **11/12** toys in every batch; CR4 fails: REP-1 6/9 (integration seeds 31, 73 pass; 101 fails), REP-2 fresh seeds 0/9, REP-3 independent D 0/6. Over every Q05 s9 search CR4 passes 6/24 | `rep_recovery.csv`, `coverage.csv` |
| Is that better than the W3 Q01 lock? | **Not demonstrably.** W3 Q01 lock: 11/12 on Lab 09 seeds (CR4 0/9) and 11/12 on the fresh seeds 307/401/503 (CR4 3/9). Both fail only CR4, both pass the other 11 in every run | `coverage.csv` (REP-2C, W3-FIN) |
| W4-02: why did Q05 change CR4? | **Finite-integration luck on a near-tie between two distinct basins**, not the Gaussian law as such. On every measure both the bad Q01 rotation and the good seed-31 Q05 rotation are ~.011-.013 *below* L(R\*) (row-weight term ≈ −.02 beats the row-score term ≈ +.007); the two differ by only ±2e-4 with a latent-block bootstrap SD ≈ 1e-4, and which one wins follows the standard-normal draw: seeds 31 / 73 pass for M-D **and** M-J (covariance irrelevant), 101 / 307 / 401 / 503 / D211 / D613 fail; empirical laws (M-E, M-I, Q01, Q02) fail every seed | `measure_recovery.csv`, `loss_decomposition.csv`, `block_bootstrap.csv`, `fig3_cr4_landscape.png` |
| W4-03: theoretical explanation? | Gap 1 (O-true): RW-GL practically adequate on 8/8 analytic targets; exact Gap 1 has confirmed global counterexamples on CR3 (both masks) and CR4 at d_SP .05-.065 (practically harmless). Gap 2: RW-GL both-marginal on **23/24** libraries at every Q05 seed (22/24 at D 211); the only miss is CR4 s9 (inferred side). The inferred CR4 landscape, not the analytic one, carries the failure | `oracle_global.csv`, `gap2_summary.csv` |
| W4-04: representation / mask specificity | s9 > s9_2 for RW-GL under every measure (s9_2 also loses CR4L). Q02 / Q03 / Q04 behave like Q01 / Q05 (all pass the 8 non-CR4 random and 4 bank toys; CR4 fails, Q03 3/9). MASK-COL and decoder-only FULL on banks: 4/4 s9. Secondary saved NPs: **5/5 incl. CR4#r2 (9/9)** under Q05 and Q01 | `robustness_recovery.csv` |
| W4-05: lock | `results/algorithm_lock_W4.json` = **W4-Q05-v1** (RW-GL, raw Z, Q05, S0, D-ranked single deployment), lock sha256 `d487357c…`; recorded as a development selection; W3 lock untouched | `freeze_algorithm.py --verify` |
| W4-05: CONF-A | see section 7 (fresh data seed 2026112801, NP seeds 2026112811 / 2026112812) | `confirmation_*.csv` |
| W4-06 | **not run**: no reviewed novel-family registry exists (section 21 requires one before any generation) | — |

Decision row of section 9 that applies: **"Less than 12/12: retain Q05 as a partial development result … may confirm its actual limited
scope under a lock."**  Q05 is retained as an equal-coverage alternative to the W3 Q01 lock (11/12, CR4 open), not as a 12/12 method.
No metric parameter (lambda, c, a) was changed.

## 2. W4-00 replay and gate

* `validate.py`: 9/9 unit tests (`results/validation_checks.csv`): GL formula re-implemented from the spec text equals `rowmetrics`
  (200 random B, signed-permutation invariance, exact .75/.25 decomposition, NaN on zero energy); moment contraction equals AD of
  g_R(u,x) = g(Rᵀu, x) at fixed points for both frames; AD vs central FD; class-C output scaling; the evaluation law of any R is the
  push-forward of the same base points (block sums == global J to 1e-12); M-D == historical Q05 bit-for-bit, deterministic replay,
  matched normals for M-J, whole-row bank replication, mask denominators; blind Q05 search under an audit hook that forbids opening
  any truth-bearing file; static import split.
* `run_replay.py` (`replay_audit.csv`): A key identity 12/12; B archived recovery 72/72; C caches 32/32 + design fit 12/12; D finalist
  re-scoring 72/72; E re-executed S0 72/72 (tolerance 1e-6 on d_SP of R Rᵀ, raw distances saved); F replay recovery 72/72
  (s9 36/36 pass); G W3 lock 2/2 (canonical LF text = committed blob; working copy differs only by CRLF; Lab 09 `--verify` passes).

## 3. W4-01 replication (RW-GL, both variants; controls)

| batch | measure / seeds | s9 toys (runs) | s9_2 toys (runs) | failing |
|---|---|---|---|---|
| REP-1 | Q05 C, i31/73/101 × s31/73/101 | 11/12 (105/108) | 10/12 (90/108) | CR4 6/9 (s9); CR4, CR4L 0/9 (s9_2) |
| REP-2 | Q05 C, fresh i307/401/503 | 11/12 (99/108) | 10/12 (90/108) | CR4 0/9 |
| REP-3 | Q05 D (1024×128 / 512×bank), i211/613 | 11/12 (66/72) | 10/12 (60/72) | CR4 0/6 |
| REP-2C | W3 lock (Q01) on fresh seeds | 11/12 (102/108) | 10/12 (93/108) | CR4 3/9 |
| W3-FIN | W3 lock (Q01), Lab 09 seeds | 11/12 (99/108) | 10/12 (90/108) | CR4 0/9 |
| REP-1 | E02 on Q05 (diagnostic) | 5/12 | 10/12 | Q05 does not repair E02 |
| REP-2C | E02 Q01 fresh seeds | 5/12 | 10/12 | CR3, CR5 (s9_2) as in W2/W3 |

REP-4 (`independent_reeval.csv`): every REP-1/2 finalist re-scored on both D caches.  The D caches agree with the C selection in 94-100%
of cells; 116/432 cells were escalated by the Lab 08/09 blind flags (mostly near-minima separation / seed spread); the 42 S1 (5×)
searches change no pass/fail outcome.  **In all 12 toys and all 18 cells, L(R̂) < L(R\*) on the search cache and on both independent
caches** — the Q05 minimum is never R\* itself; where recovery passes it is a nearby rotation (relative distance to R\* .005-.20, median .03).  CR4 failures are
"objective / estimand issue" (L̂ < L\* − τ), never optimizer failures.

Single deployment (development readout, `deployment_readout_dev.csv`): the declared blind D-ranked rule passes 11/12 s9 (it picks the
failing CR4 cell i101), the fixed pair i31/s31 passes 12/12 s9 — but that cell *is* the post-hoc FIN-SENS observation, so it is not
evidence.  D-ranked was frozen as primary; the fixed pair is reported as a declared secondary readout.

## 4. W4-02 integration-measure diagnosis (CR4 mechanism)

Controlled grid (all 12 toys, 3×3, same X panel, matched normals; `measure_recovery.csv`):

| law | s9 toys | CR4 s9 by integration seed 31 / 73 / 101 | s9_2 |
|---|---|---|---|
| M-P = Q01 (paired empirical) | 11/12 | 0 / 0 / 0 | 10/12 |
| M-E (empirical Z × common X) | 11/12 | 0 / 0 / 0 | 10/12 |
| M-I (independent empirical marginals) | 11/12 | 0 / 0 / 0 | 10/12 |
| M-J (joint Gaussian, same normals) | 11/12 | 3 / 3 / 0 | 10/12 |
| M-D = Q05 (diagonal Gaussian) | 11/12 | 3 / 3 / 0 | 10/12 |
| MIX α = .25 / .5 / .75 (J mixture E→D) | 11/12 | 3/0/0, 3/3/0, 3/3/0 | — |

Sensitivity at seed 31 (s9; `report_tables.md`): ρ = .25/.5/.75, κ = .75/1.25 and subpool fits with fixed X all 12/12 — **but they reuse
the seed-31 normals and X panel, so they are not independent evidence**; truncation (own RNG stream) TRN-2 / TRN-3 lose CR4 (0/3);
subpools with their own X rows lose CR4 in 3/6.

Interpretation (section 11.1 table): M-D = M-J with the same draws → covariance removal is **not** the cause; Gaussian laws can produce
the good CR4 basin while empirical laws never do in 9+ draws, but only for 2 of 8 Gaussian integration draws → row "new good minimum
appears only for some MC seeds: finite integration issue", not "robust metric success".  Mechanism (`loss_decomposition.csv`,
`cr4_geodesics.csv.gz`, `fig3_cr4_landscape.png`): along the geodesic between the Q05 (good) and Q01 (bad) CR4 rotations every law
shows two basins separated by a ~.003 barrier with end values within ~2e-4; both are ~.011-.013 below R\* because the row-weight term
(≈ −.02) outweighs the row-score term (≈ +.007) — the RW-GL objective itself prefers a mixed CR4 axis set; the Gaussian reference only
tilts a near-tie.  Block bootstrap (`block_bootstrap.csv`): sd of L(R_Q05) − L(R_Q01) ≈ 1e-4 against differences of ±2e-4.

Support (`support_diagnostics.csv`, `tail_energy.csv`, `predictive_by_tail.csv`): learned Z is close to, but not exactly, Gaussian
(|skew| ≤ .24, |excess kurtosis| ≤ .3 per axis; small correlations).  Q05 draws are not more off-support than empirical Z (fraction
beyond the 99% VALIDATION Mahalanobis radius ≈ .002-.01); Q05 moves UX energy toward |z| > 2 draws (median share .20 vs .16 for Q01)
but away from |z| > 3 (.001 vs .009).  The decoder predicts actual VALIDATION rows of |z| > 3 subjects slightly worse for CR4 (log-loss
.43 vs .39).  Nothing indicates that Q05 extrapolates into unsupported regions.

## 5. W4-03 three gaps under labelled measures

* **O-true** = true T of the empirical TRAIN subjects of the M-E design × the same X panel / bank and masks as Q05 (supported by
  construction).  O-ref and the Z→T bridge were not run (sections 12.2-12.3 are optional sensitivity / diagnostic).
* **Gap 1** (`oracle_global.csv`): RW-GL — empirical support on C1/C2/C3/CR5 (both masks); global counterexamples on CR3 (MASK-ALL,
  MASK-COL) and CR4, confirmed on both independent D caches, at d_SP .051-.065 with ideal MCC ≥ .996 and transfer passing → **8/8
  practically adequate, 5/8 exact empirical support**.  E02: 6/8 practical (C2, C3 tied non-permutations far from I); B01: 8/8.
* **Gap 2** (`gap2_summary.csv`, primary masks, 24 libraries): RW-GL analytic perfect 24/24 at every seed; inferred marginal 23/24
  (22/24 at D 211); both-marginal 23/24 (22/24); the only miss is CR4 s9 (s9_2 also at D 211).  E02: analytic perfect 8/24; B01:
  inferred 12/24.  Pooled Pearson (median) RW-GL .96, E02 .97, B01 .77.  W3 Q01 baseline: 22/24 (CR4 both variants).
* **Gap 3** = W4-01.  Practical Gap 1 and Gap 2 pass on 11 toys and Gap 3 passes on the same 11; CR4 fails on the inferred side
  (Gap 2) and in recovery (Gap 3) while the analytic CR4 objective is practically adequate → the CR4 problem lives in the learned
  representation / decoder surface (consistent with CR4#r2 passing 9/9), not in the metric's analytic preference.

## 6. W4-04 robustness (`robustness_recovery.csv`)

* RB-Q: Q02 / Q03 / Q04 with RW-GL, 3×3: all 4 bank toys pass every run in both variants; random toys: 7/8 s9 (CR4 0/9, Q03 3/9), 6/8 s9_2.
* RB-MASK (banks, Q05): MASK-COL and decoder-only FULL both 4/4 s9; s9_2 loses CR5L (3/9).
* RB-SEC (secondary saved NPs; S9_2 references exist only for CR3#b5e-5): RW-GL Q05 and Q01 5/5 toys (45/45 s9 runs), including
  **CR4#r2 9/9** and CR3#b5e-5 (alternative beta, not a seed replicate); E02 3/5.
* The optional context-representation diagnostic of section 16 was not run (no specific unresolved question).

## 7. W4-05 lock and CONF-A

Lock `results/algorithm_lock_W4.json` (W4-Q05-v1, schema W4-LOCK-1): exact formula / constants / active rule / dtype, raw-Z frame,
channel rule, Q05 rule as an *algorithm* (TRAIN-pool mean / population SD recomputed per dataset, RNG streams, X / bank / mask policy),
fidelity C, integration seeds 31/73/101, S0 search and VALIDATION tie, S1 escalation policy, **primary single deployment = D-ranked**
(lowest loss on an independent Q05 D cache, iseed 211, among the 9 selections; fixed pair i31/s31 secondary), controls, evaluation
criterion, policies, the W3 lock hashes (untouched) and the confirmation manifest hash.  Canonical (LF) hashes: algorithm files (a change
fails `--verify`) and orchestration / evaluation files (recorded; a change is listed as a logged deviation).

CONF-A (`configs/confirmation_protocol.json`; manifest registered before any confirmation output existed): data seed 2026112801 ×
NP training seeds 2026112811 / 2026112812 × 12 toys = 24 NPs, generated / trained by the unchanged S1/S2/S3/S5/S8/S8_2 scripts with
the output root redirected (`wave4/conf_pipeline.py`); banks byte-identical copies.  All 12 datasets qualified (S2).

<!-- CONF-A RESULTS -->

## 8. Deviations, limits and bugs found

* **CONF-A schedule.** NP training runs on one 8 GB laptop GPU (~8-10 s / epoch, historical runs need 188-764 epochs; ≈ 20 GPU-hours for
  24 NPs).  Generation / training ran in parallel with development (reading nothing from it); every confirmation blind run and evaluation
  happened after the lock.  See section 7 for what was complete at report time.
* Fixed-pair deployment readout in development is biased (it is the post-hoc cell); D-ranked was frozen instead.
* MS sensitivity laws were run at integration seed 31 only (protocol), which shares normals with historical Q05 → their 12/12 is not
  independent; the controlled grid with 3 seeds is the evidence.
* Gap 2 uses TRAIN-pool inferred caches vs O-true of TRAIN subjects (Lab 07/09 used TEST-pool Q01 tensors); labelled.
* Not run: O-ref, bridge, context-representation Q05 fits, CONF-A full 48-NP panel, CONF-B, CONF-C / W4-06 (no reviewed registry).
* Bugs fixed during the run: Gap-1 output directory created after the first save (rerun, no result used before the fix); replay
  equivalence tolerance 1e-8 → 1e-6 (4 rotations differ by 1-4e-8 with identical loss); W3-lock byte check replaced by canonical-text
  check (core.autocrlf); training queue restarted once (3 → 2 GPU workers; stray workers from the first queue killed, their partial runs
  discarded — checkpoints are written only at the end).

## 9. How to run

`run_all.sh` lists every step in order (Windows py11, `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`).  Main entry points:
`validate.py`, `run_replication.py` (`--caches`, `--jobs B`, `--batch`, `--reeval`), `run_batch.sh N [batches]`, `run_evaluation.py --stage
rep|measures|robustness`, `run_replay.py`, `run_measure_diagnostics.py`, `run_gap_evaluation.py`, `freeze_algorithm.py`,
`run_confirmation.py` (`--register / --generate / --train / --payload / --blind / --evaluate`), `conf_train.sh`, `conf_followup.sh`,
`analyze.py`, `figures.py`.

## 10. Layout

`wave4/common.py` paths & adapters · `designs.py` Q01-Q05 (historical) + controlled M-* / ρ / κ / truncation / subpool laws ·
`moments.py` AD moment caches with latent-block resolution · `search.py` blind job runner · `reeval.py` REP-4 · `conf_pipeline.py`
CONF-A adapter · `confirmation.py` CONF-A payload / blind / evaluation · `eval_only/` recovery (lock-then-truth), oracle (O-true, Gap 1,
Gap 2), diagnostics (support, tail energy, decomposition, geodesics, block bootstrap).  Configs: `primary_candidate.json`,
`replication_protocol.json`, `measure_diagnostics.json`, `gap_protocol.json`, `confirmation_protocol.json`.
