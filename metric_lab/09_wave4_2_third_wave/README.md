# Lab 09: wave 4-2 third wave

This lab implements [03 Third-wave protocol](../../docs/00_project_flow/wave4-2/03_third_wave_README.md) on the 12 formal test beds (C1, C2, C3, CR3, CR4, CR5 and their `L` variants) and their 24 primary library-variants (S9 raw Z, S9_2 marginally standardised Z). The modules are:

- **W3-00** validation and exact replay of the second wave;
- **W3-01** repair of B08's row-weighting mechanism (RW-A decomposition, RW-B local flags, RW-C blind pilots);
- **W3-02** factorial metric × estimator comparison (MX-A), same-frame fusion (MX-B), Pareto diagnostics and a blind cross-objective selector (MX-SEL);
- **W3-03** gradient (G), observed-stimulus contrast (D) and reduced-axis (R / ST-E) structural targets, with full-bank vs interior controls (ST-C);
- **W3-04** frozen-encoder disjoint-context diagnostic (DC-N nested contexts, DC-R repeat blocks);
- **W3-05** unified evaluation: exact / practical Gap 1, Gap 2 local flags, Gap 3 blind search, preregistered escalation and the 3 × 3-seed finalists;
- **W3-06** algorithm lock and confirmation protocol.

**No NP was retrained and no third-order derivative was used.** As in the earlier waves, T / R* never enter a loss, a hyperparameter or a selection: the blind code reads only the truth-free payload of Lab 08, and truth is opened by `wave3/eval_only/` only after the outputs of a stage have been hashed into a lock file.

All numbers below come from `results/` and are produced by `analyze.py`. The complete tables are in [results/report_tables.md](results/report_tables.md). **They are development results.** Every candidate was screened on the same 12 test beds, and the final rule (section 23) selects with T. Nothing in this wave is independent confirmation: CONF-A was not run, see W3-06.

Pass criterion (evaluation only, unchanged from Lab 05/08): MCC ≥ supervised − .05 **and** leak max < .2 (C families) / < .3 (CR families). A toy "passes" a seed panel only if **every** run in the panel passes.

## Headline (2026-11-27)

| Fixed blind algorithm (3 integration × 3 search seeds) | Toys passing all 9 runs, S9 | same, S9_2 | 24-library coverage | Failing toys |
|---|---|---|---|---|
| **RW-GL-l0.75-a1.0 (locked, S9)** | **11/12** | 10/12 | **21/24** | S9: CR4; S9_2: CR4, CR4L |
| MX-SEL (blind E/X selector) | 10/12 | 10/12 | 20/24 | CR3L, CR4L |
| MX-F-w0.75 (fusion, MX-SEL support job) | 10/12 | 10/12 | 20/24 | CR3L, CR4L |
| MX-F-eps0.1 / eps0.25 / w0.5 / w0.5-q9505 | 8/12 | 10/12 | 18/24 | S9_2: CR3L + CR4L or CR4 |
| E02 = B08.M03.c0.03 @ AD (W2 baseline) | 7/12 | 10/12 | 17/24 | S9_2: CR3, CR5 |
| RW-GL-l0.5-a1.0 / RW-ACT-b0.1 | 9 / 8 | 9 / 9 | 18 / 17 | CR4 (+ CR4L/CR5L or CR3/CR5) |
| X02 = SL0.UXUU.e0.01 @ RBF | 6/12 | 7/12 | 13/24 | CR3(L), CR4(L), CR5L |
| ST-DA-col+mean-l0.25 (best structural) | 2/12 | 2/12 | 4/24 | everything except C1 / C1L |

- **The row-weight repair works on its own target.** Mixing a 25% global POW(1) term into B08 (`RW-GL-l0.75-a1.0`) recovers CR3 and CR5 in every seed and both variants. Those are the two class-B toys that E02 never recovered. On the bank estimands it reaches 4/4 under MASK-ALL for Q01, Q02, Q04 and Q05 and 4/4 under MASK-COL, against 2/4 for E02.
- **A new wrong rotation appears in CR4.** CR4 (both variants) and CR4L (S9_2) fail in every run. In each case the blind loss at the found R is below the loss at R* ("objective / estimand issue"), so the optimiser is not the cause. Decision rule 28 therefore reads *"fixed old wrong R loses preference but new wrong R appears → mechanism not sufficient"*.
- **No fixed algorithm reaches 12/12.** The pilot result MX-F-eps0.25 S9_2 12/12 (one integration seed) **did not replicate**: under 3 × 3 seeds it gives 10/12 (CR3L 5/9 runs, CR4L 8/9). Only a retrospective union of candidates covers all toys. Following rule 28 (*"only retrospective union passes → no 12/12 claim"*), no 12/12 claim is made.
- **Gradient and contrast targets fail.** Every G / D target fails in practice (≤ 2/12 toys, analytic local perfect 0/24). On the identical product grid, pooled HE reaches 9/12. The full-bank vs interior comparison does not favour the new targets.
- **A larger context improves Stage A, but the encoder is not the bottleneck of the repaired objective.** Larger disjoint contexts strongly improve Stage A (CR3 R² .17 → .58 from 25 to 200 trials). The repaired objective, however, recovers the same-context ceiling at every context size (10–12/12 toys).

## W3-00: gate passed

148/148 checks pass ([results/validation_checks.csv](results/validation_checks.csv), [results/replay_audit.csv](results/replay_audit.csv)). They cover:

- The Lab 08 blind payloads, designs and moment caches are re-read unchanged.
- The W2 blind counts reproduce (E02 S9 7/12, S9_2 10/12; X02).
- The B08 local status reproduces from the Lab 07 tensors.
- The loss replay matches the archived W2 losses to 2.6e-15.
- The W2 E-RBF surrogate refit is coefficient-identical.
- The MX-A replays of E02 / X01 / X02 are bit-identical to the W2 job files.
- 13 unit tests (`tests/test_w3.py`). They include:
  - the exact row-decomposition identity;
  - fusion invariance;
  - Haar calibration determinism;
  - the antithetic sign-flip invariance of the reduced-axis estimator;
  - a truth-file firewall: an audit hook blocks the S1 dataset, every S8/S9 directory, the evaluation files and the Lab 07 tensors while the blind objective / search runs, and the test checks that the hook fires.

## W3-01: repairing B08's row weighting

### RW-A: exact decomposition of the E02 failure

With B the UX energy (k × p), the row loss is L = Σᵢ wᵢ sᵢ: wᵢ is the row's energy weight and sᵢ its sparsity score. Write ΔL between the E02 blind solution R̂ and R* as ΔL = Σ w̄Δs + Σ s̄Δw. The identity holds to 2e-16 in every cell, and the archived losses are reproduced to 2.6e-15 ([results/repair_decomposition.csv.gz](results/repair_decomposition.csv.gz)).

| Toy group | ΔL = L(R̂) − L(R*) | Score term Σ w̄Δs | Weight term Σ s̄Δw | Cells with negative weight term |
|---|---|---|---|---|
| C families (C1–C3, L variants) | −.001 to −.005 | dominates (negative) | small (±1e-3) | 0/9 (C2L: 9/9, tiny) |
| CR families (all 12 CR cells) | −.003 to −.026 | **positive** (+.005 to +.019) | **negative** (−.010 to −.037) | 9/9 in every CR cell |

So in every CR toy the blind solution wins **through re-weighting**: it moves energy onto rows whose scores are worse but whose weights fall. This happens in CR4, CR4L and CR5L as well, which E02 S9_2 recovers. This confirms the section 1.2 hypothesis: B08's energy weighting lets the objective trade row sparsity for weight.

17 of the 39 fixed recipes, plus the controls B01c03 and B07c03, make the old E02 wrong solution lose to R* on CR3 / CR5 in **every** cell ([results/repair_old_vs_ref.csv.gz](results/repair_old_vs_ref.csv.gz)). These are:

- POW with a = 0, and POW with a = .25 at c ≥ .03;
- every GL with λ ≤ .5 or a < 1;
- every SM with a = 0;
- BL with ρ = .75. RW-GL-l0.75-a1.0 still prefers the old R̂ in 75% of the CR3 / CR5 cells at the frozen E02 rotation, yet its own blind search lands on a different, correct solution.

### RW-B: local Gap 1 / Gap 2 (Q01 pilot, 24 libraries)

| Candidate | Analytic perfect | Analytic marginal | Inferred marginal | Joint | CR inferred marginal |
|---|---|---|---|---|---|
| RW-ACT-b0.3 | **24** | 24 | 23 | 23 | 11/12 |
| RW-GL-l0.75-a1.0 | **24** | 24 | 23 | 23 | 11/12 |
| RW-ACT-b0.1 | 20 | 24 | 22 | 22 | 10/12 |
| RW-GL-l0.5-a1.0 | 24 | 24 | 21 | 21 | 9/12 |
| B01.M03.c0.03 | 24 | 24 | 12 | 12 | 0/12 |
| E02 (= RW-POW-a1.0-c0.03 = RW-SM-a1.0-*) | 8 | 12 | **24** | 12 | 12/12 |

The repaired recipes close most of E02's analytic local gap (8 → 24 perfect) at the cost of one inferred-marginal library. Recipes that weaken the weights further (POW with a ≤ .75, BL, SM with a ≤ .5) reach at most 12/24 analytic perfect.

### RW-C: blind pilots (TRAIN Q01 C, integration seed 31, search seeds 31/73/101)

[results/coverage.csv](results/coverage.csv) lists all 39 recipes plus the controls. The best are:

| Recipe | S9 | S9_2 | Libraries |
|---|---|---|---|
| RW-GL-l0.75-a1.0 | 11 | 10 | 21/24 |
| E02 and its numerically identical recipes | 7 | 10 | 17/24 |
| RW-ACT-b0.1, RW-GL-l0.5-a1.0 | 9 | 9 | 18/24 |
| RW-POW-a0.75-c0.01 | 8 | 9 | 17/24 |

Recipes with weaker weights fail C2 / C3 (POW / SM with a ≤ .5, BL, GL with a < 1), and the a = 0 recipes fail every toy. GL-l0.25-a1.0 keeps the C families but loses all six CR toys. The repair works only with most of B08's weighting **plus** a modest global support term.

## W3-02: factorial comparison and same-frame fusion

### MX-A factorial: metric × estimator (seed 31 caches)

| Metric \ surface | AD (frozen decoder) | RBF (W2 E-RBF block D1) |
|---|---|---|
| B08.M03.c0.03 | S9 7 / S9_2 **10** | 6 / 6 |
| B01.M03.c0.03 | 6 / 6 | 6 / 6 |
| SL0.UXUU.e0.01 | 5 / 5 | 8 / 7 |
| SL0.UX.e0.01 | 5 / 5 | 8 / 8 |

The W2 shortlist effect (X02) belongs to the **combination**: SL0 metrics are better on RBF, while B08 is better on AD. Neither component alone covers the class-B CR toys and the C families together.

### MX-B fusion and MX-SEL

The components are E = B08.M03.c0.03 on AD and X = SL0.UXUU.e0.01 on RBF. Each component is z-scored with its own Haar-512 calibration (median / IQR; q95–q05 and sd as sensitivity). The fusions are:

- F_w with w = .25 / .5 / .75;
- F_max;
- an ε-constraint (min z_X subject to z_E ≤ z_E,min + δ, with δ = .1 / .25 / .5).

| | Pilot (seed 31) | 3 × 3 finalists |
|---|---|---|
| MX-F-eps0.25 S9_2 | 12/12 | 10/12 |
| Other fusions S9_2 | 10–11/12 | 10/12 (w0.25: 7/12) |
| MX-SEL S9 | 11/12 | 10/12 |
| MX-SEL S9_2 | 10/12 | 10/12 |

MX-SEL is the blind selector: min max(z_E, z_X) on the VALIDATION cache with TRAIN calibration.

- **MX-SEL's disagreement flag** (E and X minimisers more than .10 apart) predicts errors but is not used for selection. Unflagged runs pass 100%; flagged runs pass 83% (S9) and 67% (S9_2).
- **Pareto diagnostics** ([results/pareto_front.csv](results/pareto_front.csv)): the union of component and fusion finalists on the (L_E, L_X) front passes at the same rate as the dominated points (.80 vs .79). Both objectives are low at the wrong solutions as well, so the front does not separate right from wrong.

## W3-03: beyond averaged Hessian energy

### Setup

Product-grid decoder gradients are computed for 256 subjects × 256 X rows (medium; 128 × 500 for the bank toys). The same automatic-differentiation pass produces pooled HE moments on the identical grid (ST-HE controls). The targets are:

- G: total gradient energy C_G, its stimulus-average C0 and stimulus-dependent CΔ parts, the stimulus-resolved C_x, C^c_x and the J / Jt forms;
- D: pairwise contrast under five pair designs (uniform, nearest, far, axis-focused, coverage-balanced; 1024 pairs), each with three recipes (global / row / column) and optional mean-response fusion λ = .25 / .5 / .75;
- interior-row controls for the bank toys.

### Results

- **Local:** every G / D target has analytic local perfect **0/24**. The best, ST-DA-col, is analytic marginal 4/24 and inferred marginal 4/24.
- **Blind pilots:** ≤ 2/12 toys (C1 and C1L only). ST-DA-col+mean-l0.25, the escalated GD candidate, stays at 2/12 under 3 × 3.
- **ST-C bank toys (CR3, CR5 and L variants):**
  - The full-bank and interior contrasts all fail: MCC .68–.74, leak .52–.59.
  - Interior is slightly *better* than full.
  - Pooled HE B08 on the same grid passes 50% (S9_2).
  - Decision-table row *"full-bank contrast beats same-support HE and interior contrast"* is **not** met.
- **ST-E reduced-axis:**
  - Single-axis deletion total effect against an isotropic Gaussian reference, with antithetic ξ (sign-flip invariance 3e-10) and the capped RA budget.
  - Local Gap 1 fails everywhere (O_perfect 0).
  - The analytic arm exists only for the unbounded C families: Pearson with the decoder curve .88–.997. It is NA for the CR families (bounded prior support).
  - The blind pilot is 0/12.

**W3-03 conclusion.** First-order gradient and contrast energies, and the reduced-axis deletion effect, lose the axis information that pooled second-order energy carries. In decision-table terms this is *"all extended structure fails"* for the first-order and reduced-axis targets. Third-order derivatives (ST-F) were excluded by instruction.

## W3-04: disjoint-context diagnostic (frozen encoder)

### Setup

The frozen S5 encoder re-encodes every subject from nested, disjoint trial contexts of 25 / 50 / 100 / 200 / all trials. Each context has a fixed query set (100 trials or 20%) and context seeds 31 / 73 / 101 (DC-N). DC-R adds two disjoint repeat blocks of 50 / 100 / 200 trials. The GPU path reproduces the official S5 representation exactly at full context.

Stage A and the supervised reference are computed per representation, with TRAIN-fit frames ([results/context_diagnostics.csv](results/context_diagnostics.csv)).

### Stage A (mean R² of the supervised fit)

| Context | CR3 | CR5 | C3 | All toys, mean |
|---|---|---|---|---|
| 25 trials | .17 | .19 | .54 | .54 |
| 50 | .29 | .34 | .72 | .66 |
| 100 | .43 | .48 | .83 | .76 |
| 200 | .58 | .63 | .91 | .83 |
| all observed (N\|all) | .72 | .76 | .93 | .88 |
| official full S5 | .76 | .78 | .94 | .90 |

- **DC-R repeat-block correlation:** CR3 .30 → .45 → .60 at 50 / 100 / 200 trials; C3 .73 → .85.
- **Out-of-distribution contexts:** 25- and 50-trial contexts are out of the training context-size range in 100% of subjects, and 100-trial contexts in 75%. Those representations are flagged.

### Stage B (blind search on each representation)

The search reuses the same caches and searches, and recovery is judged against the **same-context** supervised ceiling ([results/context_recovery.csv](results/context_recovery.csv)).

| Representation | E02 S9 | E02 S9_2 | RW-GL-l0.75-a1.0 S9 | RW-GL-l0.75-a1.0 S9_2 |
|---|---|---|---|---|
| 25 trials | 9 | 7 | 12 | 10 |
| 50 | 9 | 8 | 12 | 11 |
| 100 | 9 | 8 | 12 | 11 |
| 200 | 7 | 9 | 11 | 11 |
| all observed | 6 | 10 | 11 | 10 |
| official full | 7 | 10 | 11 | 10 |

- **Mean MCC gap:** RW-GL .005–.010 against E02 .020–.024.
- **Local Gap 2:**
  - RW-GL is analytic perfect 12/12 and inferred marginal 11–12/12 at every context.
  - E02 is analytic perfect 4/12 and inferred marginal 10–12/12.

**W3-04 conclusion.** More context improves the representation substantially (rule 28: *"disjoint context improves fidelity with information"*). The blind objective follows the same-context ceiling at every size, so the remaining CR4 failure of RW-GL is not caused by representation quality alone. The controlled bridge interventions of section 19.1 were not run (see Deviations).

## W3-05: unified evaluation

### Exact / practical Gap 1 (analytic global search)

The analytic global search uses the 8 targets C1 / C2 / C3 / CR4 (FULL) and CR3 / CR5 (MASK-ALL and MASK-COL) ([results/oracle_summary.csv](results/oracle_summary.csv)).

| Candidate | Empirical support | Global counterexample | Tied | Practical adequate | Worst d_SP of best minimum |
|---|---|---|---|---|---|
| E02 | 2 | 5 | 1 | 6/8 (C2, C3 fail) | .36 |
| RW-GL-l0.75-a1.0 | 6 | 2 | 0 | **8/8** | .061 |
| RW-GL-l0.5-a1.0 | 7 | 1 | 0 | 8/8 | .057 |
| RW-ACT-b0.1 / b0.3 | 4 / 3 | 4 / 5 | 0 | 8/8 | .07 / .08 |
| RW-POW-a0.75-c0.01 | 3 | 4 | 1 | 6/8 (C2, C3 fail) | .49 |
| MX-F-eps0.1 (analytic analogue) | 6 | 2 | 0 | 8/8 | .040 |
| MX-F-eps0.25 / MX-F-w0.5-q9505 (analytic analogue) | **8** | 0 | 0 | 8/8 | .014 / .018 |

These categories rest on a finite search, so "empirical support" is never a proof. All remaining counterexamples of the RW-GL candidates are practically harmless (d_SP ≤ .061, with ideal and transferred recovery passing). The analytic fusion analogue is the only Gap 1-clean family, but it does not model the AD/RBF estimator pair of the deployed fusion.

### Escalation

Escalation follows the section 23 rule, applied by `escalate.py` to [configs/finalists.json](configs/finalists.json) after the pilots were locked. The finalists are:

- the controls E02 and X02;
- RW, empirical: RW-GL-l0.75-a1.0, RW-ACT-b0.1, RW-POW-a0.75-c0.01;
- RW, theory: RW-ACT-b0.3, RW-GL-l0.5-a1.0;
- MX, empirical: MX-F-eps0.25, MX-F-eps0.1, MX-F-w0.5-q9505;
- MX, best S9: MX-SEL, with support jobs MX-F-w0.25, w0.5 and w0.75;
- GD: ST-DA-col+mean-l0.25, the single best candidate because the family is below baseline.

Declared exclusions applied before ranking are listed under Deviations.

### Finalists: 3 integration seeds (31 / 73 / 101) × 3 search seeds

All 12 toys and both variants were run ([results/finalist_coverage.csv](results/finalist_coverage.csv), [results/finalist_recovery.csv](results/finalist_recovery.csv)). The headline table above gives the coverage.

Per-toy detail of the locked candidate RW-GL-l0.75-a1.0:

- **The C families pass every run.** MCC gap ≤ .0012 and leak ≤ .076.
- **The CR toys that pass do so with margin.**
  - S9 gap ≤ .011, S9_2 gap ≤ .031.
  - Leak max .14–.27 (threshold .3).
- **CR4 fails every run in both variants, as does CR4L in S9_2.** Gap .05–.07, leak .36–.42. In every one of these runs L(R̂) < L(R*) ("objective / estimand issue").

### Run-level pass rates (FIN, 108 runs per variant)

| Candidate | S9 | S9_2 |
|---|---|---|
| RW-GL-l0.75-a1.0 | **.917** | .833 |
| MX-SEL | .889 | .833 |
| MX-F-w0.75 | .861 | .833 |
| MX-F-eps0.25 | .787 | **.954** |
| MX-F-eps0.1 | .815 | .935 |
| E02 | .583 | .833 |

MX-F-eps0.25 S9_2 has the best run-level rate, but CR3L passes only 5/9 runs and CR4L 8/9. Under the all-run toy criterion, RW-GL S9 wins.

**The failures are complementary.** MX-SEL and the fusions pass CR4 in 9/9 runs in both variants, which RW-GL never does. In turn they fail CR3L (0/9) and CR4L, which RW-GL S9 passes. This is the "retrospective union" situation of the decision table.

### FIN-D: independent high-fidelity re-evaluation

Every refined finalist is re-evaluated on an independent 1024 × 128 TRAIN cache (D, seed 211). An S1 escalation (5× budget) plus a search directly on D runs when the blind diagnostics fire ([results/finalist_reeval.csv](results/finalist_reeval.csv)).

- **The selected solution is also best on D:**
  - 100% of cells for E02, every RW finalist on S9, and the RW-GL recipes in both variants;
  - 92–100% for RW-ACT, X02 and the w-fusions;
  - only 55–61% for the ε-fusions.
- **The ε-constraint fusions are numerically fragile.** Their diagnostics escalated 61–64% of cells.
- **MX-SEL has no FIN-D re-evaluation.** It has no single objective to re-evaluate, so it is N/A by design (12 logged failures).
- **Escalated searches did not rescue any failing toy of the locked candidate.** CR4 still fails under S1 and on D.

### FIN-SENS: measure / mask sensitivity

Seed-31 caches, 3 search seeds ([results/finalist_sensitivity_recovery.csv](results/finalist_sensitivity_recovery.csv)):

| Candidate | Q01 MASK-COL | Q02 FULL | Q02 MASK-ALL | Q04 MASK-ALL | Q05 FULL | Q05 MASK-ALL |
|---|---|---|---|---|---|---|
| RW-GL-l0.75-a1.0 (S9 / S9_2) | 4/4, 4/4 | 7/8, 6/8 | **4/4, 4/4** | **4/4, 4/4** | 8/8, 6/8 | **4/4, 4/4** |
| E02 | 1/4, 2/4 | 6/8, 8/8 | 1/4, 2/4 | 0/4, 2/4 | 5/8, 8/8 | 0/4, 2/4 |

In W2 the bank estimands (MASK-ALL / MASK-COL) were weak in every module. The repaired recipe now recovers all bank libraries under every measure tested.

## W3-06: algorithm lock and confirmation

`freeze_algorithm.py --auto` applies the section 23 final rule to the FIN runs, ranking by:

1. all-run toy coverage on a single variant;
2. worst normalised margin min((.05 − gap)/.05, (thr − leak)/thr);
3. lexical id.

The ranking is written to [results/final_selection.csv](results/final_selection.csv).

**Locked:** `RW-GL-l0.75-a1.0`, variant **S9**, frozen decoder (AD) surface ([results/algorithm_lock.json](results/algorithm_lock.json), sha256 `cf40744f…`):

- **Objective:** L = 0.75 · POW(a = 1, c = .03) + 0.25 · B01.M03.c0.03. The first term is B08's row-energy-weighted M03 sparsity; the second is the unweighted global support term.
- **Measure and moments:** Q01 TRAIN measure, 512 subjects × 64 trials. Moments are taken on MASK-ALL for fixed banks and FULL for random X.
- **Integration:** three integration designs (31 / 73 / 101).
- **Search:** the W2 S0 budget (400/8 Haar starts for k = 3, 4; 2000/30 for k = 6; identity start; Givens spans 45 → .5°). The lowest TRAIN loss wins, and ties are broken by VALIDATION.
- **Hashes:** the lock records the hashes of the code, configs and dependencies. `freeze_algorithm.py --verify` reports "changed files: none".

The runner-up is the same recipe on S9_2 (10/12), followed by MX-SEL S9 and E02 S9_2 (10/12).

**Confirmation was not run.** [configs/confirmation_protocol.json](configs/confirmation_protocol.json) specifies CONF-A (≥ 24 fresh NP trainings on new data seeds), CONF-B (one-factor robustness) and CONF-C (unseen families; registry not designed). `run_confirmation.py` refuses to run without a verified lock and a confirmation manifest. One NP training takes GPU hours, so CONF-A is outside this wave's compute. **The lock is therefore a development selection, not a confirmed algorithm.**

## Decision table (03 section 28)

| Result in this wave | Row of the table | Next step |
|---|---|---|
| RW-GL repairs CR3 / CR5 in all 9 seeds and the bank estimands, but a new wrong R with lower loss appears in CR4 (and CR4L in S9_2) | *fixed old wrong R loses preference but new wrong R appears → mechanism not sufficient* | analyse the CR4 active-row / global-support trade-off (decompose RW-GL's CR4 failure as RW-A did for E02) |
| Fusions and MX-SEL reach 10/12; only a retrospective union of RW-GL S9 + E02 / fusion S9_2 covers 12/12 | *only retrospective union passes → no deployable algorithm yet* | no 12/12 claim |
| Fusion improves pilot coverage (12/12) but not 3 × 3 (10/12) | *fusion improves coverage in common frame* **not** confirmed | fusion stays a secondary lane |
| G / D / reduced-axis targets fail analytically and blind; full-bank contrast does not beat interior or same-grid HE | *all extended structure fails* (first-order / reduced-axis) | functional ANOVA, sparse basis / separability or MDL in a new wave (third order excluded) |
| RW-GL is analytic perfect 24/24 locally and practical 8/8 globally, inferred 23/24 | Gap 1 and Gap 2 progress | keep as theory + empirical candidate |
| Disjoint context improves Stage A strongly; the blind objective tracks the same-context ceiling | *disjoint context improves fidelity with information* | quantify the CR4 conditional target; the context model is not the limiting factor of RW-GL |
| Confirmation not run | – | CONF-A with the locked RW-GL S9 and E02 baseline, 24 NP trainings |

## Deviations from the specification and what was not run

- **Not run:**
  - **ST-F**, third-order / richer derivatives: excluded by instruction ("不要搞到三阶", i.e. no third order).
  - **The HEFUS conditional-HE fusion arm** of section 14: secondary, and the G / D pilots failed first.
  - **The 19.1 matched-truth vs conditional-mean bridge and synthetic noise interventions:** needs a controlled synthetic generator; this is an evaluation-only diagnostic.
  - **CONF-A / CONF-B / CONF-C:** compute; only the protocol and guard are implemented.
  - **Retraining encoders for W3-04:** the frozen encoder supported the diagnostic, as section 18 allows.
  - **ST-E at 3 × 3:** the reduced-axis pilot uses the capped RA budget (64 starts, ≤ 3000 evaluations) and is declared pilot-only (not escalated).
- **DC-R is partial.** Repeat blocks are 50 / 100 / 200 trials, with 2 blocks per seed. Blocks at 25 trials and "all" are impossible to make disjoint.
- **DC-N Stage B** was run for E02 and the lead finalist (RW-GL-l0.75-a1.0) only, at context seed 31; X02 is not included.
- **Escalation exclusions,** declared in `escalate.py` before ranking and never chosen by truth:
  - Configurations numerically identical to a retained control by construction are not advanced again: RW-POW-a1.0-c* = B08.M03.c*, RW-SM-a1.0-t* (weights exactly eᵢ), E01, B07c03, B01c03.
  - The ST-HE-* pooled-HE controls are controls, not structural candidates.
  - ST-E-RA is not escalated (see above).
  - A family whose best pilot is below the E02 baseline advances only its best configuration (rule text).
  - MX-SEL's support jobs MX-F-w0.25 / .5 / .75 were added so the selector has its pool at every integration seed. They are reported but are not eligible in the final selection.
- **Invalid objectives.** ST-B interior-row (`-INT`) objectives have no valid start on toys without a fixed bank. Those 192 runs are recorded as "invalid objective (no valid start)" with pass = False. An earlier merge counted their NaN pass flags as passes; this was fixed and the pilot file re-evaluated.
- **The fusion Gap 1 analysis is an analogue only.** Both components are evaluated on the same analytic surface, so it does not model the AD/RBF estimator pair.
- **Code edits during the run.** `wave3/jobs.py` (the `rbf_cache` fallback for RBF seeds 73 / 101) and `run_search.py` (MX-SEL at FIN seeds) were edited while some batches were running. The job files' `code_hash` therefore differs between batches. The edits add code paths and change no existing objective; the MX-A replays stay bit-identical to W2 (W3-00).
- **Temporary preview evaluations** were made while the finalists ran: `rwmx_*`, `mxb_*` and `finpreview_*` in `results/`. They are kept for traceability; `finalist_recovery.csv` supersedes them.
- **Evaluation helpers** (`wave3/eval_only/*`, `run_local.py`, `run_oracle.py`) open truth only after the corresponding lock file has been written.

### Bugs found and fixed during the run

All affected outputs were recomputed.

- `structure.py`: missing recipe keys for the GJ / GXJ targets; interior-row matrices accessed for toys without a bank.
- Reduced-axis: the non-antithetic ξ broke sign-flip invariance (5e-4). It now uses antithetic ±ξ (3e-10).
- ST-E local analytic arm: a CUDA / CPU device mismatch. The analytic arm now runs on CPU float64.
- Escalation: NaN pass flags of invalid objectives were counted as passes (see above).

## How to run

```bash
cd metric_lab/09_wave4_2_third_wave
PY=/d/Programs/anaconda_new/envs/py11/python
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
$PY -m pytest -q tests                 # 13 unit tests incl. the truth firewall
bash run_all.sh                        # full pipeline in execution order; every step is resumable
```

`run_all.sh` lists the individual steps:

1. `prepare.py` builds the gradient caches;
2. `run_batch.sh <batch>` runs the blind pilots (MX-A, RW-C, ST-B, MX-B, MX-SEL), with `run_reduced.py` and `run_context.py --encode` on the GPU;
3. `validate.py --refit` runs the W3-00 gate;
4. the evaluation-only steps: `run_repair_diag.py`, `run_local.py`, `run_evaluation.py --stage pilot --pareto`, `run_oracle.py`;
5. `escalate.py` writes the finalists;
6. `run_batch.sh FIN`, then `run_fin_extra.py` (FIN-D / FIN-SENS), then `run_evaluation.py --stage finalists`;
7. the W3-04 searches;
8. `analyze.py` and `figures.py`;
9. `freeze_algorithm.py --auto && --verify`.

Dependencies:

- The Lab 08 W2r1 outputs must exist: blind payload, moment caches and the RBF state.
- The Lab 07 tensors are needed for the evaluation-only local / oracle steps.
- The blind payload of Lab 08 is regenerated by `metric_lab/08_wave4_2_second_wave/prepare.py --toy <t>` if missing.

## Design

- **Imports.** Lab 08 (`wave2`) and Lab 07 (`wave4_2`, `run_global_oracle`, `evaluation`) are imported read-only, as the third-wave specification asks. No earlier lab or project definition file is modified.
- **Blind side** (`wave3/common, rowmetrics, objectives, fusion, structure, reduced_axis, context, jobs`, `run_search.py`, `run_reduced.py --search`, `run_context.py --encode/--search`):
  - reads only the truth-free payload, the frozen checkpoint and its own caches;
  - selects by TRAIN loss with VALIDATION tie-break;
  - is enforced by the audit-hook test.
- **Evaluation side** (`wave3/eval_only/`, `run_evaluation.py`, `run_local.py`, `run_oracle.py`, `run_repair_diag.py`, `run_fin_extra.py --reeval` diagnostics):
  - opens T / R* only after `*_locks.csv` hashes every job file of the stage.
- **Coordinates.** U = R V with V = (Z − c)/s; d raw / dU = diag(s) Rᵀ. Squared energies are never rotated. Integration points are transported by R.

## Outputs

| results/ | Content |
|---|---|
| w3_manifest.json, config_snapshot.json, validation_checks.csv, replay_audit.csv | W3-00 snapshot, hashes, 148 checks, W2 replay |
| repair_decomposition.csv.gz, repair_geodesic.csv.gz, repair_old_vs_ref.csv.gz | W3-01 RW-A: exact row decomposition, geodesic R* → R̂ profiles, old-vs-reference ordering per recipe |
| local_flags.csv.gz | Local Gap 1 / Gap 2 flags (RW-B, ST-A, ST-E) at Q01 pilot and fidelity C (seeds 31 / 73 / 101) |
| pilot_recovery.csv, pilot_locks.csv, pilot_found_rotations.npz, pilot_search_trace.csv.gz, pilot_coverage.csv, pilot_toy_pass.csv, coverage.csv | All pilot blind runs (RW-C, MX-A, MX-B, MX-SEL, ST-B, ST-E): locks, selected R, TEST recovery, categories, coverage |
| pareto_front.csv, pareto_locks.csv | W3-02 Pareto diagnostics |
| reduced_axis_summary.csv.gz, reduced_axis_equivalence.csv | W3-03 ST-E local sweeps and estimator checks |
| context_diagnostics.csv, context_recovery.csv, context_local.csv, context_cids.txt | W3-04 Stage A, Stage B, local Gap 2 per representation |
| oracle_summary.csv | W3-05 exact / practical Gap 1 |
| finalist_recovery.csv, finalist_coverage.csv, finalist_reeval.csv, finalist_sensitivity_recovery.csv | W3-05 finalists (FIN 3 × 3, FIN-D, FIN-SENS) |
| final_selection.csv, algorithm_lock.json | W3-06 section 23 ranking and the hash-signed lock |
| report_tables.md, figures/ | All tables and figures (see below) |
| rwmx_*, mxb_*, finpreview_* | Temporary preview evaluations (superseded) |

Figures are in [results/figures/](results/figures/):

1. `fig1_row_decomposition.png`: score vs weight terms per toy;
2. `fig2_repair_coverage.png`: repair-family coverage heatmap over a, c / λ;
3. `fig3_pareto.png`: E / X Pareto front, coloured by recovery post hoc;
4. `fig4_contrast_full_vs_interior.png`;
5. `fig5_local_flags.png`;
6. `fig6_context.png`: context size vs Stage A / Gap 2 / Gap 3;
7. `fig7_blind_heatmaps.png`: TEST correlation heatmaps of the locked candidate (RW-GL S9) and the E02 S9_2 baseline, every toy, failures included;
8. `fig8_oracle_global.png`: analytic loss vs distance vs recovery.

Large arrays live in `outputs/metric_lab/09_wave4_2_third_wave/W3r1/`:

- `blind/<batch>/<toy>/<cid>.npz`: every finalist rotation, loss and trace;
- `grad_cache/`: product-grid gradients and HE;
- `context/<toy>/reps.npz`: the re-encoded representations;
- `moment_cache/`, `local/`, `oracle_global/`, `reduced_axis/`, `logs/`.

A ≤ 500 MB copy of the important outputs, with its own README, is in `outputs/metric_lab/09_wave4_2_third_wave_key/`.
