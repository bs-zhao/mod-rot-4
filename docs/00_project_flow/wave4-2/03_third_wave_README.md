# Wave 4-2 / 03 — Third wave: repair the objective, extend structural information, diagnose inference, and confirm blind recovery

**Status:** proposed implementation protocol, 2026-11-27. This is a plan, not a result report.
**Scientific definitions:** [01 Project Purpose](../01_project_purpose.md), [02 Test Beds](../02_test_beds.md), [03 Simplicity-based Search Principle](../03_simplicity_search_principle.md), [04.1 HE](../04.1_hessian_energy_simplicity.md), [04.2 Rotation Sweep](../04.2_first_experiment_rotation_sweep.md).
**Roadmap:** [总览](总览.md), [01 First wave](01_first_wave_README.md), [02 Second wave](02_second_wave_README.md).
**Evidence:** [Lab 07](../../../metric_lab/07_unified_benchmark/README.md), [Lab 08](../../../metric_lab/08_wave4_2_second_wave/README.md), [Lab 08 report tables](../../../metric_lab/08_wave4_2_second_wave/results/report_tables.md), [candidate handoff](../../../metric_lab/08_wave4_2_second_wave/results/next_wave_candidates.csv).

---

# Part I. Rationale, scientific targets and priorities

## 1. What has been established and what has not

The ultimate purpose is to recover interpretable individual-difference axes from behavioral data, not merely to predict Y or to reproduce a particular Hessian matrix. Stage A learns Z. Stage B uses only X, Y, Z and optionally a frozen decoder d to select a transformation. Truth f, T and supervised R* are evaluation-only.

The formal 12 toys remain unchanged: C1/C2/C3/CR3/CR4/CR5 (Class B binary choice), and their L counterparts (Class C noisy continuous logit). Classic denotes the first three families, not Class C. Each has S9 (raw Z) and S9_2 (marginal standardization then orthogonal rotation). The latter is not full whitening.

### 1.1 Second-wave evidence

| Result | Evidence | Implication |
|---|---|---|
| E02 = L06 B08.M03.c0.03 on decoder AD, S9_2 | 10/12 toys, all 9 integration/search seed combinations; S9 7/12 | strongest fixed algorithm; two binary failures remain |
| E01 = B08.M03.c0.01 | S9_2 9/12; S9 6/12 | sensitivity control |
| B08 local inferred marginal | 24/24, analytic perfect 8/24, analytic marginal 12/24 under primary Q01 mask | local preference is not sufficient for blind recovery |
| E02 failures | CR3 and CR5 in both variants; lower inferred loss than R* | wrong preference demonstrated, not merely failure to optimize |
| X02 = L03 SL0.UXUU.e0.01 on block RBF D1 teacher surrogate | analytic perfect 24/24, joint local marginal 20/24; blind S9 8/12, S9_2 7/12, 3 search seeds | complementary but truth-referenced shortlist, weaker replication |
| X01 = same metric on decoder AD | 5/12 in each variant | change in estimator can matter with fixed metric |
| Practical analytic recovery | 51/93 priority metrics adequate for all 8 logit targets; 44 without inconclusive targets; 19 without tied or inconclusive targets in additional filtering | preserve exact vs practical Gap 1 distinction |
| Conditional HE | tested P1/P4 degraded performance; P2 approximately P0; no escalation | do not enlarge the same conditional grid as the main task |
| Inference | A1 CR marginal rate .003 AD, .03–.04 observed polynomial, .033 RBF teacher, .08 cross-fit PSD | conventional surrogate prediction improvements do not solve Gap 2 |
| Mask coverage | fixed bank MASK-ALL uses approximately 22% of points | exploit observed stimuli without requiring off-bank X derivatives |
| Search reliability | E02 216/216 selected finalists remain best on independent high-fidelity caches | reuse optimizer; no general global-optimality proof |
| Secondary NPs | classic repeat passes; CR4 repeat and CR3 low-beta fail | development result is not cross-training confirmation |

Sources: Lab 08 README and report tables; exact historical metric definitions and moment transforms are in Lab 03, 06, 07 and 08. The 51 practical metrics are not 51 proofs of global uniqueness: empirical support may assign I when no reproducible lower solution was found, and some objectives have ties or weak search coverage.

All 12 formal toys have already been inspected in development. The union of E02 (S9_2) and X02 (S9) covers 12/12 only by truth-assisted retrospective method choice. It is NOT a universal method. With both methods restricted to S9_2 the union covers 11/12, because CR3 is not recovered by either. The raw-coordinate X02 CR3 success is close to its leakage threshold. Do not confuse the union with a single blind selection rule.

### 1.2 New diagnostic hypothesis for B08

The post-W2 moment-cache audit compared aligned rows of E02's blind solution and R*. In both failed binary models the row losses increased but energy weights moved toward the lower-loss row. Approximate values (rounded; reproduce from exact cache before treating them as verified W3 evidence):

| Toy / matched axis | reference weight | blind weight | reference row score | blind row score |
|---|---:|---:|---:|---:|
| CR3 alpha | .037 | .071 | .962 | .963 |
| CR3 lambda | .506 | .620 | .751 | .766 |
| CR3 gamma | .066 | .097 | .912 | .928 |
| CR3 tau | .391 | .212 | .956 | .959 |
| CR5 lambda | .519 | .628 | .729 | .738 |
| CR5 gamma | .060 | .106 | .910 | .936 |
| CR5 tau | .421 | .267 | .956 | .960 |

CR3 loss approximately .849508 -> .836423; CR5 .83529 -> .81799 (seed31 illustration). The parameter labels are post-lock matching labels, not inputs. A change of row weights can reduce the total while every aligned row becomes less sparse. This is a concrete mechanism to test, not a proof of the only cause, and removing it does not ensure global correct recovery.

## 2. The seven modules and dependency structure

| Module | Purpose | Parent roadmap |
|---|---|---|
| W3-00 | reproduce and freeze baselines, add unit tests | common gate |
| W3-01 | B08 weighting / active-row repair and ablation | Steps 3, 4, 6 |
| W3-02 | metric × estimator factorial, same-frame fusion, blind decision rule | Steps 2, 6 |
| W3-03 | gradient, observed-row contrast, richer structure and reduced-axis loss | Steps 4, 5C |
| W3-04 | disjoint-context representation and conditional-target diagnostics | Steps 1C, 2 |
| W3-05 | unified analytic, inference and actual blind evaluation | Steps 4–6 |
| W3-06 | locked independent confirmation | Step 6C |

W3-00 first. W3-01 and W3-02 are the first cheap jobs; most operations reuse moments. W3-03 and W3-04 can proceed in parallel after the common gate. W3-05 starts as soon as a batch of frozen candidates is ready; there is no requirement to wait for the expensive context diagnostic. W3-06 is strictly downstream of an algorithm lock and must not be used to tune the algorithm. If no 12/12 development candidate emerges, still report the best fixed algorithm and all failures; confirmation of its limited performance is valuable and can be done with the same frozen rule.

Do not redesign formal test beds or modify previous labs. Do not optimize for CR3/CR5 alone, select a metric by toy identity, or change the success thresholds because a new candidate barely misses.

---

# Part II. Shared experimental contract and W3-00

## 3. Information, coordinates and splits

### 3.1 Allowed and forbidden information

Blind objective, estimator, hyperparameter selector and optimizer may receive only:
- Z, X, observed Y, subject IDs and splits;
- frozen decoder checkpoint and fitted input/output normalizers;
- declared observation type (Bernoulli vs continuous), known stimulus design / domain metadata and empirical X rows.

Never pass T, true f, true prior, parameter names, R*, true-parameter alignments, saved sweep labels/angles or oracle-dependent masks into the blind computation. Do not use family ID to choose metric, weights, number of features, or estimator. A known fixed-bank design can trigger a generic support-aware routine, but must also be evaluated using the same mathematical rule as random designs where feasible.

Development comparisons using T/f are allowed in a separate evaluator after hashing all candidate outputs. Label truth-assisted choice of a globally fixed candidate as development selection, never as a blind per-dataset selector.

### 3.2 Coordinate convention

Use column vectors. For variant V = Z or V = D_Z^{-1}(Z-mu_Z), let U=RV and g_R(u,x)=g(R^T u,x). Derivatives transform before squaring:

$$
\nabla_U g_R=R\nabla_Vg,\quad H_{UU}^R=RH_{VV}R^\top,\quad H_{UX}^R=RH_{VX}.
$$

Keep the physical evaluation points fixed and transport them by R. Never rotate an already squared energy matrix. Raw decoder scaling is composed exactly once. X stays in the original Lab 07/08 TRAIN-standardized coordinates. Class B primary output is class1-class0 logit; Class C return decoder output to original Y units. Probability channel is a separately labeled optional sensitivity, not another toy.

S9 and S9_2 have different feasible transformation sets. If one uses R_raw and another R_std, the relative map includes D_Z and is not generally orthogonal. No direct averaging of these R matrices or evaluation of a raw optimum as though it were an S9_2 optimum. In this wave both objectives in a fusion are evaluated in one common frame. Keep S9_2 primary, S9 sensitivity; optionally compare TRAIN-only marginal scaling as a distinct variant. The historical all-subject Z scaling is transductive and must be labeled as such; for strict future confirmation use a TRAIN-fit scaler, applied to validation/test unchanged.

### 3.3 Splits and candidate freeze

Use the existing subject-disjoint TRAIN, VALIDATION, TEST IDs for development. TRAIN constructs moments/fits, VALIDATION can select prediction hyperparameters and break numerical loss ties, TEST is evaluation only. No split by individual rows for model fitting. In W3-04 separately split each subject's trials into context/query before encoding.

Primary panel: all 12 designated NPs and both variants, separately report five secondary existing runs and their missing reference limitations. Use the Lab 07 toy registry and Lab 08 payload interface; not a new selection of favorable beta/repeat.

Register all planned candidate IDs, formulas, grids, seeds, search budgets, active-row policies and escalation rules in JSON before evaluating truth. A change is a new version, never overwrite original experiments. Maintain a common baseline E02 and X02 throughout. Each pilot uses seed31 and all 12 toys; smoke tests may use a subset only for implementation correctness. Final development uses integration seeds 31/73/101 and search seeds 31/73/101. All completed and failed cells remain visible.

### 3.4 Measures

Preserve Q01 (observed subject/trial sampling) and FULL/MASK-ALL for HE comparison. MASK-COL is separate; do not fill undefined derivatives with zero. Use Q02/Q05 and Q04 on banks as sensitivity, not as a per-toy choice. Primary medium C and independent high D designs follow Lab 08 (512x64 / 1024x128 for observed/random, 128x500 / 512x500 for full-bank where relevant). Same point samples and valid denominators across R.

New gradient and contrast targets use full observed rows, including valid zero amounts, because no X derivative is taken. For random designs use empirical complete X rows independent of sampled V as product design. Compare pooled HE on identical product support if attributing effects to a different mathematical metric. Include an interior-restricted contrast control to isolate coverage from functional change. Do not interpret model predictions on arbitrary unseen X as empirical evidence of off-support derivatives.

## 4. W3-00 validation and exact replay

Inputs:
- [Lab 08 README](../../../metric_lab/08_wave4_2_second_wave/README.md);
- [blind recovery](../../../metric_lab/08_wave4_2_second_wave/results/blind_recovery.csv), [reference losses](../../../metric_lab/08_wave4_2_second_wave/results/blind_reference_loss.csv), [blind candidates](../../../metric_lab/08_wave4_2_second_wave/configs/blind_candidates.json), [shortlist implementation](../../../metric_lab/08_wave4_2_second_wave/run_shortlist.py);
- [practical oracle](../../../metric_lab/08_wave4_2_second_wave/results/oracle_metric_practical.csv), [inference comparison](../../../metric_lab/08_wave4_2_second_wave/results/inference_estimator_comparison.csv);
- locally available W2 payloads, moment caches, fitted RBF state and checkpoints.

Freeze commit/blob hashes, local artifact hashes, fitted estimator state, RNG and numeric conventions. Large ignored outputs are not guaranteed to be in Git; if missing, reconstruct from original checkpoints or report blocked cells. Do not silently substitute other NPs.

Acceptance:
1. Reproduce E01/E02 BS-B success counts (15/24, 17/24) and E02 S9_2 10/12 from raw recovery rows. Reproduce X01/X02 pilot 5/12 and 8/12 S9, 5/12 and 7/12 S9_2; do not count 3 search seeds as 9 independent runs.
2. Recompute W2 B08 inferred 24/24, analytic perfect 8/24, marginal 12/24 and mask sensitivity; verify historical flags and identical points.
3. Recompute row-weight decomposition for all toys, both variants and available seeds; recover archived scalar loss to numerical tolerance. Use exact active row implementation, not rounded table. Record matched ordering only in evaluation.
4. Rebuild X02 teacher RBF from saved state or deterministic fit replay with source code and validation rule S-min. Verify prediction/derivative moments and selected losses against stored pilot. Persist state for every W3 surrogate, not only winners.
5. Verify analytic formulas on toy functions, chain rule, fixed-point transport, signed permutation invariance, invariant energies, zero/undefined controls, active threshold continuity/sensitivity, masks, finite differences and tensor vs direct derivatives. Run regression tests for W2 dimension-indexing and zero-UU issues.
6. Test truth firewall by denying access to datasets containing T and all S8/S9/eval-only artifacts during actual blind scoring, fitting and search. Hook must demonstrably fire; blind payloads must be minimal.
7. Test all new gradient/contrast moments on explicit functions and exact finite bank contrasts; test product mean/covariance reconstruction and reduced-axis resampling invariance under signed permutation.

Gate: report historical differences before proceeding. A missing historical artifact is a blocker for exact replay, not permission to fabricate a baseline.

---

# Part III. W3-01 — Repair B08's weighting mechanism

## 5. Mathematical problem

Let B be the UX energy matrix with p columns, r_i = sum_j B_ij, E = sum_i r_i, q_ij=B_ij/r_i on active rows, and

$$
s_i(c)=\frac1p\sum_{j=1}^{p}\frac{q_{ij}}{q_{ij}+c/p}.
$$

Historical E02 is

$$
L_1(R)=\sum_i w_i(R)s_i(R),\qquad w_i=r_i/E
$$

(with the historical active-row cutoff and renormalization). E is invariant under orthogonal rotation for a fixed support and complete column moments, but individual r_i and s_i are not.

For two rotations after an evaluation-only matching of rows, decompose the loss change symmetrically:

$$
\Delta L=\sum_i\bar w_i\Delta s_i+\sum_i\bar s_i\Delta w_i,\quad
\bar w_i=(w_i^A+w_i^B)/2,\quad \bar s_i=(s_i^A+s_i^B)/2.
$$

Treat absent rows using zero weight and a declared score convention; if active sets change, report separate support changes because row normalization is undefined on exact zero. The formula is exact for the chosen convention and matched finite scores. W2's examples suggest negative weight-change term can dominate positive row-score-change term. Use this as a falsifiable mechanism audit across all models, successful and unsuccessful, not as an argument to hardcode a parameter axis.

## 6. Candidate family R — weighting ablation

### 6.1 Power weights

$$
L_{a,c}(R)=\frac{\sum_{i\in A}r_i^a s_i(c)}{\sum_{i\in A}r_i^a},
\quad a\in\{0,.25,.5,.75,1\},\quad c\in\{.01,.03,.1\}.
$$

a=0 gives equal active-row weights, a=1 the B08 baseline. B07 and historical rowsp.a* already implement related ideas; compare exact historical recipes rather than claiming every grid point is new. Keep c=.03 as primary ablation and the other c as sensitivity. All 15 parameter pairs are registered before scoring.

A crucial issue: equal active-row weights can make tiny rows discontinuously influential and exact zero rows disappear. A pure row-normalized score can also be flat in classic functions where rotating a zero-energy bias axis into a slope does not change a row's support count. Therefore do not assume a=0 fixes identification.

### 6.2 Smooth activation and controlled weighting

Let e_i=r_i/E, and use a fixed rotation-invariant cutoff delta = t/k (t=.001,.01,.05). Define

$$
A_\delta(e)=\frac{e}{e+\delta},\quad
w_i^{(a,\delta)}=\frac{(e_i+\delta)^a A_\delta(e_i)}{\sum_j(e_j+\delta)^a A_\delta(e_j)}.
$$

Use row score zero for a truly zero row; its weight is zero. To avoid division by a nearly zero r_i use float64 and explicit zero handling, not a rotation-dependent arbitrary epsilon. At a=1 the numerator is exactly e_i (up to common normalization), a useful implementation check. At a=0 weak rows turn on smoothly, high-energy rows become nearly equal-weighted. Test a in {0,.5,1} with c=.03; delta grid is fixed, not chosen from known parameter roles.

Additionally compare a normalized uniform blend:

$$
w_i^\rho=\rho w_i^{(0,\delta)}+(1-\rho)e_i,\quad \rho\in\{.25,.5,.75\},\quad \delta=.01/k.
$$

No weights may be frozen using R*. An axis-specific externally assigned weight would break signed-permutation invariance and violate the project principle.

### 6.3 Add a global support constraint

Let v_ij=B_ij/E, n=kp, G_c=(1/n) sum v_ij/(v_ij+c/n). This is the B01 family. Test

$$
L_{\lambda,a}(R)=\lambda L_{a,.03}(R)+(1-\lambda)G_{.03}(R)
$$

with a={0,.5,1}, lambda={.25,.5,.75} and endpoints from preceding ablation. Both components lie in [0,1]; use raw bounded scales first, Haar-standardized version only as declared sensitivity. This tests whether global coupling count can prevent weight compensation without destroying B08's row-local support advantage.

### 6.4 Add an active-axis penalty, not just strength balancing

Define

$$
A_{\delta}^{UX}(R)=\frac1k\sum_i\frac{e_i}{e_i+\delta}.
$$

Test L_row + beta A_delta with beta={.1,.3,1} and delta=.01/k (or bounded convex blend with equivalently registered weights). This penalizes spreading nonzero stimulus-coupling energy across more axes. It is NOT a demand for equal row energies: equalizing row energies can itself destroy real parameter structure. The bias-only C1 axis is expected to be UX-inactive, so evaluate gradient and intercept components in W3-03. An overly aggressive active-axis penalty can collapse multiple real roles into a few axes; require full panel tests.

## 7. Experimental sequence and diagnostics

| Batch | Scope | Work |
|---|---|---|
| RW-A | all 24 primary variants, both surfaces where available | exact cache replay, compare old R/R*, weight/row decomposition, fine pair-angle scan |
| RW-B | all 24 variants; analytic and decoder arms | fixed power/smooth/fusion/active-axis grid, Q01 pilot then medium cache |
| RW-C | all registered primary recipes | true blind search on TRAIN medium, seed31; compare to E01/E02 and B07/B01 |
| RW-D | fixed shortlists from W3-05, all 12 toys | 3x3 seeds, independent high D reevaluation, Q02/Q05 and MASK-COL sensitivity |

RW-A uses truth-assisted alignment only to explain the observed mechanism. RW-C must start from Haar/identity and not reuse R* or the old successful R as the only starting point. Optional old blind R warm start is a labeled additional numerical control, not the primary comparison.

For analytic and inferred surfaces save every rotation's B, r, s, normalized weights, active set and each loss term where practical. Save exact loss decompositions for finalists. Use common random starts across metric ablations. Report whether:
- the original bad solution ceases to beat R* under the modified loss;
- a newly optimized solution actually improves MCC/leakage;
- the improved score only shifts the failure to another toy;
- analytic flatness/ties increase when energy weighting is removed;
- results remain robust near the row activation cutoff.

A successful repair must be evaluated by a new blind search, not just by comparing the two historical fixed R matrices. The full-panel target is improving both binary failures without losing the ten baseline successes. A tradeoff is still a valid negative/partial result. No post-hoc per-toy a or c.

---

# Part IV. W3-02 — Understand and combine complementary objectives

## 8. Factorial comparison before fusion

E02 and X02 differ simultaneously in Metric, Infer and coordinate variant. The next experiment must isolate those factors.

| Factor | Levels |
|---|---|
| Metric | B08.c.03, SL0.UXUU.e.01, SL0.UX.e.01, B01.c.03; plus fixed RW shortlist |
| Infer | frozen decoder AD; E-RBF block D1 teacher (S-min), historical X02 replay |
| Variant | S9, S9_2 |
| Measure | Q01 primary, identical masks and points |
| Budget | S0 equal starts/evaluations, first seed31, then 3x3 for finalists |

The essential 2x2 is B08 vs SL0.UXUU, AD vs RBF. SL0.UX is a UU ablation, B01 a global-UX control. Evaluate both response surfaces on the exact same TRAIN/VALIDATION point design; save both moment caches. Do not refit a surrogate at each R.

For each candidate retain prediction loss, derivative errors, local oracle/inferred agreement and actual blind recovery separately. RBF state includes landmarks, block scaling, coefficients, bandwidth, regularization, intercept, train/validation IDs and code hashes. Use the exact previous recipe and validation selector first; no new RBF grid selected by truth.

## 9. Same-frame fusion

Given two objectives in the same frame, L_E (AD B08 or repaired row metric) and L_X (RBF SL0), calibrate against a blind Haar reference sample of 512 rotations, independent of search starts. For each component define

$$
z_m(R)=\frac{L_m(R)-\operatorname{median}_{H}L_m}{\operatorname{IQR}_{H}L_m}.
$$

If IQR is effectively zero under a registered numeric tolerance, flag that component as invariant/uninformative; do not divide by a tiny number. A fusion with unavailable component is invalid, not silently a different method. Sensitivity: mean/SD or (q95-q05) normalization. No min-max fit using oracle sweeps, no normalization by the final best solution.

Primary fusion:

$$
F_\omega(R)=\omega z_E(R)+(1-\omega)z_X(R),\quad \omega\in\{.25,.5,.75\}.
$$

Also try one minimax objective:

$$
F_{\max}(R)=\max\{z_E(R),z_X(R)\}
$$

and a preregistered epsilon-constraint control: minimize z_X subject to z_E <= its blind minimum plus delta in Haar-IQR units (delta=.1,.25,.5). Implement feasibility explicitly or use a fixed penalty with constraint-violation reporting. Both single components are mandatory endpoints.

Compute in S9_2 and S9 independently. If X02's raw-space advantage is lost when restricted to S9_2, do not claim a same-frame fusion must preserve its historical union. An invertible affine search is a different project extension and not part of this mandatory experiment.

### 9.1 Pareto diagnostics

Use union of all refinements from component and fusion searches to construct nondominated solutions in the two blind loss coordinates. Save each R and both raw/normalized component losses, with seed/provenance. The Pareto front is not a deployed selector. Post-lock T-based coloring can illustrate tradeoffs but cannot choose the answer.

Examine whether the successful solution for one component is near the other component's minimum, whether both objectives agree on a subspace but disagree on a pair of axes, and whether a joint solution exists within feasible recovery bounds. Compare aligned axes with distance modulo signed permutation.

### 9.2 A strictly blind cross-objective selector (secondary)

Generate finalists from both fixed component searches and all three fixed fusion weights, using only TRAIN. On independently sampled VALIDATION points evaluate both normalized losses; choose the finalist minimizing their maximum validation z-score, ties by sum then deterministic start ID. Haar calibration is frozen using TRAIN; do not re-fit calibration to validation finalists. This is an explicit deployable rule, not truth-assisted selection. Test alongside direct minimax optimization, which may find candidates absent from the shortlist.

Optional uncertainty/abstention: flag disagreement if relative signed-permutation distance between component minimizers exceeds .10 or loss ranks are unstable on independent blind validation caches. Such a flag may be useful but is not evidence of which solution is correct. Report coverage of accepted and rejected separately; abstained cases are not successes in 12-toy coverage. Do not train a classifier on the 12 toy IDs or truth-derived labels.

## 10. Execution and gate

MX-A reproduce factorial and all single-metric/surface controls, all primary libraries. MX-B test fixed fusions and selector with seed31; MX-C replicate shortlist 3x3 and independent D, then secondary NPs. Report common-frame and original-coordinate historical evidence separately.

Desired result: one fixed fusion or selection rule covers all 12 toys with same coordinate convention. If no such rule works, identify conflicting objective rankings and proceed to richer structural information. A union selected after looking at MCC remains oracle-assisted and is not an algorithm.

---

# Part V. W3-03 — Beyond averaged Hessian energy

## 11. Motivation and shared principles

HE compresses a function into marginal squared second derivatives. It cannot directly detect pure linear/intercept effects; it loses derivative sign and cross-derivative information after marginalization. For fixed banks, mixed X derivatives of the unknown response are not identified from isolated observed stimulus rows without smoothness assumptions. The bank-contrast pilot in 02 was not run. This wave therefore tests first-order and observed-row structural information, rather than another large conditional-HE search.

All targets below define a single common R across the full design. No gain-only/loss-only grouping, known CPT categories or parameter-specific rules in the operational objective. The declared bank rows are legitimate inputs; labels are evaluation-only. Gradient/contrast derivatives are of the scalar logit/continuous mean; never differentiate sampled responses.

Let V and X be drawn independently from a declared empirical product design for primary new estimands. Let
$$
a(v,x)=\nabla_V g(v,x),\quad a_R=Ra,\quad
\bar a(v)=E_X[a(v,X)].
$$
Take expectation over V and X on fixed complete observed-row panels. Use AD on frozen decoder first; RBF surface is a fixed sensitivity. Gradient is a different functional, not another estimator of the HE matrix.

## 12. G — gradient structure

### 12.1 Total gradient energy and active axes

$$
C_G=E_{V,X}[aa^\top],\quad G_i(R)=(RC_GR^\top)_{ii}.
$$

Define gradient support/concentration penalties on normalized G_i, with the same Soft-L0 and fractional-power recipes (c=.01,.03 and power=.1,.5). Include gradient off-diagonal covariance cost
$$
D(C;R)=\|\operatorname{offdiag}(RCR^\top)\|_F^2/(\|C\|_F^2+\epsilon)
$$
where zero C is flagged uninformative. A single symmetric matrix can be diagonalized by PCA; this is only an active-subspace or joint-structure baseline, not a guaranteed axis identifier. Total trace is rotation-invariant and is a negative control.

### 12.2 Separate stimulus-average and stimulus-dependent gradients

$$
C_0=E_V[\bar a(V)\bar a(V)^\top],\quad
C_\Delta=E_{V,X}[(a-\bar a)(a-\bar a)^\top].
$$
For a common product design, C_G=C_0+C_delta. Unit test this identity using identical finite weights. C0 retains average-response/intercept information that contrast may remove; C_delta reflects stimulus-modulated effect. Neither is necessarily sparse in true axes. Use support penalties separately, and joint approximate diagonalization of both matrices (equal normalized matrix weights). Compare with total G alone.

### 12.3 Stimulus-resolved gradient matrices

For each observed x, estimate
$$
C_x=E_V[a(V,x)a(V,x)^\top].
$$
Compare two independent structural ideas:
- mean support of diag(R C_x R^T) after local trace normalization (conditional number of active axes);
- joint diagonalization: E_X[D(C_x;R)], weighted uniformly or by fixed trace energy.

The first resembles support, the second discourages cross-axis gradient co-variation. These are not equivalent and may conflict. In particular real interacting parameters can produce off-diagonal covariance even in correct coordinates. Retain pooled C_G control and report analytic failures rather than assume diagonal structure.

## 13. D — observed-stimulus contrast

### 13.1 Pairwise contrast derivative

For actual observed rows x,x':
$$
c_{x,x'}(v)=g(v,x)-g(v,x'),\quad
b_{x,x'}(v)=a(v,x)-a(v,x').
$$
Define C_pair=E_V[b b^T]. Then the pair-resolved latent coupling energy is
$$
B^{\Delta}_{i,\ell}(R)=(R C_{\text{pair }\ell}R^\top)_{ii}.
$$
This is a k x number-of-pairs nonnegative matrix; columns now mean observed contrasts, not raw X derivatives. Apply global and row-wise Soft-L0, fractional-power and column-wise support recipes. Include normalized pair weights in the registered definition; if using nonuniform pair sampling, report target weights and do not pretend it is uniform-pair expectation.

For independent uniform draws x,x', E_pair C_pair = 2 C_delta (population or with replacement empirical identity). Consequently pooling all pair energies and then scoring can lose pair-specific structure; retain pair columns or apply per-pair nonlinear scores, and test reconstruction against C_delta. Identical pairs are uninformative. Excluding identical rows changes the finite-sample normalization; verify the corresponding correction. Arbitrary pairs may mix all stimulus features and destroy the sparse pattern; this is why pair-design comparison is required.

### 13.2 Generic pairing designs

| ID | Rule | Purpose |
|---|---|---|
| D-U | uniform distinct observed row pairs, fixed RNG | broad design coverage |
| D-N | nearest-neighbor pairs in TRAIN-standardized X, graph edges sampled with fixed seed | small observed differences |
| D-F | far pairs, e.g. upper quartile of empirical distance, uniform sample | larger response contrast |
| D-A | dimension-focused observed edges: maximize fraction of squared distance in one X coordinate, conditional on nonzero distance and fixed quantile | approximate axis-specific contrast without interpolating |
| D-C | centered contrast a(v,x)-bar a(v), one column per x | no arbitrary pair ordering |

For D-A use the generic ratio delta_x_j^2 / sum_l delta_x_l^2 and choose top quantile (.9) for each coordinate on candidate graph; if too few edges, report support shortage. Do not require exact single-coordinate matches, which may be unavailable in a bank. Do not label D-A as an estimate of a partial derivative without additional assumptions. Pair selection uses complete X rows only, no response or truth. Freeze pair IDs before optimization; test sensitivity to pair seed separately from search seed.

Optional distance-normalized contrast b/(||x-x'||+delta) is a distinct target; use fixed TRAIN distance quantile floor (e.g. 10th percentile of nonzero distances), not a denominator that blows up for nearly duplicate stimuli. Report pair distance, support and normalization. Primary raw differences first. No off-bank interpolation is needed.

### 13.3 Retain the mean response and all axes

Contrast-only methods are blind to a parameter that affects every stimulus equally. Add C0 support as a separate component, not as an artificial fake X column with arbitrary scale. Test:
$$
L_D=\lambda L_{\text{pair}}+(1-\lambda)L_{\text{mean}},\quad
\lambda\in\{.25,.5,.75\}
$$
using bounded support scores, and compare with pair-only and mean-only. Include a UX HE fusion as secondary: .5 new contrast score + .5 historical normalized HE score on a common reference scale. For all-zero gradient/contrast blocks, use explicit undefined vs genuinely inactive conventions; a rotation-invariant zero block cannot identify directions.

## 14. J — full moment structure and richer derivatives (secondary)

Rather than only using diagonals of R J^(j) R^T, evaluate normalized joint diagonalization of the whole UX derivative covariance matrices J^(j), and compare with gradient C_x and contrast C_pair. This uses signed cross moments h_i h_j before squaring them in a diagonalization penalty; it is not a new estimate of HE. Test equal weighting and invariant-energy weighting. Include negative controls where matrices are proportional to I; any rotation should be indistinguishable.

Optional higher order if primary G/D fails: compute UUX and UXX derivative energies with AD or derivative-FD on interior support, first on small pilot; do not differentiate ReLU-like nonsmooth activations as though they were smooth. Use correct tensor transforms in each U index; account for symmetry. Third derivatives are more sensitive to numerical approximation and bank support than HE, so require convergence and analytic tests before interpreting. No broad higher-order grid is mandatory in this wave.

## 15. R — reduced-axis predictive structure (non-Hessian alternative)

The researcher expects that a simple parameterization may allow prediction for a stimulus or a stimulus subset with fewer active latent axes. To test this without knowing f or T, define an effect of excluding an axis from a response function.

For U=RV and index set S, replace omitted U coordinates using a fixed reference distribution, not by zero alone. Primary product reference uses independent standard Gaussian or empirical independent marginal reference in standardized base frame with its rotation-aware transport; a genuinely independent reference in U must be labeled a different R-dependent estimand unless base measure is isotropic. For a simple implementable first target use an isotropic Gaussian reference over V, so U has the same distribution for every R:
$$
g_{R,S}(u_S,x)=E_{U_{-S}\sim N(0,I)}[g(R^\top U,x)].
$$
Define
$$
D_S(x,R)=E_U[(g_R(U,x)-g_{R,S}(U_S,x))^2].
$$
For all axes included, D=0; for no axes, D equals stimulus-specific response variance. Approximate using common product draws and independent replacement draws; for singleton axis removal use paired replacement identity 1/2 E[(g(U,x)-g(U^{(i)},x))^2] under independence. This is a total-effect variance, not a unique main effect. A Gaussian query is an extrapolation risk: monitor support and compare empirical conditional/marginal resampling as separately declared estimands; do not claim a product decomposition for correlated empirical U.

For each x calculate normalized single-axis deletion effects; use Soft-L0 or a fractional-power support score. For k<=6, pilot all subsets to find the smallest cardinality with D_S/(Var_U g+epsilon) <= tau, tau in {.01,.05,.1}. If variance is zero the response has no latent effect at that x; record it separately. Aggregate across x and include a mean-response term. A hard minimal-count objective is nonsmooth; use fixed numerical budget and smooth support proxy as comparison, not falsely claim gradient optimization is appropriate.

For continuous and binary primary use squared logit/mean difference for consistency; optional probability/likelihood-based predictive discrepancy is a separate response target. An observed-data regression of subsets could be a later direct WY alternative but refitting for every R is expensive and changes estimator capacity. Do not treat subset prediction of held-out Y as proof of correct axes: multiple subsets can perform similarly, and a pure additive model with all axes active may offer no discrimination.

## 16. W3-03 execution, compute caps and tests

Primary methods (G total, G mean/centered, G joint, D-U/D-N/D-A/D-C, D + mean; HE controls) use pilot 128 V x 128 X random rows or 32 V x complete 500-row bank; 256 pairs. Evaluate all 12 toys and both variants, analytic arm and decoder arm. Cache gradients at all product points once; form contrast moments by subtraction without additional decoder calls. Medium 256 x 256 or 128 x 500 and 1024 pairs; high 512 x 512 or 512 x 500 where feasible. Save actual counts, weights and seeds. Different designs must use the same points for the matched comparison.

| Batch | Mandatory scope |
|---|---|
| ST-A | gradient/contrast unit tests on simple functions, all-panel analytic/local and inferred/local pilot |
| ST-B | actual blind searches of a frozen compact structural grid (global/row support c=.03, two gradient components, four pair rules, mean fusion; plus controls) |
| ST-C | paired full-bank vs interior-only contrast, HE on matched support; mask/measure sensitivity |
| ST-D | medium/high and 3x3 seeds for shortlisted candidates, exact/practical oracle search |
| ST-E | reduced-axis pilot: isotropic reference, 64 base x 16 replacement draws, 64 X rows (all toys); scale to 256x64 and full bank for finalists; use fixed common points, capped search |
| ST-F | optional joint moment/higher-order extensions if primary evidence warrants |

For ST-E approximate losses may be noisy and expensive. Record independent-cache reevaluation; if runtime exceeds declared cap use a lower fidelity and label inconclusive rather than reporting only favorable models. All metrics must be invariant to permutation/sign of U; zero/constant objectives fail discrimination. Test exact pair energy reconstruction, baseline mean contribution for f=a+b x, sign permutation and sampling stability. Use same generic rules for classic and CR; diagnostic labels after locking only.

Primary scientific comparison: does a new structural target offer better blind recovery than E02 across the whole panel, and/or better exact/practical analytic preference and inference fidelity? If contrast helps only on banks, determine whether it comes from additional support or different structure by ST-C. If nothing improves, retain the negative result and consider ANOVA/basis/MDL in next wave, not ad hoc CPT-specific features.

---

# Part VI. W3-04 — Disjoint-context and statistical identification diagnostic

## 17. Why the response target may be different from the true function

W2 found excellent prediction but poor curvature fidelity, even with RBF and cross-fitting. A learned V from finite trials is not necessarily an exact invertible map of T. For a genuinely independent query response, assuming its noise is independent given T,X and the context representation, the ideal conditional target is

$$
p(V,X)=E[\sigma(\eta(T,X))\mid V,X],\quad
g_B=\operatorname{logit}p,
$$
or
$$
g_C(V,X)=E[\eta(T,X)\mid V,X].
$$

Generally these do not equal eta at E[T|V,X]. In linear-in-T continuous models with affine posterior mean and no additional X information about T, a comparable functional form may survive. In nonlinear models and binary response, conditional averaging may distort it. This is a hypothesis, not a causal result of W2 Bridge. BO injects analytic structure and cannot prove it is identifiable from Z,X,Y.

Full-context S5 may include the target query response itself. A disjoint context/query design is needed before treating prediction as a conditional response of an independent trial.

## 18. Design: frozen encoder first, retraining only if necessary

Read actual S3/S5 encoder APIs and supported context size; preserve trained preprocessing, latent output convention (posterior mean vs sample) and test deterministic encoding. Do not assume arbitrary variable-size contexts are supported. If the encoder requires a fixed size, use the supported sizes/masking path or mark out-of-distribution runs; do not pad with observed query responses. New training is a separate optional intervention if frozen checkpoint cannot support the design.

For each subject partition trial IDs using nested randomized context sets of sizes {25,50,100,200} where feasible, plus largest available non-query context, with a fixed held-out query set of 100 trials (or 20% of n if n<500), and three context seeds {31,73,101}. Use disjointness assertions, same query IDs across nested sizes, no repeated query row from the same subject in its context. Fixed bank rows are shared across subjects but subject-trial responses remain disjoint. For comparison include original S5 full-context Z with explicitly overlapping query, and the largest disjoint representation. The latter changes Z and must not be conflated with official S5.

For fixed banks, random context subsets can have varying stimulus coverage; record empirical X coverage without conditioning on known CPT trial labels. Report query prediction at exact same observed rows for all sizes. Use TRAIN subject-only preprocessing/scalers for the new representations; apply to VAL/TEST. Keep official S5 comparison separately.

### 18.1 Nested and independent representations

Two complementary designs:
- DC-N: nested context sizes per subject, fixed query; measures dependence on context information.
- DC-R: two disjoint independent context blocks of equal size (e.g. 50 and 100, feasible by n), same query; measures repeatability of inferred Z, U, and decoder gradients. Align repeated Z estimates without T only if needed, using Procrustes on paired TRAIN subjects; report alignment and unaligned results separately.

Use same trained decoder for frozen-encoder arm. If context shift causes major prediction degradation, fit a separate D0 observed-query surrogate using TRAIN pairs (Z_context, X_query, Y_query), evaluate VAL/TEST. Do not include context responses in that query fit. Teacher D1 may remain useful but estimates teacher target, not necessarily independent conditional truth.

### 18.2 Stage A and Stage B comparisons

For every context size/seed/NP:
- Z mean, covariance, SD, covariance condition number, effective rank; repeatability across independent contexts;
- held-out prediction likelihood/MSE, calibration for binary, logit comparison only evaluation;
- TRAIN-fit T~Z linear regression evaluated TEST, per-parameter R², MCC reference computed on TRAIN then evaluated on TEST; distinguish from historical supervised all-subject reference;
- raw unrotated and supervised recovery, all new blind outputs assessed both against same-context reference and official S5 reference as sensitivity;
- AD and D0/RBF derivative energy errors and local curve Pearson/Spearman, strict and legacy minimum flags;
- E02, X02, fixed repaired/contrast objective(s), all-panel blind search first seed31, then 3x3 for finalist.

New representations may fall below Stage A adequacy; do not remove those cases or call their Stage B failures pure metric failure. Report joint and conditional coverage; low representation information itself is an informative diagnostic.

## 19. Controlled evaluation-only interventions to separate causes

### 19.1 Matched truth bridge versus conditional-mean bridge

For each context representation use TRAIN truth to construct BO/BA as in W1. This is evaluation-only. Compare same-point derivative landscape to frozen decoder and observed-query surrogate. BO answers whether geometry can host analytic structure; BA is a linear map fit, not a recovered generative law.

Optional simulation of known measurement error: standardized T, V_sigma = A T_s + sigma e (fixed orthogonal A, e isotropic independent noise), sigma in {0,.05,.1,.2,.4,.6}. Use TRAIN samples for conditional approximation, held-out TEST for analysis. Evaluate both direct oracle f(A^T V_sigma,X) plug-in and conditional response E[f(T,X)|V_sigma,X] (for binary average probability then logit) using known prior and noise model, e.g. posterior quadrature/importance weighting with independent convergence checks. Exact Gaussian posterior only for Gaussian priors/linear Gaussian observation; CPT marginals require supported sampling. Do not clip invalid parameters silently. If posterior effective sample size insufficient mark inconclusive.

This separates conditional-target distortion under known synthetic measurement error from decoder approximation. It uses f/T and is strictly diagnostic; no cache or estimated prior enters blind search. It is not an exact model of NP encoder error. Compare at sigma=0 as a numerical baseline and matched noise levels using reliability, not calibrated against desired rotations.

### 19.2 Interpretation and gates

| Observation | Supported interpretation | Not established |
|---|---|---|
| Larger independent context improves Stage A and loss agreement | consistent with representation uncertainty contributing | causal uniqueness or that training error absent |
| Z reliability improves but curvature remains wrong | response target / decoder structure or metric sensitivity may dominate | impossibility of all inference |
| Query D0 beats teacher on structural fidelity | conditional-target estimation may matter | automatic proof of true generative derivative recovery |
| Synthetic conditional average alone changes minima | statistical mechanism can cause distortion | exact decomposition of NP error |
| No usable small-context encoder path | current checkpoint cannot answer intervention | failure of mathematical hypothesis |

Mandatory DC-N and primary diagnostic over all 12 if feasible. DC-R and simulated-noise diagnostic are secondary but high scientific value; report any unavailable encoder support. No general NP hyperparameter sweep. If retraining is required, isolate an experimental checkpoint namespace, freeze original architecture/beta first and train on declared context/query objective with three seeds for a small predeclared matched B/C subset (C1/CR3/CR5 and L counterparts), then extend only by fixed rule. All official NP results remain unchanged.

---

# Part VII. W3-05 — Unified analytic, inference and blind evaluation

## 20. Exact and practical Gap 1

For each metric use true standardized T and declared measure on matching support. Compute analytic loss over legacy sweeps, fine angles near I, random skew directions and random-start global search. Report:
- Eval2 perfect / marginal flags with legacy and strict minimum-set rules; constant and undefined fail;
- exact global counterexample, tied non-permutation, empirical support (with basin coverage), inconclusive;
- d_SP normalized by sqrt(2k), paired independent cache loss difference, all finalists;
- ideal Q T_s and transferred Q R* V recovery with matching, MCC, min matched, leak mean/max.

Use W2's practical diagnostic criteria (ideal MCC>=.95, leak<.2; transfer historical Lab 05), not a new definition of exact Gap 1. Displaced minima may be practically adequate; flat/tied objectives cannot uniquely identify axes. A finite search with no counterexample is empirical support only. For fusion of different estimators define analytic analogue by applying both structural metrics to the analytic surface with separately calibrated Haar reference; no claim that this exactly models a surrogate estimator. If no well-defined analytic analogue exists, record NA and classify as empirical candidate, not Gap 1 pass.

## 21. Gap 2 inference fidelity

Compute pooled Pearson and Spearman, per-curve correlations, strict and legacy minimum-set preference, margins, false near-zero energy and normalized/unnormalized block errors. Analytic and inferred comparison requires same relative rotations, coordinate convention and paired support. New gradient/contrast targets use their own matrix/tensor errors. Do not compare unrelated target magnitudes. Report high correlation with wrong minima and low correlation with correct minima as separate outcomes.

For empirical-only candidates retain inferred-only local success independent of analytic failures. Do not promote marginal scans into actual search success. Save both signed and absolute correlations, undefined fractions, response fit and compute cost.

## 22. Gap 3 actual blind search

Reuse S0 optimizer from W2: k=3/4 400 Haar starts and 8 distinct refinements, k=6 2000/30, Givens spans 45 to .5 degrees, same cap and signed-permutation dedup. Record full budget and seed. For expensive new nonlinear scores (reduced-axis) declare capped pilot budget separately and do not compare lower-budget failures as if full S0. Escalate 5x only via blind convergence, disparate near-ties or instability, not because a toy failed truth criterion. Select lowest valid TRAIN objective, tie tolerance relative 1e-9 and blind VALIDATION tie-breaker. For a fused selector follow its declared rule.

Lock R, objective ID, estimator state, frame, code/data/config hashes before opening T or R*. Use independent high D cache to reevaluate finalists; do not replace a selected solution retrospectively unless that reevaluation is part of a predeclared blind procedure, in which case report both. A lower wrong solution than R* demonstrates objective preference issue but does not prove every good-recovery R is worse or that a finite search is globally exact.

The primary operational pass remains:
$$
MCC_{\rm blind}\ge MCC_{\rm supervised}-.05,\quad
\mathrm{Leak}_{max}<.2\ (Classic),\quad <.3\ (CR).
$$
Also report gap .02, common leakage .20/.30, min matched and all raw correlations. Use Hungarian matching on abs T-U correlations after lock. A genuine stable success requires every planned 3x3 seed combination on the fixed variant. Distinguish 12-toy coverage (one fixed variant) from 24-library coverage. Nine seeds are not nine new NP trainings.

## 23. Candidate escalation and development decision rules

To bound cost, use the following preregistered rule AFTER completing all pilot blind solutions and locking them; truth is allowed only in this research comparison:
- retain E02 and X02 controls always;
- each family (RW, MX, G/D, reduced-axis) advances at most three fixed configurations ranked by number of toys passing on S9_2 at pilot, then worst MCC-gap and worst leak excess, then lexical ID; for S9 sensitivity advance its best fixed config separately, never mix best variant by toy;
- additionally retain at most two theory candidates per structural family with all analytic local passes and no observed serious practical global failure, ranked by joint strict/legacy pass and worst margin; clearly label theory selection;
- all chosen candidates rerun on all 12 toys, both variants, all 3x3 seeds, independent high D and relevant mask/measure sensitivity. Keep full pilot results. If pilots fail to meet baseline, still advance the best mechanistically informative configuration per family (within above cap).
- finalize fixed algorithm by all-seed coverage on a single variant, then worst normalized recovery margin, then smaller compute/complexity and lexical ID. This is development selection using T and must be recorded. A deployed algorithm contains no T and no family-specific tuning.

Report full Pareto/metric/estimator comparisons, per-parameter heatmaps (all failures), weight decomposition and old/new loss ordering at R*, seed stability, exact vs practical oracle and support coverage. Keep the two research routes distinct: a theoretical candidate with good Gap 1/2 and an empirical method with superior Stage B can both advance. Do not assert one scalar ranking proves all gaps.

---

# Part VIII. W3-06 — Independent confirmation

## 24. Freeze before accessing new evaluation truth

All historical toys and repetitions are development. A 12/12 result on them is a development milestone only. Before independent runs save a signed/hashable algorithm-lock manifest containing:
- one primary metric/estimator/measure/variant and exact hyperparameters or an explicit blind selection rule;
- observation likelihood rule, preprocessing fit on TRAIN, integration and search budgets, seed generation rule, tie handling and validity policy;
- planned test panels, data generation and training seed IDs, Stage A report and recovery thresholds, no-per-toy selection, failure/abstention policy;
- code version and all dependency hashes.

Do not open confirmation truth until all corresponding blind outputs are locked. If methods change after results, register a new development version and obtain a new confirmation panel. No reuse of confirmation set for repeated hyperparameter tuning.

### 24.1 Fresh-data panel (CONF-A)

All six existing formula families x both observation classes; at least two fresh independent generation seeds and two independently trained NPs per generation seed (48 NP runs if full grid). Use original documented generator/prior/noise and fixed-bank design in the first confirmation arm, preserve subject counts and split protocol. Generate new subjects, responses and random X where design permits. For fixed banks same bank is allowed for replication; alternate bank is a distinct CONF-B arm. Train using frozen architecture and beta selection rule defined before T evaluation; if hyperparameters selected by validation prediction/latent quality only, record that rule. No choosing a good seed based on T.

Resource-limited minimum first pass: one new data seed x two training seeds across all 12 toys (24 runs), then extend second data seed if possible. Report exact denominator and all missing runs. This is a proposed compute budget, not a preexisting result. Fresh data generation seeds must be verified unused in old runs; use new fixed IDs recorded in manifest.

### 24.2 Design/observation robustness (CONF-B)

Separate interventions on fresh data with fixed method:
- new generic bank or randomized bank subsampling of the same family, not hand-picked trial types to help the metric;
- context length or number of trials per subject (e.g. 100/250/500), fixed sample count and NP training rule;
- higher/lower continuous noise, binary sensitivity/response noise under declared generating settings;
- independent NP training seeds.

Change one factor at a time before any factorial expansion. Report all planned cells and Stage A failures, not just cases where an adequate supervised ceiling exists. Do not pool different observation mechanisms into one average masking failure.

### 24.3 Unseen structural families (CONF-C)

Needed for a broader scientific claim beyond these six development functions. Design and document new toy families in a separate preregistered registry before evaluation. At least four structures (both B/C where sensible): additive nonlinear stimulus basis with bias, sparse parameter-stimulus interactions, multiplicative/gated nonlinear parameter terms, mixed nonlinearity with dimension 6–8. These are design categories, not finalized generating equations. Choose generative equations and independent priors using a blinded test-bed design process, verify likelihood identifiability and researcher-assessed canonical parameterization as required by 03 section 3; do not screen families using the candidate metric's performance. Include deliberately rotationally symmetric/non-identifiable controls (e.g. dependence only on ||T||) labeled negative controls, not eligible positive test beds. Stage A adequacy and actual positive-case definition must be frozen before assessing the rotation.

To avoid circularity, structural identifiability (likelihood distinctness, independent parameter axes, support) should be checked separately from whether a specific simplicity metric prefers the axes. A known algebraically equivalent alternative may be a genuine ambiguity, not an algorithm failure to fix by tuning. Exact test-bed formulas require separate review; do not claim this protocol already defines validated unseen models.

## 25. Confirmation outcomes

Report number of datasets, NP trainings and algorithm runs separately; hierarchical aggregation by family and generation seed, not pseudo-replicating search seeds. Provide all-seed and per-run pass, exact denominator, worst model, matched correlations, leakage, Stage A failures, prediction accuracy and cost. A common bootstrap over independent datasets/NPs may be used descriptively if enough replicates; avoid treating all subjects as independent algorithm replications. Include E02 locked baseline even if the chosen candidate performs worse. No dropping a difficult family or changing thresholds.

A strong next milestone is stable recovery on the full 12 development panel plus confirmation on fresh data/NPs. Success on new structural families is needed to support broader generalization. A failure is evidence about scope, not a reason to overwrite confirmation output.

---

# Part IX. Implementation and deliverables

## 26. New isolated lab

Proposed new folder (Lab 09 is currently unused in the checked directory):

    metric_lab/09_wave4_2_third_wave/
      README.md
      configs/
        primary_registry.json
        repair_candidates.json
        fusion_candidates.json
        structure_candidates.json
        context_protocol.json
        budgets.json
        confirmation_protocol.json
      wave3/
        blind_inputs.py
        adapters.py
        row_repair.py
        fusion.py
        gradients.py
        contrasts.py
        reduced_axis.py
        context.py
        moments.py
        objectives.py
        optimizers.py
        lock.py
        eval_only/
          oracle.py
          recovery.py
          bridge.py
          noise_diagnostic.py
        reporting.py
      tests/
      prepare.py
      validate.py
      run_repair.py
      run_fusion.py
      run_structure.py
      run_context.py
      run_evaluation.py
      freeze_algorithm.py
      run_confirmation.py
      analyze.py
      figures.py
      run_all.sh
      results/

Reuse Lab 08's validated loaders, moment and search adapters read-only. Separate truth module imports and files. New large arrays under outputs/metric_lab/09_wave4_2_third_wave/<run_id>/; immutable versioned manifests, --resume, no overwrite. Code must support Windows py11 and portable path handling; normalize CRLF in job lists. CPU float64 reference, chunked gradient/derivative arrays, bounded parallelism and optional GPU only after equivalence tests. Save actual fit states for ALL new surrogate runs and final/alternate R matrices, not just reported metrics.

Illustrative future CLI (not implemented):

    python metric_lab/09_wave4_2_third_wave/prepare.py --toys all
    python metric_lab/09_wave4_2_third_wave/validate.py
    python metric_lab/09_wave4_2_third_wave/run_repair.py --batch RW-A --toys all
    python metric_lab/09_wave4_2_third_wave/run_repair.py --batch RW-B --toys all
    python metric_lab/09_wave4_2_third_wave/run_repair.py --batch RW-C --toys all
    python metric_lab/09_wave4_2_third_wave/run_fusion.py --batch MX-A --toys all
    python metric_lab/09_wave4_2_third_wave/run_fusion.py --batch MX-B --toys all
    python metric_lab/09_wave4_2_third_wave/run_structure.py --batch ST-A --toys all
    python metric_lab/09_wave4_2_third_wave/run_structure.py --batch ST-B --toys all
    python metric_lab/09_wave4_2_third_wave/run_context.py --batch DC-N --toys all
    python metric_lab/09_wave4_2_third_wave/run_evaluation.py --panel primary --stage pilot
    python metric_lab/09_wave4_2_third_wave/run_evaluation.py --panel primary --stage finalists
    python metric_lab/09_wave4_2_third_wave/analyze.py
    python metric_lab/09_wave4_2_third_wave/figures.py
    python metric_lab/09_wave4_2_third_wave/freeze_algorithm.py --candidate <registered-id>
    python metric_lab/09_wave4_2_third_wave/run_confirmation.py --batch CONF-A --lock <manifest>

Support --toy, --variant, --candidate, --measure, --mask, --seed, --fidelity, --run-id, --resume. Independent confirmation commands require a verified lock; new-model CONF-C requires its separate approved registry. Do not blindly run a command against code that has not yet been implemented.

## 27. Output contract

| File | Required content |
|---|---|
| w3_manifest.json, config_snapshot.json, validation_checks.csv | code/data versions, frozen rules, tests, known missing artifacts |
| replay_audit.csv | W2 exact reproduction, differences and reasons |
| repair_decomposition.csv.gz | row loss, row weights, active masks, exact decomposition, reference and found loss |
| repair_grid.csv.gz | all registered configurations, both arms, support and flags |
| factorial_comparison.csv | Metric x Infer x Variant with fit, structural and search evidence |
| fusion_calibration.npz, fusion_finalists.csv.gz, pareto_front.csv | Haar scale, both component losses, all finalists, selection rule |
| gradient_moments.npz / contrast_moments.npz | base moments, observed row/pair IDs, support, weights, normalization |
| contrast_design_audit.csv | distance distributions, coordinate-focused coverage, full vs interior bank |
| reduced_axis_summary.csv.gz | deletion/subset effects, variance, sample uncertainty, cost |
| context_split_manifest.json / context_representation.npz | nested context/query IDs, preprocessing, Z, no truth in blind store |
| context_diagnostics.csv / conditional_target_diagnostics.csv | separate truth-enabled evaluation, reliability, Stage A, bridge/noise comparisons |
| fit_states/ and fit_diagnostics.csv | full estimator state, train/validation data provenance, fit selection |
| analytic_solutions.npz / oracle_summary.csv | exact vs practical global findings, weak coverage, ties, independent reevaluation |
| inference_pass_summary.csv.gz | per-candidate local curve agreement, masks, error metrics |
| blind_locks.csv / found_rotations.npz / blind_search_trace.csv.gz | locked outputs, all finalists, seed, convergence, budget |
| blind_recovery.csv / recovery_correlations.npz | TEST correlations, matching, MCC/min/leak, ceiling differences, pass flags |
| coverage.csv / next_wave_candidates.csv | all planned cells, complete/invalid/missing, candidate evidence and lanes |
| algorithm_lock.json / confirmation_manifest.json | frozen deployment rule and independent panel |
| confirmation_recovery.csv / confirmation_coverage.csv | all independent runs, no post-hoc filtering |

Required figures: (1) row weight/score decomposition and contribution changes for all failures and representative successes; (2) coverage heatmap over a,c and repair families; (3) same-frame factorial and Pareto plots (truth coloring evaluation only); (4) full-bank vs interior contrast comparison and stimulus-pair support; (5) gradient/contrast/reduced-axis analytic/inferred curves; (6) context size vs Stage A/Gap2/Gap3, paired reliability; (7) all 12 blind correlation heatmaps vs unrotated/reference, all failed cells; (8) exact/practical global loss-distance-recovery; (9) development vs independent confirmation matrix with dataset/NP counts.

Every plot generated from saved numeric outputs. Results README must be generated or cross-checked by analyzer, list deviations and skipped jobs, and distinguish candidate selection from independent confirmation.

## 28. Final decision table

| Result | Interpretation | Next step |
|---|---|---|
| row-weight repair restores CR3/CR5 without losing other models | observed compensation mechanism actionable | lock candidate, replicate and confirm |
| fixed old wrong R loses preference but new wrong R appears | mechanism not sufficient | analyze active-row/global support tradeoffs, extended structure |
| fusion improves coverage in common frame | complementary information exploitable | full-panel and fresh-data test |
| only retrospective union passes | no deployable algorithm yet | no 12/12 claim, use structural diagnostics |
| full-bank contrast beats same-support HE and interior contrast | evidence of useful new target and/or coverage | separate effects by matched comparisons |
| new metric analytically good, inferred bad | Gap 1 progress, Gap 2 remains | context/conditional diagnostics and estimator work |
| new inferred-only metric recovers axes | empirical candidate | independent confirmation; do not claim fidelity |
| disjoint context improves fidelity with information | representation/conditional-target mechanism supported | quantify and refine context model |
| all extended structure fails | current loss family insufficient | functional ANOVA, sparse basis/separability, MDL in a new wave |
| development passes, fresh data or NP fails | development overfit/limited robustness | report failure; revise only under new version |
| fresh family nonidentifiable | test-bed ambiguity | report separately, do not tune criterion to it |

## 29. Acceptance checklist

- [ ] Historical replay and truth firewall pass, or blockers explicitly identified.
- [ ] W3-01 compares all fixed repair recipes on all 12, with exact loss decomposition and new blind searches.
- [ ] W3-02 isolates Metric/Infer/frame and tests same-frame fusion and a truth-free selector, not per-toy selection.
- [ ] W3-03 runs gradient and observed-row contrast with full-support/matched-interior controls, reduced-axis pilot or documented compute blocker.
- [ ] W3-04 runs disjoint-context with saved trial IDs and same-context supervised references, or documents actual encoder limitation; diagnosis not confused with algorithm.
- [ ] W3-05 reports exact/practical Gap 1, two forms of Gap 2, actual Gap 3, every seed, every failed/missing cell, frozen selection.
- [ ] W3-06 lock precedes independent evaluation; fresh dataset/NP results distinguished from new family validation; incomplete confirmation labeled.
- [ ] All previous labs and project definition documents unchanged; full estimator states, rotations, reproducible figures, and source hashes saved.

Priority under limited compute: W3-00 -> RW-A/B/C and MX-A/B -> ST-A/B/C, while DC-N proceeds in parallel if encoder support permits -> W3-05 finalists -> CONF-A. Reduced-axis, higher order, controlled synthetic conditional average and novel-family confirmation may require more compute or separate protocol review. Do not skip primary failures in favor of attractive optional experiments.

The success criterion for the research wave is a reliable answer to the scientific questions, not an obligation to manufacture a universal pass. The algorithmic milestone remains a single fixed blind method meeting the current criterion in every formal test bed, followed by independent validation. Exact analytic identification and faithful inference are separate scientific claims.

**Implementation status:** documentation only. Proposed grids, thresholds for new sensitivity analyses, code paths, experiment budgets and confirmation designs require implementation and verification. No new results or edits to earlier project definitions are implied.
