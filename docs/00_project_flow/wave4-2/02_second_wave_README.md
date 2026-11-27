# Wave 4-2 / 02 — Second-wave experiments: blind recovery, practical oracle accuracy, conditional HE and improved inference

**Status:** detailed implementation protocol, proposed after Lab 07 (2026-11-26). This is not a results report.  
**Parent roadmap:** [总览](总览.md).  
**Previous wave:** [01 First-wave protocol](01_first_wave_README.md), implemented in [Metric Lab 07](../../../metric_lab/07_unified_benchmark/README.md).  
**Fixed project definitions:** [01 Purpose](../01_project_purpose.md), [02 Test beds](../02_test_beds.md), [03 Three gaps](../03_simplicity_search_principle.md), [04.1 HE](../04.1_hessian_energy_simplicity.md), [04.2 Evaluations](../04.2_first_experiment_rotation_sweep.md), [04.3 First-wave registry](../04.3_first_wave_metric_inference_candidates.md).

---

# Part I. What the first wave changed

## 1. Evidence and revised priorities

Use Lab 07's [README](../../../metric_lab/07_unified_benchmark/README.md), [metric evidence](../../../metric_lab/07_unified_benchmark/results/metric_evidence.csv), [lane rules](../../../metric_lab/07_unified_benchmark/results/lane_rules.json), [next-wave candidates](../../../metric_lab/07_unified_benchmark/results/next_wave_candidates.csv), [global oracle summary](../../../metric_lab/07_unified_benchmark/results/global_oracle_summary.csv), and [report tables](../../../metric_lab/07_unified_benchmark/results/report_tables.md) as the inputs to this plan. Freeze their versions and the Lab 07 registry/code versions in the new manifest.

The first wave completed the 12 formal toys and 24 primary rotation libraries, using 415 historical metric entries. Its main findings are:

1. **Empirical candidate:** L06|B08.M03.c0.01 and L06|B08.M03.c0.03 have inferred-only marginal passes on all 24 primary libraries for Q01, Y-L/Y-C, FULL/MASK-ALL. Each has only 8/24 analytic perfect and 12/24 analytic marginal passes. The 12 classic analytic libraries have constant curves in the failed sweeps. Under MASK-COL, the B-CR inferred marginal count falls from 6/6 to 4/6 for both. The current metric_evidence table records seed-status changes for these candidates (1 and 2 cells respectively at fidelity C, across its configuration panel). Thus their main result is a Q01 development pass, not an unconditional claim that every mask, measure and seed succeeds. Reproduce the exact configurations before blind search. Do not copy numbers for unverified per-seed or strict passes from narrative discussions.
2. **Theoretical candidates:** 53 and 58 metrics respectively pass every analytic Q01 library under MASK-ALL and MASK-COL; 59 distinct metric IDs across the union. No joint all-class pass. A1 contains 33 development candidates satisfying its frozen rule, which allows inconclusive global checks. They are not 33 proven global minima. A2 contains 211 metric × class/family-scope entries (not necessarily 211 distinct metrics).
3. **Bridge evidence:** B-orth retains 99–100% of the CR analytic marginal passes in the logit/continuous channel; decoder retains 6–22%. B-aff is less reliable, especially CR4 and CR4L S9_2. B-orth installs known analytic structure with supervised alignment, so it shows that the current coordinate space can host a useful surface, not that the structure is identifiable without T. Representation error, conditional-response averaging and decoder approximation remain plausible contributors.
4. **Curvature problem:** the decoder's response predictions can have correlations as high as .999 with true logit while UU and fine-grained UX structures are inaccurate. A degree-2/3 observed-data polynomial surrogate also failed to retain almost all CR analytic passes. Do not merely repeat that low-degree regression or FD step tuning. Test derivative-aware smoothing, richer smooth representations and uncertainty estimation.
5. **Gap 1 problem:** among 93 priority metrics, CR logit targets show many reproducibly lower analytic losses away from signed permutations. Across the five CR logit family/mask targets there are 293/465 counterexamples beyond normalized distance .001, 142/465 beyond .05, 71/465 beyond .15. These are metric × target counts; analytic L targets may be shared by B/C pairs. A small matrix distance is not necessarily a failure of scientific parameter recovery. Quantify actual parameter correlations rather than treating .05 as an accepted recovery threshold.
6. **Measurement problem:** MASK-ALL for fixed banks often covers only about 22% of points. Fixed-bank data do not identify arbitrary off-bank X derivatives. A conditional or contrast target may better use known trial designs. Probability Y-P is a different estimand, with severe analytic/global failures in CR; it is not a primary target for this wave.

The first wave did not conduct the new B08 blind search, did not run the optional disjoint-context encoder diagnostic, and did not establish global uniqueness. All current toys and outcomes have been examined during method development; later confirmation must be on fresh data or generating models.

## 2. Research questions and execution order

| Module | Question | Primary output | Roadmap link |
|---|---|---|---|
| W2-00 | Are the selected evidence, metric implementations and interfaces reproducible? | frozen registry, reproduction audit | shared gate |
| W2-01 | Can B08 and fixed controls recover parameters by actual R*-free search? | found R, T–U recovery, search stability | early Step 6 / Gap 3 |
| W2-02 | Are lower Oracle minima scientifically consequential? | exact/global vs approximate recovery, local perturbation diagnostics | Step 1D extension / Gap 1 |
| W2-03 | Does conditional aggregation reveal structure lost by pooled HE? | new analytic and inferred losses, global diagnostics | Step 3 / Gaps 1–2 |
| W2-04 | Can inference of fixed metrics be improved without fitting to truth? | estimator comparison, curvature and support diagnostics | Step 2 / Gap 2 |
| W2-05 | Which complete method configurations should advance? | consolidated evidence and next-wave registry | synthesis |

W2-00 precedes the other modules. W2-01 and W2-02 are **mandatory first-priority jobs**, can start together, and should report even if the longer W2-03/04 computations are incomplete. W2-03's oracle arm and W2-04's estimator fitting may proceed in parallel. W2-03 inference can use the frozen decoder immediately and later the best W2-04 estimator. Do not make a universal-success metric a prerequisite for any module. No NP retraining in the mandatory scope.

The goal is not to try every possible combination. The primary grid is small and predefined; escalation rules are explicit. A failed experiment is retained as an informative result.

---

# Part II. Common protocol and W2-00

## 3. Panel, coordinates and allowed information

### 3.1 Primary panel

Use the same designated NP runs as Lab 07's [toy registry](../../../metric_lab/07_unified_benchmark/configs/toy_registry.json), with the following 12 separate test beds.

| Class B (Y-L primary) | Class C (Y-C primary) | k | p |
|---|---|---:|---:|
| C1 | C1L | 4 | 3 |
| C2 | C2L | 4 | 4 |
| C3 | C3L | 6 | 4 |
| CR3 | CR3L | 4 | 6 |
| CR4 | CR4L | 3 | 6 |
| CR5 | CR5L | 3 | 6 |

Each toy has two coordinate variants: raw Z (S9), and marginally standardized Z (S9_2). There are 24 primary library-variants, not 24 different toys. For W2-01 a variant means only preprocessing and latent dimension; do not load its saved supervised sweep. Include all available secondary NP runs as a separate robustness panel after primary results are complete. A Class B probability Y-P control may be run on the same settings, but never mixed into Y-L/Y-C coverage.

For Class B, response surfaces are decoder class1−class0 logit (Y-L) or its sigmoid (Y-P). For Class C, return the scalar decoder output to original continuous Y units using the saved TRAIN scaler (Y-C). Never differentiate sampled Y.

Let $V=Z$ or $V=D_Z^{-1}(Z-\mu_Z)$, $U=RV$, $R^\top R=I$. S9_2 is marginal standardization, not full whitening. Blind standardization can be reconstructed solely from Z and checked against saved mean/SD; never read R* to obtain it. For $g$ a fitted surface in V and standardized X coordinates,

$$
g_R(u,x)=g(R^\top u,x),\quad
H^R_{UU}=R H_{VV}R^\top,\quad H^R_{Ux}=R H_{Vx}.
$$

Use fixed integration points in the base frame and transport them by R. The X coordinate system is fixed; no per-rotation X scaling. Reuse the Lab 07 float64 chain-rule and moment contraction tests. For raw Z derivatives include the latent Jacobian once, not twice. Never rotate already squared energies.

### 3.2 Information firewall and data splits

Actual search/estimation inputs: Z, X, Y, frozen decoder d if desired, training scalers and observed or declared X-design metadata. Do not use true f, T, true priors, R*, sweep-angle labels, known CPT parameter roles, or synthetic family ID to choose directions, thresholds, estimator hyperparameters or search winners. Toy IDs may be used only by the scheduler and evaluator. Known observation class is legitimate because it determines the likelihood. The declared derivative-domain policy is also legitimate if it would be known to an analyst.

W2-01 blind module must accept a minimal blind-only payload. The generic loader may orchestrate inputs but must not pass an EvalOnly object, supervised artifact path, or precomputed T-dependent mask into the objective or optimizer. Test that the objective runs with truth files and S8/S9 reference files unavailable. The diagnostic evaluator opens them only after a blind solution is locked.

Two modes are required:
- **Historical development replay:** exact Lab 07 Q01 measure and TEST pool, seed 31 and original scalers/NP, to check that the starting objective matches the previously screened inferred landscape. Since this pool has already been examined, recovery assessed there is development evidence, not independent confirmation. This is a transductive application of a blind loss.
- **Split-safe development:** use TRAIN subjects to construct objective moments and fit new surfaces, VALIDATION subjects for T-free hyperparameter selection and independent loss stability; lock the found R, then evaluate correlations on TEST. Reuse prescribed subject splits; do not randomly split individual trials across train/test. This split-safe protocol still uses the same development toy datasets and is not new-model confirmation.

All hyperparameter grids and algorithm-selection rules are fixed before evaluation. Truth may be used to compare research candidates afterward but cannot become a per-dataset selection rule. A single common metric/measure/optimizer should be evaluated across all toys; a class-specific likelihood is not a model-specific shortcut.

### 3.3 Measures and masks

Reproduce Q01, AD, FULL (random designs) / MASK-ALL (declared derivative-valid all-interior rows of a fixed bank) as the primary development configuration. Retain MASK-COL as a separately reported estimand for fixed banks. The 22% all-interior coverage is not full design coverage, and an apparent success on it is not evidence of robustness over the full bank. CR3 zero-amount derivatives can be singular; CR5's linear amount extension may admit additional valid derivatives, but a new full-design convention must be separately specified and verified, not silently substituted.

Q02 is empirical V × complete empirical X rows, or full bank. Q03 is fitted joint Gaussian V × empirical rows. Q04 is fitted joint Gaussian V × full bank (only known fixed-bank designs). Q05 is independent marginal Gaussian V in the base frame × complete empirical X rows/full bank. The analytic counterpart of Gaussian measures uses supported true prior draws unless a separately labeled legacy sensitivity is intended. The blind arm never sees true prior draws.

For new primary computations use Q01. Sensitivity measures Q02, Q05 and, on banks, Q04 are priority comparisons; Q03 is a secondary comparison if needed. Sample complete X rows and keep the same samples/masks for every R. Do not treat different measures as interchangeable estimators of one identical target.

### 3.4 W2-00 audit and frozen registry

Create a snapshot of:
- exact Lab 07 code and result hashes, metric definitions and active-row/zero-block conventions;
- [metric_evidence.csv](../../../metric_lab/07_unified_benchmark/results/metric_evidence.csv), [next_wave_candidates.csv](../../../metric_lab/07_unified_benchmark/results/next_wave_candidates.csv), original pass summaries and available moment caches;
- NP/dataset/Z/scaler hashes, split IDs, designs and coverage.

Recompute B08 objective values on selected saved rotations in the diagnostic harness, and reproduce Q01 all-library statuses. Verify the 24/24 inferred marginal, 8/24 analytic perfect and 12/24 analytic marginal claims for the exact development configuration. Recompute per-seed/fidelity/measure/strict flags from saved losses or available tensors rather than extrapolating 24/24 to every configuration. If any required historical arrays are missing, mark them unavailable and recompute with documented input versions. Test fast single metric evaluator against Lab 07 full registry.

Unit tests: orthogonality, inverse coordinate transforms, correct output scaling, exact tensor vs direct differentiation, signed-permutation invariance, constant/undefined controls, all-zero and tiny rows, stable mask/denominator, seed reproducibility, same score irrespective of sweep label, and truth-file access denial. Record failures before interpreting results.

---

# Part III. W2-01 — Actual blind rotation of the existing candidates

## 4. Why this experiment comes first

The current B08 evidence is about a finite set of rotations surrounding a truth-assisted reference. It says nothing definitive about other basins of the objective. A candidate can be excellent in 15-degree sweeps and still fail an actual random-start search. Conversely a nonperfect local sweep need not prevent practical recovery.

This is the most direct inexpensive test of the project's actual scientific aim. A successful B08 blind search would be a significant empirical result even though the analytic metric is flat for parts of the classic models. It would not establish the original Gap 1/2 explanation.

## 5. Fixed loss and controls

For $B_{ij}=E[(\partial_{u_i x_j}^2 d_R)^2]$, $r_i=\sum_j B_{ij}$, $q_{ij}=B_{ij}/r_i$,

$$
L_{\mathrm{B08},c}(R)
=\sum_{i\in \mathcal A(R)}
\frac{r_i}{\sum_{\ell\in\mathcal A(R)}r_\ell}
\frac1p\sum_{j=1}^{p}\frac{q_{ij}}{q_{ij}+c/p},
\qquad c\in\{.01,.03\}.
$$

$\mathcal A(R)$ follows the actual historical library rule, not a new hand-selected threshold. Lab 07's fast B08 evaluates active rows with $r_i>10^{-12}\max(E_B,10^{-300})10^{-3}$, where $E_B=\sum_i r_i$, and renormalizes active energy weights. A whole-block zero flag gives undefined rather than an artificial favorable zero. Preserve the exact tested implementation in the primary arm. Document a mathematically continuous all-positive-row version only as a later sensitivity. Do not change the row weighting or c after viewing T.

Primary candidate registry (all IDs refer to their existing source libraries):

| ID | Metric | Role |
|---|---|---|
| E01 | L06\|B08.M03.c0.01 | strongest inferred-only candidate |
| E02 | L06\|B08.M03.c0.03 | pre-existing parameter sensitivity |
| C01 | L03\|UX.lq.1 | analytic-perfect control, poor CR decoder inference |
| C02 | L06\|B01.M03.c0.01 | pooled UX soft-L0 control |
| C03 | L03\|Fz.A+B+C | prior complex-continuous success |
| C04 | L06\|B09.w75.M03.c0.01 | UX+UU fusion control |
| N01 | unrotated identity | no-search baseline |
| N02 | fixed Haar rotation | random baseline |
| N03 | L06\|CTRL.EB | invariant loss; must not select informative directions |

The invariant control is a diagnostic of the optimizer/selection pipeline; do not treat its arbitrarily returned R as a successful objective. Use exact historical Haar-null normalization for Fz, with null distribution frozen per configuration.

## 6. Search algorithm and compute budget

Minimize the frozen inferred objective:

$$
\widehat R=\underset{R\in O(k)}{\arg\min}\ \widehat L(R),\qquad
U=\widehat R V.
$$

Cache the raw or standardized base derivative moments:
$$
J_{ab}^{(j)}=E[H_{V_a X_j}H_{V_b X_j}],\quad
B_{ij}(R)=(R J^{(j)}R^\top)_{ii}.
$$
UU-based controls use the corresponding fourth-order K. Compute caches once per split/measure/seed; objective evaluations need only contractions. Reuse Lab 07's validated contraction and metric adapters read-only. Do not import a truth-enabled optimizer module into blind code just because it contains a generic Givens routine; extract a neutral numerical routine or adapt it in the new lab with tests.

Search protocol S0 (mandatory):
- draw Haar starts with Gaussian QR and sign correction;
- k=3 or 4: 400 starts, refine best 8 distinct starts; k=6: 2000, refine best 30;
- use seeds 31, 73, 101 for independent start pools; search RNG is distinct from integration RNG;
- use Givens-coordinate descent with decreasing angular step and capped function evaluations, log actual sweeps/tolerances; identity may be an additional allowed initialization;
- compare final objective on fixed optimization samples, optionally use an independent blind validation moment cache as a predeclared tie-breaker within numerical/integration tolerance;
- select lowest valid blind loss; do not select the R with highest MCC.

S1 (search-budget sensitivity): repeat with 5× starts / proportionately more refinement for all primary E01/E02 cells with unresolved convergence, widely separated near-minima or inconsistent blind loss across seeds. Rule must use blind optimization diagnostics only. Record a capped budget and every attempted result.

S2 (optional optimizer comparison): smooth orthogonal-manifold optimizer with QR/polar retraction, and/or Nelder–Mead in a Givens parameterization. Use the same total loss-evaluation budget for a fair comparison; the existence of an AD Hessian does not mean a hard-threshold objective is smooth. Choose optimizer from blind objective/convergence diagnostics or freeze one for the primary result.

Use common random numbers for R. Independent sample reevaluation is needed to assess Monte Carlo overfit; a lower loss on one cache may not generalize to another integration sample. Store all final R matrices, start IDs, search traces, objective evaluations and statuses, not just the successful best.

## 7. Staged blind-search grid

| Batch | Scope | Metrics | Measures/masks | Fidelity / seeds |
|---|---|---|---|---|
| BS-A | all 24 primary variants | E01/E02, C01–C04, N01–N03 | Q01, FULL/MASK-ALL; historical replay | medium C; integration seed31; 3 search seeds |
| BS-B | all 24 primary variants | same fixed candidate set | Q01, FULL/MASK-ALL; split-safe | medium C; 3 integration seeds × 3 search seeds for E01/E02; controls seed31 × 3 |
| BS-C | all primary variants | E01/E02 plus fixed controls | Q02, Q05; Q04 on banks; MASK-COL on banks | medium; 3 search seeds; first integration seed31 |
| BS-D | all E01/E02 cells in BS-B and fragile/leading sensitivity cells | same metric IDs | original valid estimands | high D independent integration reevaluation; escalation using blind diagnostics |
| BS-E | secondary existing NP runs | E01/E02 + C01/C02 | Q01 primary, mask sensitivity | medium, 3 seeds; report separately |

Medium C follows Lab 07 (512×64 random/observed trial points, 128×500 full-bank); high D 1024×128 or 512×500 where support permits. For split-safe validation use the same count caps or explicit smaller pool and report actual counts. Nested and independent samples must be labeled correctly. Q01 includes observed bank trials rather than a full bank cross-product. Do not silently select the most favorable Q or mask per toy. BS-A first verifies comparability to historical results; BS-B is the principal new search result.

## 8. Post-lock recovery evaluation

Only after saving the blind outputs, use T and R* to evaluate. For each selected R, compute on TEST subjects

$$
C_{ij}=|\operatorname{corr}(T_i,U_j)|,
\quad \pi=\arg\max_\pi\sum_i C_{i,\pi(i)},
\quad MCC=\frac1k\sum_i C_{i,\pi(i)}.
$$

Save raw signed correlation and Hungarian-aligned absolute correlation matrices. Report matched correlations individually, minimum matched correlation, MCC, mean/max unmatched absolute correlation (leakage), difference from the same-variant supervised MCC, and signed-permutation-adjusted rotation distance to R*. Check decoder prediction invariance after inverse rotation.

Use the historical Lab 05 operational criterion as a comparison:
- MCC at least supervised MCC−.05;
- max leakage <.2 for classic families and <.3 for CR families.

These are evaluation-only thresholds, not inputs to the search, and not a replacement for the qualitative criterion of note 01. Also report common leakage <.2 and <.3 for all toys, stricter MCC gap .02, minimum matched correlation, and continuous results; do not retroactively redefine primary success. A different R can satisfy recovery without being the Procrustes R*.

For each algorithm/variant report both 12-toy coverage and 24-library coverage, success by class, all-seed success and per-seed results. Do not combine the best coordinate variant, measure or optimizer separately for each toy. A full pass of a single fixed variant across 12 toys is meaningful even if the other variant fails; distinguish that from 24/24.

Evaluate $L(R_*)$ on the exact same frozen blind loss only after locking R. Then distinguish:
- found loss appreciably lower than $L(R_*)$ but bad recovery: objective/estimand issue, not simply optimization;
- found loss appreciably higher and bad recovery: optimizer/search budget or numerical problem;
- near-tied loss but different recovery: ambiguous landscape;
- successful recovery even if Oracle failed: empirical algorithm result; still not Gap 1/2 proof.

Produce T–U heatmaps for every toy and E01/E02, including failed cells. Provide supervised, unrotated and blind columns side by side, no winner-only plots.

**Gate:** W2-01 is complete with reproducible, R*-free searches for the primary panel and the specified comparisons, whether successful or not. A stable 12-toy pass is a reason for independent confirmation, not a claim of generalization to arbitrary generators.

---

# Part IV. W2-02 — Practical accuracy of analytic global minima

## 9. Problem statement

Lab 07 evaluated analytic minima using normalized signed-permutation distance:

$$
d(Q)=\frac{\min_{P\in\mathcal P_\pm}\|Q-P\|_F}{\sqrt{2k}}.
$$

It found many counterexamples beyond 1e-3; fewer beyond .05 and .15. We need to know whether these small rotations actually break the scientific one-to-one correspondence. Matrix distance is a geometric diagnostic, not the recovery criterion. Even with a perfect inferred response, a metric might yield a slightly biased but scientifically useful solution.

This entire module is evaluation-only and must not enter the blind search configuration or hyperparameter selection.

## 10. Recover all relevant analytic solutions

Use Lab 07's [global summary](../../../metric_lab/07_unified_benchmark/results/global_oracle_summary.csv), search traces and saved matrices where available. Read [run_global_oracle.py](../../../metric_lab/07_unified_benchmark/run_global_oracle.py) and verify what was actually persisted: it saves the suspicious Q selected at the first distance threshold, not necessarily every final Q nor the overall best matrix. CSV loss/distance alone cannot reconstruct Q. Do not pretend to recover a matrix from its scalar distance.

For each of the 93 priority metrics × eight logit family/mask targets (and optional probability control), use saved Q where its identity is verified; replay the corresponding deterministic search if missing. Save all finalist Q and provenance. Prioritize the 33 A1 metrics, 17 original analytic-perfect metrics (union, no double-counting), E01/E02, and known C3/CR counterexamples. Retain all 93 in summary with exact coverage and any missing matrices.

Distinguish:
- best unconstrained loss found;
- best suspicious non-permutation solution;
- best independently re-evaluated solution;
- identity/signed-permutation loss;
- tied near-minima and search failures.

Run additional local refinements from the identity and small perturbations (diagnostic only). Lab 07 labels empirical global support even in a weak-coverage case if no reproducible lower candidate is found; record separately whether any start reached the true basin. For UX.lq.1, specifically test local escape and basin coverage instead of equating “no counterexample found” with global uniqueness.

## 11. Parameter recovery of an Oracle rotation

### 11.1 Ideal standardized-parameter recovery

Given held-out actual standardized T, define

$$
U_{\rm ideal}=QT_s.
$$

Compute signed/absolute T–U correlation matrices, MCC, min matched correlation, leak mean/max with the same Hungarian matching rule. For independent population-standardized T, $\operatorname{corr}(T_s,QT_s)=Q^\top$ (column-vector convention); empirical finite-sample covariance is not exactly I, so evaluate directly on TEST T. Do not interpret a small distance as a specific Givens angle for general rotations.

### 11.2 Recovery near the learned-space ceiling

As a **truth-assisted diagnostic only**, map the same relative analytic perturbation into each paired NP/coordinate variant:

$$
U_{\rm pert}=Q R_* V.
$$

Evaluate the same MCC/leakage metrics versus T and the reference U*=R*V. This is not a blind solution and does not imply that Q minimizes the decoder objective. It answers how much the analytic coordinate bias would harm recovery if the relative orientation were transferred to the learned space.

Report by toy, not just shared family, even when Q is computed once for matched analytic B/C targets. Verify B/C analytic point and scaler equality before reusing Q, as in Lab 07 shared_check. Maintain separate MASK-ALL and MASK-COL results.

### 11.3 Refine the local angular diagnostic

For each priority metric, examine near-identity sweeps in each pairwise plane:
$$
\theta\in\{-15,-10,-5,-2,-1,-.5,0,.5,1,2,5,10,15\}^{\circ}.
$$

Also use random unit skew-symmetric directions and matched signed-permutation endpoints. For smooth losses estimate directional derivative/curvature numerically with scale and step convergence; for nonsmooth metrics report one-sided values, discontinuities, flat intervals. Do not assert differentiability of M01/M02 or rank/tail scores.

For each analytic near-minimum save the loss improvement relative to identity, recovery degradation and distance. Verify on independent high-fidelity integration samples; use paired differences $L(Q)-L(I)$ on each sample, not only variability in $L(I)$. Label exact numerical tolerance separately from substantive integration uncertainty.

### 11.4 Outcome categories (not new formal success definitions)

| Exact analytic preference | Practical ideal recovery | Transfer recovery | Interpretation |
|---|---|---|---|
| empirically supported | good | good | theoretical priority, subject to search coverage |
| displaced minimum | good | good | approximate simplicity may be sufficient in practice |
| displaced minimum | good | poor | learned geometry/ceiling interaction matters |
| displaced minimum | poor | poor or mixed | important metric failure |
| flat / tied non-permutations | variable | variable | objective cannot identify a unique axis system |
| inconclusive | NA | NA | improve search/support, do not count as success |

For practical diagnostics use transparent sensitivity thresholds, e.g. MCC≥.95/.98/.99 and leak <.10/.15/.20 in ideal space, plus the historical ceiling-relative criteria for transfer. Report raw metrics so thresholds can later be debated. These are not a newly ratified definition of Gap 1. Keep the exact unique-minimum theoretical question distinct from approximate useful recovery.

**Deliverable:** complete analytic solution registry; distance-vs-MCC/leakage plots; ideal/transfer heatmaps; fine-angle curves and basin-coverage report. The result tells us which A1 candidates deserve inference effort even if they fail exact analytic uniqueness.

---

# Part V. W2-03 — Conditional Hessian Energy

## 12. Scientific hypothesis

The first wave used a pooled matrix
$$
M_{\rm pooled}(R)=E_{V,X}[H_R^2],\qquad L_{\rm pooled}=\Phi(M_{\rm pooled}).
$$
Pooling may combine distinct parameter–stimulus coupling patterns that never coexist on the same trial. Sparsity of the union of supports is not the same thing as sparsity within each stimulus condition.

For example, two X regions might each depend on one latent axis. Their pooled energy may look dense even though each local function is simple. The goal is to recover a **single common R** that makes these local structures simple simultaneously, without requiring knowledge of trial categories or the true function.

## 13. Conditional estimands and normalization

Let $\mu$ be a fixed reference distribution over base V, $\nu$ a fixed distribution over complete X vectors. For P1 use the declared product design; a conditional empirical distribution V|X is a different estimand. Under pushforward rotation:

$$
M_x(R)=E_{V\sim\mu}[H_R(V,x)^2],\qquad
L_X(R)=E_{X\sim\nu}[\Phi(M_X(R))].
$$

Test aggregation recipes:

| ID | Definition | Purpose |
|---|---|---|
| P0 | $\Phi(E_{V,X}H_R^2)$ | pooled baseline |
| P1 | $E_X[\Phi(E_V H_R^2)]$ | exact stimulus conditioning |
| P2 | $E_V[\Phi(E_X H_R^2)]$ | latent conditioning |
| P3 | $E_{V,X}[\Phi(H_R^2)]$ | pointwise penalty, secondary noise-sensitive arm |
| P4 | $\sum_s\omega_s\Phi(E[H_R^2\mid X\in S_s])$ | generic grouped X |
| P5 | $\lambda L_{\rm pooled}+(1-\lambda)L_X$ | pooled/conditional fusion |

The primary P0/P1 comparison must use the identical product evaluation design and mask. Do not compare P1 on product draws against P0 on observed subject-trial Q01 and attribute every change to aggregation order. For P4, the corresponding pooled P0 uses exactly the same sampled data and weights. P2 and P3 are secondary after P1/P4 correctness is verified.

For each group or x, use conditional valid denominators. A genuinely all-zero group contributes zero in an explicitly defined support loss, or is omitted with weights renormalized under a predeclared active-group rule. Undefined derivative groups are not zero. Save raw validity, omitted mass and reason, and use the same policy for every R and source. If no valid support exists, the score is unavailable.

Normalize in three ways, kept as distinct recipes:
- N0 pooled invariant energy scale (same threshold for all groups);
- N1 within-group block invariant energy scale (local support, legacy metric adapters);
- N2 local normalization plus weights proportional to each group's invariant energy.

For arbitrary \Phi implement exact legacy normalization first. For a fixed-scale separable Soft-L0 extension use
$$
\Phi_c(M_x;E_0)=\frac1n\sum_j
\frac{M_{x,j}/E_0}{M_{x,j}/E_0+c/n}.
$$
$E_0$ must be fixed before search (pooled invariant block total, with the correct symmetry multiplicities). Define separate UX and UU scales for joint recipes. For B08 local row normalization, weight active rows according to the unchanged recipe. Group weighting is separate from row weighting. Do not mix metrics with different ranges before declaring a scale; for P5 begin with the same bounded Soft-L0 components and $\lambda=.25,.5,.75$ plus endpoints.

When a nonlinear concave penalty is applied before vs after averaging, their values differ. A smaller numerical loss alone is not an improvement; the correct coordinate preference is what matters.

## 14. Evaluation design without model-specific grouping

### 14.1 Fixed-bank toys

For CR3, CR5, CR3L, CR5L, use all available complete X rows of the shared 500-item bank, with explicit masks. For P1 evaluate each row separately and integrate over empirical V or supported Gaussian V. For an all-interior derivative estimand, report the excluded fraction of the 500 rows; do not claim to use all 500 derivatives. For column-specific validity compute each valid derivative with its proper conditional denominator; specify how missing columns affect the local support score and never impute undefined as zero. A conservative complete-case P1 is primary for comparable matrix recipes, columnwise valid subspace is sensitivity.

Do not use known gain-only/loss-only/CPT categories in the primary method. Grouping can use observed X only. An explanatory category analysis may be separately labeled evaluation-only. All Toy IDs use the same aggregation algorithm.

### 14.2 Random-design toys

Build a reusable empirical product panel with 128 distinct sampled X rows × 128 latent points at pilot, 256×256 at medium and 512×512 for finalists. Sample latent V from TRAIN for split-safe inference and paired actual T for empirical oracle, or from explicitly specified supported distributions. For fixed banks use full 500 rows × 32/128/512 latent points. Use seed31 first, then 73/101 for repeatability. Exact sizes are proposed resource caps; record actual coverage and effective sample sizes.

For P4 run generic clustering of standardized X rows (k-means with fixed seed, K=4/8/16), minimum group support rule fixed before scoring; preserve complete rows. Fit groups once in X space, freeze across R. Use frequency weights as primary; equal group weights sensitivity. If a requested K creates too-small groups, merge deterministically using X-distance or mark invalid; do not remove inconvenient groups based on T or score.

For P2 use the same latent/stimulus product panel and cache by base latent draw, not by rotated coordinates independently. P3 can stream pointwise second moments; cost cap applies.

### 14.3 Efficient moments

At each x or group s store $K^{(s)},J^{(s,j)}$ rather than all Hessians. Contractions for R give $M_s(R)$ before applying \Phi. For per-X conditioning, group-specific normalization and weights remain frozen. Check that weighted conditional moments exactly reconstruct the pooled moment for the same valid support and weighting. With column-specific masks the pooled reconstruction uses each column's own valid mass; do not apply a single row weight that changes denominators.

Use float64 reference, chunk latent/stimulus points. For P1/P4 evaluate many scalar metrics cheaply after caching; do not recompute derivatives for each scalar setting.

## 15. Candidate metrics and staged grid

Initial panel: fixed legacy versions of
- L06 B01.M03.c0.01/.03, B01.M04.c0.01 (global UX support);
- L06 B08.M03.c0.01/.03, B08.M05 (row-weighted support);
- L06 B09.w75.M03.c0.01, B09.w50.M03.c0.03, B10.2blk.M05 (joint UX/UU);
- L03 UX.lq.1, UXcol.a0.lq.25, POOL.lq.1, SL0.UXUU.e0.01 (analytic candidates);
- L03 rowsp.a0.5.q0.1, Fz.A+B+C (CR candidates);
- invariant and flat controls.

For Haar-null fusions, compute null statistics for the declared conditioned/fused objective without using a supervised reference, save normalization convention. To keep the primary comparison interpretable, prioritize unfused Soft-L0 and fractional-power recipes; fusion is secondary.

Execution:
1. **CH-A**: P0/P1/P4 (K=8), N1, frequency weights, all six analytic families and masks, logit target; use matched B/C reporting only if actual points/scalers coincide. Pilot, seed31. Compute both 04.2 analytic sweep flags and strict margins. The truth surface is used only in the analytic arm.
2. **CH-B**: same frozen configurations for AD decoder, all 24 variants, Y-L/Y-C. No T in construction of conditional matrices, grouping or objective. Report inferred-only and joint flags.
3. **CH-C**: for the fixed initial panel and any new analytic-all or inferred-all candidate, compare P2/P3/P5, N0/N2 and K=4/16 with equal weighting as resource permits. Escalate across seed73/101 and medium fidelity using a frozen coverage/stability rule. Retain negative controls and failed configurations.
4. **CH-D**: for stable analytic candidates, run global Q search plus fine-angle and practical recovery diagnostics of W2-02. For stable inferred candidates, run the W2-01 blind protocol as a secondary pilot. Avoid automatically converting every local pass into a global claim.

Primary comparison: same metric, same points, P0 versus P1/P4. Report changes in analytic pass coverage, strict endpoint margin and near-permutation/global minima; then compare inference fidelity. Conditional metrics constitute a new functional, not improved estimates of the old pooled functional.

### 15.1 Small optional bank-contrast pilot

If conditional X derivatives are invalid or cover too little of the bank, test one separate derivative-free-X estimand as a preparatory Step 5C branch. For complete observed stimulus pairs x,x':

$$
c_{x,x'}(V)=g(V,x)-g(V,x'),\quad
B_i^{(x,x')}(R)=E_V[(\partial_{u_i}c_{x,x'}(R^\top U))^2].
$$

The contrast requires latent derivatives only, not off-bank X derivatives. Use a generic deterministic pairing based on standardized X distance or uniform pairs, fixed across R; e.g. 256 pairs then 1024. Include a baseline mean-response gradient component so an intercept is not invisible. Report as a separate target with its own support and normalization. Do not label contrast values classical UX Hessian energy.

**Gate:** W2-03 must establish whether conditional aggregation improves analytic discrimination, actual inferred preference, both, or neither on the full designated panel. If no improvement, retain the pooled baseline and investigate richer derivatives/alternative simplicity in the next wave.

---

# Part VI. W2-04 — Improved inference of fixed metrics

## 16. Scientific goal and information constraint

Keep the simplicity metric and integration measure fixed while changing the response-surface estimator. The purpose is to infer structural information, not just improve prediction. A model that predicts Y slightly better may have worse derivatives; a model that predicts Y slightly worse can still recover derivatives better. However choosing the estimator based on T or Oracle loss at deployment is forbidden.

Use A1's 33 metric IDs from the frozen Lab 07 handoff table, with UX.lq.1 and joint soft-L0 recipes as fixed priorities. Also retain E01/E02 and pooled UX controls. Reuse the same tensors/metric functions and Q01 masks. Report every fixed metric with every completed estimator, not just winners. The W2-02 practical/global outcomes may define a secondary research shortlist, never change the primary metric definitions after results are seen.

## 17. Two sources of data, three baseline surfaces

| Source | Target | Uses |
|---|---|---|
| D0 observed data | Class B binary Y, Class C observed continuous Y | independent statistical surface |
| D1 frozen decoder teacher | logit/continuous mean or soft probability | smoothing/distillation of existing response |
| AD direct | frozen decoder derivatives | baseline |

For binary D0 use proper Bernoulli likelihood and a common logit model, with Y-P by sigmoid if reported. For continuous D0 use squared error and recover original units. For D1 use logit/continuous MSE or probability soft-label loss as declared. A stochastic decoder choice D2 is a secondary information-loss control only, not the primary target.

D0 fits on TRAIN subjects. D1 queries the frozen teacher on TRAIN observed rows or a separately declared support-aware empirical product design; never query true f. Fit and validate by independent subject groups. Teacher distillation and observed-data fit can estimate different targets; do not call them interchangeable replicates.

The previous W1 polynomial estimator had degrees 2/3 and selected lambda=1e-4 for 11 of 12 toys. It is a historical baseline, not evidence that polynomial/PCE or smoothing in general cannot work. W2 must test broader smooth families and derivative stability explicitly.

## 18. Priority estimators

### 18.1 E-SMOOTH — smooth the frozen decoder directly

For base coordinates w=(v,x),
$$
g_h(w)=E_\xi[d(w+\operatorname{diag}(h)\xi)].
$$

Begin with latent-only isotropic perturbation in a fixed standardized latent coordinate frame, $\xi_V\sim N(0,I)$, $h_V\in\{.025,.05,.1,.2\}$, $h_X=0$. This preserves the X bank rows. Optionally compare joint smoothing in random-design interior with predeclared X-domain handling, $h_X\in\{.025,.05,.1\}$; no unconstrained off-bank perturbation treated as observed evidence. Near domain boundaries use valid-support alternatives separately or mark unavailable. Do not apply a Gaussian averaging formula requiring translation invariance if the actual truncation law depends on w without accounting for derivative terms.

For fixed noise law independent of w and suitable differentiability, $H[g_h](w)=E_\xi[H[d](w+h\xi)]$. The primary smoothed-energy estimator squares the averaged derivative:
$$
M_{ij}^{\rm smooth}=E_w[(E_\xi H_{ij}[d](w+h\xi))^2],
$$
not $E_{w,\xi}[H_{ij}[d]^2]$, which is just an enlarged sampling measure. Use common antithetic perturbations (16 at pilot, 64 for finalists), compare bandwidth zero exactly to AD, record fit-to-teacher and derivative stability. For coordinate equivariance define the kernel in the fixed base frame and transform its covariance with R; do not refit it in every rotated frame.

A blind selection rule may use held-out teacher/observed predictive loss plus derivative reproducibility under independent resamples, with a predeclared admissible prediction-loss tolerance. Report all bandwidths; the smallest Oracle error is diagnostic, not the deployed selection rule.

### 18.2 E-POLY / E-PCE — richer global smooth basis

Fit complete total-degree polynomial ridge (Class C) or logistic ridge (Class B):
$$
g(w)=\sum_{|\alpha|\le D}\beta_\alpha\psi_\alpha(w).
$$
Degrees 2/3 reproduce W1; extend to 4 and 5 if rank and memory allow. Use all generic cross terms, no CPT-specific features. Use dimensionless standardized inputs, numerically stable scaling and orthogonalized basis where appropriate. Compare monomials against product Hermite/Legendre or empirical orthogonalization. Standard PCE independence/orthogonality assumptions may fail for correlated Z or finite bank X; in that case label it a polynomial basis regression, not a valid Sobol decomposition.

Proposed ridge grid: $10^{-7},10^{-6},10^{-5},10^{-4},10^{-3},10^{-2},10^{-1},1$, with clearly normalized empirical loss and intercept convention. Select using subject-held-out validation loss; use a predeclared one-standard-error or explicit relative-loss-tolerance rule for stronger regularization as a second fixed selector. Record rank/condition, basis size, and coefficient/derivative stability. Analytic derivatives of the basis allow efficient moment estimates.

In 10 total input dimensions degree5 has thousands of terms; use matrix-free/streamed ridge or lower degree when necessary, record any skipped method. Do not silently restrict interaction order after seeing which toy fails.

### 18.3 E-KRR / E-RBF — smooth kernel surrogate

Gaussian kernel or Gaussian RBF landmark representation:
$$
g(w)=b+\sum_{\ell=1}^{m}a_\ell
\exp[-\|w-c_\ell\|^2/(2h^2)].
$$
Use 256/1024 landmarks at pilot, 2048 for finalists; choose centers from allowed TRAIN inputs using fixed random/k-means rule, not T. Bandwidth grid based on empirical median distance times .5, 1, 2; ridge grid $10^{-6},10^{-4},10^{-2},1$ (normalized objective). D0 binary uses a penalized logistic RBF model, not unbounded probability least squares relabeled as logit; D0 continuous and D1 logit/continuous use ridge/MSE.

Explicit Hessian for an RBF basis $\phi_\ell$:
$$
\partial_{ij}\phi_\ell=\phi_\ell\left[
\frac{(w_i-c_{\ell i})(w_j-c_{\ell j})}{h^4}-\frac{\delta_{ij}}{h^2}
\right].
$$
Differentiate in the declared standardized base frame; use chain rule for rotation. Compare a shared isotropic bandwidth versus fixed blockwise latent/X scaling as separate candidates. A smooth kernel does not guarantee accurate Hessians; bandwidth and design coverage are material.

### 18.4 E-MLP — derivative-stable smooth distillation

Optional second-stage estimator if the cheap families do not resolve CR inference. Use 2–3 hidden layers, widths 64/128, tanh or GELU, weight decay and fixed early stopping on subject-validation loss. D0 and D1 separately. For D1 optionally penalize roughness or instability:
$$
\mathcal J=\mathcal L_{\rm pred}
+\gamma E_{w,\xi}[\|\nabla g(w+h\xi)-\nabla g(w)\|^2],
$$
with small predeclared $\gamma$ grid on dimensionless scaled output/input. Such a generic smoothness penalty is not a true-derivative target and may erase real curvature. Compare gamma=0; no penalties involving actual T, desired rotation or Oracle Hessian. Do not pre-impose the sparsity objective on a supervised basis and then claim blind discovery.

GP, local polynomial and GAM remain extensions from 04.3; run only after the primary E-SMOOTH/E-POLY/E-RBF comparison, or if a specific identified failure warrants them. Do not turn W2 into an unrestricted estimator grid.

## 19. Cross-fitting and statistical de-noising

### 19.1 Why square bias matters

If $\hat h=h+b+e$ with zero conditional mean e, then
$$
E[\hat h^2]=(h+b)^2+E[e^2].
$$
Even a small spurious derivative can turn a structurally zero coupling into nonzero energy. This is important for Soft-L0.

For independent subject halves A and B fitted to the same allowed estimand, compute
$$
\widehat M^{\rm cross}_{ij}=E_{\rm eval}[\hat h^A_{ij}(W)\hat h^B_{ij}(W)].
$$
With conditional independence and centered errors this removes the variance contribution, not shared approximation bias. Common frozen Z/teacher, preprocessing or finite-sample dependence can invalidate independence; describe these as empirical variance diagnostics rather than guarantees. Do not multiply derivatives across different evaluation points or incompatible coordinate scalers.

Use at least 3 repeated disjoint subject half-splits for the priority observed-data surrogate(s), and same independent validation/evaluation points for derivative comparison. Compare:
- conventional single-fit squared energy;
- average of two fit energies;
- cross-product energy;
- noise-corrected energy $\frac12(E[h_A^2]+E[h_B^2])-\frac12 E[(h_A-h_B)^2]$ (algebraically identical to cross-product for a pair; not a new independent method);
- shrinkage based on repeated fits / bootstrap uncertainty.

Keep base-coordinate cross moments, symmetrized over fit A/B, so transformation is exact. Cross-estimates may be negative and need not be PSD. Save raw signed values. For metrics requiring nonnegative energies compare explicit nonnegative projection (elementwise at final R) or PSD projection of the base second-moment blocks as distinct estimators; the former can change geometry and must be audited for sign/permutation invariance and stability. Never silently truncate before reporting variance and bias diagnostics.

### 19.2 Noise floor and support uncertainty

For replicate fits estimate the derivative covariance/noise tensor in base coordinates; transform it consistently. Construct a conservative support score e.g.
$$
L_{\mathrm{support}}(R)=\frac1n\sum_{ij}\frac1B\sum_{b=1}^B
\mathbf 1\{\widehat M^{(b)}_{ij}(R)/E_0>\delta\}.
$$
Here bootstrap exceedance frequency is a stability estimate, not a Bayesian posterior probability. Use subject-level bootstrap, e.g. 20 pilot replicates and 50 finalists if feasible, with fixed evaluation points; report computational budget. A posterior probability of support would require an explicit prior/likelihood and is outside mandatory W2.

Use normalized $\delta=c/n$ for the same registered c grid or a predeclared uncertainty-based threshold. Do not learn $\delta$ from Oracle zeros. Compare uncorrected and corrected matrices, threshold sensitivity and strict margins. A noise correction that removes real weak interactions can worsen recovery; include analytic known-zero/nonzero evaluation only after the algorithm is fixed.

### 19.3 Diagnostics must distinguish uncertainty types

- Integration resampling of a frozen decoder measures numerical integration variability.
- Refit uncertainty of a surrogate measures variability of that estimator conditional on Z/teacher.
- Independent NP training repeats measure representation/decoder variation.
- Difference from Oracle includes alignment, conditional target, approximation bias and numerical effects; cannot be described as pure sampling noise.

## 20. Staged inference grid and evaluation

Use Q01 and FULL/MASK-ALL as primary, MASK-COL sensitivity on banks. Apply the same measure/coordinates to every surface. Fit in base frame once and rotate derivatives, never refit after each candidate R. Historical bridge and old polynomial outputs are evaluation/control arms. Truth bridge is not a candidate estimator.

| Batch | Scope | Methods | Data / fidelity | Purpose |
|---|---|---|---|---|
| IF-A | all 12 NPs, both variants evaluated | AD, old polynomial, E-SMOOTH | Q01 pilot, seed31; allowed perturbations | lowest-cost structure test |
| IF-B | all 12 NPs | E-POLY/PCE, E-RBF | D0/D1, max 1024 TRAIN subjects×64 trials pilot; validation up to 256×64 | broader surface approximation |
| IF-C | all primary variants, fixed best-by-blind-rule estimators + controls | cross-fit, noise correction, shrinkage | independent halves/repeated fits, Q01 medium, 3 seeds | near-zero energy stability |
| IF-D | all primary variants for leading fixed algorithms | high fidelity, mask/Q02/Q05 sensitivity; optional E-MLP | fixed seed and independent evaluation | test robustness |
| IF-E | secondary saved NPs | same frozen candidate algorithms | Q01 medium | separate repeat robustness |

Train size and validation count caps are proposals; preserve subject grouping and record actual counts. For fixed bank, if fitting from observed trials, use observed trial rows. For teacher product queries use complete X rows and TRAIN latent draws. A generic validation selector ranks by appropriate predictive loss and stability of derivative moments across independent TRAIN refits; never by agreement with T or R*. Predeclare any allowable prediction degradation (for example relative 1% validation loss above best) before truth-based evaluation.

If full-context Z includes the target response, D0 fit is conditional on that representation and may exploit target information. Do not interpret its fit as independent-query generalization. Optional disjoint-context/query encoding can be a separate diagnostic with frozen encoder, not a replacement for official S5 Z; label changed representation clearly.

Required reporting:
- prediction loss, original-scale correlation and fit cost;
- normalized and unnormalized UX/UU errors, oracle-zero false-energy fraction (truth only at evaluation);
- derivative-moment replicate variability, signed corrections, PSD/truncation diagnostics;
- Eval1 Pearson/Spearman and undefined fractions; Eval2 oracle-only/inferred-only/joint strict and legacy flags;
- coverage across 24 libraries by class; failure type and minimum margin;
- W2-01 blind search for a small frozen shortlist of improved inferred objectives after evaluation.

Improvement means a common estimator/selection rule improves CR inference or actual recovery without destroying classic controls. Do not pick separate favorable degrees/bandwidths for each toy based on T. A new metric based on support uncertainty is a different functional; report it separately from fidelity of the original HE score.

**Gate:** identify whether the bottleneck can be reduced by smooth distillation, richer observed-data surface or uncertainty-aware estimation. If none help, report that negative result and advance conditional/gradient/contrast/alternative structure methods rather than indefinitely expanding smoothing grids.

---

# Part VII. W2-05 — Reporting, decisions and implementation

## 21. Common evaluations and minimum outputs

Retain the original 04.2 metrics:
- Eval1 Pearson/Spearman over baseline plus positive-angle sweeps, exact-matrix dedup sensitivity, per-sweep correlation.
- Eval2 all-sweep perfect 0/90, marginal-or-better 0/15/75/90, constants/undefined fail. Also save strict minimum-set subset flags, interior ties and margins. Preserve legacy vs harmonized tolerance separately. For fine-angle diagnostics, report the full angle set rather than relabeling its points as the old 15-degree criterion.
- Actual blind recovery is a separate table; no local pass implies global recovery. Global analytic tests are a separate table; no failure to discover a counterexample implies proof.

Every result has toy, class, NP run, variant, channel, registry version, metric, estimator, source, measure, mask, sample sizes, split, seed, fit/search budget, validity and code/data/scaler hashes. All planned and attempted configurations have rows, including failed or unavailable configurations.

## 22. Proposed independent experiment code layout

To avoid collisions with historical Lab 01–07, create a new folder for this **integrated second wave**. The overview suggested thematic lab numbers; this proposal groups the four coordinated experiments into Lab 08 with submodules. Do not create or rename Lab 07 files. If separate thematic directories are preferred later, keep the same contracts and output namespaces.

    metric_lab/08_wave4_2_second_wave/
      README.md
      configs/
        primary_registry.json
        blind_candidates.json
        conditional_candidates.json
        inference_candidates.json
        budgets.json
      wave2/
        blind_inputs.py       # minimal allowed-data payload, no supervised files
        common.py             # versioned read-only Lab 07 adapters
        objectives.py         # exact registry and conditional score adapters
        optimizers.py         # truth-free numerical search
        evaluate_recovery.py  # evaluation-only T–U and R* diagnostics
        oracle_diagnostics.py # analytic global replay / fine-angle / transfer
        conditional.py        # P0–P5 and grouped moment caches
        surrogates.py         # D0/D1 smooth function estimation
        denoise.py            # independent fit moments, uncertainty corrections
        reporting.py
      tests/
      validate.py
      run_blind.py
      run_oracle_diagnostics.py
      run_conditional.py
      run_inference.py
      analyze.py
      run_all.sh
      results/

Large files under outputs/metric_lab/08_wave4_2_second_wave/<run_id>/{blind,oracle_diagnostics,conditional,inference}/. Use immutable run IDs and --resume; no overwrite by default. Read old checkpoints, registry and moments read-only; do not embed truth-only files in blind cache. All fresh fit models and numerical states required for replay must be saved.

Illustrative future CLI (not currently implemented):
    
    python metric_lab/08_wave4_2_second_wave/validate.py --toys all
    python metric_lab/08_wave4_2_second_wave/run_blind.py --toys all --batch BS-A
    python metric_lab/08_wave4_2_second_wave/run_blind.py --toys all --batch BS-B
    python metric_lab/08_wave4_2_second_wave/run_oracle_diagnostics.py --targets all
    python metric_lab/08_wave4_2_second_wave/run_conditional.py --batch CH-A --toys all
    python metric_lab/08_wave4_2_second_wave/run_conditional.py --batch CH-B --toys all
    python metric_lab/08_wave4_2_second_wave/run_inference.py --batch IF-A --toys all
    python metric_lab/08_wave4_2_second_wave/run_inference.py --batch IF-B --toys all
    python metric_lab/08_wave4_2_second_wave/analyze.py --panel primary

Support --toy, --variant, --channel, --metric, --seed, --measure, --mask, --fidelity, --config, --resume and bounded parallelism where appropriate. Before full panel, smoke-test C1/C3/CR4 and their L counterparts, plus CR3 or CR5 for bank masks. Smoke tests validate implementation, not favorable hyperparameter selection.

## 23. Required files and figures

| Output | Contents |
|---|---|
| w2_manifest.json / config_snapshot.json | exact versions, frozen panel, data splits, choices and rules |
| reproduction_audit.csv / validation_checks.csv | W2-00, historical reproduction and truth-firewall tests |
| coverage.csv | all planned and completed configurations, invalid/missing reason |
| blind_search_trace.csv.gz / found_rotations.npz | all starts and final matrices, selected R and loss, convergence, timing |
| blind_recovery.csv / recovery_correlations.npz | matched correlations, MCC, min, leakage, ceiling gap, R* distance, all-seed pass |
| blind_reference_loss.csv | post-lock L(R*) and blind-vs-reference loss diagnosis |
| oracle_solutions.npz / oracle_recovery.csv | exact Q, source or replay, ideal and transferred recovery, loss margin, mask/seed |
| oracle_fine_sweeps.csv.gz | fine angle/random direction loss and stability |
| conditional_moments.npz or chunked store | conditional K/J, valid mass, group labels/weights and design seeds |
| conditional_pass_summary.csv.gz | P0–P5 comparisons, analytic/inferred/joint metrics, seeds |
| estimator_fits/ and fit_diagnostics.csv | full estimator state, prediction, hyperparameters, rank, split and cost |
| derivative_error.csv / denoise_diagnostics.csv | oracle comparison only after fit, variance, negative correction, support stability |
| inference_pass_summary.csv.gz | all estimator × metric cells, both classes and variants |
| next_wave_candidates.csv | fixed candidate IDs, evidence, limitations, follow-up lane |

Figures required:
1. For E01/E02, T–U absolute correlation heatmaps: unrotated, supervised and blind for every toy/variant (failures included).
2. Blind objective loss vs MCC scatter and search-seed stability; do not select based on the scatter.
3. Oracle signed-permutation distance vs ideal and transferred MCC/max leak; mark lower-loss counterexamples and weak search coverage.
4. Fine-angle curves around identity showing small analytic deviations.
5. Pooled vs conditional oracle/inferred curves and conditional support heatmaps on common points.
6. Surrogate comparison of prediction error vs derivative error vs rotation performance; UX/UU and near-zero entries separately.
7. Per-class pass matrices and a consolidated three-gap evidence table.

Plots must be reproducible from saved outputs, not chart-only results.

## 24. Final decision rules

| Observed result | Meaning | Next action |
|---|---|---|
| B08 fixed blind algorithm passes all 12 toys robustly | strong development evidence for empirical recovery | freeze algorithm; independent new datasets, NP seeds and new model families |
| B08 fails with lower loss than R* | local success does not generalize globally | conditional/extended structure; inspect ambiguity |
| B08 fails with higher loss than R* | optimization budget/landscape may be limiting | improve optimizer by blind convergence rule |
| Oracle lower minima retain high ideal/transfer MCC and low leakage | exact Gap 1 failure may be practically small | retain candidate for inference while reporting exact failure |
| Oracle deviations destroy recovery | substantive metric problem | prioritize conditional HE / gradient / alternatives |
| Conditional improves oracle but not inferred | new Metric promising, Gap 2 remains | use best improved estimators; no blind success claim |
| Conditional improves inferred only | empirical candidate, not faithful inference | actual blind test and replication |
| Surrogate or denoising improves fixed-metric fidelity and recovery | useful Gap 2 method | lock estimator + metric and run blind/confirm |
| All surrogates fail despite high prediction accuracy | conditional representation/target or derivative structure may be limiting | disjoint-context diagnostic; gradient, UXX, contrasts, ANOVA/MDL |
| Bank mask changes conclusions | identification/support limitation | report separate estimands; prioritize observed-row contrast/reduced-axis methods |

The mandatory result is **not** “some metric must pass.” The experiment is successful as research if it cleanly separates the evidence and yields an implementable next decision. Do not relax thresholds, drop difficult toys, select different losses for different known families, or turn an inconclusive global search into proof.

## 25. Acceptance checklist and scope

- [ ] W2-00 reproduces or explains differences from Lab 07's exact candidate evidence, including B08 mask dependence and seed-status changes.
- [ ] W2-01 uses a true R*-free objective and optimizer; all 12 official toys, both coordinate variants, and specified baselines are reported.
- [ ] W2-02 reconstructs verified Q matrices or replays missing ones, and evaluates both ideal and transferred T–U recovery; exact and approximate Gap 1 claims remain distinct.
- [ ] W2-03 compares pooled and conditional objectives on the same support/points and with fixed grouping; analytic, inferred and global evidence separated.
- [ ] W2-04 includes AD, observed-data and teacher-derived estimators, fixed validation selection, derivative stability, signed/noise correction diagnostics and the declared primary metric panel.
- [ ] No truth in operational estimation/search; no per-family favorable choices; all missing/invalid/failed runs retained.
- [ ] Primary 24-library coverage, class-specific conclusions, secondary NP robustness and independent-confirmation limitations explicitly reported.
- [ ] W2-05 supplies a candidate registry for the next wave, with actual blind recovery evidence distinct from analytic or local screening.

**Priority if computing resources are limited:** complete W2-01 and W2-02 first; report CH-A/CH-B and IF-A/IF-B with explicit coverage before expensive convergence/estimator expansions. Parallel work is encouraged, but uncompleted arms must be labeled uncompleted. The third wave will be designed based on the evidence, not by assuming this plan succeeds.

**Implementation status:** documentation only. Folder names, proposed grids, budgets, evaluation sensitivities and CLI are specifications, not existing code or validated settings. No earlier experiment results, formal test beds or project definitions are changed by this note.

**Implementation (2026-11-26):** W2-00 to W2-05 are implemented and run in [Metric Lab 08](../../../metric_lab/08_wave4_2_second_wave/README.md) (results, deviations and not-run arms are listed there).
