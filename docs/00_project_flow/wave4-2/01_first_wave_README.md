# Wave 4-2 / 01 — First-wave experiments: unified benchmark and failure localization

**Status:** implementation specification, not a results report.  
**Parent:** [总览 — Next-stage research roadmap](总览.md), Step 1 and the first two work packages in its execution queue.  
**Prerequisites:** [01 Project purpose](../01_project_purpose.md), [02 Test beds](../02_test_beds.md), [03 Simplicity search principle](../03_simplicity_search_principle.md), [04.1 Hessian energy](../04.1_hessian_energy_simplicity.md), [04.2 Rotation sweep](../04.2_first_experiment_rotation_sweep.md), [04.3 First-wave metric/inference registry](../04.3_first_wave_metric_inference_candidates.md).

---

## 1. Motivation and scope

The previous experiments did not show that Hessian energy is fundamentally unsuitable. They showed that several problems are currently confounded:

- Some simplicity metrics prefer the correct analytic coordinates; others are flat or prefer wrong rotations.
- Some metrics have good analytic curves but incorrect inferred minima.
- Some inferred-only curves succeed when their analytic counterparts fail.
- The learned representations differ in their linear alignment to T and their supervised orthogonal recovery ceilings.
- The results depend on response channel, evaluation measure, fixed-bank boundary convention and possibly finite integration error.
- A local angle screen around R* does not establish the global preference of the analytic loss or the success of an actual blind search.

The immediate aim is **not** to invent another large family of sparsity scores. It is to produce a common, reproducible experiment on all twelve formal toys, and isolate which bottlenecks deserve subsequent work.

This first wave consists of six modules:

| ID | Experiment | Main scientific question |
|---|---|---|
| W1-00 | Inputs, registry, coordinate and numerical validation | Are comparisons using the same well-defined objects? |
| W1-01 | Unified metric × inference baseline | Which existing metrics succeed in each of the two classes? |
| W1-02 | Evaluation-measure and integration study | Are apparent successes/failures stable to how energy is averaged? |
| W1-03 | Representation and Oracle Bridge diagnostics | How much of the discrepancy can arise from the latent geometry and response surface? |
| W1-04 | Global analytic preference | Do locally successful analytic metrics also prefer the true axes away from R*? |
| W1-05 | Synthesis and next-wave candidate handoff | Which metric/inference questions remain after the diagnostics? |

W1-00 must finish before computing the other modules. W1-03 and W1-04 can run alongside W1-01/02 once input validation succeeds. W1-05 integrates all results. The full surrogate comparison, conditional HE, higher-order derivatives and alternative simplicity principles remain later waves. A small observed-data bridge comparator is included, but it does not replace the future Step 2 study. A full blind global search is not part of the mandatory first-wave scope.

### 1.1 Existing evidence to reproduce, not assume

The current [Lab 03](../../../metric_lab/03_s9_metric_curves/README.md) has 153 metrics evaluated on the 24 primary rotation libraries. The [class-wise angle summary](../../../metric_lab/03_s9_metric_curves/results/angle_pref_metric_by_class.csv) gives 17 analytic-only perfect and 18 analytic-only marginal-or-better metrics across each class's 12 libraries. The joint all-library pass count is zero for each class. The complex continuous subset CR3L–CR5L nevertheless has six joint-perfect and seven joint-marginal candidates.

[Lab 06](../../../metric_lab/06_wave1_screen/README.md) is a different historical Class B experiment: 262 metric variants, 17 saved binary rotation libraries, multiple measures, AD/FD and Y-L/Y-P. Its strongest Q01/AD/logit joint result is B01.M03.c0.01 at 13/17 libraries. Inferred-only B08.M03 with Q05 is a useful candidate to recheck, not evidence of analytic success or global recovery.

[Lab 05](../../../metric_lab/05_angle_pass_blind_search/README.md) already attempted actual blind search on four binary toys; no metric recovered all four. Those results are not the target of reproduction in W1-01, which is a fixed-rotation experiment.

Historical numbers are **reference checks only**. Different measures, masks and registry definitions can legitimately produce different numbers. Never adjust a new implementation to reproduce a historical count by changing its estimand.

---

## 2. W1-00 — Freeze inputs and common mathematical conventions

### 2.1 Primary toy registry

Use the formal test beds and frozen NP runs in [toys.py](../../../metric_lab/03_s9_metric_curves/toys.py). Snapshot the registry, dataset/checkpoint/latent/scaler/rotation hashes, and source commit in each run manifest.

| Family | Class B model | Class C model | latent dim k | stimulus dim p | beta B | beta C |
|---|---|---|---:|---:|---:|---:|
| C1 | C1 | C1L | 4 | 3 | .0005 | .018 |
| C2 | C2 | C2L | 4 | 4 | .0005 | .018 |
| C3 | C3 | C3L | 6 | 4 | .0005 | .006 |
| CR3 | CR3 | CR3L | 4 | 6 | .0005 | .003 |
| CR4 | CR4 | CR4L | 3 | 6 | .0005 | .001 |
| CR5 | CR5 | CR5L | 3 | 6 | .0005 | .006 |

Each toy has its own observed Y, learned Z, decoder, S8 and S8_2 reference and S9 and S9_2 sweep. All primary runs are the designated repeat01. The six additional binary libraries of Lab 03 form a secondary robustness panel, not additional toys. Use the official continuous NPs in outputs/nn, **not** the historical noisy_logit_np networks. Do not retrain any NP in the mandatory first wave.

Read-only input paths for each toy:
- outputs/models/generate/<TOY>/s1_generate__dataset.npz
- outputs/models/precheck/<TOY>/s2_precheck__report.json
- outputs/nn/<TOY>/<CONDITION>/<REPEAT>/s3_train/ (frozen checkpoint and training scalers)
- s5_extract_z/latent.npz and existing S6/S7 diagnostics
- s8_supervised_rotation/rotation.npz and s8_2_supervised_rotation/rotation.npz
- s9_rotation_sweep/ and s9_2_rotation_sweep/ manifests and rotation matrices.

Resolve exact filenames by the real loader and manifests rather than guessing. Verify subject IDs and ordering before pairing T and Z. Never rewrite source files to repair historical manifests; normalize them in the adapter and record the conversion.

### 2.2 Separate observation channels

| Class | Observed Y | Analytic response | Inferred response | Channel |
|---|---|---|---|---|
| B | Bernoulli choice | $\eta(T,X)$ | decoder class1 minus class0 logit | Y-L |
| B | Bernoulli choice | $\sigma(\eta(T,X))$ | sigmoid of decoder logit | Y-P |
| C | noisy continuous output | $\eta(T,X)$ | decoder output × training Y SD + training Y mean | Y-C |

Never differentiate individual sampled choices or noise realizations. The response surface is a deterministic conditional-response target. The six continuous toys are independent test beds, not a third channel of the binary toys.

There are 24 primary rotation libraries, with 36 library × channel cells (12 binary libraries × 2 channels + 12 continuous libraries × 1 channel). Report all 36 cells and the 24 primary libraries separately. The primary matched analytic comparison across classes is Y-L versus Y-C, not Y-P versus Y-C.

### 2.3 Coordinates and derivative transformation

Use column-vector notation. Define V as the base latent in the selected coordinate variant:

$$
V=Z \quad\text{(S9)},\qquad
V=D_Z^{-1}(Z-\mu_Z)\quad\text{(S9_2)},\qquad U=RV,\quad R^\top R=I.
$$

Use the exact saved S8_2 mean and diagonal SD, not a newly fitted scaler. Marginal standardization is not full covariance whitening. X is standardized with the saved training scaler. Use this same physical X scaling for analytic and inferred derivative evaluation within a toy.

For analytic comparison use $T_s=D_T^{-1}(T-\mu_T)$ from the saved training scaler and $U_o=Q T_s$, with $Q=G=R R_*^\top$ for the diagnostic sweep. The analytic global experiment uses independent Q, not learned R. Reconstruct raw T or Z before evaluating the original function.

Let $A$ be the base UU Hessian block in V coordinates, $b_x$ the base UX derivative vector for X coordinate x. Then

$$
H^R_{UU}=R A R^\top,\qquad H^R_{U x}=R b_x.
$$

Compute second moments in the correct base coordinates:

$$
K_{abcd}=E[A_{ab}A_{cd}],\qquad J_{ab}^{(x)}=E[b_{x,a}b_{x,b}],
$$

$$
M^{UU}_{ij}(R)=\sum_{abcd}R_{ia}R_{jb}R_{ic}R_{jd}K_{abcd},\quad
M^{UX}_{ix}(R)=\sum_{ab}R_{ia}R_{ib}J_{ab}^{(x)}.
$$

For raw derivatives apply the diagonal coordinate Jacobian before constructing these tensors, or incorporate it into the contraction, never both. Square after transformation. Rotating an already squared energy matrix is mathematically wrong.

Use float64 for reference derivatives, streaming moments and contractions. A GPU/float32 implementation may be added only after comparison with the reference. The complete-block total squared energy is invariant across orthogonal R under a fixed evaluation measure and mask. Individual diagonal shares of UU need not be invariant.

### 2.4 Rotation design

Read all saved candidates, including the unswept R*. Each subset of latent axes of size 2 through k is swept at 0,15,30,45,60,75,90 degrees using the saved ordered Givens construction. Do not replace it with a new rotation sequence.

| k | Axis subsets (2^k-k-1) | Labeled rotations including independent baseline | Evaluation 1 nominal rows (baseline + positive angles) |
|---:|---:|---:|---:|
| 3 | 4 | 29 | 25 |
| 4 | 11 | 78 | 67 |
| 6 | 57 | 400 | 343 |

Validate these counts against the actual manifests. If historical coverage differs, report missing subsets and do not label the library complete. Remove exact duplicate 0-degree rotations for pooled correlation; report both the historical labeled-positive-angle statistic and an exact-matrix-deduplicated sensitivity if other duplicates exist. Signed-permutation endpoints are distinct labeled sweep points but equivalent scientific axes.

### 2.5 Information separation

Create two separate interfaces:
1. Blind surface/metric interface: receives frozen Z, X, Y, optional decoder, observed-design metadata, scaler and candidate R only.
2. Evaluation-only interface: receives T, analytic formula, true prior, R*, parameter labels and diagnostic bridge.

The blind evaluator must return exactly the same score for the same R regardless of whether the sweep harness labels it baseline, 45-degree or random. A candidate R supplied by an evaluation harness is allowed; R* must not be an input to the score itself. Truth-controlled diagnostics are never candidate infer algorithms. The experiment scheduler may use toy IDs to load files and report results, not to select an advantageous metric, threshold or direction.

### 2.6 Required numerical acceptance tests

Before running all toys:
- Compare analytic C1/C2 derivatives to closed-form checks; test nonlinear CR4 and a fixed-bank case.
- Verify decoder output against existing predictions, including inverse Y scaling for Class C.
- Verify orthogonality and $R=G R_*$; compare contraction with direct differentiated rotated functions at randomly sampled R.
- Test prediction invariance after rotating and inverse transforming.
- Verify energy invariance, signed-permutation invariance of eligible losses, zero blocks, constant curves, tie handling, nonfinite values and X masks.
- Use synthetic nonidentifiable cases to confirm that tied directions are not falsely called strictly identified.
- Confirm blind interfaces cannot access truth or supervised references.

Proposed float64 test tolerances: orthogonality and prediction/chain-rule relative errors around 1e-7 with a scale-aware absolute tolerance; contraction and invariant checks at approximately 1e-7 relative for nonzero blocks. Tiny-energy blocks need absolute-error controls and cannot be normalized blindly. Record actual maximum errors and any necessary tolerance changes before screening.

**W1-00 gate:** all 12 designated input sets and both variants are accounted for; no silent missing library, scaler mismatch, subject-order mismatch or failed numerical control. If an artifact is unavailable, mark it missing and stop any claim of full-panel success.

---

## 3. W1-01 — Unified baseline on existing metric registries

### 3.1 Metrics to evaluate

Preserve both source registries as named, versioned libraries.

| Registry ID | Origin | Contents |
|---|---|---|
| L03 | Lab 03 | all 153 existing metrics, including candidate fusion and Haar-standardized fusion |
| L06 | Lab 06 / 04.3 | all 262 metric variants and controls, including M01–M13, B01–B12 |

The nominal registry total is 415 entries before mathematical deduplication; controls and undefined metrics are not successful candidates. A name match does not establish equality. Maintain a formula/normalization/parameter mapping table with source code hash. Freeze the registry before running. In particular, do not equate the L03 epsilon setting with L06 c without checking the normalization and scale.

Core L06 example, B01.M03:
$$
B=M^{UX},\quad v_{ix}=B_{ix}/\sum_{ja}B_{ja},\quad
L=\frac1{kp}\sum_{ix}\frac{v_{ix}}{v_{ix}+c/(kp)}.
$$

Row-weighted B08.M03:
$$
r_i=\sum_xB_{ix},\quad q_{ix}=B_{ix}/r_i,\quad
L=\sum_{i:r_i>0}\frac{r_i}{\sum_jr_j}
\left[\frac1p\sum_x\frac{q_{ix}}{q_{ix}+c/p}\right].
$$

The two are not interchangeable. B08 can be blind to splitting a single-stimulus loading across multiple axes. For a truly zero block use the predeclared whole-landscape policy; record the threshold and distinguish near-zero numerical values from structural zeros. Joint metrics must not drop blocks at individual rotations. Legacy recipe behavior, including any per-rotation active-row rule, is preserved in the compatibility run and audited in the harmonized run.

### 3.2 Inference and measures

Main arm:
- Oracle: AD of the analytic formula, never sampled Y.
- I01/AD: frozen decoder differentiated exactly.
- Q01: equal-subject empirical observed-trial average with common T/Z subject IDs and X rows.
- All valid channels and both coordinate variants.
- All metrics in both registries.

For random and bank designs alike, Q01 samples 256 distinct subjects and 32 observed trials per subject at pilot fidelity (seed 31). All subjects here means the available evaluation pool selected by the manifest, not that every subject must be differentiated in every pilot. Use TEST subjects as the primary evaluation pool; a historical all-subject protocol is a separately labeled reproduction. The finite sampled average is a Monte Carlo estimate of the full equal-subject target.

FD diagnostic: compare I02 with step .01 and .03 of fixed base-coordinate SD and I03 gradient FD at .03 against AD on a small common set of points for every toy and channel. Float64 .003 check if disagreement; .1 is a historical sensitivity. Do not sweep FD × every metric × every measure when the derivative comparison demonstrates equivalence. Report boundary validity for perturbations. FD computes the decoder derivative, not an independent estimate of the true response function.

### 3.3 Mask and compatibility arms

The primary random-X measure uses its valid domain. For fixed banks retain two labeled estimands:
- MASK-ALL: common all-interior rows for the entire energy matrix (comparable to Lab 03).
- MASK-COL: UU over its valid rows, UX column x over the points valid for that column (comparable to Lab 06).

Use identical masks for analytic and decoder arms and all latent axes, frozen before rotation. The full-design derivative target is only reported if its validity is established; otherwise mark it unavailable. CR3's power at zero and CR5's alpha=1 extension have distinct mathematical properties, so do not assert that every zero amount necessarily has a singular derivative. Masked estimates use the corresponding conditional denominator, not zero-filled unmasked averages. In the fixed bank, derivatives of a continuous extension are not identified by observations at the finite rows alone.

Retain a historical reproduction option for selected legacy configurations (Lab 03: 4096 paired points and all-interior mask; Lab 06: its historical Q01/column mask). The purpose is to verify exact formulas/loaders and isolate methodological changes, not force equal results between different protocols. Use saved seeds, original implementation and inputs where available. Any missing historic arrays or incompatible version must be reported.

### 3.4 Required output

For every metric × library × channel × measure × source save the loss at every rotation, raw energy/moment arrays, finite/undefined status, block zero flags, mask coverage and runtime. Report by toy and class; do not pool 36 channel cells as 36 independent toys. Summarize analytic-only, inferred-only and both-pass separately.

---

## 4. W1-02 — Measure and integration convergence

### 4.1 Registered measures

Distinguish Q, a measure, from a derivative estimator I. For all Q use complete X rows, never independently resample columns of a bank.

| Measure | Inferred arm | Analytic comparison | Scope |
|---|---|---|---|
| Q01 | empirical subject V × that subject's observed trials | paired actual T and the same trials | all toys |
| Q02 | empirical V × empirical joint X rows | paired actual T × the same X rows | all toys; full bank where applicable |
| Q03 | fitted joint Gaussian V × empirical joint X | independent true T draws × the same X rows | all toys |
| Q04 | fitted joint Gaussian V × all 500 fixed rows | true T draws × identical bank | only declared fixed-bank designs |
| Q05 | independent marginal Gaussian approximation in fixed base V frame × empirical X/full bank | true T prior for primary comparison; legacy independent Gaussian T only as separate valid-domain sensitivity | all toys |

Q02 is a product evaluation distribution, distinct from the observed joint subject/trial distribution. Q03–Q05 may compare different latent distributions and are sensitivity tests, not pure pointwise derivative-error tests. In the blind arm fit Gaussian parameters to observed Z/V and use empirical or known-design X only. For analytic prior integration use the actual supported parameter prior. If explicitly reproducing the legacy Gaussian T sensitivity, quantify out-of-domain draws and mark invalid; do not silently clip or interpret it as the true prior. Gaussian Z extrapolation is also monitored (marginal quantiles, Mahalanobis distance and coverage relative to empirical latent data).

Always sample in the base frame once; for each candidate R the distribution is the pushforward of the same samples. Do not choose an independent diagonal approximation separately after rotating.

### 4.2 Priority panel for detailed sensitivities

To control runtime, reuse cached moments and use all cheap scalar metrics whenever available. The priority panel for expensive replication, bridge and global checks contains:

1. All 17 Lab 03 all-library analytic-perfect metrics, and the extra marginal Fz.A+B+C+G.
2. The six complex continuous joint-perfect metrics and the extra marginal rowsp.a0.25.q0.1.
3. Inferred leaders rowsp.a1.0.q0.1, UXrow.a1.lq.25, CAND.rowsp.75+UXcol.hoy.
4. L06 B01.M03 at c=.01/.03, B01.M04 at c=.01, B08.M03 at c=.01/.03, B09.w75.M03.c0.01; include all L06 Q01 analytic-perfect candidates selected by the deterministic W1-01 rule.
5. Whole-block total energies, an undefined/constant control and a known weak UU-only metric.

Deduplicate exact registry IDs, not scientific families. The fixed historical analytic-perfect list is:
POOL.lq.1, POOL.lq.25, POOLrms.ent, POOLrms.lq.5, SL0.UX.e0.01/.03/.05, SL0.UXUU.e0.01/.03/.05, SL0m.UX, SL0m.UX.wide, SL0m.UXUU, SL0m.UXUU.wide, UX.lq.1, UXcol.a0.lq.25, UXrms.lq.5.

New W1-01 analytic-perfect candidates are an explicitly labeled development selection, not independent confirmation. Record the entire selection table before W1-02/03/04 results are examined.

### 4.3 Staged execution grid

| Batch | Libraries/channels | Measures | Fidelity | Seeds | Purpose |
|---|---|---|---|---|---|
| A | all primary | Q01 | 256 subjects × 32 trials | 31 | full baseline, all registries |
| B | all primary | Q02–Q05 where valid | 256×32; 32×500 for full bank | 31 | measure sensitivity, AD |
| C | all primary for priority panel | Q01–Q05 valid | 512×64; 128×500 for full bank | 31,73,101 | convergence and repeatability |
| D | fragile/leading configs by a frozen escalation rule | relevant measures | 1024×128; 512×500 for full bank | 31,73,101 | confirm integration stability |
| E | secondary saved libraries | Q01 plus promising registered measures | at least Batch C fidelity | same seeds | robustness, not extra primary toys |

For full bank Q02 use empirical latent draws; Q04 and bank Q05 use fitted latent distributions. If the TEST pool cannot support the requested distinct-subject size, use the largest supported subset or a predeclared with-replacement scheme, record effective sample size and label the change. Prefer nested samples within a seed when comparing fidelities, independent seeds for variability.

For Batch D use the following rule: escalate every priority candidate with a changed pass status across seeds/fidelities, a near-tie margin, or a proposed all-family pass. Also retain one failing control per metric family. Do not escalate only winners. A suggested stability criterion is same all-sweep classification across the three seeds at the last two fidelities and comparable minimum margin; report the actual range rather than interpreting stability as population truth.

### 4.4 Quantify uncertainty correctly

Report per-seed losses, rank/minimum-angle changes and variation in normalized energies. Bootstrap at subject level for empirical measures; preserve trial grouping. For the fixed bank interpreted as a census, resample subjects/latents and do not pretend bank rows are an iid random sample from an unknown population. Integration resampling of a frozen decoder does not measure NP-training uncertainty. Additional NP repeats are the distinct source of model-fit variation.

---

## 5. W1-03 — Representation, bridge and controlled surface interpolation

### 5.1 Representation diagnostics

For every designated run and variant compute on the prescribed TRAIN/TEST split:
- latent means, SDs, covariance, eigenvalues and condition number;
- per-parameter held-out linear OLS R² using full V, with intercept;
- residual covariance and singular values of the fitted V→T_s map;
- supervised S8/S8_2 matched correlations, MCC, min matched correlation, mean/max leakage, and unrotated values;
- raw vs marginally standardized coordinate geometry.

Fit the map on TRAIN subjects only; report residual and prediction errors on TEST. This is evaluation-only. A high full-V R² does not establish an orthogonal one-to-one map. The supervised Procrustes solution is a reference, not a guarantee of the best MCC among all rotations.

### 5.2 Four surfaces

For each toy and valid response channel define:

| Arm | Function | Inputs / role |
|---|---|---|
| O | true response $g(T,X)$ | analytic evaluation only |
| B-orth | $g(\mu_T+D_T R_* V_c,X)$ | supervised orthogonal bridge; V_c centered by the saved training convention |
| B-aff | $g(\mu_T+D_T(AV+b),X)$ | supervised affine bridge, TRAIN fit of standardized T on V |
| D | frozen decoder $d(V,X)$ | allowed-data surface |

B-orth must reconstruct the actual S8 alignment convention including intercept/centering, verified against saved artifacts; the displayed formula is schematic until those conventions are read. If R* is computed using a centered version of V, use that exact center. B-aff predicts standardized T, then maps back to raw parameters. The two bridges are different diagnostic targets. Neither is automatically the conditional expectation of the true response given V.

Use paired empirical TEST subject V/T and the same X rows, Q01 primary. Reuse the exact latent normalization and saved R candidates. For affine map from V to raw T with Jacobian A_raw:
$$
H_{VV}^{bridge}=A_{raw}^\top H_{TT}A_{raw},\quad
H_{VX}^{bridge}=A_{raw}^\top H_{TX}.
$$
Compare the bridge and decoder in the same V coordinate system; compare to analytic standardized-T arm using the labeled G. Include both analytic/inferred loss differences and matrix-level differences. For supported CPT domains report predicted-parameter violations by parameter and jointly. Primary comparison uses a fixed common valid subset with coverage reported; if coverage is inadequate, mark the bridge result invalid. Never clip to make the bridge successful. A restricted/clipped map, if separately explored, changes derivatives and is not the affine bridge.

### 5.3 Bridge versus decoder interpolation

For each compatible channel and fixed common points:
$$
g_a=(1-a)g_{bridge}+a\,d,\qquad a\in\{0,.25,.5,.75,1\}.
$$
Use both bridge variants where valid. In Y-P mix probabilities; do not accidentally mix a probability with a logit. For Y-L/Y-C mix quantities in the same original logit units.

The elementwise energy satisfies:
$$
M_a=(1-a)^2M_b+a^2M_d+2a(1-a)E[H_b\odot H_d].
$$
Cache cross derivative moments in the base V coordinates so all R and a can be evaluated without recomputing derivatives. Cross terms are signed; do not drop or square them individually.

For each priority metric record:
- loss landscape versus a;
- first tested a at which a flat/wrong minimum becomes an accepted minimum, or accepted becomes wrong;
- margin and minimum-set changes, not just Pearson;
- near-zero energy entries that gain or lose support;
- which changes reproduce across S9/S9_2, both classes and secondary runs.

This is a truth-assisted explanatory experiment, not a candidate search objective. Differences between O, B-orth, B-aff and D are not a formal additive error decomposition.

### 5.4 Minimal T-free conditional-response comparator

Add one limited observed-data surrogate arm S using only Z, X, Y, to assess whether a different estimator changes the same landscape. This is a diagnostic pilot, not the full later improved-inference registry.

Use standardized inputs and complete total-degree 2 and 3 polynomial ridge for Class C (MSE), and polynomial logistic ridge for Class B (Bernoulli likelihood). Select regularization from a fixed grid e.g. 1e-4, 1e-2, 1 using subject-held-out predictive loss, with the precise objective normalization stated. Fit in the base coordinate frame once, evaluate the same fitted surface at all rotations. Derive Y-L and Y-P from a common fitted binary probability model. Keep Y-C in original units. Do not use T, f, R*, oracle curves or test performance to select degree or penalty.

Proposed fit sampling: up to 1024 TRAIN subjects × 64 trials, validation up to 256 VALIDATION subjects × 64 trials; same fixed seeds and subject weighting, use full rows. Increase only by a predeclared convergence rule. Record unique subjects, design rank, regularization, predictive performance, derivative stability and compute time.

Full-context Z may include the target Y. In that setting the fitted response is conditional on that representation, not necessarily prediction on an independent query. If compatible encoder access permits, run an optional disjoint-context/query diagnostic (context and query trials disjoint, same context fraction across comparable toys, T-free encoder); label it a different Z and do not compare it as though it were the original S5 representation. This is why a discrepancy between the bridge and the best conditional response may persist even with excellent predictive fit.

### 5.5 Interpretations to test

| Result pattern | Diagnostic interpretation, not proof |
|---|---|
| O good, B-orth good, B-aff good, D bad | decoder/conditional surface or approximation error becomes a priority |
| O good, B-orth bad | alignment, scaling, orientation or representation geometry may matter |
| O good, B-aff bad | affine map geometry, residual structure or invalid parameter support may matter |
| O flat, bridge/decoder nonflat with accepted minima | investigate symmetry breaking and whether it is robust |
| O wrong minimum, D accepted | potential error compensation; not a Gap 1/2 pass |
| D bad, S better | surrogate route is promising, subject to independent confirmation |
| D and S both bad with weak representation ceiling | more inference alone may not recover information absent from V |

Save raw curves and energy comparisons; a correlation cannot isolate the mechanism. Report unresolved explanations explicitly.

---

## 6. W1-04 — Global analytic preference beyond the rotation sweep

### 6.1 Question and candidate set

For the 17 historical Lab 03 analytic-perfect metrics, additional W1-01 analytic-perfect candidates and near-pass controls, test whether the true-coordinate minimum is global on O(k), up to signs/permutations.

This experiment is **analytic only**. It uses true f and T and is explicitly not Gap 3. It will prevent spending the next wave on improving inference of metrics whose ideal objective already has an incorrect global optimum.

Primary Q01 and valid mask; secondary Q02/Q04 for bank designs and supported true-prior integrations where relevant. Use a fixed, sufficiently converged measure throughout each search. Y-L and Y-C analytic targets can share computations only when their actual physical evaluation points, scalers and masks are identical; report the two toy identities separately. Test Y-P candidates independently.

### 6.2 Global diagnostic definition

Let Q rotate standardized T, and let P range over signed permutation matrices. Define
$$
L_0=L(I),\qquad
d_{SP}(Q)=\min_P\|Q-P\|_F.
$$

Compute the nearest signed permutation with a Hungarian assignment maximizing $\sum_i |Q_{i,\pi(i)}|$. For an orthogonal Q,
$$
d_{SP}(Q)^2=2k-2\max_\pi\sum_i|Q_{i,\pi(i)}|.
$$

Report normalized distance $d_{SP}/\sqrt{2k}$ and absolute/scale-normalized loss differences $L(Q)-L_0$. Evaluate multiple signed-permutation endpoints to check invariance before interpreting global comparisons.

Candidate generation:
- Include I and random signed permutations as controls; the search starts themselves are independent Haar rotations.
- Sample Haar Q via Gaussian QR with sign correction; evaluate at least 400 initial draws for k=3/4 and 2000 for k=6, with seeds 31,73,101 as proposed budget.
- Retain best 8 or 30 distinct starts respectively for refinement using Givens-coordinate descent or the existing Lab 05-style derivative-free optimizer, with fixed maximum evaluations and convergence settings. Smooth scores may additionally use manifold gradient refinement.
- Both determinant components of O(k) are represented by the random draws; when sign invariance is verified, SO(k) search alone is equivalent up to a sign flip. Do not assume that invariance without tests.
- Save every start, initial/final loss, final Q, convergence status, signed-permutation distance and cost. Do not choose a minimum based on proximity to I.
- Re-evaluate suspicious minima and identity on independent higher-fidelity samples and multiple seeds to distinguish a finite-integration accident from a reproducible counterexample.

Use a predeclared loss tolerance (relative 1e-9 for historical numerical tie flags, plus empirical integration variability for substantive equivalence) and report distance continuously. A provisional near-permutation distance threshold such as 1e-3 is a numerical diagnostic, not a new definition of scientific success. Show sensitivity to larger tolerance. For nonsmooth threshold scores record plateaus and equally good rotations, not merely one optimizer solution.

### 6.3 Result categories

| Category | Evidence |
|---|---|
| Empirical global support | no reproducible lower or tied non-permutation solution found, adequate search coverage and invariance |
| Global counterexample | a non-permutation Q with reproducibly lower loss than signed permutations |
| Nonunique minimum | substantively tied non-permutation rotations |
| Inconclusive search | inadequate budget/convergence, unstable integrals, invalid/constant loss |

Even the first category is not a global proof. A global counterexample is sufficient to refute the stronger unique-minimum claim for that metric and measure, subject to numerical verification. This criterion concerns analytic parameters T, not recovery of learned Z.

---

## 7. Evaluation, reporting and candidate selection (W1-05)

### 7.1 Preserve both evaluations of 04.2

Evaluation 1: pooled Pearson and Spearman over unique rotation candidates and mean per-sweep Pearson. Record denominator, fraction undefined and dynamic ranges; constant correlations are NA. Also report scale-normalized loss differences, matrix relative errors and oracle-zero false-positive energy (evaluation-only). A high correlation can coexist with the wrong minimum.

Evaluation 2: for each seven-angle curve, let $S=\{\theta:L(\theta)\leq L_{\min}+\tau\}$ with a fixed tiny relative tolerance. Preserve legacy flags:
- perfect: S intersects {0,90};
- marginal-or-better: S intersects {0,15,75,90};
- nonfinite or constant curve: fail.

Audit actual historical tolerance by registry: Lab 03 uses 1e-9 times (max absolute loss + 1e-12); Lab 06 uses its separately documented relative scale. Use a common tolerance for the harmonized report, and do not silently replace the legacy flags.

Add strict flags requiring S to be a nonempty subset of the accepted angles. Report interior ties, endpoint equality, range and normalized margin
$$
m=\frac{\min_{\theta\in\{15,30,45,60,75\}}L(\theta)-\min_{\theta\in\{0,90\}}L(\theta)}
{\max_\theta L(\theta)-\min_\theta L(\theta)}.
$$
Compute only for finite nonconstant curves. Positive m is a strict endpoint advantage; negative m is an interior preference. Do not claim that a marginal pass is perfect. A library passes only if **every** sweep passes, with complete axis-subset coverage. For combined success require both analytic and inferred flags.

Produce separately:
- analytic-only, inferred-only, both-perfect/both-marginal, strict and legacy flags;
- per toy, per variant, per channel, per class, and common B/C matched-logit/continuous candidates;
- four-way table O+/I+, O+/I−, O−/I+, O−/I−, with constant and wrong-minimum failures separated;
- global analytic category for W1-04; no actual blind-search success claim in this wave.

### 7.2 Next-wave handoff rules

Freeze the output shortlist using the following lanes, rather than selecting a single best-looking metric on each toy.

| Lane | Required evidence | Next research task |
|---|---|---|
| A: Theory candidate | all-class or cross-class local analytic success, stable measures; global diagnostic supports or remains explicitly inconclusive | Step 2 improved inference |
| B: Metric repair | analytic flatness, interior preference or global counterexample | Step 3 conditional HE / Step 4 richer structure / Step 5 alternatives |
| C: Empirical candidate | inferred-only robust passes, including cases where analytic fails | separate early Step 6 blind pilot, replication and mechanism study |
| D: Representation-limited | poor supervised geometry or bridge evidence | coordinate/representation diagnostics, not metric tuning |
| E: Estimand/measurement-limited | unstable integration, boundary issues, invalid mask or off-support extrapolation | revise measure/support or consider derivative-free stimulus contrasts |

Lanes can overlap. Within each lane rank by coverage, strict margins, seed stability, and cost; report all metrics tied at the cutoff. No per-toy method selection. No tuning of a threshold, mixture or estimator on T during an operational method. Research selection across current toy truth is allowed for development but requires later frozen independent confirmation.

A successful first wave is a reliable diagnosis even if it produces no universal metric. Do not manufacture a pass by relaxing criteria, dropping failed toys, treating invalid as zero, or pooling the two observation classes.

---

## 8. Implementation and output contract

### 8.1 New independent experiment folder (proposed)

Do not modify S1–S9_2, Lab 01–06, checkpoints or historical results. Reuse their read-only adapters and mathematically validated metric functions where possible. This is the proposed code structure, not a statement that the files already exist.

    metric_lab/07_unified_benchmark/
      README.md
      configs/
        toy_registry.json
        metric_registry.json
        experiment_registry.json
      wave4_2/
        inputs.py          # frozen data/checkpoint/scaler/rotation adapters
        surfaces.py        # class-specific decoder and evaluation-only analytic adapters
        measures.py        # Q01–Q05 and masks
        derivatives.py     # AD/FD and base-coordinate chain rule
        moments.py         # UU, UX and bridge cross moments
        metrics.py         # versioned adapters to L03/L06
        evaluation.py      # Eval 1/2, legacy and strict
        bridge.py          # T-enabled evaluation-only diagnostics
        global_oracle.py   # analytic Q search; no claim of blind recovery
      tests/
      validate.py
      run_baseline.py
      run_measures.py
      run_bridge.py
      run_global_oracle.py
      analyze.py
      run_all.sh
      results/

Keep truth and blind interfaces separate in code. A shared loader may orchestrate both arms, but never pass an object containing T/f/R* to an inferred scoring function. Any module named global_oracle is restricted to the diagnostic experiment; later blind search uses a separate entry point.

Suggested CLI contract (illustrative; scripts must be implemented before these commands are executable):

    python metric_lab/07_unified_benchmark/validate.py --toys all
    python metric_lab/07_unified_benchmark/run_baseline.py --toys all --variant both --batch A
    python metric_lab/07_unified_benchmark/run_measures.py --toys all --batch B
    python metric_lab/07_unified_benchmark/run_measures.py --toys all --batch C
    python metric_lab/07_unified_benchmark/run_bridge.py --toys all
    python metric_lab/07_unified_benchmark/run_global_oracle.py --toys all
    python metric_lab/07_unified_benchmark/analyze.py --panel primary

Support --toy <ID>, --seed, --channel, --measure, --mask, --config and --resume. Default no overwrite; each run should have immutable config/data/checkpoint hashes. A quick mode may use C1/C3/CR4 and their L variants for smoke testing only, never as evidence for full coverage. Implement CPU float64 reference and optionally chunked GPU. Stream derivatives; cache moments rather than retaining every Hessian if unnecessary.

### 8.2 Saved outputs

Large arrays:

    outputs/metric_lab/07_unified_benchmark/<run_id>/
      run_manifest.json
      <toy>/<condition>/<repeat>/
        points/<measure>/<seed>/...
        tensors/<channel>/<source>/<measure>/<mask>/<seed>/...
        s9/...
        s9_2/...
        bridge/...
        global_oracle/...

Small results and analysis files under the lab results directory:

| File | Required contents |
|---|---|
| input_audit.csv | each toy/run/variant, hashes, scaler and subject alignment, missing files, validation errors |
| registry_mapping.csv | original metric names, formulas, normalization, controls, equivalence/duplicate status |
| coverage.csv | planned and observed configs, mask coverage and missing/invalid reasons |
| metrics_long.csv.gz | loss per rotation, registry version, source, channel, Q, mask, fidelity, seed, timing |
| correlations.csv | Eval 1 and energy error summaries |
| sweep_minima.csv | complete minimum set, legacy/strict pass, range, ties and margin |
| pass_summary.csv | toy/class/variant/channel and separate O/I/both pass flags |
| integration_diagnostics.csv | differences across seeds/fidelities and stability |
| representation_diagnostics.csv | linear R², singular values, MCC and leakage, evaluation-only |
| bridge_curves.csv.gz | O, B-orth, B-aff, D, S and interpolation arm results |
| bridge_energy_diagnostics.csv | support violations, paired masks, weak-entry and cross-term comparisons |
| global_oracle_search.csv.gz | starts, final loss, identity loss, distance, budgets and confirmation |
| next_wave_candidates.csv | lane assignment, reason, evidence and unresolved limitations |

Figures: per-toy paired curves with oracle/decoder/bridge, per-class pass heatmaps, minimum-angle confusion plots, integration convergence and global loss-versus-distance scatter. Every plotted value should be reproducible from saved tables; no chart-only results.

Mandatory manifest fields: toy ID, class, observation channel, NP condition/repeat/beta, coordinate variant, exact dataset/checkpoint/latent/scaler hashes, metric registry/code version, estimator, measure, mask, fidelity, seed, evaluation split, sample counts, runtime and validity status. Save random starts and any fit coefficients needed to reproduce results. Include software/device/dtype. Report attempted configurations including failures, not just successful settings.

### 8.3 Implementation acceptance checklist

- [ ] All twelve official toys × two variants are covered, with valid observation channels only.
- [ ] Both historical metric registries loaded and versioned, with no silent formula changes.
- [ ] All baseline oracle/decoder arms use correctly paired evaluation designs and explicit masks.
- [ ] Rotation/derivative/tensor invariance tests pass, and failures are recorded.
- [ ] Legacy and strict evaluations are separately reproducible; constant and invalid curves cannot pass.
- [ ] Q01–Q05 convergence report distinguishes integration uncertainty from NP uncertainty.
- [ ] Bridge diagnostics use only TRAIN T for fitting, TEST T for evaluation, and explicit support checks.
- [ ] Global analytic diagnostics include loss and signed-permutation distance, with independent re-evaluation of counterexamples.
- [ ] Final report separates both classes, analytic-only/inferred-only/joint successes, and does not claim a local pass proves global recovery.
- [ ] A reproducible next-wave candidate registry and failure table are delivered without overwriting earlier outputs.

---

## 9. What follows this first wave?

If existing analytic candidates survive the global diagnostic, prioritize improved inference of those same metrics (Step 2 of the overview): polynomial/PCE/KRR, smooth distillation, cross-fitting and uncertainty-aware support estimation.

If the analytic loss itself lacks discrimination, prioritize the conditional aggregation and derivative structure experiments in Steps 3–4. If fixed-bank stimulus derivatives are the key limitation, prioritize the observed-stimulus contrasts/reduced-axis branch of Step 5. If inferred-only candidates remain strong, independently lock and test them in the early blind-search pilot of Step 6, without relabeling them as faithful oracle inference.

The first wave is complete when the evidence and implementation contract above are satisfied, **not** when some selected metric happens to pass every toy. It is the diagnostic foundation of wave4-2.

**Current status:** proposed experimental specification only. No new experiment has been executed, no new code directory has been created by this document, and no reported historical result has been changed.
