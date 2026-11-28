# Wave 4-2 / 04 — Fourth wave: verify the Q05 breakthrough, explain the integration measure, and confirm a fixed blind algorithm

**Status:** proposed implementation protocol, 2026-11-27. No new experiment is claimed to have been executed.  
**Project definitions:** [01 Purpose](../01_project_purpose.md), [02 Test beds](../02_test_beds.md), [03 Simplicity-based Search Principle](../03_simplicity_search_principle.md), [04.1 Hessian energy](../04.1_hessian_energy_simplicity.md), [04.2 Evaluation](../04.2_first_experiment_rotation_sweep.md).  
**Earlier plans:** [总览](总览.md), [01](01_first_wave_README.md), [02](02_second_wave_README.md), [03](03_third_wave_README.md).  
**Results and implementations:** [Lab 07](../../../metric_lab/07_unified_benchmark/README.md), [Lab 08](../../../metric_lab/08_wave4_2_second_wave/README.md), [Lab 09](../../../metric_lab/09_wave4_2_third_wave/README.md), [W3 report tables](../../../metric_lab/09_wave4_2_third_wave/results/report_tables.md), [W3 row metric](../../../metric_lab/09_wave4_2_third_wave/wave3/rowmetrics.py), [blind design implementation](../../../metric_lab/08_wave4_2_second_wave/wave2/designs.py). The source bundle 09_wave4_2_third_wave_key.zip contains the original FIN-SENS recovery rows and blind matrices; use local outputs if these large files are not committed.

---

# Part I. Why this wave is deliberately narrow

## 1. Evidence from Wave 3

The purpose is still to recover individual-difference axes from behavioral X,Y without knowing T or f. The metric is an instrument for that purpose. Do not continue increasing the size of the metric library merely because more methods can be invented.

Wave 3 established a compact candidate, RW-GL-l0.75-a1.0, using only the UX block of the frozen decoder's Hessian. It retains the B08 row-energy weighting and adds a B01 global sparsity constraint. Both CR3 and CR5 binary failures of the previous best method are corrected in the primary Q01 search. CR4 is the remaining Q01 failure in the raw-coordinate variant. The prior broad tests of gradient, contrasts and reduced-axis objectives did not yield a competitive method. The disjoint-context experiments improved Stage A as context size increased but did not establish that context alone resolves Gap 2. Third derivatives were not tested and are not required in Wave 4.

The important new observation comes from independently counting rows in the uploaded W3 result file, not from relabeling the W3 headline:

| Candidate | Frame | Measure | Integration seeds × search seeds | All-run toy pass |
|---|---|---|---|---|
| E02 / AD | S9_2 | Q01 | 3 × 3 | 10/12 |
| RW-GL / AD | S9 | Q01 | 3 × 3 | 11/12 |
| RW-GL / AD | S9_2 | Q01 | 3 × 3 | 10/12 |
| RW-GL / AD | S9 | Q02 | 1 × 3 | 11/12 |
| **RW-GL / AD** | **S9** | **Q05** | **1 × 3** | **12/12** |

The Q05 results are a post-hoc discovery in the FIN-SENS sensitivity arm. All 36 runs pass the unchanged Lab 05 operational criterion. The integration seed is 31 and search seeds are 31,73,101. Three searches on the same numerical objective are not three independent integration designs, new datasets or NP trainings. The W3 Q01 algorithm lock is still the official frozen W3 output. Do not modify it or claim Q05 has already completed the W3 confirmation protocol.

### 1.1 Exact replay targets for the promising Q05 arm

From FIN-SENS, S9, RW-GL-l0.75-a1.0, Q05; medians over three search seeds (rounded here only):

| Toy | MCC | Max leakage | Pass |
|---|---:|---:|---:|
| C1 | .9730 | .0445 | 3/3 |
| C2 | .9767 | .0459 | 3/3 |
| C3 | .9705 | .0722 | 3/3 |
| CR3 | .8633 | .1841 | 3/3 |
| CR4 | .8443 | .2462 | 3/3 |
| CR5 | .8778 | .1918 | 3/3 |
| C1L | .9976 | .0368 | 3/3 |
| C2L | .9965 | .0409 | 3/3 |
| C3L | .9942 | .0584 | 3/3 |
| CR3L | .9574 | .1696 | 3/3 |
| CR4L | .9090 | .2428 | 3/3 |
| CR5L | .9524 | .1646 | 3/3 |

Random designs use FULL; known fixed-bank designs use MASK-ALL. Q02 also passes every toy except CR4; CR4 MCC .7875 and leakage .3978 under Q02. These comparisons make the latent integration distribution a promising explanatory factor, but they do not isolate it yet: existing Q03/Q05 use distinct RNG streams and different X draws, and Q01 uses paired empirical trials. W4-02 will control those differences.

## 2. Modules and priority

| ID | Task | Primary question |
|---|---|---|
| W4-00 | exact replay and numerical firewall | Is the 36/36 observation reproducible? |
| W4-01 | multi-seed Q05 blind search and independent high-fidelity checks | Does a fixed method actually pass 12/12 robustly? |
| W4-02 | controlled integration-measure diagnosis | Why does Q05 change CR4, and is it robust to covariance/tails? |
| W4-03 | exact/practical Gap 1 and both Gap 2 evaluations under matched measures | Does the successful algorithm have a defensible theoretical explanation? |
| W4-04 | fixed robustness controls on saved NPs, coordinate variants and masks | Is success specific to one representation or sampling convention? |
| W4-05 | immutable lock and fresh-data / fresh-NP confirmation | Does the frozen algorithm generalize beyond the development datasets? |
| W4-06 | novel families, scope of claims and synthesis | Does the principle extend beyond the six development functions? |

W4-00 is the shared gate. W4-01 takes first priority. W4-02/03/04 can run in parallel using the same caches. Do not postpone the report until all costly confirmation jobs finish. W4-05 must be downstream of an algorithm lock. W4-06 requires a separately reviewed novel-family registry before generating or evaluating data. This is not a mandate to invent dozens of new metrics if Q05 fails. Diagnose the failed fixed configuration first.

---

# Part II. Fixed algorithm and information protocol

## 3. Exact mathematical definition

Use the actual W3 implementation, not a new approximation to it. For V=Z in the primary S9 frame, U=RV, R^T R=I, and the frozen deterministic decoder response g(v,x), define

$$
g_R(u,x)=g(R^\top u,x),\qquad
B_{ij}(R)=E_{(V,X)\sim\mathcal Q}[(\partial_{u_i x_j}^2g_R(RV,X))^2].
$$

Class B: g is class-1 minus class-0 logit. Class C: g is the continuous decoder output transformed back to its original Y units. X uses the saved TRAIN input scaler. Probability is a separate optional estimand. Never differentiate sampled Y.

Let r_i=sum_j B_ij, E=sum_i r_i, q_ij=B_ij/r_i on positive rows and v_ij=B_ij/E. With k latent dimensions and p stimulus dimensions, c=.03:

$$
s_i=\frac1p\sum_j\frac{q_{ij}}{q_{ij}+c/p},\qquad
L_{\rm row}=\sum_{i\in A}\frac{r_i}{\sum_{\ell\in A}r_\ell}s_i,
$$

$$
L_{\rm global}=\frac1{kp}\sum_{ij}\frac{v_{ij}}{v_{ij}+c/(kp)},
\qquad L_{\rm GL}=.75L_{\rm row}+.25L_{\rm global}.
$$

Historical rowmetrics.py uses TOL=1e-12; A consists of rows with r_i > 1e-15 E, active weights renormalized; E <= TOL or nonfinite E produces NaN. Reuse exact implementation and verify against archived scores. Nonnegative energy, finite values, active mask and zero-block flags must be checked before scoring. Any new numerical convention is a separate version. Do not vary c, lambda or the exponent a=1 during this wave's primary experiment.

Compute moments once in the base frame:

$$
h_j(v,x)=\partial^2_{v x_j}g(v,x),\quad J^{(j)}=E[h_jh_j^\top],
\qquad B_{ij}(R)=e_i^\top RJ^{(j)}R^\top e_i.
$$

Here e_i denotes a unit coordinate vector, not a normalized row energy. AD calculates the frozen decoder's derivative exactly up to numerical precision, not the unknown generating derivative. Moment contraction makes thousands of objective evaluations inexpensive; there is no need to differentiate the decoder anew at each rotation. Only UX is needed for the primary metric. Reuse W2/W3 float64 checks.

## 4. Q05 is a reference measure, not a claim that Z is exactly Gaussian

Read the existing designs.py carefully. In the raw Z frame, for the selected subject pool P (TRAIN for objective),

$$
\mu_i=|P|^{-1}\sum_{s\in P}z_{si},\qquad
\sigma_i^2=|P|^{-1}\sum_{s\in P}(z_{si}-\mu_i)^2,
$$

$$
Z^\circ=\mu+D_\sigma\epsilon,\qquad \epsilon\sim N(0,I_k).
$$

This is the independent marginal Gaussian fitted to the entire eligible pool (population SD, ddof=0), not a Gaussian fitted to just the 512 sampled latent points. Under Q05 use an independent empirical X design: random-design toys sample complete observed X rows from that pool; fixed-bank toys use all complete bank rows for every sampled latent draw, then the prescribed derivative-valid mask. Do not independently sample stimulus columns.

The existing Q05 design uses RNG [seed,3,5] for latent normal draws, [seed,4,5] for empirical X rows, and MAXDRAW=4096. Preserve these for historical replay. With medium C the design sizes are ns=512, nt=64 for random/observed rows, nb=128 for the bank. Thus 32,768 point pairs in a random design and 64,000 pre-mask bank pairs. High D uses 1024×128 and 512×500 respectively. Save actual retained pair counts and valid weights. The number of distinct Gaussian latents is not the number of subjects and not the number of independent derivative pairs.

For a given cache draw all latent and stimulus points once. For every candidate R the evaluation distribution is the pushforward of those SAME base points, not a new diagonal Gaussian fitted after rotation:

$$
U^\circ=RZ^\circ,\quad U^\circ\sim N(R\mu,RD_\sigma^2R^\top).
$$

This matters: enforcing independent U at each R would change the evaluation distribution with R. Do not do it.

Q05 integrates a reference distribution inferred from observed Z; it does not require particular observed subjects as evaluation positions after moments have been fitted. It still requires learned Z and empirical stimuli to define that reference. It is a different functional from Q01. The observed decoder might be unreliable outside its training support. A success with Q05 does not prove the normality or independence assumptions are literally true.

### 4.1 Other measures

| ID | Base latent law | Stimulus law |
|---|---|---|
| Q01 | empirical subject | that subject's observed trial rows |
| Q02 | empirical subject | independent empirical complete rows / full bank |
| Q03 | fitted joint Gaussian | empirical complete rows; historical bank implementation samples empirical rows |
| Q04 | fitted joint Gaussian | full fixed bank (only fixed-bank designs) |
| Q05 | fitted independent marginal Gaussian | empirical complete rows / full bank |

The original Q03 and Q04 are not interchangeable on a bank. W4 controlled comparisons may define a joint-Gaussian × full-bank target explicitly, corresponding to Q04 on banks, to ensure Q05 vs joint Gaussian changes only covariance. Preserve original IDs for replay; use new descriptive IDs for controlled variants. No true parameter prior or true stimulus generator is permitted in a blind design except known design metadata that would genuinely be supplied in the operational setting.

### 4.2 Masks and support

Retain FULL for random designs and MASK-ALL for declared derivative-valid bank rows. MASK-COL is a separately labeled sensitivity with column-specific denominators. Never silently count undefined X derivatives as zero. The mask must be constructible from known observed X and declared mathematical/domain conventions, not T, f, family roles or oracle derivative values. If the known zero-amount convention cannot be justified without truth, report this operational assumption and perform a decoder-only full-row sensitivity separately. A frozen smooth decoder may have derivatives at all bank rows, but that does not mean classical derivatives of the unknown response are identified there.

Approximately 22% all-interior bank coverage in earlier experiments is an important limit. Save the exact 500-row IDs and retained counts rather than using the approximate proportion as a constant.

## 5. Splits, representation and evaluation

Same 12 formal test beds as note 02; do not conflate Classic and Class C. Primary raw Z (S9); S9_2 is a separate marginal standardization followed by rotation, not full covariance whitening. Use original designated NP checkpoints and latent arrays, all existing subject splits. Fit Q05 reference and moments from TRAIN only, use VALIDATION for the historical numerical tie-break, evaluate TEST only after lock. For exact replay use existing scaler conventions. Any strict TRAIN-only standardization sensitivity must be separately labeled if the historical convention uses the whole Z array.

Blind inputs are Z, X, Y, frozen decoder, subject/split IDs and permitted preprocessing/design metadata. No T, f, R*, S8/S9 reference files, oracle masks or truth-dependent hyperparameter selection. The same candidate metric and measure apply to every toy. Toy ID can route data files, not select a favorable objective. Use separate processes or minimal payload interfaces for evaluation; audit access to raw S1 files containing T and all eval_only/supervised artifacts.

Primary evaluation criterion unchanged:

$$
MCC_{\rm blind}\ge MCC_{\rm supervised}-.05,\qquad
Leak_{\max}<.2\ \text{Classic},\quad <.3\ \text{CR}.
$$

Hungarian matching uses absolute correlations on TEST. Compute signed and absolute T–U matrices, per-axis matched correlations, MCC, minimum matched correlation, mean/max unmatched leakage, reference gap and distance modulo signed permutations. Also report absolute recovery thresholds and common leakage .2/.3 plus gap .02 sensitivity. These evaluation thresholds never enter the objective. A run with weak representation may meet its relative criterion yet have low absolute MCC; report both.

---

# Part III. W4-00 — Reproduce the Q05 observation

## 6. Inputs and baseline audit

Freeze source commit, actual file hashes and data manifest from Lab 09/08/07. Inspect local W3 bundle and corresponding W2 moment caches; do not assume all ignored outputs are committed. Copy references read-only, never overwrite an existing result or lock.

Required historical records: FIN-SENS selected rotations, finalQ, trace, cache metadata, recovery rows, W3 Q01 algorithm_lock.json, W3 finalist Q01 3×3 results. In the key bundle FIN-SENS files are under W3r1/blind/FIN-SENS/<toy>/<candidate>.npz; keys encode variant, Q, mask, integration seed, search seed and budget. For each of the 36 primary Q05 rows verify key identity, no duplicate/missing seed, exact candidate, variant, channel, and correct FULL/MASK-ALL. Recompute recovery on TEST from archived R, not from rounded MCC values.

Then reconstruct the corresponding TRAIN and VALIDATION Q05 medium caches using exact historical designs. Verify their means/SDs, sample IDs or replay RNG, mask counts, J tensors, and loss of every archived finalist. Re-execute S0 with original search seeds and tie handling. Numerical differences must be quantified; if rotations differ only by signs/permutation, measure equivalence after alignment. A changed winner with essentially equal loss but different scientific recovery is not exact replication.

Compare historical Q01 and Q02 controls and the old E02 result. Use the existing Lab 08/09 optimizer rather than writing a new one for the replay. Preserve signed-permutation dedup, identity start, 400/8 (k=3,4) or 2000/30 (k=6), refinement spans 45,15,5,1.5,.5 degrees, relative 1e-9 numerical ties with VALIDATION tie-break. Read actual caps in source; report them instead of inventing unbounded optimization.

## 7. Unit tests and acceptance

- AD vs finite difference at valid points; direct rotated derivative vs moment contraction; rotation inverse and output scaling.
- For fixed points, sum_ij B invariant; signed permutation invariance of row and global metric; row cut and undefined score behavior; exact GL decomposition.
- Same Q05 sample under arbitrary R; no re-estimation in U coordinates.
- Sample mean/covariance vs fitted reference; deterministic replay and independent RNG stream separation; correct whole-row bank replication and mask denominators.
- Verify blind cache and optimizer work with truth files inaccessible; lock every selected R before evaluation opens truth.
- Reproduce 36/36 historical pass, loss/recovery numbers and search budgets, or explicitly explain any deviation.
- Verify old Q01 lock remains byte-identical. Historical lock verification may be sensitive to Git CRLF conversion; preserve original byte hashes and separately record canonical text hashes rather than rewriting the lock to pass.

Gate: no assertion of a replicated Q05 result if source artifacts are missing or exact reconstruction fails. A historical archived recovery can still be reported separately with provenance. A replay is not independent evidence.

---

# Part IV. W4-01 — Full Q05 blind-search replication

## 8. Fixed experimental grid

Primary candidate ID W4-Q05 = RW-GL-l0.75-a1.0 / decoder AD / raw Z / Q05 / FULL or MASK-ALL by design / existing S0. No hyperparameter search. Use all 12 toys, not only CR4. Preserve W3 locked Q01 method and E02+S9_2 as fixed controls.

| Batch | Design | Integration seeds | Search seeds | Purpose |
|---|---|---|---|---|
| REP-0 | C historical | 31 | 31,73,101 | exact replay, 36 runs |
| REP-1 | C | 31,73,101 | 31,73,101 | primary 108 runs, of which 36 replay |
| REP-2 | C | 307,401,503 (new proposed IDs; verify unused) | 31,73,101 | another 108 runs, no tuning |
| REP-3 | D independent | 211 and 613 (verify cache provenance) | 31,73,101 | high-fidelity 72 fresh searches |
| REP-4 | independent D reevaluation | same frozen D caches | all REP-1/2 finalists | test numerical integration overfit |

Seeds designate Monte Carlo streams, not independent datasets. For Gaussian measures used_subjects() is empty because no empirical subject is selected as a latent evaluation location. Consequently excluding those IDs does NOT make the Gaussian fit independent of TRAIN. Primary independent integration uses the same fitted TRAIN law with fresh latent/X draws; separate TRAIN-subpool diagnostics assess reference-fit uncertainty (W4-02). Do not incorrectly call a fresh Gaussian draw a new subject.

All comparisons use equal search budgets. If numerical diagnostics (separated near-minima, nonconvergence or inconsistent blind best loss) trigger S1, allow the historical 5× budget rule; report S0 and S1 separately. Do not increase budget because TEST MCC was low. For D reevaluation save all C finalists and their D losses without replacing the originally selected R; D re-search is a separate locked algorithm run. Ties and near-equivalent candidates need signed-permutation distances and recovery of all finalists.

### 8.1 Convergence beyond D if needed

For cells with unstable ranking or substantive changes in independent integration estimates, predeclare escalation to 2048×128 random points or 1024×500 full bank. Existing MAXDRAW=4096 permits these counts; if using larger values change the sampler explicitly with tests, no silent slicing. Use chunked AD, sufficient memory, float64 accumulation. Compare nested samples within an integration seed for convergence and independent seeds for variability. For bank treated as census do not bootstrap rows as if they were random draws; resample the latent integration law only.

For selected R and R* compare paired loss differences on each independent cache; report median/range and rank stability, not merely standard error of the loss alone. A very small variation in B can switch minima if the landscape is flat. Save the variance of each J and normalized B by latent-draw block; rows sharing one latent are not iid.

## 9. Pass rules and escalation

The primary development milestone is all 12 toys passing all 9 REP-1 searches with a single fixed algorithm. Stronger replication requires REP-2 and REP-3 coverage, stable independent-cache rankings, no missing/invalid cells and reported secondary-NP results. Report success count and denominator separately for each batch, plus intersection across batches; do not hide seed failures under a median.

For failure classification use post-lock L(Rhat)−L(R*) on identical objective and measure, with numerical/integration uncertainty. Lower wrong loss implies at least one inferior-recovery direction is preferred to the supervised reference. It does not prove that every scientifically good direction is worse, or that the global minimum is mathematically established. Higher wrong loss motivates optimizer audit. Ties imply ambiguity. Report both failure criterion components and matched parameters.

Decision:
- 12/12 across all primary replications: advance to lock and CONF-A; do not tune lambda/c.
- 12/12 REP-1 but failure under fresh seeds/D: diagnose integration stability; no complete robustness claim.
- Less than 12/12: retain Q05 as a partial development result, inspect W4-02 before any metric changes. May confirm its actual limited scope under a lock.
- Missing jobs: label incomplete, do not shrink denominator.

---

# Part V. W4-02 — Explain why the reference measure changes CR4

## 10. Controlled decomposition of the measure

Question: is the CR4 improvement due to the change from empirical to Gaussian latent distribution, removal of latent covariance, changed tails, product separation of Z and X, or Monte Carlo variability?

The previous Q01/Q02/Q05 results are not a factorial causal decomposition. Build a controlled family in which the same fixed TRAIN scaler, same physical X rows and masks, same latent sample count and same RNG for underlying standard normals are used wherever feasible.

| ID | Latent law | X law | Contrast |
|---|---|---|---|
| M-P | empirical paired subject/trials | own rows | historical Q01 |
| M-E | empirical Z | common independent complete rows/bank | product vs paired |
| M-J | joint Gaussian fitted on TRAIN Z | common independent rows/bank | empirical vs Gaussian with covariance |
| M-D | diagonal Gaussian fitted on TRAIN Z | same common rows/bank | covariance removal |
| M-I | independent empirical marginals (shuffle/resample each latent column in base frame) | same common rows/bank | independence without Gaussian marginals |

M-J vs M-D must use the same standard normal draws and X panel, unlike historical Q03/Q05 RNG. Use one common latent grouping scheme and weights; if empirical latent sampling uses a different finite support, report that as part of the intended contrast. M-I samples columns of Z only, not columns of X. M-J is Q04-like on a fixed bank. The M-* variants are diagnostic candidates, not replacements for historical Q05. Begin with CR4/CR4L plus C1 and CR3/CR5 and their paired L counterparts for mechanism plots, but run the fixed M grid on all 12 before making a general claim. At least C medium and three independent integration seeds. Actual blind searches for each fixed M recipe are evaluated independently.

### 10.1 Covariance interpolation

Let Sigma be empirical TRAIN covariance (ddof=0), D=diag(sqrt(diag Sigma)). Define

$$
\Sigma_\rho=(1-\rho)D^2+\rho\Sigma,\quad \rho\in\{0,.25,.5,.75,1\}.
$$

All preserve marginal variance, while off-diagonal covariance is scaled. Use matched standard normals and a continuous matrix square root or documented Cholesky convention. Check positive semidefiniteness and degenerate dimensions; any jitter must be fixed and logged. Evaluate on common X. rho=0 is the diagonal target in the controlled design; rho=1 joint Gaussian. Do not select rho by the known CR4 true axes and then call it a universal estimator.

### 10.2 Tail, variance and distribution-fit sensitivity

In the raw base frame compare

$$
Z^\circ=\mu+\kappa D\epsilon,\quad \kappa\in\{.75,1,1.25\}.
$$

Also compare componentwise truncated standard normals at |epsilon_i|<=2, 3, using rejection sampling with a fixed budget and specified conditional law (not clipping). These alter the variance; if variance-matched versions are desired label them separately and save the realized variance. A radial truncation is another distinct estimand, not the same as marginal truncation. The primary untruncated Q05 remains unchanged.

A complementary interpolation between empirical and diagonal Gaussian moments uses

$$
J_\alpha^{(j)}=(1-\alpha)J_E^{(j)}+\alpha J_D^{(j)},\quad \alpha\in\{0,.25,.5,.75,1\},
$$

where E and D use the same X/mask and normalized weights. This is the HE under a mixture of measures, not a mixture of decoder functions. It permits cheap systematic loss-landscape comparisons. Use these for understanding, not model-specific tuning.

Fit Q05 on independently sampled TRAIN subpools (e.g. 25%, 50%, full pool, 3 seeds). In one arm hold X panel fixed while refitting latent moments, in another change both as in deployment. Never fit Q05 using T. Report population versus sample SD, mean/SD/covariance error, effective rank and number of subjects used. No model-specific Gaussian choice.

## 11. Inspect support and contributions

For each measure, evaluate allowed-data distribution diagnostics:
- marginal means, SD, skewness, excess kurtosis, correlations, covariance eigenspectrum, empirical quantiles, multivariate Mahalanobis and nearest-neighbor distances (reference threshold from held-out empirical TRAIN/VAL, not oracle);
- fraction of synthetic latent draws outside empirical marginal quantile ranges (1–99%, .5–99.5%), and high-distance empirical quantile threshold;
- response/derivative norms and energy fraction contributed by tail strata, e.g. max_i |(z_i−mu_i)/sd_i| >2, >3;
- VALIDATION prediction comparison on actual observed rows; synthetic off-support predictions have no directly observed Y and must not be reported as predictive validation;
- sensitivity of J/B and loss rankings to truncation and reduced variance.

For CR4 compare archived Q01/Q02 bad R, Q05 found R and supervised R* after lock on every controlled measure. Define

$$
\Delta L_{\mathcal Q}(R_a,R_b)=L_{\mathcal Q}(R_a)-L_{\mathcal Q}(R_b).
$$

Decompose GL differences into .75 times the exact symmetric row-score and row-weight terms plus .25 times change in global score. Save per-row/column B, r, s, weights, J eigenvalues and signed cross-moments. Align rows using Hungarian or signed permutation in the evaluator only. Compare across CR3/CR5 and successes as controls. Evaluate geodesic paths modulo sign/permutation (align determinants for a continuous orthogonal path), fine pair-angle sweeps and random skew directions. Curves between fixed old R do not replace new optimization. Record which measure reverses the ordering and whether its own optimum is good.

### 11.1 Interpretation rules

| Outcome | Evidence supports | Does not prove |
|---|---|---|
| M-D beats M-J with same draws and X | covariance removal changes useful curvature geometry | independence of true parameters or universal necessity |
| M-J and M-D beat M-E | Gaussian reference/smoothing of integration distribution matters | smoothing the decoder itself |
| M-I matches M-D | independence may matter more than Gaussian shape | normality of Z |
| only untruncated M-D passes, tails dominate J | extrapolation-sensitive objective | reliable empirical support identification |
| success stable across truncation, covariance/scale perturbations | robustness to reference assumptions | broad generative generalization |
| new good minimum appears only for one MC seed | finite integration issue | robust metric success |

Do not conclude that Q05 works because VAE forces exact N(0,I). The learned representation's empirical covariance and marginal shape must be measured, and Q05 is fitted, not necessarily standard normal.

---

# Part VI. W4-03 — Three-gap evaluation under Q05

## 12. Define two separate analytic estimands

The blind Q05 law is fitted to Z; it is not the true parameter prior. Blind and oracle frames are not literally the same space. A comparison must say which measure each arm uses.

**O-true (primary scientific Oracle):** use actual supported T prior or paired empirical T (clearly labeled), standardized by the fixed TRAIN T scaler, independent of the same physical X panel; evaluate analytic logit/continuous mean and UX derivatives on valid points. This answers whether the metric prefers true axes under the original generative distribution. CPT parameters may be bounded or positive. Do not draw an independent normal T outside its domain, silently clip T or substitute an invalid derivative.

**O-ref (sensitivity):** use a supported, explicitly defined reference distribution in standardized T coordinates. If an independent normal approximation to T is attempted, report invalid fraction and only evaluate a declared supported conditional law, with corresponding weights and changed estimand. This is not exact Q05 in Z. For non-Gaussian bounded priors the supported conditional law may differ substantially from a Gaussian. Do not claim it is a faithful Oracle for every Q05 query.

**Bridge (diagnostic only):** map Q05 draws in Z using TRAIN-fit BO/BA as in W1, evaluate analytic response in that frame with chain rule and valid support. Compare bridge and decoder at identical Q05 positions. Report invalid predicted T and common subset coverage. Neither bridge nor true-prior draws enter actual search. Bridge inserts known f and alignment; it cannot prove the unknown surface is identifiable.

For matched measures save all distribution and scaling choices. Use the original independent parameter laws and complete X rows to distinguish the effect of the metric from a changed integration law. For B/C pairs share analytic moments only if actual T law/scalers, physical X, masks and seeds match; still report both toy identities.

## 13. Gap 1: local, exact global, practical global

Use the fixed RW-GL, historical E02 and B01.c03 controls (no large grid). Repeat all saved angle sweeps (0,15,30,45,60,75,90 across all axis subsets), fine near-identity angles (±.5,1,2,5,10,15), random directions and analytic Haar/global search. Use true standardized T frame, fixed support and common draws. The diagnostic global search can start at I as an extra check. Save every finalist and independent high-fidelity reevaluation, identity loss and loss improvement.

Exact Metric Success is argmin_Q L_O(Q)=signed permutations. Finite empirical search cannot establish this identity. Categorize observed global counterexamples, tied nonpermutations, empirical support and inconclusive cases with basin coverage.

Practical Gap 1 evaluates U_ideal=Q T_s and U_transfer=Q R* Z, both using TEST truth after lock. Report d_SP=min_P ||Q−P||_F/sqrt(2k), matched correlations and leakage. Use W2/W3 practical criteria (ideal MCC≥.95 and leak<.2; transferred historical ceiling-relative criterion), plus sensitivity. Small displacements are not automatically failure of practical recovery. In W3 Q01 the new metric passed 8/8 practical logit targets with worst observed displaced minimum about .061; preserve that historical comparison. Do not change exact Gap 1 to practical success.

## 14. Gap 2: fidelity and preference

For every toy and both coordinate variants, compare Oracle and inferred curves under labeled measures, using saved relative Givens rotations only in evaluation. Let R=G R* and analytic Q=G. Compute both historical evaluations:

1. Eval 1: pooled Pearson/Spearman on baseline plus positive-angle rotations, per-sweep correlation, undefined fraction, loss/matrix error and near-zero energy discrepancies. A high correlation does not establish a correct minimum. For an analytic constant curve report NA, not zero correlation as a numerical fact.
2. Eval 2: all-sweep perfect (0/90), marginal-or-better (0/15/75/90), legacy and strict all-minimizers subset rule, tied/interior/constant/invalid categories and normalized loss margins.

Use Q01 baseline and Q05, both C and D, seeds 31/73/101 and independent reevaluation where available. For Q05 pair O-true with inferred Q05 as a distribution-labeled scientific comparison; also report the common-point bridge separately to isolate geometry/response differences. Do not equate distribution differences with pure derivative estimator error. Compare normalized/unnormalized UX and, if already cached, UU diagnostic only; primary objective uses UX alone.

Report analytic-only, inferred-only, and both-pass across the full 24 libraries, by class and family. Keep probability channel separate if run. The W3 Q01 medium result (22/24 joint marginal, CR4 both variants fail) is a baseline to reproduce, not a Q05 result. A new 24/24 local pass is not by itself proof of full inference fidelity.

## 15. Gap 3: actual blind recovery

W4-01 is the authoritative actual search experiment. Report the selected R, all runs, supervised reference gap, T–U heatmaps for each toy and variant, independent cache rankings and failure types. Distinguish 12 toys under a single fixed variant from 24 libraries and from seed-level pass rate. No retrospective union of Q01 and Q05 across toys counts as a fixed algorithm.

If practical Gap 1, local Gap 2 and actual Gap 3 all pass on the same candidate and full development panel, that is strong converging evidence. It is not a theorem of exact global identification, and not independent confirmation. If the theoretical route fails but blind recovery succeeds, retain the empirical result with appropriate limits.

---

# Part VII. W4-04 — Robustness without new NP training

## 16. Fixed controls and secondary panel

Run W4-Q05 and W3-Q01 on both S9 and S9_2 separately; E02+S9_2 as historical control. The primary algorithm remains S9 regardless of individual toy results. Compare Q01/Q02/Q03/Q04/Q05 where defined with same fixed metric. W4-02 controlled laws are a different experiment. No per-toy frame or measure selection.

Use the same official secondary saved NP runs available in Lab 08/09: classic repeats and difficult CR repeats/alternative beta. Record exact checkpoint/latent paths and available supervised reference; if S9_2 reference missing, report not evaluated rather than fabricate it. Use C, 3×3 seeds for primary candidate where available, D reevaluation of fragile cells. Report secondary panels separately; alternative beta is not a pure training-seed replicate.

Bank sensitivity: evaluate Q05 MASK-COL and decoder-only full-row extension if valid by observed-design rule. Same mathematical metric, separate estimands. Include interior support counts and compare results to historical Q01/Q05 MASK-ALL. Do not choose mask by T.

Optional Q05 reference fit stability across context representations already saved in W3 (100,200,largest disjoint, original full), using exact saved encodings and same-context references. Smaller out-of-training-range contexts are flagged. No further encoder training in this module; absolute MCC and Stage A must accompany ceiling-relative passes. Do not repeat the full W3 context grid unless this diagnostic has a specific unresolved question.

## 17. What NOT to do

No mandatory new sparsity library, gradient/contrast expansion, third derivatives, RBF search, conditional HE expansion or broad beta sweep. These are reserved for a new development cycle if W4 diagnoses a substantive failure. Do not allow those optional projects to block W4-01. Keep compute focused on the candidate that has actual 36/36 evidence.

---

# Part VIII. W4-05 — Freeze and independently confirm

## 18. Versioned algorithm lock

Preserve W3 algorithm_lock.json (Q01, S9) as immutable. Create a new lock in Lab 10, e.g. W4-Q05-v1, only after W4 development results are complete. Explicitly record that selection of Q05 was motivated by inspecting FIN-SENS recovery and is therefore development selection. Do not repurpose the existing freeze script if it hardcodes Q01; implement a measure-aware schema and tests.

Lock contains:
- exact RW-GL formula, .75/.25, c=.03, a=1, active cutoff, zero/invalid policy, dtype;
- raw Z frame, observation channel rule, saved training scaler and fitted TRAIN-only Q05 marginal law (algorithm rule, not fixed historical numerical mu/sigma);
- X sampling and mask policy, seed rules, integration fidelity and search S0/S1 rules, VALIDATION tie-break;
- one primary deployable rotation rule. If all nine seed combinations are run as robustness assessments, do not choose their winner by T. For a single deployment either fix one seed pair in advance or use a declared blind independent-cache ranking across finalists; these are different algorithms and must be frozen as such before confirmation;
- full code/config/dependency hashes, exact historical data/checkpoint versions, lock hash and canonical path format; old lock untouched;
- confirmation dataset/NP training registry, stage A and recovery evaluation, all-run vs single-deployment definitions, missing/abstention and failure policy.

Use W4-Q05 lock if the development evidence justifies advancing it; if unstable, freeze the limited candidate for honest confirmation or defer expansion with explicit blocker. Never use confirmation results to modify the lock. Verify lock hash before every confirmation run, save truth-free outputs and manifest before any evaluation process sees confirmation T/R*.

## 19. CONF-A — same formula families, new data and new NPs

Reuse the generator definitions and nominal data/NP protocol in note 02 and W3 confirmation_protocol.json. Freeze exact existing per-toy architecture, beta and training selection rule before new truth-based evaluation. Existing Class B beta .0005; Class C per official registry; verify actual configuration rather than choosing a new beta on recovery. Keep 8000 subjects and documented per-toy trial count, observation noise/priors, subject split and stimulus design. Fixed bank may be reused for this replication, with fresh subject responses; record bank hash.

Minimum mandatory confirmation panel proposed: one previously unused data seed × two independent NP training seeds × 12 toys = 24 trained NPs. Full panel: two data seeds × two training seeds × 12 = 48 NPs. Example proposed IDs data 2026112801/2026112802, NP 2026112811/2026112812; verify they are legal in the actual RNG/API and unused in historical manifests before generation, freeze final IDs. If generation and split streams are separate save all IDs and provenance. B/C paired designs can share T/X streams where specified; observation noise and NP training are distinct. Do not describe them as independent formula families.

Use isolated dataset/checkpoint output namespace with new run IDs. Existing S1 code may preserve a dataset when a path exists; verify hashes and seed rather than mistakenly retrain on old data. No overwrite of existing formal outputs. Use read-only generator adapters or parameterized output paths with tests. Run documented MLE/precheck and Stage A diagnostics in evaluation-only workflow, not to select a favorable NP seed. If a new NP is weak, keep it in all-attempted denominator and report separate adequate-Stage-A subset as descriptive. Do not silently drop or regenerate unfavorable cases.

### 19.1 Confirmation pipeline

1. Register all planned datasets and training runs before accessing their recovery results.
2. Generate and validate fresh data, export minimal truth-free payload; separate T and actual generator code from blind search worker.
3. Train NP using frozen rule, store full checkpoint/scalers and history, extract Z.
4. Fit Q05 to TRAIN Z only; create fixed X design/mask; build moments; run locked candidate and fixed controls W3-Q01 and E02 if resources permit (declare scope before).
5. Lock selected rotations and independent integration reevaluations. Only after all required blind outputs for a panel are locked, evaluate T–U and same-variant TRAIN-fit supervised reference on TEST.
6. Report Stage A linear R², raw-Z alignment, prediction, reference MCC/leak, blind recovery, min matched, all seeds, runtime and failures. Use same criterion as development and common threshold sensitivity; do not claim perfect one-to-one mapping from relative pass alone.

The original supervised reference convention may have used all subjects in historical runs; for strict confirmation fit supervised alignment on TRAIN and evaluate TEST, with the historical convention as separate comparison if needed. This change must be frozen before confirmation. It is evaluation-only and not a different blind objective.

Do not count 9 integration/search repeats as 9 independent NPs. Summarize by family, observation class, generation seed and NP training run. For 24 runs give number of runs and number of 12 toy identities with both new NPs passing, and all attempted failures. Bootstrap uncertainty only at appropriate dataset/training hierarchy; two seeds are limited evidence.

## 20. CONF-B — distribution and design shifts (secondary)

After CONF-A lock remains fixed. Change one factor at a time: a new valid generic bank / random bank subset, number of trials per subject (100/250/500 with model feasibility), noise level or observation mechanism as documented. New data and NP train rule stay fixed within arm. Q05 reference fit adapts to each TRAIN Z automatically by the same formula; no choice of kappa/rho/lambda by toy. Report support changes and Stage A adequacy. This assesses robustness beyond identical data generating settings, not a new method-tuning panel.

---

# Part IX. W4-06 — Unseen families and final synthesis

## 21. Novel-family confirmation (CONF-C)

Before running a novel generator, obtain a separately reviewed frozen test-bed registry, as already required by W3. At least four structural families spanning additive nonlinear basis with bias, sparse cross-feature interactions, multiplicative/gated effects, and nonlinear six-to-eight parameter structure. Include clearly nonidentifiable/rotationally symmetric negative controls. These are proposed design categories, not validated equations. The final registry must define parameter priors, stimulus distributions, observation likelihood, valid derivatives, parameter dimension, identifiability checks and training protocol. Do not choose equations by observing whether RW-GL likes their axes.

Use paired B/C when scientifically meaningful, independent fresh generation and training seeds, and the same locked blind rule. A rotation-symmetric negative control should reveal ambiguity, not a manufactured successful original labeling. Separate identifiable positive test beds from negative controls. Passing the original six formula families does not establish performance on arbitrary cognitive models. Novel-family validation may be a later compute phase and must be labeled incomplete if registry or compute unavailable.

## 22. Final scientific report

Report separately:
- Stage A adequacy across old/new representations and absolute recovery;
- exact and practical Gap 1 (oracle, supported distribution, finite search limitations);
- Gap 2 Eval1 shape agreement and Eval2 minima preference (labeled measures);
- actual fixed-algorithm Gap 3, all-seed and single-deployment performance;
- robustness across distributions, tails, masks, NP training runs and observation classes;
- fresh-data confirmation vs novel-family generalization.

The development milestone is a single fixed method passing all 12 official toys. Stronger scientific validation requires a locked method on independent data/NPs and then unseen structures. Even these experiments do not prove universal identifiability or exact global optimality.

---

# Part X. Implementation details and deliverables

## 23. Independent Lab 10

Proposed new folder (Lab 10 unused at planning time):

    metric_lab/10_wave4_2_fourth_wave/
      README.md
      configs/
        primary_candidate.json
        replication_protocol.json
        measure_diagnostics.json
        gap_protocol.json
        confirmation_protocol.json
      wave4/
        inputs.py
        designs.py
        moments.py
        objectives.py
        search.py
        support.py
        locks.py
        eval_only/
          recovery.py
          oracle.py
          bridge.py
          diagnostics.py
        reporting.py
      tests/
      prepare.py
      validate.py
      run_replay.py
      run_replication.py
      run_measure_diagnostics.py
      run_gap_evaluation.py
      run_robustness.py
      freeze_algorithm.py
      run_confirmation.py
      analyze.py
      figures.py
      run_all.sh
      results/

Reuse read-only W2 design/optimizer and W3 rowmetrics where possible. Do not modify any Lab 01–09 code, outputs or locks to implement W4. All new large caches under outputs/metric_lab/10_wave4_2_fourth_wave/<run_id>/, with separate blind and eval-only interfaces. Immutable configuration manifests, --resume, no overwrite by default, atomic output writes, Windows py11 and portable path handling; strip CRLF from job lists. CPU float64 reference and optional chunked GPU AD after equivalence tests.

Illustrative commands (must implement before execution, not existing commands):

    python metric_lab/10_wave4_2_fourth_wave/prepare.py --toys all
    python metric_lab/10_wave4_2_fourth_wave/validate.py --toys all
    python metric_lab/10_wave4_2_fourth_wave/run_replay.py --batch REP-0 --toys all
    python metric_lab/10_wave4_2_fourth_wave/run_replication.py --batch REP-1 --toys all
    python metric_lab/10_wave4_2_fourth_wave/run_replication.py --batch REP-2 --toys all
    python metric_lab/10_wave4_2_fourth_wave/run_replication.py --batch REP-3 --toys all
    python metric_lab/10_wave4_2_fourth_wave/run_replication.py --batch REP-4 --toys all
    python metric_lab/10_wave4_2_fourth_wave/run_measure_diagnostics.py --toys all
    python metric_lab/10_wave4_2_fourth_wave/run_gap_evaluation.py --toys all
    python metric_lab/10_wave4_2_fourth_wave/run_robustness.py --panel secondary
    python metric_lab/10_wave4_2_fourth_wave/analyze.py --panel development
    python metric_lab/10_wave4_2_fourth_wave/figures.py
    python metric_lab/10_wave4_2_fourth_wave/freeze_algorithm.py --candidate W4-Q05-v1
    python metric_lab/10_wave4_2_fourth_wave/run_confirmation.py --batch CONF-A --lock <verified_lock>
    python metric_lab/10_wave4_2_fourth_wave/analyze.py --panel confirmation

Support --toy, --variant, --Q, --mask, --iseed, --sseed, --fidelity, --run-id, --resume and bounded parallelism. Confirmation needs a separately registered data/training manifest. Do not invoke a script that still hardcodes Q01. Run smoke tests on C1, CR4, CR3 and paired continuous toys, then full coverage; never select favorable parameters from smoke-test truth.

## 24. Result files and figures

| Output | Required fields |
|---|---|
| w4_manifest.json / validation_checks.csv / replay_audit.csv | frozen source hashes, exact replay discrepancies, tests, known missing artifacts |
| integration_designs.json / q05_fit.npz | TRAIN subject pool IDs, mean/SD/cov, RNG, X rows or bank hash, mask denominators, actual draw counts |
| ux_moments.npz | base J per split/measure/fidelity/seed; numeric validity, dtype, cache hash |
| replication_locks.csv / found_rotations.npz / search_trace.csv.gz | all starts, final matrices, selected R, blind losses, tie rules, budgets, elapsed, code/cache hashes |
| replication_recovery.csv / recovery_correlations.npz | TEST correlation matrices, Hungarian matching, MCC/min/leak, gap, pass flags, variant, seed, candidate |
| independent_reeval.csv | independent-cache rank and loss changes, original winner, D best, numerical equivalence |
| measure_comparison.csv / covariance_tail_sensitivity.csv | M-grid, rho/kappa/truncation, coverage, recovery, declared estimand |
| support_diagnostics.csv / tail_energy.csv | empirical vs synthetic support, tail weights and contributions, invalid rows |
| loss_decomposition.csv / cr4_geodesics.csv.gz | exact row score, weight and global terms, fixed and optimized R, matched measure |
| oracle_global.csv / oracle_solutions.npz | supported law, identity/finalist losses, d_SP, exact/practical categories, ideal/transfer recovery |
| gap2_curves.csv.gz / gap2_summary.csv | paired relative rotation loss, correlation, strict/legacy minima, error, measure labels |
| robustness_recovery.csv / coverage.csv | all primary, secondary, mask and context planned/attempted cells, invalid/missing reason |
| algorithm_lock_W4.json / confirmation_manifest.json | immutable method and independent panel, no replacement of W3 lock |
| confirmation_locks.csv / confirmation_recovery.csv | all new blind outputs locked before truth; new data/NP IDs, absolute and relative recovery, Stage A |
| report_tables.md / next_wave_candidates.csv | analyzer-generated tables, deviations, decision rationale |

Figures (generated from saved numeric data):
1. All 12 T–U correlation heatmaps for Q05 vs Q01 vs supervised, include failures and consistent matching/color scale.
2. REP-1/2/3 seed-by-toy success, MCC gap and max leakage distributions; exact denominator.
3. CR4 fixed-R and reoptimized landscape for M-E/M-J/M-D/M-I, covariance interpolation and tail sensitivity.
4. Q05 support/energy contribution plots and full-bank mask coverage.
5. Oracle exact/practical loss-distance-recovery and Gap2 curve/minimum comparisons.
6. Secondary NP robustness and independent confirmation panel, no winner-only plots.

Every row records toy/class/family (reporting only), checkpoint and data IDs, measure, mask, channel, coordinate frame, integration and search seeds, split, actual sample counts, validity, metric/estimator versions and hash. Save all failures, not only successes. Report missing or optional unrun arms in README.

## 25. Ordered execution and final decisions

| Order | Task | Gate |
|---|---|---|
| 1 | W4-00 | exact reproduction or documented discrepancy |
| 2 | REP-1, REP-2, REP-3 and D reevaluation | all planned primary cells reported, stable or identified failure |
| 3 | W4-02/03/04 in parallel | measure mechanism, matched-gap evaluation, robustness |
| 4 | lock one fixed algorithm | no future truth-based modification; retain historical lock |
| 5 | CONF-A 24 NP minimum then 48 full | independent data/NP outcomes, exact denominator |
| 6 | CONF-B/C as resources and registry allow | independent design and structural generalization |

If W4-Q05 reliably achieves 12/12 on the development panel, move to independent confirmation rather than endless new metric engineering. If CR4 or another model fails on fresh integration draws, diagnose the reference distribution and exact loss ordering first. If independent new NPs fail despite development success, report the limitation and only revise method in a new development version with a fresh confirmation panel. If new families are nonidentifiable, report ambiguity rather than forcing axes.

### Acceptance checklist

- [ ] All 36 historical Q05 results reproduced or deviations explicitly explained.
- [ ] All 12 toys evaluated with the fixed Q05 method at full 3×3 and fresh seeds, independent high fidelity and saved finalists.
- [ ] Q05 is fitted in base Z coordinates on TRAIN, points transported consistently, complete X rows and masks verified; no hidden T/f/R* usage.
- [ ] CR4 mechanism tested with matched draws and X, covariance/tail/support diagnostics; no per-toy measure tuning.
- [ ] Exact/practical Gap1, both Gap2 evaluations and actual blind Gap3 distinguished under declared supported measures.
- [ ] Saved NP repeat and coordinate/mask sensitivities reported, all failures retained.
- [ ] W3 Q01 lock unchanged; new Q05 lock records development selection and deterministic deployment rule.
- [ ] Independent CONF-A begins only after lock, fresh dataset/NP provenance verified, Stage A failures included.
- [ ] Novel family confirmation not claimed until separate registry and runs exist.

**Implementation status:** only the research protocol has been written. Proposed seeds, budgets, Lab 10 paths, controlled measure diagnostics and confirmation design are not yet validated or executed. Q05 12/12 is a verified historical FIN-SENS observation with one integration seed, not the outcome of this fourth wave.
