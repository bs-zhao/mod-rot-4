# 05 — Next-stage research roadmap: from local Hessian screening to blind parameter recovery

**Status:** research plan, proposed on 2026-11-26. No experiment or new method in this note is claimed to have been implemented or validated.  
**Purpose:** turn the findings of the current experiments into a sequence of major research steps. Preserve the simplicity-based search principle, but separate failures of the metric, statistical identification/inference, representation, and search. Avoid an endless grid of minor modifications to existing sparsity penalties.

Related notes: [01 Project purpose](01_project_purpose.md), [02 Test beds](02_test_beds.md), [03 Simplicity search principle](03_simplicity_search_principle.md), [04.1 Hessian energy](04.1_hessian_energy_simplicity.md), [04.2 Rotation sweep](04.2_first_experiment_rotation_sweep.md), [04.3 First-wave candidates](04.3_first_wave_metric_inference_candidates.md).

---

# Part I. Starting point and scientific objectives

## 1. The twelve test beds and experimental units

Use the formal test beds of note 02, not the historical stand-alone noisy_logit_np networks.

| Model family | Class B: binary choice | Class C: continuous output | dim(T) | dim(X) |
|---|---|---|---:|---:|
| C1 | C1 | C1L | 4 | 3 |
| C2 | C2 | C2L | 4 | 4 |
| C3 | C3 | C3L | 6 | 4 |
| CR3 | CR3 | CR3L | 4 | 6 |
| CR4 | CR4 | CR4L | 3 | 6 |
| CR5 | CR5 | CR5L | 3 | 6 |

Class B observes $Y\sim\mathrm{Bernoulli}(\sigma(\eta))$; Class C observes $Y=\eta+\epsilon$, with the designed signal ratio 0.8. The two toys in a family share the definition of $\eta(T,X)$, the parameter prior and stimulus design, not the trained NP, observation mechanism, latent space, or search result. Existing datasets share T and X streams; thus the 12 independent test-bed identities are not 12 statistically independent generating mechanisms or independent replications.

All 12 formal toys have S1–S9_2 according to the current note 02. Use the specified NP of each toy in [toys.py](../../metric_lab/03_s9_metric_curves/toys.py). The primary panel contains 12 toys × 2 coordinate variants (S9 and S9_2) = 24 rotation libraries. Additional saved NP runs are a secondary robustness panel, not extra toy identities. Do not use a model family name as an input to the loss or to select an estimator. The observation class and known data design are legitimate method inputs.

Response surfaces:
- Class B, Y-L: decoder class-1 minus class-0 logit, oracle $\eta$.
- Class B, Y-P: sigmoid of this logit, oracle $\sigma(\eta)$.
- Class C, Y-C: continuous conditional mean, decoder output converted back to original Y units with the saved training scaler, oracle $\eta$.

Class B Y-L and Class C Y-C have the same analytic target in a family, but the inferred targets may differ. Y-P is a separate mathematical estimand. Report by toy, class, channel, NP run and variant; do not count channels as new toys.

## 2. What the existing evidence establishes

Sources: [Lab 03: 12-toy S9 metric curves](../../metric_lab/03_s9_metric_curves/README.md), [class-wise angle statistics](../../metric_lab/03_s9_metric_curves/results/angle_pref_metric_by_class.csv), [Lab 06: wave 1](../../metric_lab/06_wave1_screen/README.md), and [Lab 05: blind search](../../metric_lab/05_angle_pass_blind_search/README.md). These are different experimental protocols; their counts must not be pooled as if they were one study.

### 2.1 Lab 03, current 12-toy panel

153 metrics, logit/continuous channels, paired analytic and decoder evaluations. The number of metrics satisfying the local angle criterion on **every** library of the indicated subset:

| Scope | Both analytic and inferred perfect | Both marginal or better |
|---|---:|---:|
| C1–C3, 6 libraries | 44 | 49 |
| CR3–CR5, 6 libraries | 0 | 0 |
| C1L–C3L, 6 libraries | 45 | 49 |
| CR3L–CR5L, 6 libraries | 6 | 7 |
| All Class B, 12 libraries | 0 | 0 |
| All Class C, 12 libraries | 0 | 0 |

The separate analytic-only statistics are crucial: 17 metrics are perfect on all 12 libraries in each class, 18 marginal or better. The corresponding analytic values are shared across classes in this protocol. The best inferred-only counts are 9/12 for Class B and 11/12 for Class C; the best across both is 18/24 for rowsp.a1.0.q0.1. Thus a claim that *no* metric can work on both classic and CR families at the analytic level is not supported by these data.

The six metrics with both-perfect passes on all complex continuous libraries are Fz.A+B+C, Fz.A+K+C, UXrow.a0.5.lq.25, rowsp.a0.5.q0.1, rowsp.a0.5.q0.5, rowsp.a0.75.q0.1. These are valuable empirical candidates but not universal solutions.

Median pooled Pearson across 153 metrics, combining S9/S9_2:

| Family | Class B | Class C |
|---|---:|---:|
| C1 | .998 | .999 |
| C2 | .993 | .998 |
| C3 | .987 | .996 |
| CR3 | .806 | .864 |
| CR4 | .486 | .708 |
| CR5 | .709 | .761 |

The larger agreement and higher supervised MCC in Class C suggest a role of representation quality and the observation/learning mechanism. They do not isolate a causal effect of binary versus continuous data: the NP beta and training regime also differ. CR4 remains a priority diagnostic because it has only three parameters and random X but is difficult.

### 2.2 Lab 06, historical Class B first-wave screen

262 metric variants, 17 saved binary rotation libraries, several derivative schemes and integration measures, two channels Y-L/Y-P. Under the joint criterion the best Q01/AD/logit combination B01.M03.c0.01 passed 13/17. AD and small-step finite differences were almost identical in the reported UX errors, so repeated tuning of finite-difference step alone is low priority. Inference-only candidates such as B08.M03.c0.01 and c0.03 with Q05 were identified in the earlier analysis; reverify exact counts from raw results under the harmonized protocol before adopting them. Their analytic failures on classic families are not erased by inferred-only passes.

Y-P was notably harder at the analytic level for CR models. A successful logit metric is not automatically a probability metric, and the continuous toy is not a third decoder channel of a binary toy.

Q05 was a sensitivity measure, not the true CPT prior. The existing implementation approximates T by an independent Gaussian in the analytic arm; this may leave its legal parameter support. Do not treat this as evidence under the original T distribution. Lab 03 uses an all-interior mask for some bank models while Lab 06 uses column-specific masks. Their energy targets are different.

### 2.3 Lab 05, actual blind search

Historical four-binary-toy experiment: 94 candidates selected from local angle screens, R*-free Haar starts and local refinement. Success counts were C1 77, C2 67, C3 3, CR4 3, with no common successful metric. These are actual search results, not local sweep counts. It is incorrect to say blind search has never been attempted. The missing experiment is a unified, independently confirmed blind search on the current 12-toy panel using the improved objectives/inference methods.

### 2.4 Interpretation

- Gap 1 has promising **local** evidence; no universal global theorem or global empirical guarantee.
- Gap 2 is not solved: correlation of complete curves, accuracy of M, and the location of the minimum are distinct.
- Gap 3 has partial historical successes, but not a common validated algorithm across either current class.
- A successful inferred curve with a failed analytic curve is an empirical recovery hypothesis, not a proof of inference fidelity. Constant analytic curves and wrong analytic minima must be distinguished.
- The first priority is to isolate why the available information does or does not support the correct coordinate system, rather than select a new threshold for each toy.

## 3. Common experiment contract

Keep the language of 03–04.3. Let $V$ be raw Z or its saved marginally standardized version, $U=RV$ with $R^\top R=I$, and $g_R(u,x)=g(R^\top u,x)$. The baseline for the analytic arm uses standardized T and an orthogonal Q. The decoder must receive exactly the coordinate transform and scalers used by the saved network. S9_2 is marginal standardization, not necessarily full whitening.

For the conventional HE estimand,

$$
h_R(w)=\nabla_w^2g_R(w),\quad M_{ij}(R)=E_{\mathcal Q_R}[h_{R,ij}(W)^2],
\quad L(R)=\Phi(M(R)).
$$

$\mathcal Q_R$ is the pushforward of a fixed base evaluation measure. Do not refit the integration distribution independently in every rotated frame. X is fixed in its physical coordinate system, with a single saved scaling convention. For a fixed bank, sample or average **complete rows**.

**Allowed-information rule:** every operational infer/search method receives only Z, X, Y, optionally frozen d, and declared known design/preprocessing metadata. No T, true f, true parameter distribution, oracle outputs, R*, or synthetic-family identity used to choose favorable directions. Supervised R* is permitted for constructing the labeled diagnostic sweeps only. Truth is used to compute analytic benchmarks and after-the-fact evaluations. Any bridge or truth-controlled intervention is explicitly a diagnostic arm and cannot be a candidate algorithm.

**Three separate result columns:** analytic preference; inferred preference; actual search recovery. Inference fidelity also reports curve correlations and loss/matrix errors. Never replace the official combined criterion of 04.2 by an inferred-only result. Keep both as separate outcomes.

**Evaluation 1:** pooled Pearson/Spearman over R* plus positive-angle sweep rotations, and per-curve correlation; constant/undefined correlations are NA. **Evaluation 2:** official all-sweep perfect (0/90) and marginal (0/15/75/90) results. Additionally record strict interior margin, ties and dynamic range. Audit the tie rule across implementations: a minimum set intersecting the accepted angles can hide an equally good wrong interior rotation. Retain historical pass flags and add a strict all-minimizers view; do not silently change the official numbers.

Every score is minimized, and sign/permutation invariance is checked. Use random orthogonal Q/R beyond the saved sweep to test local versus global claims. Whole-block Frobenius energy is a rotation-invariant control for the appropriate fixed-measure orthogonal transformation, not a useful standalone rotation loss.

---

# Part II. The six major steps

## Step 1 — Unified benchmark and failure localization

**Scientific question:** are the failures caused by metric geometry, Z representation, the statistical response target, function approximation, or numerical estimation?

### 1A. Extend the baseline to every formal toy

Create a new independent experiment directory (proposed metric_lab/07_unified_benchmark). Reuse the common read-only NP loader, official toys registry and saved rotation manifests. Do not change S1–S9_2 or overwrite Lab 03/06 outputs. Port the Lab 06 registry of 262 variants and the Lab 03 153-metric registry as two explicitly versioned registries; deduplicate only after mathematical equivalence is established. Keep names and mapping table. Run all 24 primary libraries, with Y-L/Y-P only for Class B and Y-C only for Class C. Use AD as the reference decoder derivative method and retain FD spot checks.

Primary measure: equal-subject empirical observed-trial average. Secondary measures: empirical latent × complete empirical X rows; fitted joint Gaussian latent × empirical X; fixed bank averaging when appropriate; independent marginal Gaussian as a declared sensitivity. For oracle comparisons use the true prior or paired actual T according to the estimand. An independent normal approximation to T is a separate sensitivity, not the baseline. Do not give the blind arm the synthetic X generator unless that design law is explicitly supplied to the analyst.

Convergence: start at the existing pilot fidelity, then expand to 512×64 and 1024×128 for random designs; use full bank and 32, 128, 512 latent draws for bank designs as resources permit. Common random numbers across rotations; at least three fixed integration seeds; report metric ranges, minimum stability and computational cost. Gaussian tails outside observed support are monitored separately.

Boundary policy:
- All-interior design, column-specific interior masks, and full valid design must be separately labeled estimands.
- CR3/CR3L power derivatives at zero amounts may not exist or may diverge. Do not replace undefined derivatives by zeros. CR5/CR5L have alpha=1; one-sided/domain-extension conventions can differ, and should be documented rather than declaring every zero singular.
- Fixed continuous-valued bank data alone do not identify arbitrary X derivatives off the bank. A surrogate/decoder is a modeling extension; more subjects at identical X do not solve that identification problem.
- Later finite-contrast metrics on observed bank rows avoid claiming classical X derivatives and are explicitly different targets.

### 1B. Verify coordinates and representation ceiling

For each NP and variant report Z covariance, marginal variances, condition number of fitted Z→T map, train/test linear R² by parameter, supervised MCC/leakage, and stability across repeats. Diagnose raw vs standardized Z; optional full covariance whitening must be a separate coordinate variant with saved inverse, not called S9_2. Full-Z regression success does not prove exact orthogonal recovery.

Record the alignment model on training subjects:

$$
\widehat T=AV+b.
$$

Use held-out subjects for diagnostics. If considering affine maps or general invertible transformations in the future, define separate constraints and success criteria; unrestricted scaling can trivially shrink derivatives and make the loss meaningless.

### 1C. Oracle bridge and controlled interpolation (diagnostics only)

Compare on common physical evaluation points:
1. true standardized-T surface $\eta(T,X)$ (or sigmoid);
2. bridge surface $g_{\rm bridge}(V,X)=\eta(AV+b,X)$ (or sigmoid);
3. observed-data conditional-response surrogate, fitted without T;
4. frozen decoder $d(V,X)$.

For the bridge, the latent derivative blocks obey

$$
H_{VV}^{\rm bridge}=A^\top H_{TT}A,\qquad H_{VX}^{\rm bridge}=A^\top H_{TX}.
$$

The bridge tests the effect of a fitted affine geometry; it is not generally the optimal conditional mean. Predicted T can leave the valid parameter domain, so report this fraction and use a predeclared common support or mark invalid points, not silent clipping. Compare a bridge in supervised orthogonal coordinates versus unrestricted linear fit as separate diagnostics.

For a deeper diagnostic interpolate two compatible response surfaces in the same coordinate frame:
$$
g_\alpha=(1-\alpha)g_{\rm bridge}+\alpha d,\qquad \alpha\in\{0,.25,.5,.75,1\}.
$$
On a probability channel mix probabilities, not arbitrary inconsistent links. The Hessian is linear in the surfaces, but HE includes cross-products:
$$
M_\alpha=(1-\alpha)^2M_b+\alpha^2M_d+2\alpha(1-\alpha)E[H_b\odot H_d].
$$
This helps locate which energy changes cause false minima, lifted ties, and inferred-only success.

A binary optimal predictor conditional on an independent context embedding V may be
$$
m_p(v,x)=E[\sigma(\eta(T,x))\mid V=v,X=x],\qquad m_L=\operatorname{logit}(m_p).
$$
A continuous optimal predictor is $m_C(v,x)=E[\eta(T,x)\mid V=v,X=x]$ if noise has conditional mean zero. Neither need equal the true surface at $E[T\mid V]$. Context embeddings that include the target response can invalidate those conditional-independence interpretations; use disjoint context/query diagnostics or report full-context leakage explicitly. Do not label a difference between bridge and decoder as a rigorous additive error decomposition.

For an optional representation/noise study, train a limited set of NPs with different context sizes, training seeds and beta under an identical predeclared T-free selection rule. Keep the formal NP fixed for the primary panel. Compare paired outcomes on the same toy and record changes in supervised ceiling only afterward.

### 1D. Test the analytic landscape beyond the sweep

For the 17 Lab 03 analytic-perfect metrics and comparable Lab 06 candidates, perform analytic-only random orthogonal starts and multiple refinements. Report best loss versus signed-permutation loss, distance to a signed permutation, and ties/near-ties. A counterexample refutes the claimed global preference for that metric and measure; failure to find one is empirical evidence, not a proof. This can run concurrently with the other Step 1 diagnostics.

**Deliverables:** unified run manifest, full rotation-level loss registry, per-class pass tables, analytic/global diagnostics, representation diagnostics, bridge curves and integration-convergence report.

**Decision:** retain local analytic-success metrics for improved inference; if the analytic loss itself has competing minima, send it to Steps 3–5 rather than trying to repair it solely with better estimation. Retain inferred-only successes in a separate empirical candidate lane for Step 6.

## Step 2 — Improved inference of existing metrics (Gap 2)

**Scientific question:** can we infer the structure of the unknown response surface more faithfully than differentiating the current decoder?

Freeze the primary latent vectors, metric registry, evaluation measures and rotation manifests from Step 1. Prioritize the analytic-perfect candidates and include row-wise inferred-only candidates and negative controls. Do not tune degree, bandwidth or regularization against T, oracle loss curves or R*.

### 2A. Fit new smooth response surfaces

Two independent data sources:
- D0: observed $(Z,X,Y)$; no use of the decoder.
- D1: deterministic frozen decoder outputs on observed or permitted queried $(Z,X)$; smooth distillation.
- D2: decoder-generated stochastic choice, optional loss-of-information control.

For continuous D0, fit conditional mean with MSE. For binary D0, fit Bernoulli likelihood with a declared link or estimate conditional probability directly; Y-L requires a logistic/logit estimator, and Y-P a probability estimator. Do not regress raw choices as if they were differentiable logits.

For D1, match the selected channel with MSE on logits or original-scale continuous means, and soft-label cross-entropy or MSE for probabilities. Synthetic queries should use observed support or an explicitly declared extrapolation measure. No oracle labels.

First run these surrogate families from 04.3:

| Method | Concrete first configuration | Main risk |
|---|---|---|
| Polynomial ridge / logistic ridge (I05) | complete total degrees 2, 3, 4; optional 5 | global truncation bias; high-degree instability |
| Polynomial chaos / orthogonal basis (I06) | Hermite/Legendre where justified; degrees 2–4 | product basis orthogonality may fail for correlated or fixed design |
| RBF / KRR (I08) | smooth Gaussian kernel, 256/1024/2048 landmarks, held-out ridge/bandwidth | derivative bias from kernel length scale |
| Smooth MLP (I11) | 2–3 GELU/tanh layers; width 64/128 | prediction accuracy may not improve derivative structure |
| Local polynomial (I07) | quadratic/cubic, kNN neighborhoods 256/1024/4096 rows if feasible | insufficient independent subjects and local rank |
| Sparse GP (I09) | RBF or Matern 5/2, 256/512 inducing points | cost and prior-dependent derivative behavior |
| Penalized spline/GAM (I10) | univariate/bivariate tensor terms, explicit higher-order sensitivity | pairwise-only model can erase true higher-order structure |

Use the same generic basis rule for all families; do not inject known CPT expressions or known sufficient features. Fit on the base coordinate frame, not separately on every R unless testing coordinate-dependent estimation deliberately. Use the chain rule to transform derivatives. Save predictions and fitted objects/coefficients. Hyperparameters use held-out prediction, numerical conditioning and derivative stability across independent resamples, not matching T.

A rotation-invariant smoothing variant is useful:
$$
g_h(w)=E_\xi[d(w+h\xi)].
$$
Use isotropic perturbations in a fixed standardized latent frame and support-aware X perturbations; smoothness bandwidth is fixed from blind data criteria. Then differentiate the smoothed surface or use averaged derivatives where justified. It suppresses high-frequency curvature, but can erase genuine sharp structure; compare no smoothing and several predeclared bandwidths.

### 2B. Estimate the small entries rather than merely predict Y

Let $\hat h=h+b+e$ with conditional zero-mean stochastic error e. Then $E[\hat h^2]=(h+b)^2+E[e^2]$. The positive squaring term is particularly harmful when true energy is zero. Cross-fit independent subject splits:
$$
\widehat M_{ij}^{\rm cross}=E_{\rm eval}[\widehat h_{ij}^{(1)}\widehat h_{ij}^{(2)}].
$$
If errors are independent and centered conditional on evaluation points, this removes the variance contribution, not shared bias. Shared NP or common teachers can induce dependence; resampling fixed decoder evaluation points measures integration variability, not decoder-fit uncertainty.

Compare:
- I12 independent subject-level surrogate fits and cross-products; repeat splits.
- Bootstrap at the subject level, including refitting the surrogate; independent NP replicates where possible.
- Noise-floor correction from estimated derivative variance; preserve signed raw estimates and flag truncation.
- Shrinkage of weak energies using common blind uncertainty rules; compare against unshrunk controls.
- Probabilistic support, e.g. $L=\sum_{ij}P(M_{ij}>\delta\mid\mathcal D)$ under a declared statistical model, or confidence-bound-based support score. A bootstrap exceedance frequency is not automatically a Bayesian posterior probability.

Thresholds should be expressed in a fixed invariant energy scale, not tuned to oracle zeros. A correction defined cellwise in rotated coordinates may itself alter invariance and create bias. Record such choices; preferably estimate derivative-error tensors in the base frame and transform consistently. If corrected estimates become negative, save them before clipping; a cross-covariance estimator need not be positive semidefinite.

### 2C. Estimate energy directly from WY

Investigate I13/I14 and direct estimation of L after 2A/2B. For a fixed orthonormal basis $\psi$ and continuous mean or probability response, define
$$
D_{rs}^{ij}=E[\partial_{ij}\psi_r\partial_{ij}\psi_s],\qquad
\mu=E[Y\psi(W)],\qquad M_{\rm proj}^{ij}=\mu^\top D^{ij}\mu.
$$
Use products of independent subject-level mean feature-response vectors to estimate the quadratic functional. The target is projected energy at finite basis size, not exact true energy. The linear moment identity does not directly estimate binary logit energy. Direct observed local contrasts need matched designs or local approximation, not fictitious exact finite-difference neighbors.

### 2D. Evaluation

Primary comparison with AD:
- analytic/inferred Pearson and Spearman; curve minima and pass counts;
- energy matrix/tensor error and false-positive energy in oracle-zero entries (evaluation only);
- stability of near-zero support and minimum margins;
- prediction performance, independent-fit variance and cost.

Use subject-level train/validation/evaluation splits, no row leakage. A fitted surrogate can reduce predictive performance slightly and still improve the scientific metric, but selection of that trade-off must be fixed from allowed-data diagnostics. No candidate is accepted just because the fit to Y improves.

**Deliverable:** inference leaderboard with one row per metric × estimator × source × measure × channel × toy × variant, and ablation of smoothing/variance correction. A convincing improvement should increase CR pass coverage without destroying classic controls. All-family success is evaluated independently in B and C; a shared method across both is an additional goal.

## Step 3 — Conditional and multiscale Hessian energy

**Scientific question:** does averaging Hessian energy over every stimulus before measuring sparsity erase the structural differences that identify parameters?

This extends the existing metric framework. It is not merely another numerical estimator of the original pooled M.

### 3A. Change the aggregation order

Current global estimator:
$$
L_{\rm pooled}(R)=\Phi(E_{V,X}[H_R^2]).
$$

Stimulus-conditioned energy:
$$
M_R(x)=E_{V\mid X=x}[H_R(V,x)^2],\qquad
L_X(R)=E_X[\Phi(M_R(X))].
$$
For a declared product evaluation measure replace $V\mid X=x$ by the same fixed $V$ measure for all x. The two laws are distinct. Nonlinearity of $\Phi$ means $L_X$ is generally not $L_{\rm pooled}$. For concave separable penalties with fixed scale, Jensen gives $E[\Phi(M_X)]\leq\Phi(E[M_X])$; normalization or variable weights may invalidate that particular inequality. A smaller objective is not itself proof of better recovery.

Evaluate the same metric in several orders:
- P0: pooled latent and stimulus averages, then metric (baseline).
- P1: condition on X, average over latent, compute metric, average X.
- P2: condition on latent, average over X, compute metric, average latent.
- P3: compute pointwise metric of $H_R^2$, then average both (more sensitive to noise).
- P4: groups of complete X rows, compute one energy matrix per group, then aggregate.

Study whether conditional support is complementary across trials. For example, two different groups may each require a small subset of axes although the pooled matrix is dense. A useful criterion should reward a common coordinate frame, not independently choose a different R for every group.

### 3B. Grouping and measures without model-specific rules

For fixed bank designs compute P1 on each of the 500 complete rows where the chosen derivative target is defined. For random designs use sampled complete X vectors and latent integration. Start with no labels and no special case for CPT. For P4, use generic X-only clustering or predeclared quantile bins, at several resolutions (e.g. 4/8/16 clusters) with minimum group coverage. Freeze grouping before rotation. Use identical algorithms and settings across toys; known metadata groups can be a separately labeled explanatory arm, never a hidden model-specific shortcut.

For all groups:
$$
L_{\rm group}=\sum_s\omega_s\Phi(M_s(R)).
$$
Compare design-frequency weighting and equal group weighting. Use pooled invariant scale, group-local invariant scale, and energy-weighted normalization as explicitly different settings. Never drop weak or invalid groups at selected rotations. Handle all-zero groups using a fixed rule. Report pooled and conditional information, group sizes, effective sample sizes and sensitivity to partition seeds.

### 3C. Initial metric panel

Apply Soft-L0 (rational/exponential/multiscale), fractional powers, Hoyer, entropy and row/column recipes from 04.3 and the fixed baseline registry. In addition, compare a generic combined score
$$
L_\lambda=\lambda L_{\rm pooled}+(1-\lambda)L_{\rm conditional},\quad
\lambda\in\{0,.25,.5,.75,1\}.
$$
Components must have comparable scales fixed before search; no per-model fitting of lambda. Analyze Pareto trade-offs as diagnostics, not an excuse for selecting a toy-specific winner. Explore condition-specific active-axis counts (including first derivatives in Step 4), so a parameter invisible to second-order curvature need not be treated as absent.

### 3D. Efficient implementation and validity

Cache group-specific derivative second moments $K^{(s)}$ and $J^{(s,x)}$ as in 04.3. For per-X analysis, streaming or chunking is appropriate; fixed bank evaluation can reuse common latent points. For P2 condition on latent values in the base frame and push them forward with R. Maintain identical masks and weights for each rotation; conditional estimates with insufficient support are flagged, not interpreted as evidence of zero interactions.

For fixed banks also consider a derivative-free condition score based on latent response differences between observed stimuli (Step 5), so the bank boundary does not require extrapolated X derivatives.

**Deliverable:** pooled versus conditional angle landscapes, conditional matrices, stability across partitions, and a clear distinction between metric improvement and inference improvement. First establish analytic preference, then run the same candidate with AD and best Step 2 estimators. Carry inferred-only successes in a separate lane.

## Step 4 — Extend structural information beyond the averaged Hessian

**Scientific question:** is the elementwise mean-squared Hessian too lossy to distinguish scientifically distinct axes?

This step has two parallel branches: richer use of derivatives of order two, and additional derivative orders. Both should be tested at the analytic level before costly inference.

### 4A. Use the full derivative covariance structure

For UU and UX blocks, cache
$$
K_{abcd}=E[H_{ab}H_{cd}],\qquad
J_{ab}^{(x)}=E[H_{ax}H_{bx}].
$$
The existing M is only a set of diagonal entries of these second moments. Retaining cross-products makes rotation contraction exact. It does not automatically make a new metric: a scalar depending only on rotation-invariant spectra, norms or ranks of a complete covariant tensor cannot select a rotation.

Study coordinate-sensitive quantities:
- joint sparsification of derivative matrices across X or latent regions;
- sparsity of cross-products between different coordinate couplings;
- approximate simultaneous diagonalization of PSD matrices $J^{(x)}$ or their group-conditioned versions;
- signed mean derivatives versus squared energies, keeping cancellation limitations visible;
- joint tensor factorization whose loadings are constrained to a common orthogonal latent basis.

For instance a candidate diagnostic is
$$
L_{\rm JD}(R)=\sum_x \omega_x\|\operatorname{offdiag}(RJ^{(x)}R^\top)\|_F^2.
$$
A diagonalizable family with distinct joint eigenvalue profiles can identify axes up to sign/permutation. General CR surfaces need not satisfy simultaneous diagonalizability; joint diagonalization may favor a basis different from T. Include it as a hypothesis, not a universal principle. Test eigenvalue degeneracy, conflicting matrices, and whether diagonalizing one matrix introduces extra couplings elsewhere.

### 4B. First-order information

Compute $G_i=E[(\partial_{u_i}g_R)^2]$ and signed/conditional gradient statistics. C1 intercept is invisible to the logit Hessian but not to the gradient. A global gradient energy sum is invariant under orthogonal rotation, so use conditional support or structured sparsity, not total energy.

One useful descriptor is the covariance of gradients over X, at fixed latent:
$$
C_\nabla(v)=\operatorname{Cov}_X(\nabla_v g(v,X)).
$$
This separates constant/bias effects from stimulus-dependent effects. Investigate the sparsity of gradient mean and conditional covariance jointly with UX and UU energies. Do not assume diagonalization identifies true axes when covariance eigenvalues are repeated.

### 4C. Higher-order information

The third derivative tensor:
$$
D_{ijk}^{(3)}=E[(\partial_{w_iw_jw_k}g_R)^2].
$$
Focus on blocks UXX, UUX, UUU and optionally fourth-order terms only if the third-order oracle adds clear information. C2/C3's linear and quadratic effects may have overlapping UX support but distinct UXX structure. For each order use scale-fixed, sign/permutation-invariant coordinate sparsity; count unique symmetric index combinations and declare multiplicities. Do not optimize raw total squared norm of a full rotation-covariant block.

Combine orders only with dimensionless normalization, fixed weights and a structural-zero-block rule. Evaluate first, second, third separately before any fusion. A candidate family could be
$$
L=\omega_1\Phi_1(G)+\omega_2\Phi_2(M)+\omega_3\Phi_3(D^{(3)}),
$$
with weights predeclared across all toys and components evaluated independently. Third derivatives amplify approximation error; use AD on smooth surrogates and step-convergence checks. At boundaries record the derivative domain and do not hide invalid entries.

### 4D. Theoretical checks

Use the examples of note 03 plus linear-intercept, bilinear, quadratic, repeated-eigenvalue and functionally equivalent rotations as unit-test generators. In particular:
- $f=a x_1+b x_2$ tests unnecessary coupling.
- $f=a x_1+ab x_2$ tests the inadequacy of UU-offdiagonal alone.
- $f=a+bx_1+cx_2$ tests bias invisibility and row-normalization ties.
- $f=ax+bx^2$ tests same-stimulus support.
- $f=(a+b)x_1+(a-b)x_2$ is a counterexample to arbitrary original parameter labels always being the simplest basis.

Prove invariances and derive failure cases wherever possible. A metric cannot recover a scientifically labeled axis if the observable function and input law have a genuine rotational symmetry. Finite oracle scans do not establish identifiability. A high-order method must earn its complexity by solving failures of simpler baselines.

**Deliverable:** mathematically defined structural metrics, unit tests, analytic/global preference results and inference comparison. Only advance expensive higher-order estimates if the oracle demonstrates improved discrimination.

## Step 5 — Independent alternatives to Hessian-based simplicity

**Scientific question:** can the underlying simplicity principle be realized through a different statistical object with better identifiability or robustness?

The metric of 04.1 is specifically a function of Hessian energy. Methods in this step are a separate research branch under note 03, not redefinitions of the original H matrix. Apply the same analytic/ inferred/ blind evaluation framework.

### 5A. Functional ANOVA / Sobol decomposition

Under a fixed independent product measure, use
$$
g(W)=g_0+\sum_i g_i(W_i)+\sum_{i<j}g_{ij}(W_i,W_j)+\cdots,
$$
and variance components $V_S=E[g_S^2]$. Study support and concentration of components involving U, including U–X interactions and higher-order terms, not just their total magnitude. Total interaction variance can favor an incorrectly rotated additive surface.

For correlated latent coordinates, ordinary independent-input Sobol decomposition is not automatically valid. Either declare a fixed independent reference product measure in a suitably standardized/whitened frame (a different estimand if independence is approximate), or use a dependence-aware decomposition with explicit identifiability conventions. Do not independently permute X columns of a fixed bank. For bank X, treat the whole design row as one grouping variable or use empirical contrasts; attribute-wise effects require additional support assumptions.

Compare exact analytic/quadrature evaluation, PCE coefficients when orthogonality is justified, and Monte Carlo estimators on fitted surrogates.

### 5B. Sparse basis representation and functional separability

A generic complete smooth basis:
$$
g_R(u,x)\approx\sum_{\nu,\mu}c_{\nu\mu}(R)\psi_\nu(u)\varphi_\mu(x).
$$
Use effective support, grouped coefficient sparsity, expansion complexity and held-out prediction subject to the same error tolerance. Rotate latent polynomial tensors analytically where possible instead of refitting a different biased model at each R. Compare GAM-type separability and low-order interaction decompositions. Avoid a deliberately restricted basis that encodes the true generator in advance.

### 5C. Reduced-axis prediction and observed-stimulus contrasts

At each X or group of X, estimate how well g or observed Y can be predicted using fewer U axes. For a subset S define
$$
\Delta_S(x)=\inf_{a}E[\ell(Y,a(U_S,x))\mid X=x]
-\inf_b E[\ell(Y,b(U,x))\mid X=x],
$$
with proper binary or continuous loss, and comparable model capacity. Use held-out subjects, fixed tolerances, and search over subsets with k≤6 in this panel. Estimate effective required dimension and stable axis support across conditions. For bank designs repeated rows across subjects make within-item response modeling feasible without needing continuous X derivatives.

A related derivative-free interaction score uses paired complete stimuli:
$$
c_{x,x'}(u)=g(u,x)-g(u,x').
$$
Measure dependence on latent coordinates via gradient energy, conditional variance, or prediction-drop scores. This removes common intercept effects and is an estimand on observed X contrasts. Include a separate score for overall baseline dependence so intercept axes are not lost. Pairing rules must be based on allowed design or distances, not known parameter roles.

### 5D. Description length and symbolic approximations

Consider minimum description length, sparse symbolic regression and generic expression grammars. Compare candidate rotations under the same grammar, coefficient precision, noise model, prediction tolerance and complexity budget. Complexity should include model structure, numeric parameters and possibly coding cost of R if that is part of the declared objective. Do not hand the grammar the exact CPT equation or tune its operators after seeing which true family fails. Test simple expression generators first; full symbolic search is expensive and stochastic.

**Deliverable:** a separate non-HE metric registry, analytic/inferred angle evaluations, cost and assumption comparison against HE. A successful alternative is a result about simplicity-based search, even if Hessian-only criteria fail. Conversely, failure of an alternative is informative about what kind of structural information is insufficient.

## Step 6 — Actual blind search, robustness and generalization (Gap 3)

**Scientific question:** can a completely T-free algorithm select a rotation that recovers scientifically meaningful latent axes, rather than merely pass an R*-centered local screen?

Run an early pilot in parallel with Steps 2–5, using already promising candidates, including the Lab 06 inferred-only B08 combinations after revalidation, Lab 03 inferred leaders and universal analytic candidates. This is a separate empirical lane; do not claim Gap 1/2 based on its outcome.

### 6A. Candidate registry and search

Each candidate is a complete fixed algorithm: metric functional, infer method, source, integration measure, coordinate preprocessing, hyperparameter rule, search procedure and budget. No model-ID-specific weights or metric selection. Class-specific estimators are allowed because the observation type is known. A data-adaptive selector is legitimate only if it uses blind diagnostics, is frozen as an algorithm, and validated on independent toys.

Use orthogonal matrices, with equivalent sign/permutation solutions. Explore:
- Haar-random starts and repeated local refinement (existing Lab 05 baseline);
- Givens coordinate descent / multi-start optimization;
- differentiable objectives on the orthogonal manifold with retraction (QR or matrix exponential), where smooth;
- derivative-free search for hard thresholds or tail ranks;
- additional starts for higher latent dimension, with an explicit budget rule independent of T.

Cache moments in the base frame so repeated evaluation is cheap. When the objective uses fitted responses, freeze the fit and evaluation samples before optimization; optionally validate the candidate loss on independent integration draws to avoid Monte Carlo overfitting. No R* initialization, early stopping by T, or choosing the run with the highest supervised correlation. Pick the best blind objective according to a predeclared rule. Save every start's final loss, convergence, matrix and runtime.

### 6B. Evaluation against the real scientific target

Only after the blind solution is locked, compute $U=RV$ on held-out subjects. Match columns to T with Hungarian matching on absolute correlations. Report matched correlations individually, MCC, min matched correlation, mean/max leakage, and distance to the supervised reference modulo signed permutations. Also report predictive invariance before/after rotation and orthogonality error.

The formal Search Success Criterion in note 01 is qualitative (one-to-one correspondence). For comparability retain the historical operational Lab 05 criterion MCC within 0.05 of supervised MCC and max leakage <0.2 for classic families / <0.3 for CR; these are historical evaluation thresholds, not inputs to search and not a newly ratified universal definition. Add continuous-valued results and a stricter common-threshold sensitivity table. Fix any new final criterion before confirmation.

For diagnostic comparison only evaluate the locked blind objective at R*:
- If an incorrect blind solution has substantially lower loss than R*, the objective/inference/representation is implicated rather than a mere search failure.
- If the found loss exceeds L(R*) and the recovered directions are poor, search may be insufficient; increase budget with a T-free rule or diagnose numerical issues.
- If losses tie for different directions, identifiability is weak for this objective.
- If blind recovery succeeds despite analytic failure, report the empirical success but do not relabel it as Gap 1/2 success.

A high MCC alone can hide a poorly recovered individual axis. A rotation may be close to R* but far from perfect T recovery if the supervised ceiling is low. Conversely a different R may satisfy the scientific criterion; R* is a reference, not unique ground truth.

### 6C. Confirm on genuinely independent data and models

The current 12 toys are the development panel. Screening many methods on them uses synthetic truth for research selection; it cannot be treated as an unbiased generalization test. Freeze the winning algorithm(s) before confirmation.

- New subjects, noise and NP training seeds; keep each toy's prescribed observation mechanism.
- Additional signal/noise levels and trials per subject; compare paired B/C designs with controlled training when testing observation effects.
- New generating functions not in the original six families, constructed before method evaluation, spanning additive, multiplicative, sparse nonlinear and context-dependent effects.
- Include deliberate nonidentifiable controls where arbitrary original parameter axes are not uniquely simplest; expect ties/ambiguity rather than false confidence.
- Evaluate sensitivity to dimensionality, latent covariance, fixed-bank support and domain boundaries.

A model-dependent switch based on knowing that a toy is CR is prohibited. A common adaptive rule based on observed X/Y/Z or derivative evidence is allowed, but its selection itself is part of the algorithm that must generalize.

**Deliverable:** blind search manifests, loss traces, found rotations, independent held-out T–U heatmaps, full parameter-level metrics, seed-level uncertainty, class-specific and cross-class success tables, and a confirmation report. A universal claim needs consistent recovery with the same algorithm or explicitly justified observation-class variants, not a collection of individually best methods.

---

# Part III. Execution plan, priorities and decision rules

## 7. Major dependencies and parallel work

    Step 1: unified 12-toy benchmark + failure diagnostics + analytic global screen
       |
       +-- Step 2: improve inference for fixed promising metrics
       +-- Step 3: conditional/multiscale metric family
       +-- Step 4: richer derivative structure and additional orders
       +-- Step 5: non-Hessian simplicity alternatives
       |
       +-- Step 6 pilot: R*-free search of existing empirical candidates
                     |
                Step 6 confirmation: frozen algorithms on independent evidence

The arrows are logical dependencies, not a prohibition on parallel work. Steps 3–5 have analytic-first branches independent of surrogate development. Step 6 may run early with existing objectives. Full confirmation is only after a registry/algorithm is frozen.

## 8. First implementation queue

| Order | Work package | Why now | Gate to continue |
|---|---|---|---|
| 1 | 07 unified baseline, 24 libraries, all valid channels, reproducible masks | establish comparable results | data/loaders/units/tie controls pass; missing/invalid states explicit |
| 2 | Oracle bridge + global analytic screen | distinguish representation vs metric vs surface | identify competing minima and bridge failures; no truth in blind registry |
| 3 | Polynomial/PCE/KRR plus smooth distillation | high potential, lower cost than GP | improved fidelity or recovery without sacrificing classical controls |
| 4 | Conditional HE, per-X and X-only grouping | directly tests information lost by pooling | new analytic preference or robust inferred improvement |
| 5 | Cross-fitting and noise-floor correction | address near-zero bias | reduced false energy without suppressing real effects; independent-fit checks |
| 6 | Gradient/UX covariance, joint diagonalization, UXX | test whether M loses essential structure | demonstrable analytic discrimination; only then costly estimation |
| 7 | ANOVA/PCE sparsity and reduced-axis/contrast prediction | independent test of simplicity | same local/global diagnostics, clear assumptions |
| 8 | Blind pilot and independent confirmation | evaluate final scientific goal | predeclared criteria on unseen runs/models |

These are proposed priorities, not measured costs or deadlines. Start with manageable grids; escalate evaluation sample size for finalists and for fragile apparent passes. Use representative C1, C3, CR4 and their L counterparts for implementation smoke tests, never for selecting toy-specific hyperparameters. Report full primary-panel coverage before claiming generality.

## 9. File and information architecture (proposal)

Keep each experimental wave independent; never overwrite historic experiments. Suggested new lab folders:

    metric_lab/
      07_unified_benchmark/
      08_improved_inference/
      09_conditional_hessian/
      10_extended_derivatives/
      11_alternative_simplicity/
      12_blind_search_confirmation/

These names are proposed and not existing paths. Step 1 bridge and analytic search can live under 07. Each lab contains a README, a versioned config registry, shared read-only loaders or adapters, execution scripts, numerical/unit tests, a results analyzer, and small reproducible summaries. Large arrays go under the corresponding outputs/metric_lab/ directory with manifests. Reuse code instead of copying incompatible loaders into every lab.

Information separation:

    inputs_blind/   Z, X, Y, frozen decoder, permitted preprocessing/design metadata
    eval_only/      T, f, oracle energies, R*, parameter-matching results
    configs/        metric/infer/measure/search registry and seeds
    results/        frozen outputs, diagnostics, evaluation tables

Separate actual process interfaces or modules, not just folder labels. Blind estimation and global search functions must not accept an oracle object or path to supervised rotations. The sweep harness may pass a rotation matrix R as a fixed candidate, but must not pass its angle label or privileged reference to the inference method. Candidate scores must be identical for the same R regardless of its label.

Suggested manifests/results:
- toy_registry.json, config_registry.json, run_manifest.json, rotation_manifest.csv;
- metrics_long.csv, correlations.csv, sweep_minima.csv, pass_summary.csv;
- energy_matrices.npz, derivative_moments.npz, integration_diagnostics.csv;
- fit_diagnostics.csv, representation_diagnostics.csv, bridge_diagnostics.csv (eval only);
- search_trace.csv, found_rotations.npz, recovery_metrics.csv, confirmation_summary.csv.

Every row should carry toy ID/class, family for reporting only, NP run and beta, coordinate variant, channel, metric version, estimator/source, evaluation measure/mask, seed, rotation ID, numerical validity, reason for failure and runtime. Save dataset/checkpoint/latent hashes, scalers, integration points or deterministic replay seeds, and exact code/config versions. Record all attempted settings, not only winners. Preserve signed energy estimates and missing values; never impute invalid as zero.

Unit tests: chain-rule equality for random rotations, contraction versus direct derivatives, prediction invariance, signed-permutation loss invariance, constant/invariant controls, known nonidentifiable cases, all-zero blocks, tiny-energy behavior, fixed-bank mask coverage, train/eval isolation, subject-level resampling and R*-free search inputs.

## 10. Decision table: what to do when an experiment fails

| Observed outcome | Interpretation to investigate | Next action |
|---|---|---|
| Analytic loss flat or wrong minimum | metric information/identifiability | Steps 3–5; global diagnostic; do not tune inference to fix oracle |
| Analytic good, bridge bad | geometry/representation or bridge misspecification | inspect affine alignment, supervised ceiling, support, preprocessing |
| Bridge good, decoder poor | surface/conditional target or derivative approximation | Step 2; independent observed-data surrogate |
| AD and FD disagree | numerical or implementation problem | convergence, precision, domain/chain-rule audit |
| Energy accurate, minima wrong | weak margin, sensitive small entries, aggregation | noise-aware estimates and conditional metrics |
| Analytic fails, inferred passes | symmetry breaking, smoothing or error compensation hypothesis | report separately; replication and blind pilot |
| Both pass local, blind search fails | global competing minima or optimization | compare blind loss to reference after lock, random landscapes, budget study |
| Both fail, blind search occasionally succeeds | possible accidental/weak recovery | repeat, examine margins and selection bias |
| Fixed-bank derivatives unstable | support/extension assumptions | report separate target; contrast/reduced-axis methods |
| No candidate generalizes | principle or representation insufficient under assumptions | document limits; investigate identifiability rather than endlessly tune |

## 11. Scientific end points

We seek two related but distinct contributions.

1. **Theory-oriented:** characterize a simplicity functional whose true response-surface landscape identifies the parameter coordinates (when the generating family has that identifiability), and infer that landscape accurately without truth.
2. **Algorithm-oriented:** a fully blind procedure that finds useful, stable, interpretable axes from observed data and a learned representation, even if its effectiveness partly comes from the learned conditional response geometry rather than faithfully reconstructing the analytic metric.

Do not collapse these claims. The original three-gap theory remains the primary hypothesis; inferred-only success is a secondary empirical path that needs its own explanation. A useful negative result can specify which structural assumptions fail, which observables do not identify axes, or which representation quality is needed.

The project is not committed to the Hessian as the only possible representation of simplicity. It is committed to evaluating every proposed method honestly against the same scientific target, with no true parameters or generating function available to the actual algorithm.

---

**Implementation status:** this roadmap adds no experiments, changes no official test bed, and changes no existing success criterion. All new folder names, grids, prioritization and statistical extensions are proposals to be implemented and evaluated. Historical numbers above are snapshots of the cited results, not newly run experiments.