# Metric Lab 01 — Mean-Squared Hessian Curvature Energy (HE)

Independent experiment for [mod-rot-4](../../README.md) and the [HE mathematical specification](../../docs/01_metrics/01_hessian_mean_squared_curvature_energy.md). This code reads upstream assets; **it never modifies S1–S9**, or saves into their directories. Outputs are written to `outputs/metric_lab/01/<MODEL>/<NN condition>/<repeat>/<experiment tag>/`.

## Two *strictly separate* scientific paths

1. **Analytic Oracle (ideal T representation)**: draw T from the true generating prior, use the original generating logit/probability formula and X design, and define an ideal rotation with **G = R @ R_star.T**. Standardize T using S8's **TRAIN T mean and std**; `U_oracle = G @ D_T^{-1} (T-mu_T)`. Hence `T = mu_T + D_T @ G.T @ U_oracle`. Compute exact input Hessians of the generating formula, transform with `J_T = D_T @ G.T`, square each element, and integrate. **No fitted T-hat, S5 Z, NN decoder or inference method enters oracle Hessians.** S8 is read only for the coordinate anchors R*, T mean and std.
2. **Inference (actual learned U)**: `U = R @ D_Z^{-1} (Z-mu_Z)`, `W=[U,X_raw]`. Methods either differentiate the **frozen** S3 decoder, fit a function using only W,Y, or estimate the quadratic functional directly (future extension). S9 uses `mu_Z=0,D_Z=I`; S9_2 uses saved *all-subject* Z statistics, including TEST (the existing transductive convention). The decoder really receives `[X_standardized,Z_original]`; reverse U and the TRAIN-only X scaler before network input. All HE output is in **W=[U,X_raw]** coordinates, not decoder X_std coordinates.

The Oracle and inference functions are not guaranteed identical: S8 R* is a best-fit alignment, not proof that Z is an affine exact representation of T. Comparing landscapes tests **recovery of ideal T-space rotation geometry**, not unbiased estimation of an actual Z-space ground truth.

**HE, not a Hessian norm or a PSD Gram matrix:**

`M_ij=E_Q[(d²f/dW_i dW_j)²]`, with full `(dimU+dimX)²` matrix and UU / UX / XX blocks. `M` is symmetric and elementwise nonnegative for plug-in methods, but **need not be positive-semidefinite**.

## What is actually implemented

| Layer | Methods | Notes |
| --- | --- | --- |
| Oracle | `analytic_oracle` | C1/C2/C3/CR3/CR4/CR5 torch differentiable faithful generating formula, cross-moment cache, two target functions: true logit/true probability |
| Existing frozen decoder | `nn_autograd`, `nn_finite_difference` | Differentiates decoder **inputs**, not model weights; base Z/X Hessian re-used across R |
| Blind W,Y functions | `wy_polynomial`, `wy_local_quadratic`, `wy_kernel_ridge` | Fits per R, not using oracle formulas or true T |
| Direct W,Y quadratic functionals | **Not yet implemented** | Registered only when a statistically valid D1-style estimator with appropriate measure and repeated-subject treatment is available. Never mislabel a fitted polynomial as direct. |
| Integration | `theory_x`, `observed_joint` | `theory_x` = known T prior (oracle) or fitted *joint* Gaussian Z (inference), model X uniforms or the **entire exact 500-question fixed CPT bank** crossed with sampled latents. `observed_joint` = real paired (T,X) or (Z,X) rows sampled subject-equally. |
| Matrix metrics | Total energy, diagonal/off-diagonal UU, HHI, entropy, effective number, Hoyer, top-k, tail-L1/L2, UX row concentration | Registered by name and scope, evaluated identically for oracle and inferred matrices |
| Diagnostics | Oracle–inference Pearson/Spearman, MAE/RMSE, R-index ranking, multi-Givens path angle curves | Distinct `baseline_unswept` output retained, duplicate physical R removed for correlations |

The *full* CPT X Hessian is generally **not a well-defined classical two-sided derivative** at fixed bank zero amounts or p=0,1. Default `--boundary-policy mask` retains complete fixed bank and marks any affected UX/XX matrix cells as NaN, rather than silently imposing derivatives of a numerical continuation. `--boundary-policy interior` explicitly changes the integration target: only fully interior stimuli are retained, and the exclusion fraction is reported. Zero-alpha fractional powers have a safe mathematical continuation for T derivatives at zero amounts.

**Important practical limit:** a NaN HE block produces NaN block metrics, not arbitrary zero. For CR3/CR5 many UX/XX full-bank oracle metrics can be undefined. Use UU diagnostics or an explicitly matched interior-domain evaluation, and do not claim full-bank HE correctness on masked blocks.

## Inputs and response semantics

Existing files:
- S1 `outputs/models/generate/MODEL/s1_generate__dataset.npz`: T, raw X, choice, indexing
- S3 `.../s3_train/checkpoint.pt`: frozen model weights, decoder X mean/std
- S5 `.../s5_extract_z/latent.npz`: original subject-level Z and hashes
- S8/S8_2 `.../rotation.npz`: R*, T TRAIN mean/std and S9_2 global Z normalization
- S9/S9_2 `.../rotation_index.csv` and rotation NPZ including `delta_R`: complete R and baseline

The decoder represents a categorical **class-1 minus class-0** logit; its sigmoid is the NN probability. Four prepared Y versions are **saved once and reused for all rotations**: `true_choice` (S1 binary), `nn_choice` (once sampled Bernoulli NN p), `nn_probability`, `nn_logit`.

Binary choices do **not** have differentiable Hessians: blind logistic polynomial estimates smooth p via the logit chain rule; local polynomial / kernel ridge approximate the conditional probability by WLS / KRR. A frozen NN method differentiates its learned p or logit, **not the sampled 0/1**; it does not refit to a selected Y variant. All metric output records the target, inference source and Y label.

## Quick start (Windows, from repository root)

Install in your existing `py11` environment:

```bat
conda activate py11
cd /d D:\Projects\rotation\mod-rot-4
python -m pip install -r metric_lab/01/requirements.txt
```

Use the **exact** existing NN run directory (its condition string is hashed by S3; do not guess it):

```bat
python metric_lab/01/run.py --model CR3 --run-dir "outputs/nn/CR3/YOUR_CONDITION/repeat01" --stage validate --variant s9
python metric_lab/01/run.py --model CR3 --run-dir "outputs/nn/CR3/YOUR_CONDITION/repeat01" --stage all --variant both --methods nn_autograd,wy_polynomial,wy_kernel_ridge
```

Separate stages can be rerun:

```bat
python metric_lab/01/run.py --model C1 --run-dir "outputs/nn/C1/YOUR_CONDITION/repeat01" --stage oracle
python metric_lab/01/run.py --model C1 --run-dir "outputs/nn/C1/YOUR_CONDITION/repeat01" --stage prepare
python metric_lab/01/run.py --model C1 --run-dir "outputs/nn/C1/YOUR_CONDITION/repeat01" --stage infer --methods all
python metric_lab/01/run.py --model C1 --run-dir "outputs/nn/C1/YOUR_CONDITION/repeat01" --stage metrics
python metric_lab/01/run.py --model C1 --run-dir "outputs/nn/C1/YOUR_CONDITION/repeat01" --stage diagnose
```

**All S9/S9_2 rotations**, including the independent unswept baseline, are analyzed by default. A cheap initial end-to-end smoke run:

```bat
python metric_lab/01/run.py --model C1 --run-dir "outputs/nn/C1/YOUR_CONDITION/repeat01" --stage all --variant s9 --max-rotations 4 --fit-rows 256 --n-eval 32 --latent-draws 2 --methods nn_autograd,wy_polynomial
```

For CR3/CR5 with `--measure theory_x`, `--latent-draws 16` evaluates 16×500 = 8,000 paired input points **in both** the ideal T prior and Gaussian Z measures; torch Hessians on such grids can take substantial CPU/GPU time. Start with `--latent-draws 2`, use `--chunk 16` to bound memory, then increase to test convergence. Use `--boundary-policy interior` only when your research question explicitly changes to the interior-domain estimand.

`--fit-rows 0` uses all available W,Y rows (not recommended for exact kernel methods). Nonzero fit counts are sampled subject-equally, reproducibly with the seed. `--n-eval 0 --measure observed_joint` uses **all** S1 trial rows; nonzero counts sample by subject then by valid trial. When using `--measure theory_x`, `--n-eval` is the sample count for non-fixed X, and `--latent-draws` controls the fixed 500-question cross-product.

## Output format

```text
outputs/metric_lab/01/<MODEL>/<condition>/<repeat>/<eval+fit+hash>/
  run_config.json
  validation/smoke.json
  prepared/responses.npz + responses.json
  cache/{oracle,nn}/...npz + ...json
  oracle/{s9,s9_2}/<measure>/analytic_oracle/true_{logit,probability}/<rotation>.npz
  inference/{s9,s9_2}/<measure>/<method>/<y_name>/<rotation>.npz
  metrics/{s9,s9_2}/all_metrics.csv
  diagnostics/{s9,s9_2}/correlations.csv
  diagnostics/{s9,s9_2}/curves/*.png
  inference_failures.json
```

HE NPZ includes M, M_UU, M_UX, M_XX, RMS, valid mask and dim_u. Each has matching JSON (R ID, axes, angle, source, target, evaluation measure, method-specific metadata). A shared deterministic experiment hash includes S1/S3/S5 SHA256 values and numerical settings, preventing reuse across changed upstream datasets or checkpoints.

## Correctness constraints and extensions

- Never derive rotated `M` by `R @ M @ R.T`. Each entry is **expected square of a transformed Hessian element**; store the full cross moment `E[vec(H) vec(H).T]` and contract with the coordinate Jacobian. The all-sweep implementation does this for oracle and frozen NN.
- `X` is always the original physical X, including for the NN (internal TRAIN X standardization is undone by differentiation). `Z` is the original S5 posterior mean, not the entire encoder computational graph.
- Changing evaluation measure changes the estimand. Do not regard Gaussian Z with fitted covariance as an exact representation of T's true prior; or marginally re-fit independent U distributions per angle.
- An estimator fitting W,Y does not have access to T, the analytical generator or oracle H. This implementation intentionally does not auto-tune hyperparameters using oracle results. The immediate goal is descriptive metric-landscape comparison, not leakage-free claims about generalization.
- Add new W,Y estimators to `he_lab/estimators.py:REGISTRY`, new metrics to `he_lab/metrics.py`, and preserve the common HEResult interface. Add independent future experiments as `metric_lab/02`, not by importing implementation from `01`.
