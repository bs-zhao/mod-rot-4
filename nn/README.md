# Neural Process module (Mod-Rot-4)

Nine independent scripts reproduce the scientific functionality of
`mod-rot-3/np` S3–S9 for C1, C2, C3, CR3, CR4 and CR5. They consume
**only** `outputs/models/generate/<MODEL>/s1_generate__dataset.npz` and the
corresponding qualified S2 report from `models/`. No model simulation,
legacy `outputs/choice` data or second Generate/Precheck exists here.

## Common rule: standardize X once, before every neural-network operation

All six models share the **same X preprocessing**:

1. Split **subjects** into TRAIN/VALIDATION/TEST (70/15/15).
2. Fit one `mean_j`, `std_j` for each raw X feature using **TRAIN X only**.
3. Set `X_std[...,j]=(X[...,j]-mean_j)/std_j` for **every subject and every trial**.
4. Pass **only `X_std`** to both the context Encoder and prediction Decoder.
5. Save the `input_standardization` inside S3 checkpoint; S5 reuses the
   identical numbers. Original X in `models/` is never modified.

The context choice channel has its own normalization, as in the original NP.
No further learned or model-specific X normalization is used.

## Commands

From repository root (after `pip install -r requirements-nn.txt`):

```bat
python nn/s3_train.py --model CR5 --latent-dim 3 --beta 0.0005 --repeat 1
python nn/s4_evaluate.py --model CR5 --latent-dim 3 --beta 0.0005 --repeat 1
python nn/s5_extract_z.py --model CR5 --latent-dim 3 --beta 0.0005 --repeat 1
python nn/s6_z_diagnostics.py --model CR5 --latent-dim 3 --beta 0.0005 --repeat 1
python nn/s7_regression.py --model CR5 --latent-dim 3 --beta 0.0005 --repeat 1
python nn/s8_supervised_rotation.py --model CR5 --latent-dim 3 --beta 0.0005 --repeat 1
python nn/s9_rotation_sweep.py --model CR5 --latent-dim 3 --beta 0.0005 --repeat 1
python nn/s8_2_supervised_rotation.py --model CR5 --latent-dim 3 --beta 0.0005 --repeat 1
python nn/s9_2_rotation_sweep.py --model CR5 --latent-dim 3 --beta 0.0005 --repeat 1
```

S3 saves the checkpoint, validation/test summaries, history and data/config
provenance. S4 saves original fixed-context validation/test performance and
learning curves without retraining. S5 computes the Gaussian posterior mean
and logvariance for **all** subjects using **all observed trials**, and
saves immutable `z`, `z_logvar`, TRAIN/VAL/TEST IDs and checkpoint/S1 hashes.
S5 does not save true T into the Z data: downstream S6/S7 load T from S1
for diagnostics only.

S6 saves: posterior mean distributions, split-wise correlations, axis-wise
variance, posterior uncertainty, KL, PCA eigenvalues/ratios, selected original
Z/PC pairwise diagnostics, ICA, FactorAnalysis, all component-to-T matrices,
heatmaps and scatter plots. All unsupervised reductions fit TRAIN only.

S7 saves: independently fitted Ridge probes of Full Z, selected Z and selected
PCs; TEST R², Pearson r and RMSE for every true parameter, three true-T
versus predicted-T correlation heatmaps, held-out predictions, and the
variable-number-of-PCs diagnostic curves. TRAIN fits normalization, VALIDATION
selects alpha; refitting on TRAIN+VAL precedes held-out TEST scoring.

## S8: supervised orthogonal rotation to true T

S8 reads the immutable S5 Z file, the S1 true parameter T and the S3
subject splits. It trains a **genuine orthogonal matrix R*** **on TRAIN
subjects only**, without retraining NP or touching S5:

```text
L = latent_dim; d = T_dim; require L >= d
TRAIN target = (T - TRAIN_mean(T)) / TRAIN_std(T)
TRAIN Z is centered, but is NOT variance-standardized or whitened
R* = argmin_{R R^T=I} ||(Z_train - TRAIN_mean(Z)) R^T - [T_train_std, 0]||²
Full rotated output = Z @ R*.T
U = first d columns of full rotated output (first d axes of RZ)
```

The closed-form solution uses orthogonal Procrustes (SVD). In particular,
for L > d it finds d T-aligned directions and completes them to a full
orthogonal L × L matrix, without pre-selecting the first d **original** Z
coordinates. The first d **rotated** axes align to T in its known parameter
order. This is a supervised diagnostic oracle; it is **not** a blind rotation.

Then S8 uses **only U** as predictor of every true T in a Ridge probe:
TRAIN-only standardization, VALIDATION-selected alpha, TRAIN+VAL refit,
untouched TEST scoring (the same policy as S7). It saves two full TEST
Pearson matrices and their all-pairs scatter grids:

- `rz_vs_t_correlation.csv/png` and `rz_vs_t_scatter.png`: true
  `T_i` versus each `U_j` (including off-diagonal cells)
- `pred_vs_t_correlation.csv/png` and `pred_vs_t_scatter.png`: true
  `T_i` versus each `pred_T_j` from Ridge(U)

Outputs under `s8_supervised_rotation/`:

- `rotation.npz`: orthogonal `R_star`, TRAIN means/SD of T, TRAIN
  mean of Z, source hashes and subject split IDs
- `rotated_z.npz`: `rz` for **all** L axes and `u` for first d axes
- `test_predictions.npz`: TEST ID, T, predicted T, rotated Z and Ridge alpha
- `summary.json`: optimization objective, orthogonality, TEST R²/r/correlations
- Four CSV/PNG correlation and scatter outputs described above

This script requires only S3 checkpoint, S5 Z and S1 T: S6/S7 are **not**
prerequisites. Archive an existing S8 output folder before regenerating
it; no silent overwrite is performed.

## S9: exhaustive rotations around the frozen S8 oracle R*

S9 is independent from S6/S7 and requires **S8 `rotation.npz`**, S5 Z
and the original S1 true parameters. It does not retrain NP or refit R*.

By default, S9 rotates the first **d = T_dim** S8-aligned axes (the
"parameter axes"), leaving any L-d extra latent coordinates untouched.
Use `--axes all` to include all L axes instead. Given d selected axes,
it enumerates every subset of size k=2,3,...,d and applies seven angles:
**0, 15, 30, 45, 60, 75, 90 degrees**. The number of distinct subsets is
`2**d - d - 1`; the number of saved R matrices is 7 times that.

Because a k>2-dimensional "rotation by angle θ" is not mathematically
unique, S9 defines a reproducible **ordered Givens sweep**. For each
sorted subset `(a1,...,ak)`, rotate adjacent pairs in order
`(a1,a2)`, `(a2,a3)`, ..., `(a(k-1),ak)`, using angle θ for
each elementary plane rotation. This is **one deterministic construction**,
not an exhaustive search over every possible k-dimensional rotation.
The same rule is applied to all models; k=2 is an ordinary plane rotation.

Every saved matrix is a full L×L orthogonal matrix:

```text
G(axes, theta) = ordered product of adjacent Givens rotations
R(axes, theta) = G(axes, theta) @ R_star
RZ = Z @ R(axes, theta).T
```

S9 writes **each rotation separately**, grouped by subset size. In addition,
the **unswept original R*** is a standalone first-class baseline, not merely
a duplicate hidden among the 0-degree sweeps. The baseline has its own
complete orthogonal matrix, metadata, diagnostic figures and a distinct
`candidate_type=baseline_unswept` row in `rotation_index.csv`.
Sweep entries use `candidate_type=sweep`. Neither the baseline nor any
0-degree sweep is removed; both are kept for reproducible comparisons.



```text
s9_rotation_sweep/
  rotation_index.csv
  summary.json
  z_variance_diagnostics/
    z_axis_variances.png
    z_variances_all.csv
    z_variances_train.csv
    z_variances_validation.csv
    z_variances_test.csv
    paired_variance_tests_{all,train,validation,test}.csv
    z_variance_equal_tests.json
  rotations/
    baseline/unswept.npz
    k02/axes_01_02__deg_000.npz
    k02/axes_01_02__deg_015.npz
    ...
    k03/axes_01_02_03__deg_045.npz
    ...
  figures_baseline/
    rz_vs_t_correlation.csv
    rz_vs_t_heatmap.png
    rz_vs_t_scatter.png
    rz_axis_correlations.csv
    rz_axis_correlations.png
  figures_45deg/
    k02/axes_01_02/
      rz_vs_t_correlation.csv
      rz_vs_t_heatmap.png
      rz_vs_t_scatter.png
      rz_axis_correlations.csv
      rz_axis_correlations.png
    k03/axes_01_02_03/
      rz_vs_t_correlation.csv
      rz_vs_t_heatmap.png
      rz_vs_t_scatter.png
    ...
```

The saved NPZ rotation files include `R`, `R_star`, the incremental
orthogonal `delta_R`, zero-based and one-based participating axes,
angle, convention and S8/S5 source hashes.

For **every 45° subset**, the other folder provides a correlation
heatmap and all-pairs scatter grid across **all L rotated Z coordinates**
against **all d true parameters**. Pearson correlations use only the
held-out TEST subjects and every TEST observation. Scatter plots may
limit visible points (default up to 2000 TEST subjects per panel,
`--max-test-scatter-points`), without changing the computed correlations.

For T_dim=3: 4 subsets × 7 angles = 28 sweep matrices + 1
independent baseline = **29 candidates**, with 4 sweep heatmap/scatter
sets and 1 baseline set. For T_dim=4: 77 sweep matrices +
1 baseline = **78 candidates**, with 11 sweep figure sets plus baseline. With `--axes all` the count uses L instead and can be
very large; for example L=16 requires 458,633 matrices and 65,519
figure pairs, which can consume substantial time and storage.

In addition, S9 tests whether the **original S5 posterior-mean Z axes have
equal between-subject variances**. It saves per-axis sample variances (ddof=1)
for ALL/TRAIN/VALIDATION/TEST, max/min variance ratios, bar chart, and
pairwise **Pitman–Morgan tests** accounting for the paired observations
(across dimensions of the same subjects) with Holm familywise correction.
TRAIN is the primary inference split; other splits are descriptive replication.
These tests assume bivariate normality for exact p-values; failure to reject
is not proof of equal variance, and equal variance alone does not imply the
joint Z distribution is isotropic Gaussian.

Every 45° subset now **also** produces `rz_axis_correlations.csv` and
`rz_axis_correlations.png` in the same `figures_45deg/kXX/axes_.../`
directory. They show the complete L×L Pearson correlation matrix among
all rotated Z axes on TEST subjects. This is a different diagnostic from
the existing d×L T-vs-RZ heatmap.

No existing S8 or S5 artifacts are modified. S9 and S9_2 replace only their own existing sweep output folders on rerun
(including stale rotation matrices and figure files); S8, S8_2 and S5 are
not overwritten by a sweep. This matches the current sweep implementation.

## S8_2 and S9_2: orthogonal rotations after global per-axis Z-score

These are **separate** pipelines and leave S8/S9 outputs unchanged.

**S8_2:** From S5 all-trial posterior means `Z` (one row per subject),
calculate for *each latent dimension* its mean `mu_all` and population
standard deviation `sd_all` over **ALL subjects** (TRAIN+VAL+TEST).
Define `Zz=(Z-mu_all)/sd_all`. Fail if any latent standard deviation
is effectively zero; never silently replace it with 1. Then run exactly
S8's orthogonal Procrustes T-supervised optimizer on `Zz`, fitting R*
using only TRAIN subject T. Ridge T prediction uses the S8 TRAIN,
VALIDATION, TEST policy, but **the global Z-score includes TEST
subjects**: any TEST metrics are **transductive diagnostics**, not
strictly inductive held-out validation.

```text
Zz = (Z - mean_all_subjects(Z)) / std_all_subjects(Z, ddof=0)
R*_2 = argmin orthogonal R ||(Zz_train - mean_train(Zz)) R.T - [Tstd,0]||²
U = Zz @ R*_2.T
Z = mu_all + (U @ R*_2) * sd_all  [row convention]
```

`R*_2` is orthogonal **in Zz coordinates**, while the complete map
from original Z to U is affine, not a pure orthogonal rotation on
original Z. Because this is per-axis z-scoring, it is **not whitening**:
any cross-dimensional correlations in Z can remain.

Output `s8_2_supervised_rotation/` mirrors S8 (rotation.npz,
rotated_z.npz, test_predictions.npz, both T×RZ/T×pred correlations,
scatter grids, summary.json) and additionally saves
`z_standardized.npz`. A standalone `z_normalizer.npz` saves
`mean`, `variance`, `std`, number of subjects, `ddof=0` and the
S5 source hash; `z_normalizer.csv` lists all axes for inspection.
`rotation.npz` and `z_standardized.npz` also store
`global_z_mean`, `global_z_variance`, `global_z_std`.
For inversion, use **the saved** means and stds, never recompute them.

**S9_2:** Loads the **exact** scaler and frozen R* from S8_2,
validates that they match the source S5 Z, and generates all the
same S9 Givens subset/angle combinations (0 to 90 degrees in 15-degree
steps) in **Zz space**. All R files are full L×L orthogonal matrices
**relative to Zz**, and each R NPZ also contains global means/variances/stds so
the raw-space affine map is fully specified. Its separate directory
`s9_2_rotation_sweep/` mirrors S9:

- `rotations/` and `rotation_index.csv` for every R, including
  `rotations/baseline/unswept.npz` as its own indexed candidate
- `figures_baseline/` for the unswept R*_2 T×U and U×U diagnostics
- `figures_45deg/` for each 45-degree subset's T×U heatmap,
  T×U scatter grid, and U×U correlation heatmap/CSV
- `z_variance_diagnostics/` for the *standardized Zz* axis variances
- `original_z_variance_diagnostics/` for raw S5 Z variance tests
- `summary.json` recording the scaler, oracle stage, and variant

Both S9 and S9_2 use `--axes theta` by default and `--axes all`
optionally. S9_2 requires S8_2 but not original S8 or S9; neither
S8_2 nor S9_2 requires a new NP training or latent extraction.

## Outputs

```text
outputs/nn/<MODEL>/<condition>_<config-hash>/repeat01/
    config.json
    manifest.json
    splits.npz
    s3_train/checkpoint.pt
    s3_train/history.csv
    s3_train/summary.json
    s4_evaluate/metrics.json
    s4_evaluate/metrics.csv
    s4_evaluate/learning_curves.png
    s5_extract_z/latent.npz
    s6_z_diagnostics/z_statistics.csv
    s6_z_diagnostics/z_histograms.png
    s6_z_diagnostics/z_variance_kl.png
    s6_z_diagnostics/s12__pca.npz
    s6_z_diagnostics/s12__ica.npz
    s6_z_diagnostics/s12__factor_analysis.npz
    s6_z_diagnostics/... (more figures and CSV files)
    s7_regression/s12__diagnostics.json
    s7_regression/s12__theta_hat_test_predictions.npz
    s7_regression/... (more plots and CSV files)
    s8_supervised_rotation/rotation.npz
    s8_supervised_rotation/rotated_z.npz
    s8_supervised_rotation/test_predictions.npz
    s8_supervised_rotation/rz_vs_t_correlation.csv/png
    s8_supervised_rotation/rz_vs_t_scatter.png
    s8_supervised_rotation/pred_vs_t_correlation.csv/png
    s8_supervised_rotation/pred_vs_t_scatter.png
    s8_supervised_rotation/summary.json
    s9_rotation_sweep/z_variance_diagnostics/z_variance_equal_tests.json
    s9_rotation_sweep/z_variance_diagnostics/z_axis_variances.png
    s9_rotation_sweep/rotation_index.csv
    s9_rotation_sweep/rotations/baseline/unswept.npz
    s9_rotation_sweep/figures_baseline/
    s9_rotation_sweep/summary.json
    s9_rotation_sweep/rotations/kXX/axes_...__deg_XXX.npz
    s9_rotation_sweep/figures_45deg/kXX/axes_.../rz_vs_t_heatmap.png
    s9_rotation_sweep/figures_45deg/kXX/axes_.../rz_vs_t_scatter.png
    s9_rotation_sweep/figures_45deg/kXX/axes_.../rz_axis_correlations.png
    s8_2_supervised_rotation/rotation.npz
    s8_2_supervised_rotation/z_standardized.npz
    s8_2_supervised_rotation/z_normalizer.npz
    s8_2_supervised_rotation/z_normalizer.csv
    s8_2_supervised_rotation/rotated_z.npz
    s8_2_supervised_rotation/test_predictions.npz
    s8_2_supervised_rotation/... (the same S8 heatmaps/scatters)
    s9_2_rotation_sweep/rotation_index.csv
    s9_2_rotation_sweep/rotations/baseline/unswept.npz
    s9_2_rotation_sweep/figures_baseline/
    s9_2_rotation_sweep/rotations/kXX/axes_...__deg_XXX.npz
    s9_2_rotation_sweep/figures_45deg/kXX/axes_.../... (the same S9 figures)
    s9_2_rotation_sweep/z_variance_diagnostics/...
    s9_2_rotation_sweep/original_z_variance_diagnostics/...
```

Training conditions include beta/lambda/context/L and a hash of all relevant
training hyperparameters. `--repeat` identifies independent repetitions.
Pass identical run selector parameters for S3–S9. Use `--run-id <name>`
to replace the default training-condition tag when required.

Both S3 checkpoints and S5 Z files reject in-place overwrites. The user is
responsible for moving any results they want to preserve before rerunning
diagnostic stages.

The Neural Process network architecture and original NP loss/training loop
are adapted from
[mod-rot-3](https://github.com/bs-zhao/mod-rot-3/tree/main/np).
Only the X standardizer/data adapters, run paths and diagnostics stage
organization are redesigned.

## Smoke test

```bat
python -m unittest discover -s nn/tests -v
```

## `<MODEL>L` (continuous noisy-logit response)

The same entry points accept C1L … CR5L. The decoder has one output, the loss is MSE on the TRAIN-standardized `y` (plus β·KL), and S3/S4 report R² instead of accuracy (`ContinuousNPTrainer`). The scaler is stored in the checkpoint as `input_standardization.y_mean / y_std`: predicted logit = output·y_std + y_mean. S5–S9_2 are unchanged. See the root README.

Stopping defaults differ by model family: `<MODEL>L` uses `--max-epochs 5000 --patience 50`, the choice models keep `500 / 50`. Both values enter the run-folder fingerprint, so L runs trained before this change (500 / 50) are reached only by passing `--max-epochs 500 --patience 50` explicitly to every stage.

