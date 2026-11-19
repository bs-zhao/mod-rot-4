# Neural Process module (Mod-Rot-4)

Seven independent scripts reproduce the scientific functionality of
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

S9 writes **each rotation separately**, grouped by subset size:

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
    k02/axes_01_02__deg_000.npz
    k02/axes_01_02__deg_015.npz
    ...
    k03/axes_01_02_03__deg_045.npz
    ...
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

For T_dim=3: 4 subsets × 7 angles = 28 matrices, 4 heatmaps + 4 scatter
grids. For T_dim=4: 11 subsets × 7 = 77 matrices, 11 heatmaps +
11 scatter grids. With `--axes all` the count uses L instead and can be
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

No existing S8 or S5 artifacts are modified. S9 refuses to overwrite a
nonempty previous S9 directory.

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
    s9_rotation_sweep/summary.json
    s9_rotation_sweep/rotations/kXX/axes_...__deg_XXX.npz
    s9_rotation_sweep/figures_45deg/kXX/axes_.../rz_vs_t_heatmap.png
    s9_rotation_sweep/figures_45deg/kXX/axes_.../rz_vs_t_scatter.png
    s9_rotation_sweep/figures_45deg/kXX/axes_.../rz_axis_correlations.png
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
