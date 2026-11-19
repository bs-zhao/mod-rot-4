# Neural Process module (Mod-Rot-4)

Five independent scripts reproduce the scientific functionality of
`mod-rot-3/np` S3–S7 for C1, C2, C3, CR3, CR4 and CR5. They consume
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
```

Training conditions include beta/lambda/context/L and a hash of all relevant
training hyperparameters. `--repeat` identifies independent repetitions.
Pass identical run selector parameters for S3–S7. Use `--run-id <name>`
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
