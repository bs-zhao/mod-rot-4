# Mod-Rot-4

A reproducible, modular research platform for six behavioral decision models, simulated data, Neural Processes, latent diagnostics and orthogonal rotations.

**Current implementation:** S1–S2 live in `models/`; S3–S9 (plus S8_2/S9_2) live in `nn/`. The `nn/` module consumes the **single** dataset/precheck produced by `models/`; it does not regenerate data or use legacy `mod-rot-3/outputs/choice/` directories.

For detailed module internals, see [models/README.md](models/README.md) and [nn/README.md](nn/README.md).

## 1. Project architecture

```text
mod-rot-4/
├── README.md                           <- overview + complete six-model commands
├── requirements-models.txt
├── requirements-nn.txt
├── models/
│   ├── README.md
│   ├── s1_generate.py                  <- common simulator for all six models
│   ├── s2_precheck.py                  <- common diagnostics and qualification
│   ├── src/modrot4/
│   │   ├── cli.py                      <- --model, --subjects, seeds
│   │   ├── core.py                     <- one data-generation & NPZ contract
│   │   ├── model_specs.py              <- model-specific T, X, logit, MLE/Fisher
│   │   ├── fixed_bank.py               <- CR3/CR5 fixed designs
│   │   ├── unified_precheck.py         <- shared S2 diagnostics
│   │   └── pipeline.py                 <- model output paths and hashes
│   ├── reference/                      <- mathematical reference generators
│   └── tests/
├── nn/
│   ├── README.md
│   ├── s3_train.py                     <- NP training, TRAIN-only X scaling
│   ├── s4_evaluate.py                  <- saved checkpoint evaluation
│   ├── s5_extract_z.py                 <- full-trial posterior mean Z
│   ├── s6_z_diagnostics.py             <- PCA/ICA/FA and latent statistics
│   ├── s7_regression.py                <- TEST-only true-T recovery diagnostics
│   ├── s8_supervised_rotation.py       <- Procrustes on original Z
│   ├── s9_rotation_sweep.py            <- Givens sweeps around S8 R*
│   ├── s8_2_supervised_rotation.py     <- z-score Z then Procrustes
│   ├── s9_2_rotation_sweep.py          <- Givens sweeps on standardized Z
│   ├── src/modrot4_nn/
│   │   ├── cli.py                      <- shared NN hyperparameters
│   │   ├── pipeline.py                 <- NN paths, fingerprint, X scaler
│   │   ├── dataset.py                  <- consume models/S1 dataset
│   │   ├── net/                        <- discrete Neural Process
│   │   ├── trainer.py                  <- NP training
│   │   ├── analysis.py                 <- regression / frozen-data access
│   │   ├── supervised_rotation.py      <- orthogonal Procrustes
│   │   ├── s9_rotation.py              <- Givens sweep mechanics
│   │   ├── s9_diagnostics.py           <- variance and correlation analysis
│   │   └── training_helpers.py         <- splits / checkpoint utilities
│   └── tests/
└── outputs/                            <- generated locally; not tracked as code
    ├── models/
    │   ├── generate/<MODEL>/
    │   └── precheck/<MODEL>/
    └── nn/<MODEL>/<condition>_<hash>/repeatXX/
```

Some research extensions (e.g. MLP surrogates, Hessian metrics and blind rotation searches) are **research directions**, not part of this documented S1–S9 execution contract. Do not confuse them with implemented S1–S9 stages.

### Stage dependencies and interfaces

| Stage | Script | Inputs | Primary outputs |
|---|---|---|---|
| **S1** | `models/s1_generate.py` | model specification, subject count and seeds | `T`, `X`, `logit`, `p`, `choice` dataset |
| **S2** | `models/s2_precheck.py` | S1 dataset | qualification report, MLE recovery CSV, Fisher checks |
| **S3** | `nn/s3_train.py` | S1 + qualified S2 | trained NP checkpoint, training history, splits, X scaler |
| **S4** | `nn/s4_evaluate.py` | S3 checkpoint | validation/test metrics and learning curves |
| **S5** | `nn/s5_extract_z.py` | S1 + S3 | frozen all-trial posterior-mean `Z` and log-variance |
| **S6** | `nn/s6_z_diagnostics.py` | S5 + S3 | PCA/ICA/FA, variance, KL, correlations and figures |
| **S7** | `nn/s7_regression.py` | S5 + S6 + true T | Ridge recovery and TEST diagnostics |
| **S8** | `nn/s8_supervised_rotation.py` | S5 + S3 + true T | original-Z Procrustes `R*`, TEST diagnostics |
| **S9** | `nn/s9_rotation_sweep.py` | S8 + S5 + S3 + true T | standalone baseline and Givens-sweep rotations |
| **S8_2** | `nn/s8_2_supervised_rotation.py` | S5 + S3 + true T | global-z-scored `Zz`, scaler, orthogonal `R*_2` |
| **S9_2** | `nn/s9_2_rotation_sweep.py` | S8_2 + S5 + S3 + true T | standalone baseline and sweeps in `Zz` space |

S6/S7 are not mathematical prerequisites of S8/S9, but commands below follow the complete numbered S1–S9 order for reproducibility.

### Shared data and training contracts

S1 saves `parameters` (true T), `trial_counts`, `subject_index`, `trial_index`, `x` (raw physical X), `logit`, `p`, and `choice`. S2 recomputes choice probabilities from true T and X for validation; it does not use stored p/logit as privileged inputs.

S3 splits **subjects**, not trials, into TRAIN/VALIDATION/TEST (70%/15%/15%). It fits one X mean/SD vector using only TRAIN-subject stimuli. Both the NP encoder and decoder receive this same standardized X, while S1 raw X remains unchanged. S5 encodes each subject from **all** of their observed trials and freezes subject-level posterior means in `latent.npz`. True T is not used to train the NP or extract its Z.

S8 rotates the original centered latent coordinate system; S8_2 first globally z-scores every S5 Z axis, saving exact means and variances to invert the transform. Both R* matrices are orthogonal **in their respective coordinates**, but they need not be equal. S8_2 uses **all subjects including TEST** to compute the global Z moments, so associated TEST analysis is **transductive**, not strictly inductive. The inverse decoder input transformation is `Z = mu_all + (U @ R) * sd_all` (row convention).

S9 and S9_2 both save an **independent unswept R*** baseline plus all 0°, 15°, 30°, 45°, 60°, 75° and 90° ordered-Givens subset sweeps. By default `--axes theta` sweeps the first `T_dim` aligned axes; optional `--axes all` can produce combinatorially many files.

## 2. Six model configurations

| Model | T dimension | X dimension | Example L | Example β | Stimulus-generation policy |
|---|---:|---:|---:|---:|---|

| C1 | 4 | 3 | 4 | 0.0005 | Per-subject/per-trial independent U(-1,1); 300–500 trials |
| C2 | 4 | 4 | 4 | 0.0005 | Per-subject/per-trial independent U(-1,1); 300–500 trials |
| C3 | 6 | 4 | 6 | 0.0005 | Per-subject/per-trial independent U(-1,1); 300–500 trials |
| CR3 | 4 | 6 | 4 | 0.00005 | Shared fixed 500-trial structured CPT bank |
| CR4 | 3 | 6 | 3 | 0.0005 | Independent random X for every person/trial; amounts U(5,200), probabilities U(.05,.95); 500 trials |
| CR5 | 3 | 6 | 3 | 0.0005 | Shared fixed 500-trial CR3-style CPT bank (separately stored) |

The `Example L` and `Example β` values below are **explicit runnable illustration settings**, not claims of empirically optimal hyperparameters. C3 has **six** true parameters and therefore requires `--latent-dim >= 6` for S8. For CR3 this guide uses β=0.00005; the other five examples use β=0.0005.

All commands below assume **8,000 generated subjects**, **batch size 128**, `lambda-sparse=0`, `context-min=.25`, `context-max=.80`, and `repeat=1`. These latter values match the current NN CLI defaults. The default run selector hashes the complete S3 training configuration, including **batch size**, so supply the same settings on every downstream NN command. No separate `--batch-size` flag exists in S1 or S2.

## 3. Setup

Run from the repository root on Windows:

```cmd
cd /d D:\Projects\rotation\mod-rot-4
conda activate py11
git pull
python -m pip install -r requirements-models.txt
python -m pip install -r requirements-nn.txt
```

## 4. Full commands: one complete S1–S9 pipeline per model

Each code block includes S1, S2, S3, S4, S5, S6, S7, **both S8 variants**, and **both S9 variants**. These are independent entrypoints. Run top-to-bottom for each model.


### C1 — T_dim=4, L=4, β=0.0005

```cmd
python models/s1_generate.py --model C1 --subjects 8000
python models/s2_precheck.py --model C1
python nn/s3_train.py --model C1 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s4_evaluate.py --model C1 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s5_extract_z.py --model C1 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s6_z_diagnostics.py --model C1 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s7_regression.py --model C1 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s8_supervised_rotation.py --model C1 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s9_rotation_sweep.py --model C1 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s8_2_supervised_rotation.py --model C1 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s9_2_rotation_sweep.py --model C1 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
```

### C2 — T_dim=4, L=4, β=0.0005

```cmd
python models/s1_generate.py --model C2 --subjects 8000
python models/s2_precheck.py --model C2
python nn/s3_train.py --model C2 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s4_evaluate.py --model C2 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s5_extract_z.py --model C2 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s6_z_diagnostics.py --model C2 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s7_regression.py --model C2 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s8_supervised_rotation.py --model C2 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s9_rotation_sweep.py --model C2 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s8_2_supervised_rotation.py --model C2 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s9_2_rotation_sweep.py --model C2 --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
```

### C3 — T_dim=6, L=6, β=0.0005

```cmd
python models/s1_generate.py --model C3 --subjects 8000
python models/s2_precheck.py --model C3
python nn/s3_train.py --model C3 --latent-dim 6 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s4_evaluate.py --model C3 --latent-dim 6 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s5_extract_z.py --model C3 --latent-dim 6 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s6_z_diagnostics.py --model C3 --latent-dim 6 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s7_regression.py --model C3 --latent-dim 6 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s8_supervised_rotation.py --model C3 --latent-dim 6 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s9_rotation_sweep.py --model C3 --latent-dim 6 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s8_2_supervised_rotation.py --model C3 --latent-dim 6 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s9_2_rotation_sweep.py --model C3 --latent-dim 6 --beta 0.0005 --batch-size 128 --repeat 1
```

### CR3 — T_dim=4, L=4, β=0.00005

```cmd
python models/s1_generate.py --model CR3 --subjects 8000
python models/s2_precheck.py --model CR3
python nn/s3_train.py --model CR3 --latent-dim 4 --beta 0.00005 --batch-size 128 --repeat 1
python nn/s4_evaluate.py --model CR3 --latent-dim 4 --beta 0.00005 --batch-size 128 --repeat 1
python nn/s5_extract_z.py --model CR3 --latent-dim 4 --beta 0.00005 --batch-size 128 --repeat 1
python nn/s6_z_diagnostics.py --model CR3 --latent-dim 4 --beta 0.00005 --batch-size 128 --repeat 1
python nn/s7_regression.py --model CR3 --latent-dim 4 --beta 0.00005 --batch-size 128 --repeat 1
python nn/s8_supervised_rotation.py --model CR3 --latent-dim 4 --beta 0.00005 --batch-size 128 --repeat 1
python nn/s9_rotation_sweep.py --model CR3 --latent-dim 4 --beta 0.00005 --batch-size 128 --repeat 1
python nn/s8_2_supervised_rotation.py --model CR3 --latent-dim 4 --beta 0.00005 --batch-size 128 --repeat 1
python nn/s9_2_rotation_sweep.py --model CR3 --latent-dim 4 --beta 0.00005 --batch-size 128 --repeat 1
```

### CR4 — T_dim=3, L=3, β=0.0005

```cmd
python models/s1_generate.py --model CR4 --subjects 8000
python models/s2_precheck.py --model CR4
python nn/s3_train.py --model CR4 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s4_evaluate.py --model CR4 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s5_extract_z.py --model CR4 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s6_z_diagnostics.py --model CR4 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s7_regression.py --model CR4 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s8_supervised_rotation.py --model CR4 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s9_rotation_sweep.py --model CR4 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s8_2_supervised_rotation.py --model CR4 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s9_2_rotation_sweep.py --model CR4 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
```

### CR5 — T_dim=3, L=3, β=0.0005

```cmd
python models/s1_generate.py --model CR5 --subjects 8000
python models/s2_precheck.py --model CR5
python nn/s3_train.py --model CR5 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s4_evaluate.py --model CR5 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s5_extract_z.py --model CR5 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s6_z_diagnostics.py --model CR5 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s7_regression.py --model CR5 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s8_supervised_rotation.py --model CR5 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s9_rotation_sweep.py --model CR5 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s8_2_supervised_rotation.py --model CR5 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
python nn/s9_2_rotation_sweep.py --model CR5 --latent-dim 3 --beta 0.0005 --batch-size 128 --repeat 1
```

## Noisy-logit models: C1L, C2L, C3L, CR3L, CR4L, CR5L

Every model also exists as `<MODEL>L`, run through the **same S1–S9 / S8_2 / S9_2 entry points**. `C1`–`CR5` behave exactly as before (their regenerated datasets are bit-identical to the existing ones).

| | `<MODEL>` | `<MODEL>L` |
|---|---|---|
| T, trial counts, X, true logit | same | same (same seeds and streams; CR3L / CR5L use the fixed bank of CR3 / CR5) |
| Observation | `choice ~ Bernoulli(sigmoid(logit))` | `y = logit + N(0, s²)`, `s² = Var(logit)·0.2/0.8`, so a perfect predictor has R² = 0.8; noise stream `default_rng(777 + model index)` |
| S1 fields | …, `logit`, `p`, `choice` | …, `logit`, `y` |
| S2 | Bernoulli balance, logistic MLE / Fisher | residual mean and realised R² (0.8 ± 0.01), Gaussian MLE (least squares) / Fisher; same other thresholds |
| NP | 2 class logits, cross-entropy | 1 output, MSE on y standardized with TRAIN mean/SD (saved in the checkpoint: `y_mean`, `y_std`); decoder logit = output·y_std + y_mean |
| Encoder, decoder body, latent, context sampling, splits, β·KL, optimizer, early stopping | same | same |
| S3/S4 metrics | CE, accuracy, Brier | MSE, R² on y |
| S5–S9_2 | | unchanged |

Outputs go to `outputs/models/{generate,precheck}/<MODEL>L/` and `outputs/nn/<MODEL>L/…`. Use the commands of section 4 with the `L` name, for example:

```cmd
python models/s1_generate.py --model C1L --subjects 8000
python models/s2_precheck.py --model C1L
python nn/s3_train.py --model C1L --latent-dim 4 --beta 0.0005 --batch-size 128 --repeat 1
```

then `s4`–`s9_2` with the same arguments. The earlier stand-alone experiment `noisy_logit_np/` is unchanged; its C1–CR4 datasets are identical to the new `C1L`–`CR4L` S1 datasets (its networks are not part of this interface).

On a Chinese-locale Windows console set `PYTHONIOENCODING=utf-8` (S7 prints `R²`, which the GBK codec cannot encode).

## 5. Output directory structure

```text
outputs/
├── models/
│   ├── generate/<MODEL>/
│   │   ├── s1_generate__dataset.npz      <- T, raw X, logit, p, choice
│   │   ├── s1_generate__manifest.json
│   │   └── s0_design/                    <- applicable only to CR3/CR5
│   └── precheck/<MODEL>/
│       ├── s2_precheck__report.json
│       └── s2_precheck__recovery.csv
└── nn/<MODEL>/<condition>_<hash>/repeat01/
    ├── config.json
    ├── manifest.json
    ├── splits.npz
    ├── s3_train/
    │   ├── checkpoint.pt
    │   ├── history.csv
    │   └── summary.json
    ├── s4_evaluate/                      <- metrics and curves
    ├── s5_extract_z/
    │   └── latent.npz                    <- frozen Z, logvar, splits, hashes
    ├── s6_z_diagnostics/                 <- PCA/ICA/FA, variance, plots
    ├── s7_regression/                    <- TEST Ridge recovery diagnostics
    ├── s8_supervised_rotation/
    │   ├── rotation.npz                  <- R* on original Z
    │   ├── rotated_z.npz
    │   └── ...                           <- TEST correlation/scatter figures
    ├── s9_rotation_sweep/
    │   ├── rotation_index.csv
    │   ├── rotations/baseline/unswept.npz
    │   ├── rotations/kXX/*.npz
    │   ├── figures_baseline/
    │   └── figures_45deg/
    ├── s8_2_supervised_rotation/
    │   ├── rotation.npz                  <- R*_2 on globally standardized Z
    │   ├── z_standardized.npz
    │   ├── z_normalizer.npz              <- saved mu, var, std
    │   ├── z_normalizer.csv
    │   └── ...
    └── s9_2_rotation_sweep/
        ├── rotation_index.csv
        ├── rotations/baseline/unswept.npz
        ├── rotations/kXX/*.npz
        ├── figures_baseline/
        ├── figures_45deg/
        ├── z_variance_diagnostics/
        └── original_z_variance_diagnostics/
```

Do not manually guess the condition-hash. It is derived from all relevant training hyperparameters; the commands above use the same selector consistently.

## 6. Rerun and overwrite policy

- **S1:** refuses to overwrite an existing model dataset. Archive/rename that model's `outputs/models/generate/<MODEL>` (and associated precheck) before generating new data.
- **S2:** validates the exact current S1 dataset and stores its report.
- **S3:** refuses to overwrite an existing checkpoint under the same run selector.
- **S5:** refuses to overwrite an existing frozen Z.
- **S8 and S8_2:** refuse to overwrite nonempty existing rotation output folders.
- **S9 and S9_2:** **automatically remove and regenerate only their respective output stage folders** when rerun (current code policy). Do not assume a previous S9 sweep is preserved.
- **S4/S6/S7:** diagnostic outputs may be rerun; inspect their respective scripts for artifact-level write policies.

If running on an already completed configuration, do **not** rerun S1/S3/S5/S8 blindly. Use the existing artifacts or create a new run (e.g. `--repeat 2` for a new NN run). Do not change parameters on only one stage: that may select a different condition hash.

## 7. Validation and additional documentation

```cmd
python -m unittest discover -s models/tests -v
python -m unittest discover -s nn/tests -v
```

- [Model generation, CPT designs and unified S2 precheck](models/README.md)
- [Neural Process methods, S8/S9 details, S8_2/S9_2 and full outputs](nn/README.md)

All modeling and rotation analyses should retain exact source hashes, subject split identities, coordinate conventions and normalization parameters to ensure that recovered latent axes and Hessian analyses remain reproducible.
