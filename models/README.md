# Models module: one Generate, one Precheck, six model specifications

This module produces binary-choice synthetic datasets and checks their scientific
validity. The two user-facing scripts are **identical for every model**:

```bat
python models/s1_generate.py --model C1 --subjects 8000
python models/s2_precheck.py --model C1
```

Replace `C1` by `C2`, `C3`, `CR3`, `CR4`, or `CR5`. No extra Oracle flag
exists: individual MLE parameter recovery is part of S2 by default.

## Model specifications

| Model | True T dimensionality | Trials / subject | Stimulus X |
| --- | --- | --- | --- |
| C1 | 4 | Uniform integer 300–500 | 3 raw features, fresh iid U(-1,1) |
| C2 | 4 | Uniform integer 300–500 | 4 raw option features, fresh iid U(-1,1) |
| C3 | 6 | Uniform integer 300–500 | 4 raw option features, fresh iid U(-1,1) |
| CR3 | 4 (alpha, lambda, gamma, tau) | 500 | Locally built 500×6 bank, shared across this model's people; independent of CR2 outputs |
| CR4 | 3 (lambda, gamma, tau), alpha fixed to 1 | 500 | Fresh 6D X for **every person and every trial**, independent amount U(5,200) and probability U(.05,.95) |
| CR5 | 3 (lambda, gamma, tau), alpha fixed to 1 | 500 | Independently constructed local 500×6 bank, same deterministic recipe and default seed as CR3 |

Original generating formulas and parameter priors remain unchanged. CR4's X
sampling changed to match C1/C2/C3 **algorithmically**, not just in statistical
independence. CR4's 6D amount/probability ranges and fixed 500-trial count remain
as before. Its old X random-number sequences are NOT identical to the new ones.
CR4 and CR5 retain the original paired T prior/seed and response RNG streams.

## Implementation architecture

```text
models/s1_generate.py                    same S1 driver for every model
    └── src/modrot4/model_specs.py       T/X/probability/choice generators
        ├── original C1/C2/C3 generator
        └── original CR3/CR4/CR5 math and bank preparation
    └── src/modrot4/fixed_bank.py         independent per-model CR3/CR5 bank construction
    └── common dataset, manifest and summary writer

models/s2_precheck.py                    same S2 driver for every model
    ├── src/modrot4/model_specs.py       per-model prediction / MLE / Fisher / domain
    └── src/modrot4/unified_precheck.py  same six checks / gates / output / logging
```

`model_specs.py` is the **only** standard pipeline implementation that selects
model-specific mathematics, sampling, provenance or fitting operators. The entry
scripts do not select a separate generator or precheck suite based on model ID.
The original `models/reference/` formulas and bank builders remain the source of
truth. The old `modlearn_rot/diagnostics/precheck/` and standalone
`reference/cr*/s2_oracle_*.py` remain for historical comparison; standard S2
**does not** run those alternative routines.

C1/C2/C3 and CR4 use **exactly the same** `generate_trialwise_dataset`
per-subject loop and `sample_independent_uniform_x` helper. For each subject,
a single `inputs` Generator is used to invoke
`rng.uniform(low,high,size=(n_trials,x_dim))`. That Generator is the
**same third SeedSequence child** of `data_seed` in all four models.
C1/C2/C3 pass scalar bounds (-1,1), CR4 passes per-column lower/upper
bounds [5,5,.05,5,5,.05] and [200,200,.95,200,200,.95].
There are no shared questions, amount/probability substreams, X design banks,
acceptance/rejection steps, or correlations imposed across trials.

For **non-X** random variables, CR4 specifically preserves its original
`data_seed+4` prior stream and `data_seed+1004` Bernoulli uniform stream,
so the original CR4/CR5 subject parameter pairing is preserved. This is
intentional model-specific T/Y behavior and does not affect the shared X
algorithm. CR3/CR5 continue using their original CPT seed policies and
immutable **locally generated** banks. **Every model can Generate and
Precheck independently, including CR3 and CR5. No prior CR2/CR3 run is required.**

## Unified S2 checks

Every dataset goes through the same checks, with the same output keys:

1. `structure`: subject/trial indexing, T/X/Y/P dimensions, finite values,
   response support, parameter limits, trial counts.
2. `generator_fidelity`: exact model probability recomputation on up to
   32 actual S1 subjects; max absolute error ≤ 1e-12.
3. `stimulus_design`: model-provided prior/stimulus constraints including
   frozen bank integrity for CR3/CR5 or CR4's freshly generated single-RNG X
   verified against the common C1/C2/C3 SeedSequence input stream.
4. `response_balance`: global Y rate compared with original P
   (within 5 Bernoulli standard errors).
5. `fisher_information`: information spectrum on the actual subject-specific
   stimulus X. Fisher full-rank fraction ≥ 0.95.
6. `parameter_recovery`: sample **100 actual S1 subjects**, fit their existing
   X/Y with the model's original MLE, then compare estimates with the saved
   true T. Report Pearson r, uncalibrated R², RMSE and Bias for **every**
   parameter. Each Pearson r must be ≥ 0.80 and ≥ 30 subjects are required for
   a conclusive gate. C1/C2/C3 use convex GLM MLE; CR3/CR4/CR5 use four-start
   bounded CPT MLE. The *evaluation rule*, not the mathematical estimator,
   is uniform.

These thresholds are a new Mod-Rot-4 diagnostic policy, not claims about
mod-rot-3's prior model-specific thresholds. Failed checks still write their
reports and return a nonzero exit status. An underpowered small test
(`--subjects 5`) is intentionally **not** considered scientifically qualified.

## Commands: six models, same Generate and Precheck

```bat
conda activate py11
python -m pip install -r requirements-models.txt

python models/s1_generate.py --model C1 --subjects 8000
python models/s1_generate.py --model C2 --subjects 8000
python models/s1_generate.py --model C3 --subjects 8000
python models/s1_generate.py --model CR3 --subjects 8000
python models/s1_generate.py --model CR4 --subjects 8000
python models/s1_generate.py --model CR5 --subjects 8000

python models/s2_precheck.py --model C1
python models/s2_precheck.py --model C2
python models/s2_precheck.py --model C3
python models/s2_precheck.py --model CR3
python models/s2_precheck.py --model CR4
python models/s2_precheck.py --model CR5
```

S1 defaults to 8,000 participants even when `--subjects` is omitted.
Data seeds: C1/C2/C3 `20261006`, CR3/CR4/CR5 `20261028`.
Precheck sample seed: `20261006`; design seed: `20261023`.

## Paths (module → stage → dataset)

```text
outputs/
  models/
    generate/
      C1/ C2/ C3/ CR3/ CR4/ CR5/
        s1_generate__dataset.npz
        s1_generate__manifest.json
        s1_generate__summary.json         # for fresh generation
      CR3/s0_design/                       # CR3 independently built 500-X bank
      CR5/s0_design/                       # CR5 independently built 500-X bank
    precheck/
      C1/ C2/ C3/ CR3/ CR4/ CR5/
        s2_precheck__report.json
        s2_precheck__summary.csv
        s2_precheck__oracle_recovery.csv
        s2_precheck__oracle_subjects.npz
```

All generated NPZ files use the same fields:
`parameters`, `trial_counts`, `subject_index`, `trial_index`, `x`,
`y`, `predictive_true`. Old original C1/C2/C3 Y/P may be stored as 1D,
and CR3/4/5 Y/P as single-column 2D; the common evaluator handles both
without changing previously generated data.

**Preservation:** if a dataset already exists, S1 prints `status=EXISTS`
and does not overwrite it. **Exception:** a legacy CR4 dataset is rejected
even if its 8,000 subjects match, because v1 used a different X RNG policy.
Archive the old `outputs/models/generate/CR4/` and
`outputs/models/precheck/CR4/` directories, then regenerate CR4. S2 also
rejects the old manifest policy; it cannot silently report a new-method pass
for an old-method dataset. An existing 5,000-person dataset is not silently
converted into 8,000 persons. Old `outputs/choice/` files are not read.
Do not replace either model's immutable CR3/CR5 bank without preserving its
own provenance and regenerating that model's dependent dataset. Neither model
reads CR2/CR3/CR5 generated outputs owned by another model.
The original CR2 design optimization is executed **within each model's own
Generate** when a local bank is absent, retaining all original hyperparameters:
6,000 candidates, 48 reference parameters, greedy shortlist 96 and 600 swaps.
The resulting CR3-style 500-row bank has 60 gain-only, 60 gamma-neutral,
40 loss-only and 340 general items; strict dominance is forbidden.
The only difference between the CR3 and CR5 bank files is their parameter
metadata and output folder. With the same design seed, code and numeric
environment, the X arrays are deterministic and should be identical.
The first standalone Generate of CR3 *or* CR5 may perform expensive bank
optimization; subsequent runs can use its own cached bank. No global CR2
bank needs to be generated. Optimizer floating-point behavior can differ
across NumPy/SciPy versions; do not assume bitwise match across environments.

## Verification and source provenance

```bat
python -m unittest discover -s models/tests -v
```

GitHub Actions runs these tests on each push. Included tests cover classic
model deterministic generation and S1/S2, CR4 legacy original bitwise IID sampling plus v2 single-RNG X equivalence,
CR3/CR5 original mathematical RNG sampling with a fixed test bank,
an end-to-end CR5 Generate + Precheck test with **both CR2 and CR3 output
directories absent**, independent CR3/CR5 equal-design construction, and
the common S2 MLE/Fisher reporting. **The workflow has not run the full
8,000-person datasets or the expensive original CR2 bank optimizer.**

The source baseline from mod-rot-3 is
[`98979c9ef522`](https://github.com/bs-zhao/mod-rot-3/tree/98979c9ef522caa7d24de7ccc683148b762d92a5/np).
See `models/UPSTREAM_PROVENANCE.md` for subsequent Mod-Rot-4 changes.

## Run CR5 independently (without CR3)

```bat
cd /d D:\Projects\rotation\mod-rot-4
git pull
conda activate py11
python models/s1_generate.py --model CR5 --subjects 8000
python models/s2_precheck.py --model CR5
```

CR5's own bank is stored at
`outputs/models/generate/CR5/s0_design/CR5_3P_cr3bank500.npz`.
CR3 and CR2 output folders are **not consulted**. If an existing CR5 bank
was created by the old workflow, its own stored bank and SHA are still
validated locally, without requiring the old CR3 source folder.
