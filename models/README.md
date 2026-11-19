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
| CR3 | 4 (alpha, lambda, gamma, tau) | 500 | Original frozen CR3 500×6 battery, shared across people |
| CR4 | 3 (lambda, gamma, tau), alpha fixed to 1 | 500 | Fresh 6D X for **every person and every trial**, independent amount U(5,200) and probability U(.05,.95) |
| CR5 | 3 (lambda, gamma, tau), alpha fixed to 1 | 500 | Exact same frozen 500×6 stimulus battery as CR3 |

Original generating formulas, parameter priors and sampling seeds remain unchanged;
only their orchestration and outputs are unified. Exact X-generation semantics are
never changed to force the six datasets to use the same distribution.

## Implementation architecture

```text
models/s1_generate.py                    same S1 driver for every model
    └── src/modrot4/model_specs.py       T/X/probability/choice generators
        ├── original C1/C2/C3 generator
        └── original CR3/CR4/CR5 math and bank preparation
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

C1/C2/C3 use the original generator's deterministic SeedSequence streams and
Bernoulli draws. CR3/CR4/CR5 preserve their original `data_seed+4` parameter
stream, `data_seed+1004` choice stream, and CR4's `data_seed+2004` X stream.
CR3 uses the original optimized CR2 bank and its CR3 conditioning; CR5 copies
the immutable CR3 bank. **Run CR3 Generate before CR5 Generate.**

## Unified S2 checks

Every dataset goes through the same checks, with the same output keys:

1. `structure`: subject/trial indexing, T/X/Y/P dimensions, finite values,
   response support, parameter limits, trial counts.
2. `generator_fidelity`: exact model probability recomputation on up to
   32 actual S1 subjects; max absolute error ≤ 1e-12.
3. `stimulus_design`: model-provided prior/stimulus constraints including
   frozen bank integrity for CR3/CR5 or fresh iid X for CR4.
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
      CR2/s0_design_recovery/              # CR3 upstream optimized bank
      CR3/s0_design/                       # CR3 immutable bank
      CR5/s0_design/                       # CR5 immutable bank copy
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
and does not overwrite it. An existing 5,000-person dataset is not silently
converted into 8,000 persons. Old `outputs/choice/` files are not read.
Do not replace an immutable CR2/CR3/CR5 bank without preserving provenance and
regenerating dependent datasets. CR2 bank optimization can differ slightly
across NumPy/SciPy versions even with fixed seeds, so preserve frozen banks
when strict binary reproducibility is required.

## Verification and source provenance

```bat
python -m unittest discover -s models/tests -v
```

GitHub Actions runs these tests on each push. Included tests cover classic
model deterministic generation and S1/S2, CR4 original bitwise IID sampling,
CR3/CR5 original mathematical RNG sampling with a fixed test bank, and
the common S2 MLE/Fisher reporting. **The workflow has not run the full
8,000-person datasets or the expensive original CR2 bank optimizer.**

The source baseline from mod-rot-3 is
[`98979c9ef522`](https://github.com/bs-zhao/mod-rot-3/tree/98979c9ef522caa7d24de7ccc683148b762d92a5/np).
See `models/UPSTREAM_PROVENANCE.md` for subsequent Mod-Rot-4 changes.
