# Models: faithful Mod-Rot-3 choice generators and S2 checks

This module mirrors [mod-rot-3/np](https://github.com/bs-zhao/mod-rot-3/tree/main/np) at upstream source commit `98979c9ef522caa7d24de7ccc683148b762d92a5`. The generating mathematics, parameter priors, RNG streams, dataset field names, fixed-bank design, precheck and Oracle implementations remain available as references. The default S2 now uses a uniform six-check protocol and fits actual saved S1 participants. CR4 X now uses an explicit per-subject, per-trial IID generation loop that is byte-equivalent to the old vectorized IID sampler. **The default subject count is now 8,000**; `--subjects` overrides it for **all six** models, including C1/C2/C3.

## Models

| ID | True parameter count | Trials per subject | Stimulus X |
|----|----------------------|--------------------|------------|
| C1 | 4 (b1–b4) | Uniform integer 300–500 | Independent trial-specific uniform inputs, 3 features |
| C2 | 4 (b1–b4) | Uniform integer 300–500 | Independent trial-specific 4 raw option attributes; linear and quadratic X1 |
| C3 | 6 (b1–b6) | Uniform integer 300–500 | Independent trial-specific 4 raw option attributes; 2 quadratic + interaction |
| CR3 | 4 (alpha, lambda_loss, gamma, tau) | Exactly 500 | Shared immutable 500×6 CR3 design built from original CR2 optimized bank |
| CR4 | 3 (lambda_loss, gamma, tau), alpha=1 | Exactly 500 | **Fresh independent 6D X for each trial of each subject, like C1/C2/C3**: amounts U(5,200), gain probs U(0.05,0.95) |
| CR5 | 3 (lambda_loss, gamma, tau), alpha=1 | Exactly 500 | **Exact CR3 immutable 500×6 X** with the CR4 3-parameter response model |

CR3 is not CR4: their X designs differ. CR4 and CR5 share the same priors/function, but not the X-design policy. CR3/CR5 preserve the original 60 gain-only + 60 gamma-neutral + 40 loss-only + 340 general trial categories. **C3 has six parameters**, as in the upstream source.

## Dependencies

Python 3.11 recommended (the original simulator uses NumPy and SciPy only; **PyTorch and NP are not required for S1/S2**).

```cmd
pip install -r requirements-models.txt
```

Run commands from repository root (each stage separately):

```cmd
python models/s1_generate.py --model C1
python models/s2_precheck.py --model C1

python models/s1_generate.py --model C2
python models/s2_precheck.py --model C2

python models/s1_generate.py --model C3
python models/s2_precheck.py --model C3

python models/s1_generate.py --model CR3
python models/s2_precheck.py --model CR3

python models/s1_generate.py --model CR4
python models/s2_precheck.py --model CR4

python models/s1_generate.py --model CR5
python models/s2_precheck.py --model CR5
```

By default each S1 command uses **8,000 subjects**. For a small test run, e.g. `python models/s1_generate.py --model CR4 --subjects 10`.

**Run CR3 S1 before CR5 S1.** CR3 S1 uses the frozen CR2 4P Fisher-optimized design if already present in `outputs/models/generate/CR2/s0_design_recovery/`; otherwise it executes the unchanged original CR2 bank optimizer then constructs CR3's 500-item bank. CR5 **requires the actual CR3 bank** and will not fabricate or resample it. CR4 does not create a shared S0 bank.

## Seeds and frozen format

- C1/C2/C3 data seed: `20261006`.
- CR3/CR4/CR5 data seed: `20261028`; original design seed `20261023`.
- CR4 independent streams: `theta = data_seed+4`, `choice = data_seed+1004`, `X = data_seed+2004`; CR5 shares CR4's theta/choice stream policies.
- Precheck seed: `20261006`.
- All datasets use the original `GeneratedDataset` keys: `parameters, trial_counts, subject_index, trial_index, x, y, predictive_true`.

Paths, arranged by module, stage, then model (breaking the Mod-Rot-3 path layout):

```text
outputs/models/generate/<MODEL>/s1_generate__dataset.npz
outputs/models/generate/<MODEL>/s1_generate__manifest.json
outputs/models/generate/<MODEL>/s1_generate__summary.json (C1/C2/C3)
outputs/models/precheck/<MODEL>/s2_precheck__report.json
```

CR3 and CR5 store their fixed-bank files under `outputs/models/generate/<MODEL>/s0_design/`. Standard S2 MLE recovery outputs are written under `outputs/models/precheck/<MODEL>/`.

**Data preservation:** existing datasets are never silently overwritten, even when the requested subject count differs. To regenerate with a different N, archive/move the relevant `outputs/models/generate/<MODEL>/` directory first. CR3 and CR5 should use a consistent saved CR3/CR2 bank when comparing runs.

The original CR2 design optimization may depend on the NumPy/SciPy versions. Same seed + same source does not guarantee an **identical binary bank file** across environments: if byte-level replication is required, reuse and verify the original CR2/CR3 bank rather than rerunning the optimizer. This is also why bank checksums are retained.

## Code layout and provenance

- `s1_generate.py`, `s2_precheck.py`: original stage logic with import/path adaptation and configurable N for C1/C2/C3.
- `src/modrot4/cli.py`, `src/modrot4/pipeline.py`: S1/S2-only CLI, paths, subprocess invocation; no network/training dependencies.
- `src/modlearn_rot/generators/`, `src/modlearn_rot/diagnostics/precheck/`, `src/modlearn_rot/utils/`: original generation, dataset serialization, and full C1–C3 precheck implementations.
- `reference/`: original CR2 bank optimizer, CR3/CR4/CR5 generators, Oracle implementations and CR3 identifiability helpers.

Deliberate source adjustments: `np/` -> `models/` filesystem imports; original standalone reference generator `--subjects` default 5,000 -> 8,000; C1/C2/C3 class default 5,000 -> 8,000; S1 sets `model.n_subjects=args.subjects` so the flag actually controls these three generators; CR3 optional Oracle subprocess reference path fixed for this module. **No formula, prior, underlying X distribution, RNG offsets, or precheck threshold was changed.**

## Tests

```cmd
python -m unittest discover -s models/tests -v
```

The smoke tests cover deterministic C1/C2/C3 generation, 3P CPT equivalence and CR4's IID-X generation/S2 provenance. They do not regenerate the expensive CR2 optimized bank, and they are **not** full 8,000-subject numerical equivalence tests.

## Stage-first output organization (Mod-Rot-4)

```text
outputs/
└── models/
    ├── generate/
    │   ├── C1/  C2/  C3/
    │   ├── CR2/s0_design_recovery/     # CR3 upstream optimized bank
    │   ├── CR3/s0_design/             # CR3 frozen shared bank
    │   ├── CR4/                       # S1 data; NO shared bank
    │   └── CR5/s0_design/             # Exact copy of CR3 bank
    └── precheck/
        ├── C1/  C2/  C3/             # S2 report and check arrays
        ├── CR3/s2_oracle/            # Optional CR3 Oracle comparison
        ├── CR4/                      # S2 + optional Oracle
        └── CR5/                      # S2 + optional Oracle
```

Each model's dataset and S1 manifest are directly in its `generate/<MODEL>/` directory; each S2 report is directly in `precheck/<MODEL>/`. Subfolders are reserved for supplementary bank design and CR3 Oracle studies.

**Do not rename old `outputs/choice/` assets into the new structure without confirming provenance.** They are not read by this version. CR3 and CR5 require a consistent original frozen CR2/CR3 bank. To reuse an existing immutable bank, copy only the verified reference bank files into the corresponding new `models/generate/` design folder, and keep their original manifests/checksums consistent. Generated 5k-person datasets are not 8k-person datasets; regenerate from S1 when changing N.

CR4 per-subject trialwise sampling has the same distribution and reproducibility as before. Its code now loops over subjects, drawing all six X features independently for their trials, matching the C1/C2/C3 sampling organization. The tests require bitwise equivalence with the former flattened vectorized sampler.

## Universal S2 precheck: same command for all six models

No separate Oracle flag or optional Oracle mode is necessary. Each standard S2
invocation always performs the following six checks:

1. `structure`: every stored S1 subject/trial, parameter, X, Y and P is valid;
2. `generator_fidelity`: verify calculated exact-model probabilities equal saved
   `predictive_true` for a seeded sample of up to 32 actual subjects;
3. `stimulus_design`: verify model-specific original X generation and provenance
   (CR3/CR5 immutable bank, CR4 independent X, C1/C2/C3 input support);
4. `response_balance`: verify the observed global choice rate agrees with stored
   Bernoulli predictions within five sampling standard errors;
5. `fisher_information`: report exact-model Fisher eigenvalues and the
   full-rank fraction on the actual sampled S1 subject-specific X;
6. `parameter_recovery`: fit original per-subject MLE using the **stored X/Y**.
   Use true T **only for post-fit assessment**. Report true/estimated
   parameters, Pearson r, uncalibrated R², RMSE and bias.

Default MLE cohort: **100 S1 subjects sampled reproducibly without replacement**
(seed `20261006`), four multi-start bounded MLEs per CPT subject (CR3/4/5),
one convex GLM MLE per C1/2/3 subject. Same threshold for all models:
**Pearson r >= 0.8 on every parameter**, with at least 30 assessed subjects.
The Fisher full-rank fraction must be >=0.95.
These are Mod-Rot-4's NEW unified S2 gates, distinct from legacy stage gates;
no parameter recovery result is guaranteed to pass. Small diagnostic N<30
is explicitly inconclusive and fails qualification.

Outputs, always under `outputs/models/precheck/<MODEL>/`:

```text
s2_precheck__report.json
s2_precheck__summary.csv
s2_precheck__oracle_recovery.csv
s2_precheck__oracle_subjects.npz
```

Previous standalone `models/reference/cr*/s2_oracle_*.py` scripts still exist
for historical comparison but are NOT invoked by the standard S2 stage.
In particular, standard S2 does NOT generate a second artificial
cohort to test recoverability. Run `python models/s2_precheck.py --model C1`
through `--model CR5` with identical command syntax.
