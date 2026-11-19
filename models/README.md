# Models: faithful Mod-Rot-3 choice generators and S2 checks

This module mirrors [mod-rot-3/np](https://github.com/bs-zhao/mod-rot-3/tree/main/np) at upstream source commit `98979c9ef522caa7d24de7ccc683148b762d92a5`. The generating mathematics, parameter priors, X sampling policies, RNG streams, dataset field names, fixed-bank design, precheck tests and optional Oracle implementations are retained. **The default subject count is now 8,000**; `--subjects` overrides it for **all six** models, including C1/C2/C3.

## Models

| ID | True parameter count | Trials per subject | Stimulus X |
|----|----------------------|--------------------|------------|
| C1 | 4 (b1–b4) | Uniform integer 300–500 | Independent trial-specific uniform inputs, 3 features |
| C2 | 4 (b1–b4) | Uniform integer 300–500 | Independent trial-specific 4 raw option attributes; linear and quadratic X1 |
| C3 | 6 (b1–b6) | Uniform integer 300–500 | Independent trial-specific 4 raw option attributes; 2 quadratic + interaction |
| CR3 | 4 (alpha, lambda_loss, gamma, tau) | Exactly 500 | Shared immutable 500×6 CR3 design built from original CR2 optimized bank |
| CR4 | 3 (lambda_loss, gamma, tau), alpha=1 | Exactly 500 | **Independent IID 6D X per subject/trial**: amounts U(5,200), gain probs U(0.05,0.95) |
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
python models/s2_precheck.py --model CR3 --oracle-precheck --oracle-subjects 100 --oracle-starts 4

python models/s1_generate.py --model CR4
python models/s2_precheck.py --model CR4 --oracle-precheck --oracle-subjects 100 --oracle-starts 4

python models/s1_generate.py --model CR5
python models/s2_precheck.py --model CR5 --oracle-precheck --oracle-subjects 100 --oracle-starts 4
```

By default each S1 command uses **8,000 subjects**. For a small test run, e.g. `python models/s1_generate.py --model CR4 --subjects 10`.

**Run CR3 S1 before CR5 S1.** CR3 S1 uses the frozen CR2 4P Fisher-optimized design if already present in `outputs/choice/CR2/s0_design_recovery/`; otherwise it executes the unchanged original CR2 bank optimizer then constructs CR3's 500-item bank. CR5 **requires the actual CR3 bank** and will not fabricate or resample it. CR4 does not create a shared S0 bank.

## Seeds and frozen format

- C1/C2/C3 data seed: `20261006`.
- CR3/CR4/CR5 data seed: `20261028`; original design seed `20261023`.
- CR4 independent streams: `theta = data_seed+4`, `choice = data_seed+1004`, `X = data_seed+2004`; CR5 shares CR4's theta/choice stream policies.
- Precheck seed: `20261006`.
- All datasets use the original `GeneratedDataset` keys: `parameters, trial_counts, subject_index, trial_index, x, y, predictive_true`.

Paths, compatible with the previous pipeline:

```text
outputs/choice/<MODEL>/s1_generate/s1_generate__dataset.npz
outputs/choice/<MODEL>/s1_generate/s1_generate__manifest.json
outputs/choice/<MODEL>/s1_generate/s1_generate__summary.json (C1/C2/C3)
outputs/choice/<MODEL>/s2_precheck/s2_precheck__report.json
```

CR3 and CR5 store their fixed-bank files under `outputs/choice/<MODEL>/s0_design/`. Optional CR3–CR5 Oracle diagnostics are written to the original S2 output locations.

**Data preservation:** existing datasets are never silently overwritten, even when the requested subject count differs. To regenerate with a different N, archive/move the relevant `outputs/choice/<MODEL>/` directory first. CR3 and CR5 should use a consistent saved CR3/CR2 bank when comparing runs.

The original CR2 design optimization may depend on the NumPy/SciPy versions. Same seed + same source does not guarantee an **identical binary bank file** across environments: if byte-level replication is required, reuse and verify the original CR2/CR3 bank rather than rerunning the optimizer. This is also why bank checksums are retained.

## Code layout and provenance

- `s1_generate.py`, `s2_precheck.py`: original stage logic with import/path adaptation and configurable N for C1/C2/C3.
- `src/modrot4/cli.py`, `src/modrot4/pipeline.py`: S1/S2-only CLI, paths, subprocess invocation; no network/training dependencies.
- `src/modlearn_rot/generators/`, `src/modlearn_rot/diagnostics/precheck/`, `src/modlearn_rot/utils/`: original generation, dataset serialization, and full C1–C3 precheck implementations.
- `reference/`: original CR2 bank optimizer, CR3/CR4/CR5 generators, Oracle implementations and CR3 identifiability helpers.

Only deliberate source adjustments: `np/` -> `models/` filesystem imports; original standalone reference generator `--subjects` default 5,000 -> 8,000; C1/C2/C3 class default 5,000 -> 8,000; S1 sets `model.n_subjects=args.subjects` so the flag actually controls these three generators; CR3 optional Oracle subprocess reference path fixed for this module. **No formula, prior, sampling policy, RNG offsets, or precheck threshold was deliberately changed.**

## Tests

```cmd
python -m unittest discover -s models/tests -v
```

The smoke tests cover deterministic C1/C2/C3 generation, 3P CPT equivalence and CR4's IID-X generation/S2 provenance. They do not regenerate the expensive CR2 optimized bank, and they are **not** full 8,000-subject numerical equivalence tests.
