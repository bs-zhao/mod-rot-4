# Models — six models, two common stages

All six models run **one Generate** and **one Precheck**. The only
model-dependent code is T sampling, X sampling, the true logit formula,
and the model's mathematical MLE/Fisher functions.

## Generate

```text
T ~ parameter prior
For each subject:
    X ~ its stimulus design
    logit = f_model(T, X)
    p = sigmoid(logit)
    choice ~ Bernoulli(p)
Save one dataset
```

The single entrypoint is `models/s1_generate.py`. Generation/serialization
is implemented in `src/modrot4/core.py`; the six functions are specified in
`src/modrot4/model_specs.py`.

**Saved NPZ fields (all six models, same names and shapes):**

| Key | Description | Shape |
| --- | --- | --- |
| `parameters` | True T, not estimated | (subjects, T_dim) |
| `trial_counts` | Trials for each subject | (subjects,) |
| `subject_index` | Subject ID per row | (total_trials,) |
| `trial_index` | Trial within subject | (total_trials,) |
| `x` | Raw X | (total_trials, X_dim) |
| `logit` | Exact theoretical logit | (total_trials,) |
| `p` | Exact sigmoid(logit) | (total_trials,) |
| `choice` | Observed Bernoulli 0/1 response | (total_trials,) |

Only `choice` is the observed response. `logit` and `p` are known
generating quantities and are saved for analysis. These replace the old
`y` / `predictive_true` dataset-field names; there are no redundant copies.

| Model | T_dim | X_dim | X generation | Trials/person |
| --- | ---: | ---: | --- | ---: |
| C1 | 4 | 3 | iid U(-1,1) for every person/trial | 300–500 |
| C2 | 4 | 4 | iid U(-1,1) for every person/trial | 300–500 |
| C3 | 6 | 4 | iid U(-1,1) for every person/trial | 300–500 |
| CR3 | 4 | 6 | own fixed 500-trial CPT design | 500 |
| CR4 | 3 | 6 | independent U(5,200) amounts / U(.05,.95) probabilities per person/trial | 500 |
| CR5 | 3 | 6 | own fixed 500-trial CPT design, same deterministic design recipe as CR3 | 500 |

CR3 and CR5 independently construct their own fixed bank when needed.
Neither needs another model's output directory.
C1/C2/C3/CR4 share the same per-person X RNG and uniform sampling function.
The original model formula/priors are preserved.

## Precheck

`models/s2_precheck.py` imports the saved **T, X and choice**, then uses
the same evaluation code in `src/modrot4/unified_precheck.py` for every model.
It recalculates p from the true T and X; **it does not take the saved logit/p
as inputs**. This avoids giving one model extra information.

Uniform diagnostics:

- Data shape, indexing, parameter support and binary choices
- Response-rate calibration against the theoretical Bernoulli probabilities
- Fisher information rank (at least 95% of assessed subjects must be full rank)
- Per-subject MLE on 100 randomly sampled *actual* subjects; Pearson r,
  R², RMSE and bias for each T parameter, with r >= 0.8 required for all
  parameters and 30 subjects required for a conclusive test

Outputs are `s2_precheck__report.json` and
`s2_precheck__recovery.csv`, under
`outputs/models/precheck/<MODEL>/`.

## Commands

```bat
cd /d D:\Projects\rotation\mod-rot-4
conda activate py11
git pull
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

**Old datasets must be archived before using the new generation format.**
Generate never overwrites existing datasets. Move or rename each existing
`outputs/models/generate/<MODEL>/` directory and its
`outputs/models/precheck/<MODEL>/` directory before regenerating that
model. New Precheck intentionally rejects old NPZ files that lack
`logit`, `p`, or `choice`.

## Tests

```bat
python -m unittest discover -s models/tests -v
```

There are only three focused tests: one six-model dataset-field contract,
one six-model shared precheck, and one shared iid-X RNG test.
The tests use small datasets and do not establish full 8,000-person
MLE recovery quality.
