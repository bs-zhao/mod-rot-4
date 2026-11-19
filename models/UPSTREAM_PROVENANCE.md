# Provenance and modifications

Copied from [bs-zhao/mod-rot-3/np](https://github.com/bs-zhao/mod-rot-3/tree/98979c9ef522caa7d24de7ccc683148b762d92a5/np) (commit `98979c9ef522caa7d24de7ccc683148b762d92a5`). Source files are transplanted under `models/` retaining their original relative structure, in particular the original generators, prechecks, and Oracle implementations.

Only imports and paths referencing `np/` were changed to `models/`; C1/C2/C3 class sample-size defaults and CR3/CR4/CR5 reference script CLI defaults changed from 5000 to 8000; S1 now passes `--subjects` through for C1/C2/C3; the S1/S2 orchestration path/CLI was separated from NP training dependencies. Model formulas, priors, RNG algorithms, fixed-bank construction and precheck thresholds were not changed.

The source input SHA for each vendored file is available through the upstream Git history at this commit. Do not modify the frozen model mathematics without creating a new model version.

## Mod-Rot-4 output + CR4 sampling revision

- All model outputs moved to `outputs/models/generate/<MODEL>/` and `outputs/models/precheck/<MODEL>/` (including CR2/CR3/CR5 frozen design banks and optional S2 Oracle artifacts). Previous `outputs/choice/` files are not automatically read or overwritten.
- CR4 X is now drawn with a per-subject loop, using two continuous RNG streams and independent samples across all trials and subject boundaries. The X distribution, parameter sampling, logit function, seeds and trial order are unchanged. Regression test checks bitwise identity with the original vectorized `sample_bank(seed, n=subjects*trials).reshape(subjects,trials,6)` implementation.
- Existing C1/C2/C3 generators already draw new X at each participant's trials; they were not altered.

## Standardized S2 precheck (Mod-Rot-4-specific policy)

The previous asymmetric scheme (C1/C2/C3 all-subject Oracle MLE;
CR3/CR4/CR5 optional separately simulated cohorts) was replaced by one
S2 routine. Every model now uses stored S1 X/Y for a reproducible
100-subject MLE sample; every model receives the same six checks and
the same thresholds: per-parameter Pearson r >=0.8 (min 30 subjects),
Fisher full-rank fraction >=0.95. These are NEW diagnostic policy
choices, not retrospective claims about upstream thresholds. Old
reference MLE implementations and mathematical generators were not changed.
