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

## 2026-11-19 — CR4 X algorithm fully aligned with C1/C2/C3

Only the CR4 X sampler and associated data provenance were intentionally
changed. The old standalone reference `sample_subject_x` in
`reference/cr4/s0_s1_cr4.py` (2 independent streams, data_seed+2004/2005)
remains as an archived v1 reference, but **the active S1 driver no longer
uses it**. C1/C2/C3 and CR4 now all use:

- the same `generate_trialwise_dataset` per-subject loop;
- the same `sample_independent_uniform_x(n_trials,rng,low,high,n_features)`;
- one shared subject-continuous `inputs` RNG, produced by
  `SeedSequence(data_seed).spawn(4)[2]`;
- one `rng.uniform(low,high,size=(n_trials,x_dim))` call for every participant;
- independent continuous values for each trial and each X dimension.

Only per-feature X bounds, X dimensionality, and trial-count configuration
differ. C1/C2/C3 remain bitwise identical to the original scalar-bound
`rng.uniform(-1,1,...)`. CR4's X numbers change with identical data_seed
because the RNG algorithm/stream changed; its X distribution remains four
independent U(5,200) amounts and two independent U(.05,.95) probabilities.
CR4 and CR5 continue using the *paired* T stream `data_seed+4`; CR4 also
keeps its choice stream `data_seed+1004`. T priors, logits and Bernoulli
response semantics remain unchanged.

New CR4 outputs use `x_sampling_policy = "iid_uniform_single_inputs_seedsequence_v2"`
and record the X RNG stream. Legacy CR4 datasets are explicitly rejected at
both S1 reuse and S2 audit; they are never overwritten or silently re-labeled.
Archive original CR4 generate and precheck directories, then run S1/S2 again
to compare results under the new X mechanism.

## 2026-11-19 — Cross-model output independence

Standard S1/S2 no longer require running one model before another. CR5 used
to require `outputs/models/generate/CR3/s0_design/CR3_4P_fixed500.npz`;
CR3 used to require a separately generated CR2 bank. Both requirements are
removed. `src/modrot4/fixed_bank.py` constructs the **original CR2-4P
optimized 500-trial design in memory** separately for each model and then
applies the original CR3 60/60/40/340 no-FOSD conditioning algorithm. The bank
is saved *only* to the requesting model's `s0_design/`. CR5's local bank
includes its original 3P parameter labels. On the same seed and numerical
environment the CR3 and CR5 X matrices should coincide, but each is computed
and stored independently; neither S1 nor S2 reads or writes the other model's
output tree. If a bank exists from the previous pipeline, the local bank and
local checksums may still be used without CR3/CR2 source files. The vendored
`models/reference/` scripts remain historical and may preserve old paths;
the public S1/S2 entrypoints use the new standalone adapter.
