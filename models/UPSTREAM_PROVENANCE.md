# Provenance and modifications

Copied from [bs-zhao/mod-rot-3/np](https://github.com/bs-zhao/mod-rot-3/tree/98979c9ef522caa7d24de7ccc683148b762d92a5/np) (commit `98979c9ef522caa7d24de7ccc683148b762d92a5`). Source files are transplanted under `models/` retaining their original relative structure, in particular the original generators, prechecks, and Oracle implementations.

Only imports and paths referencing `np/` were changed to `models/`; C1/C2/C3 class sample-size defaults and CR3/CR4/CR5 reference script CLI defaults changed from 5000 to 8000; S1 now passes `--subjects` through for C1/C2/C3; the S1/S2 orchestration path/CLI was separated from NP training dependencies. Model formulas, priors, RNG algorithms, fixed-bank construction and precheck thresholds were not changed.

The source input SHA for each vendored file is available through the upstream Git history at this commit. Do not modify the frozen model mathematics without creating a new model version.
