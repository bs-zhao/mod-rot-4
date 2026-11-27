# Metric Lab

Numbered, independent metric experiments. Each `NN/` experiment owns its code, tests, configuration, and outputs; no imports between experiments. Modules may **read** the existing `models/` and `nn/` products but must never overwrite S1–S9 assets.

- [01 — Mean-squared Hessian curvature energy (HE)](01_hessian_curvature_energy/README.md)
- [03 — Metric-vs-angle curves on saved S9/S9_2 rotations, analytic vs NN](03_s9_metric_curves/README.md)
- [07 — Wave 4-2 first wave: unified 12-toy benchmark, bridges, global analytic search](07_unified_benchmark/README.md) (imports the Lab 03 loaders / metric code and the Lab 06 metric library read-only, as the wave 4-2 specification asks: an explicit exception to the no-import rule above)
- [08 — Wave 4-2 second wave: blind recovery, practical oracle accuracy, conditional HE, improved inference](08_wave4_2_second_wave/README.md) (imports the validated Lab 07 numerical modules read-only, as the second-wave specification asks)

Generated results belong under `outputs/metric_lab/<number>/` (ignored by the repository's existing outputs policy).
