# Metric Lab

Numbered, independent metric experiments. Each `NN/` experiment owns its code, tests, configuration, and outputs; no imports between experiments. Modules may **read** the existing `models/` and `nn/` products but must never overwrite S1–S9 assets.

- [01 — Mean-squared Hessian curvature energy (HE)](01_hessian_curvature_energy/README.md)
- [03 — Metric-vs-angle curves on saved S9/S9_2 rotations, analytic vs NN](03_s9_metric_curves/README.md)

Generated results belong under `outputs/metric_lab/<number>/` (ignored by the repository's existing outputs policy).
