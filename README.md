# Mod-Rot-4

A modular research platform for model simulation, Neural Processes, MLP surrogate decoders, functional metrics, and latent rotations.

## Implemented now

- [`models/`](models/README.md): faithful C1/C2/C3/CR3/CR4/CR5 simulation and S2 prechecks imported from `bs-zhao/mod-rot-3/np`, with **8,000 subjects by default**.

Other project modules will be added separately.

Model output convention: `outputs/models/generate/<MODEL>/` and `outputs/models/precheck/<MODEL>/`. Other modules (NP, MLP, metrics, rotations) own separate folders under `outputs/` when implemented.
