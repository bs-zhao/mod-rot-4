"""One independent per-trial X sampler for C1/C2/C3 and CR4.

All models use a *single* np.random.Generator, one call per subject:
rng.uniform(low, high, size=(n_trials, x_dim)). Model specifications can
provide scalar bounds or separate bounds for each feature. No X bank, resampling,
pre-generated separate amount/probability streams, or per-feature RNG objects.
"""
from __future__ import annotations

import numpy as np


def sample_independent_uniform_x(n_trials, rng, *, low, high, n_features):
    count = int(n_trials)
    dimension = int(n_features)
    if count < 1 or dimension < 1:
        raise ValueError("Number of trials and X dimensions must be positive")
    lo, hi = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    if lo.ndim not in (0, 1) or hi.ndim not in (0, 1):
        raise ValueError("Uniform bounds must be scalars or 1D vectors")
    for label, arr in (("low", lo), ("high", hi)):
        if arr.ndim == 1 and arr.shape != (dimension,):
            raise ValueError(f"{label} must have exactly {dimension} X bounds")
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()
            and np.all(lo < hi)):
        raise ValueError("Uniform input bounds must be finite with low < high")
    # Scalar branch is intentionally bitwise identical to the original
    # C1/C2/C3 rng.uniform(-1.0,1.0,size=(n,d)).
    if lo.ndim == 0 and hi.ndim == 0:
        return rng.uniform(float(lo), float(hi), size=(count, dimension))
    return rng.uniform(lo, hi, size=(count, dimension))
