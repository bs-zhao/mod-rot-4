"""S9 deterministic rotations of the T-aligned axes of an S8 oracle R*.

For a sorted subset (a_1,...,a_k), apply the same-angle Givens rotations
in the ordered adjacent planes (a_1,a_2),...,(a_{k-1},a_k). At k=2
this is an ordinary plane rotation; for k>2 this is one explicitly
defined ordered multi-axis sweep, NOT a unique k-dimensional angle.

Column-vector convention: R(angle,axes)=G(angle,axes) @ R_star.
Row-vector subject Z convention: RZ = Z @ R(angle,axes).T.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

ANGLES = (0, 15, 30, 45, 60, 75, 90)


def count_subsets(n_axes: int) -> int:
    if n_axes < 2:
        return 0
    return (1 << int(n_axes)) - int(n_axes) - 1


def axis_subsets(n_axes: int):
    """Enumerate size 2..n, then ascending one-based axes within each size."""
    for size in range(2, n_axes + 1):
        yield from combinations(range(n_axes), size)


def givens_sweep(n_latent: int, axes, angle_degrees: int) -> np.ndarray:
    """Return full orthogonal perturbation G, identity outside chosen axes."""
    axes = tuple(int(i) for i in axes)
    if n_latent < 2 or len(axes) < 2:
        raise ValueError("At least two latent axes are required")
    if len(set(axes)) != len(axes) or tuple(sorted(axes)) != axes:
        raise ValueError("Axes must be unique and strictly ascending")
    if any(i < 0 or i >= n_latent for i in axes):
        raise ValueError("Axis out of bounds")
    if angle_degrees not in ANGLES:
        raise ValueError(f"Angle must be one of {ANGLES}")
    a = np.deg2rad(float(angle_degrees))
    c, s = np.cos(a), np.sin(a)
    out = np.eye(n_latent, dtype=np.float64)
    # Left-multiply the current sweep by the next elementary rotation.
    # This updates two rows without modifying any other axis.
    for i, j in zip(axes[:-1], axes[1:]):
        row_i = out[i].copy()
        row_j = out[j].copy()
        out[i] = c * row_i - s * row_j
        out[j] = s * row_i + c * row_j
    return out


def rotate_from_oracle(r_star, axes, angle_degrees):
    r_star = np.asarray(r_star, dtype=np.float64)
    if r_star.ndim != 2 or r_star.shape[0] != r_star.shape[1]:
        raise ValueError("Oracle R* must be square")
    n = r_star.shape[0]
    if not np.isfinite(r_star).all() or not np.allclose(
            r_star @ r_star.T, np.eye(n), atol=1e-7):
        raise ValueError("S8 R* is not an orthogonal matrix")
    g = givens_sweep(n, axes, angle_degrees)
    rotation = g @ r_star
    return rotation, g


def rotation_id(axes, angle_degrees):
    axes = tuple(int(a) for a in axes)
    tag = "_".join(f"{axis + 1:02d}" for axis in axes)
    return f"axes_{tag}__deg_{int(angle_degrees):03d}"


def expected_rotation_count(n_axes):
    return len(ANGLES) * count_subsets(n_axes)
