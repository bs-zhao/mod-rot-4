"""Supervised orthogonal T-aligned rotation of frozen subject-level Z.

R is always a genuine L x L orthogonal matrix. No Z rescaling is hidden
inside R: fit uses centered Z, and applying R to original data is Z @ R.T.
T is standardized on TRAIN for equal target weights during orientation fit.
All supervised fitting uses TRAIN; VAL tunes Ridge and TEST reports outcomes.
"""
from __future__ import annotations

import numpy as np


def fit_supervised_rotation(z, theta, train_ids):
    z = np.asarray(z, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    train = np.asarray(train_ids, dtype=np.int64)
    if z.ndim != 2 or theta.ndim != 2 or z.shape[0] != theta.shape[0]:
        raise ValueError("Z and T must be 2D with the same subject count")
    n, latent_dim = z.shape
    t_dim = theta.shape[1]
    if latent_dim < t_dim:
        raise ValueError(f"latent_dim={latent_dim} < T_dim={t_dim}; orthogonal S8 needs L >= T_dim")
    if len(train) < max(t_dim+1, 3) or len(np.unique(train)) != len(train):
        raise ValueError("Insufficient or duplicate TRAIN subjects for supervised rotation")
    if np.any(train < 0) or np.any(train >= n):
        raise ValueError("TRAIN IDs out of bounds")
    if not np.isfinite(z).all() or not np.isfinite(theta).all():
        raise ValueError("Z and T must be finite")

    z_mean = z[train].mean(axis=0)
    theta_mean = theta[train].mean(axis=0)
    theta_sd = theta[train].std(axis=0)
    if np.any(theta_sd <= 1e-12):
        raise ValueError("At least one true T has zero TRAIN variance")

    z_train = z[train] - z_mean
    target = (theta[train] - theta_mean) / theta_sd
    target_padded = np.zeros((len(train), latent_dim), dtype=np.float64)
    target_padded[:, :t_dim] = target

    # Orthogonal Procrustes: argmin_{R R^T=I} ||(Z-trainmean) R^T - [Tstd,0]||_F^2.
    # C = Zc^T [Tstd,0] = U Σ V^T => Q = U V^T, R = Q^T.
    cross = z_train.T @ target_padded
    left, singular_values, right_t = np.linalg.svd(cross, full_matrices=True)
    q = left @ right_t
    rotation = q.T
    if not np.allclose(rotation @ rotation.T, np.eye(latent_dim), atol=1e-9):
        raise AssertionError("Computed supervised rotation is not orthogonal")
    rotated = z @ rotation.T
    zcenter_rotated = (z - z_mean) @ rotation.T
    residual = zcenter_rotated[train] - target_padded
    return dict(
        R=rotation, rz=rotated, u=rotated[:, :t_dim],
        z_train_mean=z_mean, t_train_mean=theta_mean,
        t_train_std=theta_sd, singular_values=singular_values,
        procrustes_train_mse=float(np.mean(residual**2)),
        orthogonality_max_error=float(np.max(np.abs(rotation @ rotation.T - np.eye(latent_dim)))),
        objective="min TRAIN ||(Z - mean_train Z) R^T - [standardize_train(T), zero]||_F^2",
        coordinate_convention="Z rows; RZ column-style, rows computed Z @ R.T",
    )
