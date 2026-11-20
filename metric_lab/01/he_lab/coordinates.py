"""Coordinate conventions for oracle and learned rotations (row samples)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Rotation:
    name: str
    variant: str
    candidate_type: str
    axes: tuple
    angle: float
    R: np.ndarray
    R_star: np.ndarray
    G: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray
    t_mean: np.ndarray
    t_std: np.ndarray

    @property
    def dim(self):
        return self.R.shape[0]

    def z_to_u(self, z):
        return ((np.asarray(z) - self.z_mean) / self.z_std) @ self.R.T

    def u_to_z(self, u):
        return (np.asarray(u) @ self.R) * self.z_std + self.z_mean

    def t_to_oracle_u(self, t):
        return ((np.asarray(t) - self.t_mean) / self.t_std) @ self.G.T

    def oracle_u_to_t(self, u):
        return (np.asarray(u) @ self.G) * self.t_std + self.t_mean

    def jacobian(self, source):
        """J = d([original_latent,Xraw])/d([U,Xraw])."""
        if source not in ("oracle", "nn"):
            raise ValueError(source)
        sd = self.t_std if source == "oracle" else self.z_std
        rot = self.G if source == "oracle" else self.R
        return np.diag(sd) @ rot.T

    def validate(self, atol=1e-7):
        d = self.dim
        if self.R.shape != (d, d) or self.R_star.shape != (d, d):
            raise ValueError("Rotations must be square and complete")
        if not np.allclose(self.R @ self.R.T, np.eye(d), atol=atol):
            raise ValueError("R is not orthogonal")
        if not np.allclose(self.G, self.R @ self.R_star.T, atol=atol):
            raise ValueError("G must equal R @ R_star.T")
        if np.any(self.z_std <= 0) or np.any(self.t_std <= 0):
            raise ValueError("Standard deviations must be positive")
        if self.z_mean.shape != (d,) or self.t_mean.shape != (d,):
            raise ValueError("Latent and parameter dimensions must match")


def full_jacobian(latent_jacobian, x_dim):
    k = latent_jacobian.shape[0]
    result = np.eye(k + x_dim, dtype=np.float64)
    result[:k, :k] = latent_jacobian
    return result
