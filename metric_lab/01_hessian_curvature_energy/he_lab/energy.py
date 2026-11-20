"""All estimators return the same elementwise expected squared Hessian.

M = E[H ⊙ H], NOT E[H.T @ H]; M need not be positive semidefinite.
"""
from dataclasses import dataclass, field
import numpy as np
from .coordinates import full_jacobian


@dataclass
class HEResult:
    M: np.ndarray
    dim_u: int
    method: str
    target: str
    rotation_id: str
    measure: str
    valid_mask: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def M_UU(self):
        return self.M[:self.dim_u, :self.dim_u]

    @property
    def M_UX(self):
        return self.M[:self.dim_u, self.dim_u:]

    @property
    def M_XX(self):
        return self.M[self.dim_u:, self.dim_u:]

    @property
    def rms(self):
        return np.sqrt(np.maximum(self.M, 0))

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, M=self.M, M_UU=self.M_UU,
                            M_UX=self.M_UX, M_XX=self.M_XX,
                            rms=self.rms, valid_mask=self.valid_mask,
                            dim_u=self.dim_u)


def moment_from_hessians(hessians, weights=None):
    """Second moment of vec(H); enables ALL orthogonal sweeps without re-differentiating."""
    h = np.asarray(hessians, dtype=np.float64)
    if h.ndim != 3 or h.shape[1] != h.shape[2]:
        raise ValueError("Expected [N,d,d] Hessian array")
    n, d, _ = h.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid nonnegative evaluation weights")
    hf = h.reshape(n, d * d)
    if not np.all(np.isfinite(hf)):
        raise ValueError("Nonfinite Hessian: inspect support/boundary")
    return (hf.T * w) @ hf / w.sum()


def transform_moment(moment, jacobian):
    """M_ab = E[(J[:,a].T H J[:,b])**2] via a cross-second-moment tensor."""
    j = np.asarray(jacobian, dtype=np.float64)
    d = j.shape[0]
    if j.shape != (d, d) or moment.shape != (d*d, d*d):
        raise ValueError("Incompatible moment/Jacobian")
    basis = np.einsum("ia,jb->abij", j, j, optimize=True).reshape(d*d, d*d)
    m = np.einsum("ai,ij,aj->a", basis, moment, basis, optimize=True).reshape(d, d)
    return (m + m.T) / 2


def transform_coverage(coverage, jacobian, tolerance=1e-12):
    """Conservative support validity: any undefined base derivative contaminates transformed cell."""
    d = jacobian.shape[0]
    if coverage.shape != (d, d):
        raise ValueError("Expected derivative-coverage matrix")
    active = abs(jacobian) > tolerance
    possible = np.einsum("ia,jb->abij", active, active)
    missing = ~coverage
    invalid = np.any(possible & missing[None, None, :, :], axis=(2, 3))
    return ~invalid


def result_from_moment(moment, rotation, x_dim, source, target, measure,
                       method, base_coverage=None, meta=None):
    j = full_jacobian(rotation.jacobian(source), x_dim)
    m = transform_moment(moment, j)
    valid = (np.ones(m.shape, bool) if base_coverage is None
             else transform_coverage(base_coverage, j))
    m = np.where(valid, m, np.nan)
    return HEResult(m, rotation.dim, method, target, rotation.name,
                    measure, valid, dict(meta or {}))
