"""Second-moment tensors of the base-coordinate Hessian and their exact contraction to M^UU (A) and M^UX (B).

Base coordinates: [latent_raw, X_std].  K_abcd = E[H_ab H_cd] (latent block), J^(x)_ab = E[H_a,x H_b,x].
Masks (frozen before any rotation, identical for every arm and latent axis):
  FULL      no X boundary (C families, CR4): every point counts.
  MASK-ALL  only points whose every X column is interior (Lab 03 estimand); one common denominator.
  MASK-COL  UU over all points; UX column x over the points where column x is interior (Lab 06 estimand); per-column denominators.
Rotation enters only through J = d raw / d U = diag(scale) Q^T; squared energies are never rotated.
"""
import numpy as np

MASKS_BANK = ("MASK-ALL", "MASK-COL")


def mask_modes(blind):
    return MASKS_BANK if blind.x_domain is not None and blind.fixed_bank is not None else ("FULL",)


class Moments:
    def __init__(self, k, p, mode):
        self.k, self.p, self.mode = k, p, mode
        self.K = np.zeros((k, k, k, k)); self.J = np.zeros((p, k, k)); self.nK = 0; self.nJ = np.zeros(p); self.n_points = 0
        self.nonfinite = 0

    def add(self, H, cm):
        k = self.k; H = np.asarray(H, float); self.n_points += len(H)
        if self.mode == "MASK-ALL":
            keep = cm.all(1); H = H[keep]; cm = np.ones((len(H), self.p), bool)
        elif self.mode == "FULL":
            cm = np.ones((len(H), self.p), bool)
        Huu = H[:, :k, :k]
        bad = ~np.isfinite(Huu).all(axis=(1, 2))
        for j in range(self.p):
            bad |= cm[:, j] & ~np.isfinite(H[:, :k, k + j]).all(1)
        if bad.any():                                    # never zero-filled: counted and the configuration is flagged invalid
            self.nonfinite += int(bad.sum())
        self.K += np.einsum("nab,ncd->abcd", Huu, Huu, optimize=True); self.nK += len(H)
        for j in range(self.p):
            hb = H[cm[:, j], :k, k + j]; self.J[j] += hb.T @ hb; self.nJ[j] += len(hb)

    def finish(self):
        K = self.K / max(self.nK, 1); J = self.J / np.maximum(self.nJ, 1)[:, None, None]
        cov = self.nJ / max(self.n_points, 1)
        return dict(K=K, J=J, nK=self.nK, nJ=self.nJ.copy(), n_points=self.n_points, coverage=cov, nonfinite=self.nonfinite,
                    valid=bool(self.nonfinite == 0 and self.nK > 0 and (self.nJ > 0).all()))


def contract(K, J, Jz):
    """Jz (k_raw, k_rot) = d raw / d U.  -> A (k,k) = M^UU, B (k,p) = M^UX   (exact; symmetrised, clipped at 0 for round-off)"""
    B = np.einsum("ia,xik,ka->ax", Jz, J, Jz, optimize=True)
    W = np.einsum("ijkl,ia,ka->ajl", K, Jz, Jz, optimize=True); A = np.einsum("ajl,jb,lb->ab", W, Jz, Jz, optimize=True)
    return np.maximum((A + A.T) / 2, 0), np.maximum(B, 0)


def jac(scale, Q):
    """U = Q * (raw / scale) (up to centring):  d raw / d U = diag(scale) Q^T"""
    return np.asarray(scale)[:, None] * np.asarray(Q).T
