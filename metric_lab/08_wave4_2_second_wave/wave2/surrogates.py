"""W2-04 blind response-surface estimators (02_second_wave sections 17-18).  Truth-free: inputs are Z, X, Y, the frozen decoder (teacher)
and the saved scalers.  Every surface is a scalar eta(w), w = [raw Z, X standardised by the S3 scaler], fitted ONCE in this base frame;
derivatives are rotated exactly by the moment contraction afterwards.

E-SMOOTH  g_h(w) = E_xi d(w + diag(h) xi), latent-only isotropic perturbation in the standardised latent frame (xi_V ~ N(0, I), h_X = 0),
          16 antithetic draws (pilot).  Hessian H[g_h](w) = E_xi H[d](w + h xi) (fixed noise law); the energy squares the AVERAGED Hessian.
E-POLY    complete total-degree basis (degree 2-5) on dimensionless standardised inputs, monomial ('mono') or product probabilists' Hermite
          ('herm'; a polynomial basis regression, not a Sobol decomposition: inputs are correlated), ridge (MSE) or logistic ridge (class B, D0).
E-RBF     Gaussian landmark expansion, m landmarks drawn from TRAIN rows (fixed seed), h = factor x median pairwise distance, shared isotropic
          or blockwise (latent / X blocks each scaled to unit median distance) input scaling; ridge or penalised logistic.
Sources   D0 observed data (class B binary choice: Bernoulli / logit link; class C continuous y: squared error) and D1 frozen-decoder teacher
          (logit or continuous output: squared error).  TRAIN subjects fit, VALIDATION subjects select.
"""
import numpy as np, torch
from itertools import combinations_with_replacement
from scipy.optimize import minimize

# ---------------------------------------------------------------- data rows (subject-grouped)
def rows_of(pl, pool, n_sub, n_trials, seed, exclude=None):
    P = np.sort(pl.splits[pool]) if isinstance(pool, str) else np.sort(np.asarray(pool))
    if exclude is not None:
        P = np.setdiff1d(P, exclude)
    rng = np.random.default_rng([seed, 41]); sub = rng.permutation(P)[:n_sub]; idx = []
    for s in sub:
        c = int(pl.counts[s]); idx.append(pl.starts[s] + np.random.default_rng([seed, 42, int(s)]).permutation(c)[:min(n_trials, c)])
    return np.concatenate(idx), sub


def inputs(pl, idx):
    return np.column_stack((pl.z[pl.subject_index[idx]], (pl.x[idx] - pl.x_mean) / pl.x_std))


def teacher(pl, W):
    from wave4_2.surfaces import DecoderSurface
    return DecoderSurface(pl).predict(W[:, :pl.k], W[:, pl.k:])


# ---------------------------------------------------------------- polynomial bases
def exponents(D, degree):
    out = []
    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(D), d):
            a = np.zeros(D, int)
            for i in comb:
                a[i] += 1
            out.append(a)
    return np.array(out, int)


def _univariate(u, deg, basis):
    """u (n, D) -> (n, D, deg+1) monomial powers or probabilists' Hermite He_0..He_deg"""
    out = [torch.ones_like(u), u]
    for n in range(2, deg + 1):
        out.append(out[-1] * u if basis == "mono" else u * out[-1] - (n - 1) * out[-2])
    return torch.stack(out[:deg + 1], -1)


def phi_torch(u, E, basis):
    deg = int(E.max()); H = _univariate(u, deg, basis)                      # (n, D, deg+1)
    Et = torch.as_tensor(E, dtype=torch.long)                                # (R, D)
    g = torch.gather(H.unsqueeze(1).expand(-1, Et.shape[0], -1, -1), 3, Et.unsqueeze(0).unsqueeze(-1).expand(H.shape[0], -1, -1, 1)).squeeze(-1)
    return g.prod(-1)                                                        # (n, R)


def phi_np(U, E, basis, chunk=2048):
    out = []
    with torch.no_grad():
        for i in range(0, len(U), chunk):
            out.append(phi_torch(torch.as_tensor(U[i:i + chunk], dtype=torch.float64), E, basis).numpy())
    return np.concatenate(out)


class PolyEta:
    def __init__(self, m, s, E, basis, w, b):
        self.m, self.s, self.E, self.basis, self.w, self.b = (torch.as_tensor(m), torch.as_tensor(s), E, basis, torch.as_tensor(w), float(b))

    def eta(self):
        m, s, E, basis, w, b = self.m, self.s, self.E, self.basis, self.w, self.b
        return lambda v: b + (phi_torch(((v - m) / s)[None], E, basis)[0] * w).sum()

    def predict(self, W):
        U = (W - self.m.numpy()) / self.s.numpy(); return self.b + phi_np(U, self.E, self.basis) @ self.w.numpy()


# ---------------------------------------------------------------- RBF
class RBFEta:
    def __init__(self, m, s, C, h, a, b):
        self.m, self.s, self.C, self.h, self.a, self.b = m, s, C, float(h), a, float(b)

    def features(self, W):
        U = (W - self.m) / self.s; d2 = (U ** 2).sum(1)[:, None] + (self.C ** 2).sum(1)[None] - 2 * U @ self.C.T
        return np.exp(-np.maximum(d2, 0) / (2 * self.h ** 2))

    def predict(self, W, chunk=8192):
        return np.concatenate([self.b + self.features(W[i:i + chunk]) @ self.a for i in range(0, len(W), chunk)])

    def hessian(self, W, chunk=512):
        """analytic Hessian in the w coordinates: d2 phi / du du = phi [(u-c)(u-c)^T / h^4 - I / h^2], chain rule through u = (w - m) / s"""
        out = []
        for i in range(0, len(W), chunk):
            U = (W[i:i + chunk] - self.m) / self.s; F = self.features(W[i:i + chunk]) * self.a[None]     # (n, m) a_l phi_l
            Dl = U[:, None, :] - self.C[None]                                                             # (n, m, D)
            H = np.einsum("nl,nli,nlj->nij", F, Dl, Dl, optimize=True) / self.h ** 4 - F.sum(1)[:, None, None] * np.eye(U.shape[1])[None] / self.h ** 2
            out.append(H / (self.s[None, :, None] * self.s[None, None, :]))
        return np.concatenate(out)


# ---------------------------------------------------------------- fitting
def ridge_path(F_iter, y, lambdas, n, p_dim):
    """streamed Gram ridge with unpenalised intercept: columns centred by their TRAIN means.  F_iter yields feature chunks in row order."""
    G = np.zeros((p_dim, p_dim)); s = np.zeros(p_dim); Fy = np.zeros(p_dim); off = 0
    for F in F_iter():
        yy = y[off:off + len(F)]; off += len(F); G += F.T @ F; s += F.sum(0); Fy += F.T @ yy
    mu = s / n; ym = y.mean(); Gc = G - n * np.outer(mu, mu); Fyc = Fy - n * mu * ym
    sc = np.sqrt(np.maximum(np.diag(Gc) / n, 1e-300)); Gs = Gc / np.outer(sc, sc); Fs = Fyc / sc
    ev, V = np.linalg.eigh(Gs); out = []
    for lam in lambdas:
        ws = V @ ((V.T @ Fs) / (ev + lam * n)); w = ws / sc; b = ym - mu @ w; out.append((lam, w, b))
    return out, dict(cond=float(ev.max() / max(ev.min(), 1e-300)), rank=int((ev > ev.max() * 1e-12).sum()))


def logistic_path(F, y, lambdas, maxiter=400):
    """penalised logistic (intercept unpenalised), objective mean NLL + lam ||w_scaled||^2 on column-standardised features; warm start
    along the decreasing lambda path."""
    mu = F.mean(0); sc = F.std(0); sc[sc <= 0] = 1; Fs = (F - mu) / sc; n, p_dim = Fs.shape; out = []; x0 = np.zeros(p_dim + 1)
    for lam in sorted(lambdas, reverse=True):
        def fg(x):
            z = Fs @ x[1:] + x[0]; pz = 1 / (1 + np.exp(-z)); nll = np.mean(np.logaddexp(0, z) - y * z)
            g = np.r_[np.mean(pz - y), Fs.T @ (pz - y) / n + 2 * lam * x[1:]]
            return nll + lam * (x[1:] ** 2).sum(), g
        r = minimize(fg, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter)); x0 = r.x
        w = r.x[1:] / sc; b = r.x[0] - mu @ w; out.append((lam, w, b, bool(r.success), int(r.nit)))
    return out


def nll_or_mse(pred_eta, y, cls_b_obs):
    if cls_b_obs:
        p = np.clip(1 / (1 + np.exp(-pred_eta)), 1e-12, 1 - 1e-12); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean((pred_eta - y) ** 2))


def median_dist(U, n=2000, seed=3):
    rng = np.random.default_rng(seed); S = U[rng.choice(len(U), min(n, len(U)), replace=False)]
    d = np.sqrt(np.maximum((S ** 2).sum(1)[:, None] + (S ** 2).sum(1)[None] - 2 * S @ S.T, 0)); return float(np.median(d[np.triu_indices(len(S), 1)]))


# ---------------------------------------------------------------- smoothing of the frozen decoder
def smooth_hessians(eta, P, z_sd, k, h, n_draws=16, seed=0, chunk=1024):
    """averaged Hessian E_xi H[d](w + h xi) with latent-only antithetic Gaussian perturbations (standardised latent frame)"""
    from wave4_2.derivatives import ad_chunks
    rng = np.random.default_rng([seed, 51]); half = rng.standard_normal((n_draws // 2, k)); xi = np.concatenate([half, -half])
    Hsum = np.zeros((len(P), P.shape[1], P.shape[1]))
    for e in xi:
        Pp = P.copy(); Pp[:, :k] += h * z_sd * e
        Hsum += np.concatenate([H for _, _, _, H in ad_chunks(eta, Pp, chunk)])
    return Hsum / len(xi)
