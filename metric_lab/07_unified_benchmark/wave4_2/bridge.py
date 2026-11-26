"""EVALUATION-ONLY (W1-03): representation diagnostics and the supervised bridge surfaces.

B-orth(z) = g(t_mean + t_std * R* (V(z) - c), X)          R*, c = saved S8 / S8_2 rotation and Procrustes centre (V coordinates)
B-aff(z)  = g(t_mean + t_std * (A V(z) + b), X)           A, b = TRAIN OLS of standardised T on V (with intercept)
V(z) = z (S9) or (z - mu_global) / sd_global (S9_2, the saved S8_2 normaliser).  Both are functions of raw z, so they are rotated
with exactly the same Jacobian as the decoder.  They are diagnostic targets, not candidate inference methods.
"""
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from .truth import oracle_eta


def v_of_z(z, ref, variant):
    return z if variant == "s9" else (z - ref["z_mean"]) / ref["z_std"]


def affine_fit(blind, ev, ref, variant):
    V = v_of_z(blind.z, ref, variant); tr = blind.splits["train"]
    Ts = (ev.T - ref["t_mean"]) / ref["t_std"]
    X1 = np.column_stack((np.ones(len(V)), V)); W = np.linalg.lstsq(X1[tr], Ts[tr], rcond=None)[0]
    return W[1:].T, W[0]                                       # A (dimT x k), b


def bridge_eta(blind, ev, ref, variant, kind, A=None, b=None):
    """eta of a bridge as a function of v = [z_raw, x_std]"""
    fo = oracle_eta(ev.family, blind.x_mean, blind.x_std); k = blind.k
    zm = torch.as_tensor(ref["z_mean"]); zs = torch.as_tensor(ref["z_std"]); tm = torch.as_tensor(ref["t_mean"]); ts = torch.as_tensor(ref["t_std"])
    if kind == "B-orth":
        M = torch.as_tensor(ref["R_star"]); c = torch.as_tensor(ref["z_center"])
        lin = lambda V: M @ (V - c)
    else:
        M = torch.as_tensor(A); c = torch.as_tensor(b)
        lin = lambda V: M @ V + c
    std = variant != "s9"

    def f(v):
        z = v[:k]; V = (z - zm) / zs if std else z
        return fo(torch.cat((tm + ts * lin(V), v[k:])))
    return f


def bridge_T(blind, ev, ref, variant, kind, Z, A=None, b=None):
    V = v_of_z(Z, ref, variant)
    Ts = (V - ref["z_center"]) @ ref["R_star"].T if kind == "B-orth" else V @ A.T + b
    return ref["t_mean"] + ref["t_std"] * Ts


def matched(Tc, U):
    """Hungarian matching on |corr| -> (matched |r| per T, MCC, min, leak mean, leak max)"""
    d = Tc.shape[1]; C = np.corrcoef(Tc.T, U.T)[:d, d:]; Aab = np.abs(C); r, c = linear_sum_assignment(-Aab); m = Aab[r, c]
    M = Aab.copy(); M[r, c] = np.nan; off = M[~np.isnan(M)]
    return m, float(m.mean()), float(m.min()), float(off.mean()) if off.size else 0.0, float(off.max()) if off.size else 0.0


def representation(blind, ev, refs, variant):
    ref = refs[variant]; V = v_of_z(blind.z, ref, variant); tr, te = blind.splits["train"], blind.splits["test"]
    Ts = (ev.T - ref["t_mean"]) / ref["t_std"]; cov = np.cov(V[tr], rowvar=False, ddof=0); eig = np.linalg.eigvalsh(cov)
    A, b = affine_fit(blind, ev, ref, variant); pred = V[te] @ A.T + b; res = Ts[te] - pred
    r2 = 1 - (res ** 2).mean(0) / Ts[te].var(0); sv = np.linalg.svd(A, compute_uv=False)
    U = (V[te] - ref["z_center"]) @ ref["R_star"].T
    m8, mcc8, mn8, lm8, lx8 = matched(Ts[te], U); m0, mcc0, mn0, lm0, lx0 = matched(Ts[te], V[te])
    out = dict(latent_mean=V[tr].mean(0).tolist(), latent_sd=V[tr].std(0).tolist(), cov_eigenvalues=eig.tolist(), cov_condition=float(eig.max() / eig.min()),
               linear_R2_test=r2.tolist(), linear_R2_min=float(r2.min()), linear_R2_mean=float(r2.mean()),
               residual_cov_eigenvalues=np.linalg.eigvalsh(np.cov(res, rowvar=False, ddof=0)).tolist(),
               map_singular_values=sv.tolist(), map_condition=float(sv.max() / sv.min()),
               supervised_matched=m8.tolist(), supervised_MCC=mcc8, supervised_min=mn8, supervised_leak_mean=lm8, supervised_leak_max=lx8,
               unrotated_MCC=mcc0, unrotated_min=mn0, unrotated_leak_mean=lm0, unrotated_leak_max=lx0)
    return out, (A, b)
