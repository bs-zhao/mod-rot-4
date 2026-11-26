"""EVALUATION-ONLY: analytic response, parameter prior and support.  Never imported by blind modules (tests enforce this)."""
import numpy as np, torch
from s9curves.oracle import oracle_scalar                      # Lab 03 differentiable generating formulas (exact eta)
from modrot4.model_specs import get_spec
from modrot4.pipeline import base_model

# parameter support (CR families): the S1 bounds of the CPT prior (models/reference/s0_cr2_design_recovery.py BOUNDS)
_CPT4 = np.array([[.02, .999], [.3, 5.], [.02, .999], [.5, 4.]])
BOUNDS = {"C1": None, "C2": None, "C3": None, "CR3": _CPT4, "CR4": _CPT4[1:], "CR5": _CPT4[1:]}


def oracle_eta(family, x_mean, x_std):
    """eta(T_raw, X) as a function of v = [T_raw, X standardised with the toy's saved S3 scaler]"""
    k = {"C1": 4, "C2": 4, "C3": 6, "CR3": 4, "CR4": 3, "CR5": 3}[family]
    xm = torch.as_tensor(x_mean, dtype=torch.float64); xs = torch.as_tensor(x_std, dtype=torch.float64)
    return lambda v: oracle_scalar(family, torch.cat((v[:k], v[k:] * xs + xm)), "logit")


def eta_numpy(family, T, X):
    f = lambda v: oracle_scalar(family, v, "logit")
    V = torch.as_tensor(np.column_stack((T, X)), dtype=torch.float64)
    with torch.no_grad():
        return torch.vmap(f)(V).numpy()


def prior_draws(family, n, rng):
    return np.asarray(get_spec(family).sample_parameters(n, rng), float)


def in_support(family, T):
    b = BOUNDS[family]
    if b is None:
        return np.ones(len(T), bool), np.ones(T.shape, bool)
    per = (T >= b[:, 0]) & (T <= b[:, 1])
    return per.all(1), per
