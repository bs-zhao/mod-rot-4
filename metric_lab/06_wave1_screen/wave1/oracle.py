"""Differentiable exact generating formulas; derivatives, not simulated labels.

CR3/CR5 may have undefined two-sided X derivatives at zero amounts or
p=0/1. Those matrix cells are explicitly masked, never silently 'fixed'.
"""
import numpy as np
import torch
from .energy import moment_from_hessians
from .derivatives import moment_hessians
from .sources import get_spec

def _basis(model, x):
    s3 = 3.0 ** .5
    q5 = 5.0 ** .5 / 2.0
    if model == "C1":
        return torch.stack((x.new_tensor(1.), s3*x[0], s3*x[1], s3*x[2]))
    a, b, c, d = x.unbind()
    items = [x.new_tensor(1.), s3*(a-b),
             q5*3*(a*a-b*b), s3*(c-d)]
    if model == "C3":
        items.extend((q5*3*(c*c-d*d), 3*(a*c-b*d)))
    return torch.stack(items)


def _amount_pow(a, alpha):
    # 0**alpha == 0 for alpha >0; exact T derivatives at a=0.
    positive = a > 0
    loga = torch.log(torch.clamp(a / 100., min=1e-100))
    return torch.where(positive, torch.exp(alpha*loga), torch.zeros_like(a))


def _value(lottery, alpha, lam, gamma):
    gain, loss, p = lottery.unbind()
    pc = torch.clamp(p, 1e-12, 1-1e-12)  # match source generator
    pg = torch.sigmoid(gamma*(torch.log(pc)-torch.log1p(-pc)))
    pl = torch.sigmoid(gamma*(torch.log1p(-pc)-torch.log(pc)))
    return pg*_amount_pow(gain, alpha)-lam*pl*_amount_pow(loss, alpha)


def oracle_scalar(model, flat, target="logit"):
    k = {"C1":4, "C2":4, "C3":6, "CR3":4, "CR4":3, "CR5":3}[model]
    t, x = flat[:k], flat[k:]
    if model in ("C1", "C2", "C3"):
        eta = t @ _basis(model, x)
    else:
        if model == "CR3":
            alpha, lam, gam, tau = t.unbind()
        else:
            lam, gam, tau = t.unbind()
            alpha = flat.new_tensor(1.)
        eta = tau*(_value(x[:3], alpha, lam, gam)
                   - _value(x[3:], alpha, lam, gam))
    if target == "logit":
        return eta
    if target == "probability":
        return torch.sigmoid(eta)
    raise ValueError(target)


def validity(model, points):
    """Conservative classical two-sided X-derivative support mask."""
    n = len(points.x)
    k = points.latent.shape[1]
    p = points.x.shape[1]
    good = np.ones((n, k+p, k+p), dtype=bool)
    if model.startswith("CR"):
        x = points.x
        positive = np.ones((n, p), bool)
        for j in (0, 1, 3, 4):
            positive[:, j] = x[:, j] > 0
        for j in (2, 5):
            positive[:, j] = (x[:, j] > 0) & (x[:, j] < 1)
        for j in range(p):
            # All derivatives involving this physical X direction are
            # conservatively invalid if the point is on its support boundary.
            invalid = ~positive[:, j]
            good[invalid, k+j, :] = False
            good[invalid, :, k+j] = False
    return good


def oracle_base_moment(src, points, target, chunk=64, boundary_policy="mask"):
    k = src.parameters.shape[1]
    if boundary_policy not in ("mask", "interior"):
        raise ValueError(boundary_policy)
    legal = validity(src.model, points)
    if boundary_policy == "interior":
        selected = legal.all(axis=(1, 2))
        if not selected.any():
            raise ValueError("No fully interior CPT evaluation points")
        selected_points = np.column_stack((points.latent[selected], points.x[selected]))
        weights = points.weights[selected]
        coverage = np.ones((k+src.x_dim, k+src.x_dim), bool)
    else:
        selected_points = np.column_stack((points.latent, points.x))
        weights = points.weights
        coverage = legal.all(axis=0)
    fn = lambda v: oracle_scalar(src.model, v, target=target)
    moment = moment_hessians(fn, selected_points, weights=weights, chunk=chunk)
    # A boundary continuation is only used for numerical differentiability;
    # affected physical-X cells are NEVER declared oracle-valid.
    return moment, coverage, {
        "source": "generating_formula", "target": target,
        "points": len(selected_points), "boundary_policy": boundary_policy,
        "base_valid_fraction": float(np.mean(legal)),
        "invalid_matrix_cells": int((~coverage).sum())
    }
