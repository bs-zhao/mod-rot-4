"""Full input Hessians for a scalar torch function; no parameter Hessians."""
import numpy as np
import torch


def hessians(fn, points, chunk=64):
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("points must be [N,d]")
    transformed = torch.func.vmap(torch.func.hessian(fn))
    out = []
    for start in range(0, len(x), chunk):
        batch = torch.as_tensor(x[start:start+chunk], dtype=torch.float64)
        hh = transformed(batch).detach().cpu().numpy()
        if not np.all(np.isfinite(hh)):
            raise ValueError("Nonfinite Hessian at evaluation points")
        out.append(hh)
    return np.concatenate(out, axis=0)


def moment_hessians(fn, points, weights=None, chunk=64):
    """Stream full input-Hessian cross moments without holding [N,d,d] in RAM."""
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or len(x) < 1:
        raise ValueError("Expected nonempty [N,d] input points")
    n, d = x.shape
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid weights")
    cross = np.zeros((d*d, d*d), dtype=np.float64)
    transformed = torch.func.vmap(torch.func.hessian(fn))
    for start in range(0, n, chunk):
        xx = torch.as_tensor(x[start:start+chunk], dtype=torch.float64)
        hh = transformed(xx).detach().cpu().numpy().reshape(len(xx), d*d)
        if not np.all(np.isfinite(hh)):
            raise ValueError("Nonfinite Hessian at evaluation points")
        ww = w[start:start+len(xx)]
        cross += hh.T @ (ww[:, None]*hh)
    return cross/w.sum()
