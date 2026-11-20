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
