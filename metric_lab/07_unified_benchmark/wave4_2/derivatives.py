"""Derivatives of a scalar eta(v), float64, streamed in chunks.

AD (I01, reference): eta, gradient and full Hessian by torch.func.  Channels: YL / YC -> H(eta);  YP -> H(sigmoid(eta)) =
s'(eta) H + s''(eta) g g^T (exact chain rule, so Y-L and Y-P share one pass).
FD (I02): central second differences with step = h * SD of the base coordinate;  GFD (I03): central differences of the AD gradient.
"""
import numpy as np, torch


def _fgh(f):
    g = torch.func.grad(f); H = torch.func.hessian(f)
    return torch.func.vmap(lambda v: (f(v), g(v), H(v)))


def channel_hessian(eta, g, H, channel):
    if channel in ("YL", "YC"):
        return H
    if channel == "YP":
        s = 1 / (1 + np.exp(-eta)); s1 = s * (1 - s); s2 = s1 * (1 - 2 * s)
        return s1[:, None, None] * H + s2[:, None, None] * g[:, :, None] * g[:, None, :]
    raise ValueError(channel)


def ad_chunks(f, P, chunk=1024):
    """yields (slice, eta, grad, Hessian) for consecutive chunks of the points P (N, D)"""
    fn = _fgh(f)
    for i in range(0, len(P), chunk):
        e, g, H = fn(torch.as_tensor(P[i:i + chunk], dtype=torch.float64))
        yield slice(i, i + len(e)), e.detach().numpy(), g.detach().numpy(), H.detach().numpy()


def ad_all(f, P, chunk=1024):
    E, G, H = [], [], []
    for _, e, g, h in ad_chunks(f, P, chunk):
        E.append(e); G.append(g); H.append(h)
    return np.concatenate(E), np.concatenate(G), np.concatenate(H)


def fd_hessian(f, P, step, sd, channel="YL"):
    """I02 central FD of the channel function itself (eta or sigmoid(eta))"""
    ch = (lambda v: torch.sigmoid(f(v))) if channel == "YP" else f
    fv = torch.vmap(ch); D = P.shape[1]; h = torch.as_tensor(step * sd, dtype=torch.float64); E = torch.eye(D, dtype=torch.float64)
    Pt = torch.as_tensor(P, dtype=torch.float64); H = torch.zeros(len(P), D, D, dtype=torch.float64)
    with torch.no_grad():
        f0 = fv(Pt)
        for i in range(D):
            H[:, i, i] = (fv(Pt + h[i] * E[i]) - 2 * f0 + fv(Pt - h[i] * E[i])) / h[i] ** 2
            for j in range(i + 1, D):
                a = fv(Pt + h[i] * E[i] + h[j] * E[j]); b = fv(Pt + h[i] * E[i] - h[j] * E[j])
                c = fv(Pt - h[i] * E[i] + h[j] * E[j]); d = fv(Pt - h[i] * E[i] - h[j] * E[j])
                H[:, i, j] = H[:, j, i] = (a - b - c + d) / (4 * h[i] * h[j])
    return H.numpy()


def gfd_hessian(f, P, step, sd, channel="YL"):
    """I03 central differences of the AD gradient, symmetrised"""
    ch = (lambda v: torch.sigmoid(f(v))) if channel == "YP" else f
    gv = torch.vmap(torch.func.grad(ch)); D = P.shape[1]; h = torch.as_tensor(step * sd, dtype=torch.float64); E = torch.eye(D, dtype=torch.float64)
    Pt = torch.as_tensor(P, dtype=torch.float64); H = torch.zeros(len(P), D, D, dtype=torch.float64)
    for i in range(D):
        H[:, i, :] = (gv(Pt + h[i] * E[i]) - gv(Pt - h[i] * E[i])) / (2 * h[i])
    return ((H + H.transpose(1, 2)) / 2).detach().numpy()
