"""Evaluation measures Q01-Q05, channels (Y-L / Y-P), inference I01 (AD), I02 (central FD), I03 (gradient FD), interior-X masks,
second-moment tensors (04.3 section 6) and the exact contraction to A = M^UU, B = M^UX for any saved rotation.

Base coordinates of every Hessian: [latent (oracle: raw T; inference: raw Z), X standardised with the S3 TRAIN scaler].
"""
import numpy as np, torch
from .oracle import oracle_scalar
from .sources import get_spec, load_sources, load_rotations
from .decoder import DiscreteNPConfig, DiscreteNeuralProcess

NSUB, NTRIAL, NLAT_BANK = 256, 32, 32          # pilot fidelity of 04.3 section 5 (bank: 32 latents x all 500 rows)
CHANNELS = {"L": "logit", "P": "probability"}


def col_mask(model, x):
    m = np.ones(x.shape, bool)
    if model.startswith("CR"):
        for j in (0, 1, 3, 4): m[:, j] = x[:, j] > 0
        for j in (2, 5): m[:, j] = (x[:, j] > 0) & (x[:, j] < 1)
    return m


class Eval:
    def __init__(self, model, run_dir, seed=31):
        self.model = model; self.src = load_sources(model, run_dir); self.k = self.src.latent_dim; self.p = self.src.x_dim; self.seed = seed
        self.spec = get_spec(model); self.bank = self.src.fixed_x
        st = torch.load(self.src.checkpoint_file, map_location="cpu", weights_only=True); cfg = dict(st["model_config"]); cfg["decoder_hidden_dims"] = tuple(cfg["decoder_hidden_dims"])
        net = DiscreteNeuralProcess(DiscreteNPConfig(**cfg)); net.load_state_dict(st["model_state_dict"], strict=True); net.eval().double()
        for q in net.parameters(): q.requires_grad_(False)
        self.net = net.decoder; self.xm = torch.as_tensor(self.src.x_mean, dtype=torch.float64); self.xs = torch.as_tensor(self.src.x_std, dtype=torch.float64)
        s = self.src; self.starts = np.r_[0, np.cumsum(s.counts[:-1])]; self.Tmu = s.parameters.mean(0); self.Tsd = s.parameters.std(0)
        self.Zmu = s.z.mean(0); self.Zsd = s.z.std(0); self.Zcov = np.cov(s.z, rowvar=False, ddof=0)

    # ---------------- functions of v = [latent, x_std] ----------------
    def fn(self, source, channel):
        k, tgt = self.k, CHANNELS[channel]
        if source == "oracle":
            return lambda v: oracle_scalar(self.model, torch.cat((v[:k], v[k:] * self.xs + self.xm)), tgt)
        def f(v):
            o = self.net(torch.cat((v[k:], v[:k]))); eta = o[1] - o[0]
            return eta if tgt == "logit" else torch.sigmoid(eta)
        return f

    # ---------------- evaluation point sets ----------------
    def points(self, Q):
        """-> dict(T, Z, x) with the same rows for oracle (T) and inference (Z); None if Q is not defined for this model."""
        s = self.src; rng = np.random.default_rng(self.seed + int(Q[1:])); S = len(s.counts)
        if Q in ("Q04",) and self.bank is None: return None
        def rows_of(sub, n):
            idx = []
            for sb in sub:
                c = int(s.counts[sb]); idx.append(self.starts[sb] + rng.choice(c, n, replace=c < n))
            return np.concatenate(idx)
        if Q == "Q01":
            sub = rng.choice(S, NSUB, replace=False); r = rows_of(sub, NTRIAL); sid = np.repeat(sub, NTRIAL)
            return dict(T=s.parameters[sid], Z=s.z[sid], x=s.x[r])
        if Q == "Q02":
            if self.bank is not None:
                sub = rng.choice(S, NLAT_BANK, replace=False); sid = np.repeat(sub, len(self.bank)); return dict(T=s.parameters[sid], Z=s.z[sid], x=np.tile(self.bank, (NLAT_BANK, 1)))
            sub = rng.choice(S, NSUB, replace=False); sid = np.repeat(sub, NTRIAL); r = rng.choice(len(s.x), NSUB * NTRIAL, replace=False)
            return dict(T=s.parameters[sid], Z=s.z[sid], x=s.x[r])
        if Q in ("Q03", "Q04", "Q05"):
            if Q == "Q04" or (Q == "Q05" and self.bank is not None) or (Q == "Q03" and False):
                nl = NLAT_BANK; x = np.tile(self.bank, (nl, 1)); rep = len(self.bank)
            else:
                nl = NSUB; rep = NTRIAL; x = s.x[rng.choice(len(s.x), nl * rep, replace=False)]
            eps = rng.standard_normal((nl, self.k))
            if Q == "Q05": Zl = self.Zmu + self.Zsd * eps; Tl = self.Tmu + self.Tsd * eps
            else:
                Zl = self.Zmu + eps @ np.linalg.cholesky(self.Zcov).T; Tl = np.asarray(self.spec.sample_parameters(nl, np.random.default_rng(self.seed + 900 + int(Q[1:]))))
            return dict(T=np.repeat(Tl, rep, 0), Z=np.repeat(Zl, rep, 0), x=x)
        raise ValueError(Q)

    # ---------------- Hessians ----------------
    @staticmethod
    def hess_ad(f, P, chunk=512):
        hv = torch.func.vmap(torch.func.hessian(f)); out = []
        for i in range(0, len(P), chunk): out.append(hv(torch.as_tensor(P[i:i + chunk], dtype=torch.float64)).detach().numpy())
        return np.concatenate(out)

    @staticmethod
    def hess_fd(f, P, step, chunk=2048):
        sd = P.std(0); h = torch.as_tensor(step * sd); D = P.shape[1]; fv = torch.vmap(f); E = torch.eye(D, dtype=torch.float64); out = []
        for c in range(0, len(P), chunk):
            Pt = torch.as_tensor(P[c:c + chunk], dtype=torch.float64); f0 = fv(Pt); H = torch.zeros(len(Pt), D, D, dtype=torch.float64)
            with torch.no_grad():
                for i in range(D):
                    H[:, i, i] = (fv(Pt + h[i] * E[i]) - 2 * f0 + fv(Pt - h[i] * E[i])) / h[i] ** 2
                    for j in range(i + 1, D):
                        a = fv(Pt + h[i] * E[i] + h[j] * E[j]); b = fv(Pt + h[i] * E[i] - h[j] * E[j]); c_ = fv(Pt - h[i] * E[i] + h[j] * E[j]); d_ = fv(Pt - h[i] * E[i] - h[j] * E[j])
                        H[:, i, j] = H[:, j, i] = (a - b - c_ + d_) / (4 * h[i] * h[j])
            out.append(H.numpy())
        return np.concatenate(out)

    @staticmethod
    def hess_gfd(f, P, step, chunk=2048):
        sd = P.std(0); h = torch.as_tensor(step * sd); D = P.shape[1]; gv = torch.vmap(torch.func.grad(f)); E = torch.eye(D, dtype=torch.float64); out = []
        for c in range(0, len(P), chunk):
            Pt = torch.as_tensor(P[c:c + chunk], dtype=torch.float64); H = torch.zeros(len(Pt), D, D, dtype=torch.float64)
            for i in range(D): H[:, i, :] = (gv(Pt + h[i] * E[i]) - gv(Pt - h[i] * E[i])) / (2 * h[i])     # row i = d/dw_i of the gradient
            out.append(((H + H.transpose(1, 2)) / 2).detach().numpy())
        return np.concatenate(out)

    def hessians(self, source, channel, method, pts):
        lat = pts["T"] if source == "oracle" else pts["Z"]; xs = (pts["x"] - self.src.x_mean) / self.src.x_std; P = np.column_stack((lat, xs)); f = self.fn(source, channel)
        if method == "AD": H = self.hess_ad(f, P)
        elif method.startswith("FD"): H = self.hess_fd(f, P, float(method[2:]))
        elif method.startswith("GFD"): H = self.hess_gfd(f, P, float(method[3:]))
        else: raise ValueError(method)
        return H, P

    # ---------------- second-moment tensors with column-specific interior masks ----------------
    def tensors(self, H, x_raw):
        k, p = self.k, self.p; m = col_mask(self.model, x_raw); H = H.copy(); nonfinite = ~np.isfinite(H)
        for j in range(p):                                   # entries involving a boundary X coordinate are undefined/non-integrable there: removed (and counted), never zeroed silently
            bad = ~m[:, j]; H[bad, :, k + j] = 0; H[bad, k + j, :] = 0
        if not np.isfinite(H).all(): raise FloatingPointError("non-finite Hessian entries outside the masked X columns")
        Cuu = np.einsum("nab,ncd->abcd", H[:, :k, :k], H[:, :k, :k], optimize=True) / len(H)
        Cb = np.stack([np.einsum("na,nb,n->ab", H[:, :k, k + j], H[:, :k, k + j], m[:, j].astype(float)) / max(m[:, j].sum(), 1) for j in range(p)])
        return Cuu, Cb, dict(coverage=m.mean(0), nonfinite_frac=float(nonfinite.any(axis=(1, 2)).mean()))


def contract(Cuu, Cb, J):
    """A_ij = sum R.. K ; B_ix: exact contraction of the cached second moments (no rotation of squared matrices)."""
    UX = np.einsum("ia,bik,ka->ab", J, Cb, J, optimize=True)
    W = np.einsum("ijkl,ia,ka->ajl", Cuu, J, J, optimize=True); UU = np.einsum("ajl,jb,lb->ab", W, J, J, optimize=True)
    return np.maximum((UU + UU.T) / 2, 0), np.maximum(UX, 0)
