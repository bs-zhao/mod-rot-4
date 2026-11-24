"""Ten inference variants -> D^2 x D^2 second-moment tensor C = E[vec(H) vec(H)^T] of the frozen decoder's Hessian,
D = latent_dim + x_dim, coordinates [z (raw), x (standardised)].  Output in logit units.  No retraining.
usage: python infer.py MODEL [MODEL...]
"""
import sys, itertools, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from common import *
from np_model import RegNP
from modrot4.model_specs import get_spec
MOMS = EXP / "rotation" / "moms"; MOMS.mkdir(exist_ok=True)


def hess_at(fn, P, chunk=512):
    hv = torch.func.vmap(torch.func.hessian(fn)); out = []
    for s in range(0, len(P), chunk): out.append(hv(torch.as_tensor(P[s:s + chunk], dtype=torch.float64)).detach().numpy())
    return np.concatenate(out)


def moment(H):
    Hf = H.reshape(len(H), -1); return Hf.T @ Hf / len(H)


for model in sys.argv[1:]:
    t0 = time.time(); ck = torch.load(ckpt_path(model), map_location="cpu", weights_only=False); L, p = ck["latent_dim"], ck["x_dim"]; D = L + p
    net = RegNP(p, L); net.load_state_dict(ck["model_state"]); net.double().eval()
    for q in net.parameters(): q.requires_grad_(False)
    ym, ys, xm, xs = ck["y_mean"], ck["y_std"], ck["x_mean"], ck["x_std"]
    def f(v):                                            # v = [z, x_std] -> predicted logit
        return net.decoder(torch.cat([v[L:], v[:L]]))[0] * ys + ym
    Z = np.load(latent_path(model))["z"]; spec = get_spec(base(model))
    if base(model) == "CR3": spec.bank = np.load(PROJECT / "outputs" / "models" / "generate" / "CR3" / "s0_design" / "CR3_4P_fixed500.npz")["X"]
    def sample(N, seed, gauss=False):
        rng = np.random.default_rng(seed)
        if spec.bank is not None:
            nd = int(np.ceil(N / len(spec.bank))); lat = (rng.multivariate_normal(Z.mean(0), np.cov(Z, rowvar=False), size=nd) if gauss else Z[rng.choice(len(Z), nd)])
            x = np.tile(spec.bank, (nd, 1))[:N]; lat = np.repeat(lat, len(spec.bank), 0)[:N]
        else:
            lat = rng.multivariate_normal(Z.mean(0), np.cov(Z, rowvar=False), size=N) if gauss else Z[rng.choice(len(Z), N)]
            x = np.asarray(spec.sample_inputs(N, rng), float)
        return np.column_stack((lat, (x - xm) / xs))
    P = sample(4096, 2); sd = P.std(0)
    def save(tag, C, info=""): np.savez(MOMS / f"{tag}_{model}.npz", C=C, info=np.array(info)); print(f"  [{model}] {tag} {info} ({time.time()-t0:.0f}s)", flush=True)
    save("i01_AD", moment(hess_at(f, P)), "autograd, empirical Z")
    save("i03_gaussZ", moment(hess_at(f, sample(4096, 2, gauss=True))), "autograd, Z ~ N(mean, cov)")
    Pt = torch.as_tensor(P, dtype=torch.float64); fn = torch.vmap(f); f0 = fn(Pt); E = torch.eye(D, dtype=torch.float64)
    for tag, s in (("i04_FD005", .05), ("i05_FD02", .2), ("i06_FD05", .5)):
        h = torch.as_tensor(s * sd); Hf = torch.zeros(len(P), D, D, dtype=torch.float64)
        for i in range(D):
            Hf[:, i, i] = (fn(Pt + h[i] * E[i]) - 2 * f0 + fn(Pt - h[i] * E[i])) / h[i] ** 2
            for j in range(i + 1, D):
                v = (fn(Pt + h[i] * E[i] + h[j] * E[j]) - fn(Pt + h[i] * E[i] - h[j] * E[j]) - fn(Pt - h[i] * E[i] + h[j] * E[j]) + fn(Pt - h[i] * E[i] - h[j] * E[j])) / (4 * h[i] * h[j])
                Hf[:, i, j] = v; Hf[:, j, i] = v
        save(tag, moment(Hf.numpy()), f"central differences step {s} sd")
    ptr = sample(20000, 11); m_ = ptr.mean(0); s_ = ptr.std(0)
    with torch.no_grad(): y = fn(torch.as_tensor(ptr, dtype=torch.float64)).numpy()
    for tag, deg in (("i07_poly3", 3), ("i08_poly4", 4)):
        idx = [c for k in range(1, deg + 1) for c in itertools.combinations_with_replacement(range(D), k)]; I = np.full((len(idx) + 1, deg), D, dtype=np.int64)
        for r, c in enumerate(idx, 1): I[r, :len(c)] = c
        It = torch.as_tensor(I); phi = lambda ws: torch.cat([ws, torch.ones_like(ws[..., :1])], -1)[..., It].prod(-1)
        Phi = phi(torch.as_tensor((ptr - m_) / s_)).numpy(); lam = 1e-6 * np.trace(Phi.T @ Phi) / Phi.shape[1]; ntr = int(.8 * len(y))
        th = np.linalg.solve(Phi[:ntr].T @ Phi[:ntr] + lam * np.eye(Phi.shape[1]), Phi[:ntr].T @ y[:ntr]); r2 = 1 - ((Phi[ntr:] @ th - y[ntr:]) ** 2).sum() / ((y[ntr:] - y[ntr:].mean()) ** 2).sum()
        T_ = torch.as_tensor(np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ y)); mt, st = torch.as_tensor(m_), torch.as_tensor(s_)
        save(tag, moment(hess_at(lambda w: (phi((w - mt) / st) * T_).sum(-1), P, 256)), f"poly{deg} {len(idx)+1} features, held-out R2 {r2:.4f}")
    Nc = 3000; Wt = (ptr[:Nc + 1000] - m_) / s_; Cc = Wt[:Nc]; sub = Wt[:600]; ell = float(np.median(np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(-1))))
    K = np.exp(-((Cc[:, None] - Cc[None]) ** 2).sum(-1) / (2 * ell ** 2)); ymn = y[:Nc].mean(); al = np.linalg.solve(K + 1e-4 * np.eye(Nc), y[:Nc] - ymn)
    r2 = 1 - ((np.exp(-((Wt[Nc:, None] - Cc[None]) ** 2).sum(-1) / (2 * ell ** 2)) @ al + ymn - y[Nc:Nc + 1000]) ** 2).sum() / ((y[Nc:Nc + 1000] - y[Nc:Nc + 1000].mean()) ** 2).sum()
    Ct, At, mt, st = torch.as_tensor(Cc), torch.as_tensor(al), torch.as_tensor(m_), torch.as_tensor(s_)
    save("i09_krr", moment(hess_at(lambda w: (torch.exp(-((((w - mt) / st)[None] - Ct) ** 2).sum(-1) / (2 * ell ** 2)) * At).sum() + float(ymn), P, 64)), f"RBF KRR, held-out R2 {r2:.4f}")
    for tag, s in (("i10_smooth01", .1), ("i11_smooth03", .3)):
        rng = np.random.default_rng(5); Hs = 0
        for _ in range(8): Hs = Hs + hess_at(f, P + s * sd * rng.standard_normal(P.shape))
        save(tag, moment(Hs / 8), f"mean Hessian over 8 Gaussian perturbations, sigma {s} sd")
    print(f"## {model} done [{time.time()-t0:.0f}s]", flush=True)
