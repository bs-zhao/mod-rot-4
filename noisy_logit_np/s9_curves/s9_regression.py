"""S9 / S9_2-style rotation sweeps for the REGRESSION NPs (noisy logit): metric vs angle, analytic vs NN.

Mirrors the binary-choice S9 / S9_2 construction (nn/src/modrot4_nn/s9_rotation.py, supervised_rotation.py; copied, nothing imported from nn/):
  S9   : R* = orthogonal Procrustes of RAW Z (TRAIN-centred) onto [TRAIN-standardised T, 0];   U = R Z
  S9_2 : same on the globally z-scored Zz = (Z - mean_all)/std_all (all subjects);              U = R Zz
  sweep: for every sorted subset of the first d axes (k = 2..d) and angle 0,15,...,90 deg, G = ordered product of adjacent Givens
         rotations (each by the angle),  R = G R*.   Baseline = the unswept R*.
Energy tensors are computed fresh (same recipe as run_database.py of metric_lab/03): 4096 points = subjects resampled x the model's own X design,
same draws for both sources; analytic latent = true T, NN latent = Z; coordinates [latent raw, X standardised by the NP train scaler];
output = logit (analytic: true logit; NN: regression decoder output in logit units); autograd full input Hessians; C = E[vec(H)vec(H)^T].
Oracle Jacobian = diag(T train std) G^T,  NN Jacobian = diag(z_std) R^T (S9: z_std = 1; S9_2: global std).

usage: python s9_regression.py [MODEL ...]      (default: the 5 NN1 models and the 5 replicate NPs *r2)
Writes results/raw/<MODEL>/regNP/<nn1|r2>/<s9|s9_2>/{rotations.csv,metrics_long.csv}
"""
import os, sys, csv, time, json
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(k, "2")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from itertools import combinations
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import numpy as np, torch
from scipy.stats import special_ortho_group
from common import *
from np_model import RegNP
from modrot4.model_specs import get_spec
import core, metrics152 as MET

RAW = HERE / "results" / "raw"; ANGLES = (0, 15, 30, 45, 60, 75, 90)


def procrustes_R(z, theta, train):
    zc = z[train] - z[train].mean(0); tm, ts = theta[train].mean(0), theta[train].std(0); target = (theta[train] - tm) / ts
    L, d = z.shape[1], theta.shape[1]; pad = np.zeros((len(train), L)); pad[:, :d] = target
    u, _, vt = np.linalg.svd(zc.T @ pad, full_matrices=True); return (u @ vt).T, tm, ts


def givens(n, axes, deg):
    a = np.deg2rad(float(deg)); c, s = np.cos(a), np.sin(a); out = np.eye(n)
    for i, j in zip(axes[:-1], axes[1:]):
        ri, rj = out[i].copy(), out[j].copy(); out[i] = c * ri - s * rj; out[j] = s * ri + c * rj
    return out


def hess(fn, P, chunk=256):
    hv = torch.func.vmap(torch.func.hessian(fn)); out = []
    for s in range(0, len(P), chunk): out.append(hv(torch.as_tensor(P[s:s + chunk], dtype=torch.float64)).detach().numpy())
    return np.concatenate(out)


def moment(H): Hf = H.reshape(len(H), -1); return Hf.T @ Hf / len(H)


def blocks(C, d, p):
    D = d + p; C4 = C.reshape(D, D, D, D); return C4[:d, :d, :d, :d].copy(), np.stack([C4[:d, d + b, :d, d + b] for b in range(p)])


def uxuu(Cuu, Cb, Jz):
    UX = np.einsum("ia,bik,ka->ab", Jz, Cb, Jz); W = np.einsum("ijkl,ia,ka->ajl", Cuu, Jz, Jz); UU = np.einsum("ajl,jb,lb->ab", W, Jz, Jz)
    return np.maximum(UX, 0), np.maximum(UU, 0)


def build_metrics():
    M = MET.build(); comp = M["_comp"]
    comp["A75"] = lambda UX, UU: MET.rowsp(UX, 0.75, 0.1); comp["K0"] = lambda UX, UU: float(np.mean([MET.hoy(UX[:, j]) for j in range(UX.shape[1])]))
    M["CAND.rowsp.75+UXcol.hoy"] = lambda UX, UU, st: sum((comp[k](UX, UU) - st["null"][k][0]) / st["null"][k][1] for k in ("A75", "K0"))
    return M


def run(model, n_pts=4096):
    t0 = time.time(); b = base(model); ck = torch.load(ckpt_path(model), map_location="cpu", weights_only=False); d, p = ck["latent_dim"], ck["x_dim"]
    net = RegNP(p, d); net.load_state_dict(ck["model_state"]); net.double().eval()
    for q in net.parameters(): q.requires_grad_(False)
    ym, ys, xm, xs = ck["y_mean"], ck["y_std"], ck["x_mean"], ck["x_std"]
    D_ = np.load(data_path(model)); T = D_["parameters"].astype(float); Z = np.load(latent_path(model))["z"].astype(float); train = np.asarray(ck["split"]["train"])
    spec = get_spec(b)
    if b == "CR3": spec.bank = np.load(PROJECT / "outputs" / "models" / "generate" / "CR3" / "s0_design" / "CR3_4P_fixed500.npz")["X"]
    bank = getattr(spec, "bank", None); rng = np.random.default_rng(2)
    if bank is not None:
        nd = int(np.ceil(n_pts / len(bank))); sub = np.repeat(rng.choice(len(Z), nd), len(bank))[:n_pts]; x = np.tile(bank, (nd, 1))[:n_pts]
    else:
        sub = rng.choice(len(Z), n_pts); x = np.asarray(spec.sample_inputs(n_pts, rng), float)
    if b.startswith("CR"):
        import types; keep = core.validity(b, types.SimpleNamespace(x=x, latent=T[sub])).all(axis=(1, 2)); sub, x = sub[keep], x[keep]
    xst = (x - xm) / xs; Pz = np.column_stack((Z[sub], xst)); Pt = np.column_stack((T[sub], xst))
    xm_t, xs_t = torch.as_tensor(xm, dtype=torch.float64), torch.as_tensor(xs, dtype=torch.float64)
    f_nn = lambda v: net.decoder(torch.cat((v[d:], v[:d])))[0] * ys + ym
    f_or = lambda v: core.oracle_scalar(b, torch.cat((v[:d], v[d:] * xs_t + xm_t)), "logit")
    C = {"oracle": moment(hess(f_or, Pt)), "nn": moment(hess(f_nn, Pz))}; blk = {s: blocks(C[s], d, p) for s in C}
    Mx = build_metrics(); names = [k for k in Mx if k != "_comp"]; haar = special_ortho_group.rvs(d, size=200, random_state=4)
    tag = "r2" if model.endswith("r2") else "nn1"; print(f"[{model}] d={d} p={p} points {len(x)} (Hessians {time.time()-t0:.0f}s)", flush=True)
    for var in ("s9", "s9_2"):
        if var == "s9": Zv, zmean, zstd = Z, np.zeros(d), np.ones(d)
        else: zmean, zstd = Z.mean(0), Z.std(0); Zv = (Z - zmean) / zstd
        Rs, tm, ts = procrustes_R(Zv, T, train)
        rots = [("baseline_unswept", "baseline_unswept", (), 0, Rs, np.eye(d))]
        for k in range(2, d + 1):
            for axes in combinations(range(d), k):
                for ang in ANGLES: G = givens(d, axes, ang); rots.append(("axes_" + "_".join(f"{a+1:02d}" for a in axes) + f"__deg_{ang:03d}", "sweep", axes, ang, G @ Rs, G))
        rows = []
        for source in ("oracle", "nn"):
            Cuu, Cb = blk[source]; sd = ts if source == "oracle" else zstd; J = lambda Q: sd[:, None] * Q.T
            vals = {c: [] for c in Mx["_comp"]}
            for Q in haar:
                UX, UU = uxuu(Cuu, Cb, J(Q))
                for c, fn in Mx["_comp"].items(): vals[c].append(fn(UX, UU))
            st = dict(null={c: (float(np.median(v)), float(np.std(v)) + 1e-12) for c, v in vals.items()})
            for name, kind, axes, ang, R, G in rots:
                UX, UU = uxuu(Cuu, Cb, J(G if source == "oracle" else R)); rows.append((source, name, [float(Mx[nm](UX, UU, st)) for nm in names]))
        out = RAW / model.replace("r2", "") / "regNP" / tag / var; out.mkdir(parents=True, exist_ok=True)
        with open(out / "rotations.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["rotation", "candidate_type", "k", "axes_1based", "angle_deg"])
            for name, kind, axes, ang, R, G in rots: w.writerow([name, kind, len(axes), "-".join(str(a + 1) for a in axes), ang])
        with open(out / "metrics_long.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["source", "rotation", *names]); [w.writerow([s, r, *v]) for s, r, v in rows]
        np.savez_compressed(out / "R_star.npz", R_star=Rs, t_mean=tm, t_std=ts, z_mean=zmean, z_std=zstd)
        print(f"  {var}: {len(rots)} rotations x {len(names)} metrics x 2 sources [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    for m in (sys.argv[1:] or list(MODELS) + list(MODELS_R2)): run(m)
