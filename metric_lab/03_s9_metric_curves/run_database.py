"""Metric-vs-angle curves on the SAVED S9 / S9_2 rotations.  ANALYTIC (generating formula) vs NN (frozen decoder).

Only the rotation matrices R are taken from the saved S9/S9_2 sweeps (via sources.load_rotations).  Everything else
is computed fresh here, with the same recipe as the noisy-logit rotation tests:

  * evaluation points: 4096 points = subjects resampled from the empirical subject set (seed 2) x the model's own X
    design (fixed 500-row bank for CR3/CR5, `spec.sample_inputs` otherwise); the SAME subject draws and X rows are
    used for the analytic and the NN Hessians.   Analytic latent = the true T of that subject, NN latent = its S5 Z.
  * coordinates [latent raw, X standardised by the S3 train scaler]; output = logit.
  * base tensor C = E[vec(H) vec(H)^T]  (D^2 x D^2), H = full input Hessian
      analytic: autograd of the exact generating logit        (oracle.py)
      NN      : autograd of the frozen decoder: class1 - class0 logit (choice models); for the <MODEL>L models the single
                continuous output in logit units, out * y_std + y_mean (S3 train y scaler; the Y-C channel)
  * for each saved R:  Jz = diag(sd) Q^T (oracle: sd = T train std, Q = G = R R*^T;  NN: sd = Z std of the variant, Q = R)
        UX[a,b] = E[(d2f/dU_a dX_b)^2],  UU[a,b] = E[(d2f/dU_a dU_b)^2],   then the 152 simplicity metrics (+ candidate).
  * z-fusion metrics use a null of 200 Haar rotations computed from the SAME source's base tensor.

usage: python run_database.py MODEL RUN_DIR [--n 4096]
"""
import os, sys, json, time, argparse
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(k, "2")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from scipy.stats import special_ortho_group
from s9curves.sources import load_sources, load_rotations, output_base, get_spec, sha256
from s9curves.oracle import oracle_scalar, validity
from s9curves import metrics152 as MET
from s9curves.decoder import DiscreteNPConfig, DiscreteNeuralProcess  # noqa
import types


def hess(fn, P, chunk=256):
    hv = torch.func.vmap(torch.func.hessian(fn)); out = []
    for s in range(0, len(P), chunk): out.append(hv(torch.as_tensor(P[s:s + chunk], dtype=torch.float64)).detach().numpy())
    return np.concatenate(out)


def moment(H):
    Hf = H.reshape(len(H), -1); return Hf.T @ Hf / len(H)


def blocks(C, d, p):
    D = d + p; C4 = C.reshape(D, D, D, D)
    return C4[:d, :d, :d, :d].copy(), np.stack([C4[:d, d + b, :d, d + b] for b in range(p)])


def uxuu(Cuu, Cb, Jz):
    UX = np.einsum("ia,bik,ka->ab", Jz, Cb, Jz)
    W = np.einsum("ijkl,ia,ka->ajl", Cuu, Jz, Jz); UU = np.einsum("ajl,jb,lb->ab", W, Jz, Jz)
    return np.maximum(UX, 0), np.maximum(UU, 0)


def build_metrics():
    M = MET.build(); comp = M["_comp"]
    comp["A75"] = lambda UX, UU: MET.rowsp(UX, 0.75, 0.1)                       # pre-registered candidate pieces
    comp["K0"] = lambda UX, UU: float(np.mean([MET.hoy(UX[:, j]) for j in range(UX.shape[1])]))
    M["CAND.rowsp.75+UXcol.hoy"] = lambda UX, UU, st: sum((comp[k](UX, UU) - st["null"][k][0]) / st["null"][k][1] for k in ("A75", "K0"))
    return M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("run_dir"); ap.add_argument("--n", type=int, default=4096)
    a = ap.parse_args(); t0 = time.time()
    src = load_sources(a.model, a.run_dir); d, p = src.latent_dim, src.x_dim; D = d + p
    root = output_base(src); root.mkdir(parents=True, exist_ok=True)
    # ---------- evaluation points (same subject draws / X rows for both sources) ----------
    rng = np.random.default_rng(2); spec = get_spec(a.model); bank = src.fixed_x
    if bank is not None:
        nd = int(np.ceil(a.n / len(bank))); sub = rng.choice(len(src.z), nd)
        x = np.tile(bank, (nd, 1))[:a.n]; sub = np.repeat(sub, len(bank))[:a.n]
    else:
        sub = rng.choice(len(src.z), a.n); x = np.asarray(spec.sample_inputs(a.n, rng), float)
    pts = types.SimpleNamespace(x=x, latent=src.parameters[sub])
    keep = validity(src.base, pts).all(axis=(1, 2)) if src.base.startswith("CR") else np.ones(len(x), bool)   # classical-derivative (interior) points only
    sub, x = sub[keep], x[keep]; xs_ = (x - src.x_mean) / src.x_std
    Pz = np.column_stack((src.z[sub], xs_)); Pt = np.column_stack((src.parameters[sub], xs_))
    print(f"[{a.model}] {src.run_dir.name}: {len(x)}/{a.n} interior points d={d} p={p}", flush=True)
    # ---------- the two functions of [latent, x_std] ----------
    state = torch.load(src.checkpoint_file, map_location="cpu", weights_only=True); cfg = dict(state["model_config"]); cfg["decoder_hidden_dims"] = tuple(cfg["decoder_hidden_dims"])
    net = DiscreteNeuralProcess(DiscreteNPConfig(**cfg)); net.load_state_dict(state["model_state_dict"], strict=True); net.eval().double()
    for q in net.parameters(): q.requires_grad_(False)
    xm_t = torch.as_tensor(src.x_mean, dtype=torch.float64); xsd_t = torch.as_tensor(src.x_std, dtype=torch.float64)

    if cfg["num_choices"] == 1:                                # <MODEL>L: continuous output -> logit units
        ym, ys = float(src.y_mean), float(src.y_std)

        def f_nn(v):
            return net.decoder(torch.cat((v[d:], v[:d])))[0] * ys + ym
    else:
        def f_nn(v):
            out = net.decoder(torch.cat((v[d:], v[:d]))); return out[1] - out[0]

    def f_or(v):
        return oracle_scalar(a.model, torch.cat((v[:d], v[d:] * xsd_t + xm_t)), "logit")
    C = {"oracle": moment(hess(f_or, Pt)), "nn": moment(hess(f_nn, Pz))}
    np.savez_compressed(root / "base_moments.npz", oracle=C["oracle"], nn=C["nn"], n_points=len(x), subjects=sub)
    blk = {s: blocks(C[s], d, p) for s in C}
    Mx = build_metrics(); names = [k for k in Mx if k != "_comp"]
    haar = special_ortho_group.rvs(d, size=200, random_state=4)
    # ---------- per variant ----------
    for variant in ("s9", "s9_2"):
        try: rots = load_rotations(src, variant)
        except FileNotFoundError: print(f"  {variant}: not available"); continue
        rows = []; meta = []
        for source in ("oracle", "nn"):
            Cuu, Cb = blk[source]
            def J(rot, Q): return (rot.t_std if source == "oracle" else rot.z_std)[:, None] * Q.T
            r0 = rots[0]; vals = {c: [] for c in Mx["_comp"]}
            for Q in haar:
                UX, UU = uxuu(Cuu, Cb, J(r0, Q))
                for c, f in Mx["_comp"].items(): vals[c].append(f(UX, UU))
            st = dict(null={c: (float(np.median(v)), float(np.std(v)) + 1e-12) for c, v in vals.items()})
            for rot in rots:
                Q = rot.G if source == "oracle" else rot.R
                UX, UU = uxuu(Cuu, Cb, J(rot, Q))
                rows.append((source, rot.name, [float(Mx[nm](UX, UU, st)) for nm in names]))
        for rot in rots: meta.append((rot.name, rot.candidate_type, len(rot.axes), "-".join(str(i + 1) for i in rot.axes), rot.angle))
        out = root / variant; out.mkdir(exist_ok=True)
        import csv
        with open(out / "rotations.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["rotation", "candidate_type", "k", "axes_1based", "angle_deg"]); w.writerows(meta)
        with open(out / "metrics_long.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["source", "rotation", *names])
            for s, r, v in rows: w.writerow([s, r, *v])
        print(f"  {variant}: {len(rots)} rotations x {len(names)} metrics x 2 sources written [{time.time()-t0:.0f}s]", flush=True)
    (root / "run_info.json").write_text(json.dumps(dict(model=a.model, base_model=src.base, response="noisy_logit" if src.continuous else "choice",
        nn_output="continuous*y_std+y_mean" if src.continuous else "class1-class0 logit", run_dir=str(src.run_dir), n_points=int(len(x)), d=d, p=p,
        dataset_sha=sha256(src.dataset_file), checkpoint_sha=sha256(src.checkpoint_file), latent_sha=sha256(src.latent_file)), indent=2))


if __name__ == "__main__":
    main()
