"""Extract the frozen posterior-mean Z of every subject (all observed trials used as context) and score
how well Z recovers the true parameters T (post-hoc only).

usage: python extract_z.py [MODEL ...]
Writes results/<MODEL>_latent.npz and the rotation-test metadata rotation/meta_<MODEL>.npz
(d, p, xs, zstd, Zs, Ts, R = supervised Procrustes rotation = the evaluation ceiling).
"""
import json, sys
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from common import *
from np_model import RegNP

(EXP / "rotation").mkdir(exist_ok=True)


def procrustes(Zs, Ts):
    """Q* minimising ||Zs Q^T - Ts||_F over orthogonal Q (supervised reference; T used for evaluation only)."""
    U, _, Vt = np.linalg.svd(Zs.T @ Ts); return (U @ Vt).T


def matched_corr(U, Ts):
    d = Ts.shape[1]; C = np.abs(np.corrcoef(np.c_[U, Ts].T)[:d, d:]); r, c = linear_sum_assignment(-C)
    m = C[r, c]; mask = np.ones_like(C, bool); mask[r, c] = False
    return float(m.mean()), float(m.min()), float(C[mask].mean()), float(C[mask].max())


def extract(model):
    ck = torch.load(ckpt_path(model), map_location="cpu", weights_only=False); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegNP(ck["x_dim"], ck["latent_dim"]).to(dev); net.load_state_dict(ck["model_state"]); net.eval()
    _D = np.load(data_path(model)); D = {k: _D[k] for k in _D.files}; counts = D["trial_counts"]; S = len(counts); Tmax = int(counts.max()); p = ck["x_dim"]
    X = np.zeros((S, Tmax, p), np.float32); Y = np.zeros((S, Tmax), np.float32); M = np.zeros((S, Tmax), bool); s0 = 0
    for sid, n in enumerate(counts):
        n = int(n); X[sid, :n] = D["x"][s0:s0 + n]; Y[sid, :n] = D["y"][s0:s0 + n]; M[sid, :n] = True; s0 += n
    Xs = torch.as_tensor(((X - ck["x_mean"]) / ck["x_std"]).astype(np.float32)); Ys = torch.as_tensor(((Y - ck["y_mean"]) / ck["y_std"]).astype(np.float32)); Mv = torch.as_tensor(M)
    Xs = Xs.masked_fill(~Mv.unsqueeze(-1), 0); Ys = Ys.masked_fill(~Mv, 0); Z = []
    with torch.no_grad():
        for s in range(0, S, 256):
            mu, _ = net.encode(Xs[s:s + 256].to(dev), Ys[s:s + 256].to(dev), Mv[s:s + 256].to(dev)); Z.append(mu.cpu().numpy())
    Z = np.concatenate(Z).astype(np.float64); T = D["parameters"].astype(np.float64)
    Zs = (Z - Z.mean(0)) / Z.std(0); Ts = (T - T.mean(0)) / T.std(0); Q = procrustes(Zs, Ts)
    ceil = matched_corr(Zs @ Q.T, Ts); raw = matched_corr(Zs, Ts)
    # ridge regression Z -> T on the held-out test subjects (how much parameter information Z carries)
    te = ck["split"]["test"]; tr = ck["split"]["train"]; A = np.c_[Zs[tr], np.ones(len(tr))]; W = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Ts[tr])
    pred = np.c_[Zs[te], np.ones(len(te))] @ W; r2 = 1 - ((pred - Ts[te]) ** 2).sum(0) / ((Ts[te] - Ts[te].mean(0)) ** 2).sum(0)
    np.savez(latent_path(model), z=Z, parameters=T, trial_counts=counts)
    np.savez(EXP / "rotation" / f"meta_{model}.npz", d=Z.shape[1], p=p, xs=np.ones(p), zstd=Z.std(0), Zs=Zs, Ts=Ts, R=Q)
    out = dict(model=model, latent_dim=Z.shape[1], test_R2_per_parameter_Z_to_T=[float(v) for v in r2],
               supervised_rotation_MCC=ceil[0], supervised_min_matched=ceil[1], supervised_leak_mean=ceil[2], supervised_leak_max=ceil[3],
               unrotated_MCC=raw[0], unrotated_min_matched=raw[1], unrotated_leak_mean=raw[2], unrotated_leak_max=raw[3], np_test_R2_vs_y=ck["test"]["r2_vs_y"], np_test_R2_vs_true_logit=ck["test"]["r2_vs_true_logit"],
               noise_ceiling_R2=R2_TARGET)
    (RESULTS / f"{model}_latent_summary.json").write_text(json.dumps(out, indent=2))
    print(f"{model}: NP test R2(y) {out['np_test_R2_vs_y']:.3f} (ceiling {R2_TARGET}), R2(true logit) {out['np_test_R2_vs_true_logit']:.3f} | Z->T test R2 {np.round(r2,3)} | "
          f"supervised-rotation MCC {ceil[0]:.3f} (min {ceil[1]:.3f}) leak {ceil[2]:.3f}/{ceil[3]:.3f} | unrotated MCC {raw[0]:.3f}", flush=True)


if __name__ == "__main__":
    for m in (sys.argv[1:] or MODELS): extract(m)
