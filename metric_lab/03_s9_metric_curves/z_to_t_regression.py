"""Can T be regressed linearly from Z?  For every NP used in the S9/S9_2 analyses (11 binary-choice NPs, 10 regression NPs).
OLS (with intercept) fitted on the TRAIN subjects, R^2 / Pearson r on the untouched TEST subjects, per true parameter; plus a degree-2 ridge
(all squares and cross products) as a nonlinear reference, and the multivariate share of T variance explained.   T is used only for this diagnostic."""
import sys, glob, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "noisy_logit_np"))
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import PolynomialFeatures
from s9curves.sources import load_sources, get_spec


def fit_eval(Z, T, tr, te):
    mz, sz = Z[tr].mean(0), Z[tr].std(0); Zs = (Z - mz) / sz; mt, st = T[tr].mean(0), T[tr].std(0); Ts = (T - mt) / st
    A = np.c_[Zs[tr], np.ones(len(tr))]; W = np.linalg.lstsq(A, Ts[tr], rcond=None)[0]; P = np.c_[Zs[te], np.ones(len(te))] @ W
    r2 = 1 - ((P - Ts[te]) ** 2).sum(0) / ((Ts[te] - Ts[te].mean(0)) ** 2).sum(0); r = np.array([np.corrcoef(P[:, j], Ts[te][:, j])[0, 1] for j in range(T.shape[1])])
    pf = PolynomialFeatures(2, include_bias=False); Q = pf.fit_transform(Zs); qm, qs = Q[tr].mean(0), Q[tr].std(0) + 1e-12; Q = (Q - qm) / qs
    B = Q[tr]; Wq = np.linalg.solve(B.T @ B + 1.0 * len(tr) * 1e-4 * np.eye(B.shape[1]), B.T @ (Ts[tr] - Ts[tr].mean(0))); Pq = Q[te] @ Wq + Ts[tr].mean(0)
    r2q = 1 - ((Pq - Ts[te]) ** 2).sum(0) / ((Ts[te] - Ts[te].mean(0)) ** 2).sum(0)
    return r2, r, r2q


rows = []
for run in sorted(glob.glob(str(ROOT / "outputs/nn/*/*/repeat*"))):
    run = Path(run); m = run.parents[1].name
    if not (run / "s9_rotation_sweep").is_dir(): continue
    src = load_sources(m, run); names = list(get_spec(m).parameter_names); cond = run.parents[0].name.replace("_lambda0_ctx0p25to0p8", "")
    r2, r, r2q = fit_eval(src.z, src.parameters, src.splits["train"], src.splits["test"])
    for j, n in enumerate(names): rows.append(dict(kind="binary-choice NP", model=m, np=f"{cond[:16]}|{run.name}", param=n, d_latent=src.z.shape[1], r2_linear=r2[j], r_linear=r[j], r2_quadratic=r2q[j]))
for mo in ("C1", "C2", "C3", "CR3", "CR4"):
    for sfx in ("", "r2"):
        k = mo + sfx; ck = torch.load(ROOT / "noisy_logit_np" / "checkpoints" / f"{k}_np_mse.pt", map_location="cpu", weights_only=False)
        Z = np.load(ROOT / "noisy_logit_np" / "results" / f"{k}_latent.npz")["z"].astype(float); T = np.load(ROOT / "noisy_logit_np" / "data" / f"{k}_noisy_logit.npz")["parameters"].astype(float)
        names = list(get_spec(mo).parameter_names); r2, r, r2q = fit_eval(Z, T, np.asarray(ck["split"]["train"]), np.asarray(ck["split"]["test"]))
        for j, n in enumerate(names): rows.append(dict(kind="regression NP", model=mo, np="nn1" if not sfx else "r2", param=n, d_latent=Z.shape[1], r2_linear=r2[j], r_linear=r[j], r2_quadratic=r2q[j]))
D = pd.DataFrame(rows); D.to_csv(HERE / "results" / "z_to_t_regression.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
S = D.groupby(["kind", "model", "np"]).agg(r2_lin_mean=("r2_linear", "mean"), r2_lin_min=("r2_linear", "min"), r2_quad_mean=("r2_quadratic", "mean"), r2_quad_min=("r2_quadratic", "min")).round(3)
S.to_csv(HERE / "results" / "z_to_t_regression_summary.csv"); print(S.to_string()); print(D[D.r2_linear < 0.9].round(3).to_string())
