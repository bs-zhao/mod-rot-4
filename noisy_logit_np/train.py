"""Train the regression NP (MSE reconstruction + beta*KL) on the noisy-logit data of one model.

usage: python train.py MODEL [--max-epochs 500] [--patience 50] [--device auto]
Settings follow the project's categorical NP: lr 5e-4, batch 128, beta 5e-4, context fraction U[0.25,0.80] per
presentation, 70/15/15 subject split, early stopping on validation loss; cross-entropy is replaced by MSE.
"""
import argparse, copy, csv, json, time
import numpy as np, torch
from common import *
from np_model import RegNP, kl_loss, sample_keep

ap = argparse.ArgumentParser(); ap.add_argument("model", choices=MODELS + MODELS_R2)
ap.add_argument("--max-epochs", type=int, default=500); ap.add_argument("--patience", type=int, default=50)
ap.add_argument("--batch", type=int, default=128); ap.add_argument("--beta", type=float, default=5e-4)
ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seed", type=int, default=31)
ap.add_argument("--device", default="auto"); ap.add_argument("--fresh", action="store_true", help="ignore a saved resume file")
a = ap.parse_args()
dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.set_num_threads(6)
if dev.type == "cuda": torch.cuda.set_per_process_memory_fraction(0.55)   # fail early instead of spilling into system RAM

_D = np.load(data_path(a.model)); D = {k: _D[k] for k in _D.files}      # decompress each array ONCE
counts = D["trial_counts"]; S = len(counts); Tmax = int(counts.max()); p = D["x"].shape[1]
X = np.zeros((S, Tmax, p), np.float32); Y = np.zeros((S, Tmax), np.float32); E = np.zeros((S, Tmax), np.float32); M = np.zeros((S, Tmax), bool)
start = 0
for sid, n in enumerate(counts):
    n = int(n); X[sid, :n] = D["x"][start:start + n]; Y[sid, :n] = D["y"][start:start + n]; E[sid, :n] = D["logit"][start:start + n]; M[sid, :n] = True; start += n
g = torch.Generator().manual_seed(a.seed); perm = torch.randperm(S, generator=g)
ntr, nva = round(0.70 * S), round(0.15 * S); split = {"train": perm[:ntr], "validation": perm[ntr:ntr + nva], "test": perm[ntr + nva:]}
tr = split["train"].numpy(); w = M[tr][..., None]
xm = (X[tr] * w).sum((0, 1)) / w.sum(); xs = np.sqrt((((X[tr] - xm) ** 2) * w).sum((0, 1)) / w.sum()) + 1e-6
ym = float(Y[tr][M[tr]].mean()); ys = float(Y[tr][M[tr]].std()) + 1e-6
Xs = torch.as_tensor(((X - xm) / xs).astype(np.float32), device=dev); Ys = torch.as_tensor(((Y - ym) / ys).astype(np.float32), device=dev); Es = torch.as_tensor(((E - ym) / ys).astype(np.float32), device=dev); Mv = torch.as_tensor(M, device=dev)
Xs = Xs.masked_fill(~Mv.unsqueeze(-1), 0); Ys = Ys.masked_fill(~Mv, 0)
gen = torch.Generator(device=dev).manual_seed(a.seed + 1000)
fixed = {k: sample_keep(Mv[split[k].to(dev)], 0.25, 0.80, torch.Generator(device=dev).manual_seed(a.seed + 10000 + i)) for i, k in enumerate(("validation", "test"))}

model = RegNP(p, LATENT_DIM[base(a.model)]).to(dev); opt = torch.optim.Adam(model.parameters(), lr=a.lr)


def evaluate(name):
    idx = split[name].to(dev); model.eval(); tot = dict(mse=0., kl=0., mse_truth=0.); nobs = 0; nb = 0
    with torch.no_grad():
        for s in range(0, len(idx), 128):
            b = idx[s:s + 128]; keep = fixed[name][s:s + 128]
            pred, mu, lv, _ = model(Xs[b], Ys[b], keep, sample=False); v = Mv[b]
            tot["mse"] += float(((pred - Ys[b]) ** 2)[v].sum()); tot["mse_truth"] += float(((pred - Es[b]) ** 2)[v].sum()); nobs += int(v.sum())
            tot["kl"] += float(kl_loss(mu, lv)) * len(b); nb += len(b)
    var_y = float(Ys[idx][Mv[idx]].var()); var_e = float(Es[idx][Mv[idx]].var())
    mse = tot["mse"] / nobs; kl = tot["kl"] / nb
    return dict(loss=mse + a.beta * kl, mse=mse, kl=kl, r2_vs_y=1 - mse / var_y, r2_vs_true_logit=1 - tot["mse_truth"] / nobs / var_e)


hist = {k: [] for k in ("train_loss", "train_mse", "train_kl", "val_loss", "val_mse", "val_kl", "val_r2_vs_y", "val_r2_vs_true_logit")}
best = (float("inf"), None, -1); bad = 0; t0 = time.time(); tri = split["train"].to(dev)
RES = CKPT / f"{a.model}_resume.pt"; ep0 = 0
if RES.exists() and not a.fresh:                      # resume after a crash
    r = torch.load(RES, map_location=dev, weights_only=False); model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"])
    hist = r["hist"]; best = (r["best_val"], r["best_state"], r["best_ep"]); bad = r["bad"]; ep0 = r["ep"]; print(f"[{a.model}] RESUMED from epoch {ep0}", flush=True)
for ep in range(ep0, a.max_epochs):
    te0=time.time(); model.train(); order = tri[torch.randperm(len(tri), generator=torch.Generator().manual_seed(a.seed + ep)).to(dev)]; sm = sk = 0.; nb = 0
    for s in range(0, len(order), a.batch):
        b = order[s:s + a.batch]; v = Mv[b]; keep = sample_keep(v, 0.25, 0.80, gen)
        pred, mu, lv, _ = model(Xs[b], Ys[b], keep, sample=True)
        mse = ((pred - Ys[b]) ** 2)[v].mean(); kl = kl_loss(mu, lv); loss = mse + a.beta * kl
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sm += float(mse); sk += float(kl); nb += 1
    torch.cuda.synchronize(); te1=time.time(); va = evaluate("validation"); torch.cuda.synchronize(); te2=time.time()
    for k, val in (("train_loss", sm / nb + a.beta * sk / nb), ("train_mse", sm / nb), ("train_kl", sk / nb), ("val_loss", va["loss"]), ("val_mse", va["mse"]), ("val_kl", va["kl"]), ("val_r2_vs_y", va["r2_vs_y"]), ("val_r2_vs_true_logit", va["r2_vs_true_logit"])): hist[k].append(val)
    if va["loss"] < best[0]: best = (va["loss"], copy.deepcopy(model.state_dict()), ep); bad = 0
    else:
        bad += 1
        if bad >= a.patience: break
    if ep % 10 == 0 or ep < 3: print(f"[{a.model}] epoch {ep+1} train_mse {sm/nb:.4f} val_mse {va['mse']:.4f} val_R2(y) {va['r2_vs_y']:.4f} (ceiling 0.80) val_R2(true logit) {va['r2_vs_true_logit']:.4f} kl {va['kl']:.2f} [{time.time()-t0:.0f}s; train {te1-te0:.1f}s eval {te2-te1:.1f}s]", flush=True)
    if ep % 5 == 4: torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), hist=hist, best_val=best[0], best_state=best[1], best_ep=best[2], bad=bad, ep=ep + 1), RES)
model.load_state_dict(best[1]); val = evaluate("validation"); test = evaluate("test")
torch.save(dict(model_state=model.state_dict(), x_dim=p, latent_dim=LATENT_DIM[base(a.model)], x_mean=xm, x_std=xs, y_mean=ym, y_std=ys, history=hist,
                best_epoch=best[2], validation=val, test=test, split={k: v.numpy() for k, v in split.items()}, args=vars(a)), ckpt_path(a.model))
with open(RESULTS / f"{a.model}_history.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["epoch", *hist]); [wr.writerow([i + 1, *(hist[k][i] for k in hist)]) for i in range(len(hist["val_loss"]))]
(RESULTS / f"{a.model}_train_summary.json").write_text(json.dumps(dict(best_epoch=best[2] + 1, epochs_run=len(hist["val_loss"]), validation=val, test=test, minutes=(time.time() - t0) / 60), indent=2))
print(f"[{a.model}] DONE best epoch {best[2]+1}, epochs {len(hist['val_loss'])}, TEST R2(y) {test['r2_vs_y']:.4f} (perfect-predictor ceiling 0.80), TEST R2(true logit) {test['r2_vs_true_logit']:.4f}, {(time.time()-t0)/60:.1f} min", flush=True)
RES.unlink(missing_ok=True)
