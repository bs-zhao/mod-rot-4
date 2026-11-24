"""One NN database -> all first-wave configurations (batches A-C of 04.3) on every saved S9 / S9_2 rotation.
usage: python run_db.py MODEL RUN_DIR [--quick]
configs:  oracle x {Q01..Q05} x {Y-L, Y-P}   and   inference I01(AD) x {Q01..Q05},  I02(FD 0.003/0.01/0.03/0.1) and I03(GFD 0.03) x Q01   (x {Y-L, Y-P})
Writes outputs/metric_lab/06_wave1_screen/<model>/<condition>/<repeat>/{base_tensors.npz, run_manifest.json, s9|s9_2/{metrics.npz, rotation_manifest.csv}}
"""
import os, sys, json, time, csv, argparse
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ.setdefault(k, "2")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from wave1.engine import Eval, contract, CHANNELS
from wave1.library import metrics, metric_names
from wave1.sources import load_rotations, output_base, sha256

QS = ("Q01", "Q02", "Q03", "Q04", "Q05"); INFER_Q01 = ("AD", "FD0.003", "FD0.01", "FD0.03", "FD0.1", "GFD0.03")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("run_dir"); ap.add_argument("--quick", action="store_true"); a = ap.parse_args(); t0 = time.time()
    ev = Eval(a.model, a.run_dir); root = output_base(ev.src); root.mkdir(parents=True, exist_ok=True); names = metric_names(); k, p = ev.k, ev.p
    rots = {}
    for v in ("s9", "s9_2"):
        try: rots[v] = load_rotations(ev.src, v)
        except FileNotFoundError: pass
    cfgs = []                                                     # (Q, channel, method)
    for Q in QS:
        pts = ev.points(Q)
        if pts is None: continue
        for ch in CHANNELS:
            cfgs.append((Q, ch, "oracle"))
            for m in (INFER_Q01 if Q == "Q01" else ("AD",)): cfgs.append((Q, ch, m))
    if a.quick: cfgs = [c for c in cfgs if c[0] == "Q01" and c[2] in ("oracle", "AD")]
    tens = {}; info = {}
    ptsc = {}
    for (Q, ch, m) in cfgs:
        pts = ptsc.setdefault(Q, ev.points(Q)); src_ = "oracle" if m == "oracle" else "nn"
        H, P = ev.hessians(src_, ch, "AD" if m == "oracle" else m, pts); Cuu, Cb, meta = ev.tensors(H, pts["x"])
        key = f"{Q}|{ch}|{m}"; tens[key] = (Cuu, Cb); info[key] = dict(n_points=len(P), coverage=meta["coverage"].tolist(), nonfinite_point_frac=meta["nonfinite_frac"])
        print(f"  [{a.model}] {key}: {len(P)} pts  interior coverage min {meta['coverage'].min():.2f}  [{time.time()-t0:.0f}s]", flush=True)
    np.savez_compressed(root / "base_tensors.npz", **{f"{kk}|Cuu": v[0] for kk, v in tens.items()}, **{f"{kk}|Cb": v[1] for kk, v in tens.items()})
    for var, rl in rots.items():
        out = root / var; out.mkdir(exist_ok=True); arr = {"names": np.array(names)}
        for key, (Cuu, Cb) in tens.items():
            m = key.split("|")[2]; Ms = np.full((len(rl), len(names)), np.nan); As = np.zeros((len(rl), k, k), np.float32); Bs = np.zeros((len(rl), k, p), np.float32)
            Js = [(r.t_std if m == "oracle" else r.z_std)[:, None] * (r.G if m == "oracle" else r.R).T for r in rl]
            AB = [contract(Cuu, Cb, J) for J in Js]; EA = AB[0][0].sum(); EB = AB[0][1].sum()        # energies are rotation invariant
            zero = dict(A=bool(EA <= 1e-9 * (EA + EB)), B=bool(EB <= 1e-9 * (EA + EB)))
            inv_err = max(abs(x[0].sum() - EA) / (EA + 1e-300) for x in AB)
            for i, (A, B) in enumerate(AB):
                d = metrics(A, B, zero); Ms[i] = [d[n] for n in names]; As[i] = A; Bs[i] = B
            arr[f"M|{key}"] = Ms; arr[f"A|{key}"] = As; arr[f"B|{key}"] = Bs; info[key].update(zero_A=zero["A"], zero_B=zero["B"], EA=float(EA), EB=float(EB), EA_invariance_err=float(inv_err))
        np.savez_compressed(out / "metrics.npz", **arr)
        with open(out / "rotation_manifest.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["rotation", "candidate_type", "k", "axes_1based", "angle_deg"])
            for r in rl: w.writerow([r.name, r.candidate_type, len(r.axes), "-".join(str(i + 1) for i in r.axes), r.angle])
        print(f"  [{a.model}] {var}: {len(rl)} rotations x {len(names)} metrics x {len(tens)} configs [{time.time()-t0:.0f}s]", flush=True)
    (root / "run_manifest.json").write_text(json.dumps(dict(model=a.model, run_dir=str(ev.src.run_dir), k=k, p=p, subjects=256, trials=32, configs=info, dataset_sha=sha256(ev.src.dataset_file), checkpoint_sha=sha256(ev.src.checkpoint_file), latent_sha=sha256(ev.src.latent_file), seed=ev.seed,
        notes="Q04 only for fixed-bank models; CR models use the interior-X estimand (column-specific masks, coverage recorded); full-design M is undefined for CR (boundary points) and not computed."), indent=1, default=float))


if __name__ == "__main__":
    main()
