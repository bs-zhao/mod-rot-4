"""W4-05 CONF-A blind runs and evaluation (04_fourth_wave section 19.1).  Called only through run_confirmation.py after the W4 lock verifies.

payloads()  ORCHESTRATOR: for every registered run whose S3 / S5 / S8 / S8_2 outputs exist, write (a) a truth-free blind payload (Lab 08
            BlindPayload layout: Z, X, Y, subject bookkeeping, splits, saved X / Y scalers, checkpoint path, bank / domain metadata) and
            (b) a separate eval file (T, TEST ids, S8 / S8_2 TRAIN-fit supervised references).  Toy id of a run: <toy>@D<data>T<train>.
blind(id)   the LOCKED rule on the payload only (no eval file is opened): Q05 C, integration seeds 31/73/101 x search seeds 31/73/101, both
            variants, + the declared single-deployment ranking on the independent Q05 D (seed 211) cache, + fixed controls (W3-Q01 lock =
            RW-GL Q01 S9 / s9_2, E02 Q01), same seeds.
evaluate()  hash EVERY CONF-A blind job file into results/confirmation_locks.csv, THEN open T / R*: recovery of every run / cell, Stage A,
            single-deployment readout, coverage with the exact denominator (attempted runs, failures and missing runs kept).
"""
import json
from dataclasses import replace
import numpy as np, pandas as pd
from .common import OUT, RES, CAND, E02, prim_mask, sha256, toy_entry, leak_threshold, family_group
from . import search as SE

CONF = OUT / "confirmation"
PAY = CONF / "payload"
EV = CONF / "eval_only"


def runs():
    return json.loads((CONF / "confirmation_manifest.json").read_text())["runs"]


def tid(r):
    return f"{r['toy']}@D{r['data_seed']}T{r['train_seed']}"


def run_dir_of(r):
    f = CONF / "runs" / f"{r['run_id']}.json"
    return None if not f.is_file() else json.loads(f.read_text())["run_dir"]


def payloads():
    import torch
    from wave2.blind_inputs import _detect_bank
    from wave2.common import toy_registry
    for r in runs():
        rd = run_dir_of(r); out = PAY / f"{tid(r)}.npz"
        if rd is None or out.is_file():
            continue
        from pathlib import Path
        rd = Path(rd); ds = OUT / r["root"] / "outputs" / "models" / "generate" / r["model"] / "s1_generate__dataset.npz"
        ck = rd / "s3_train" / "checkpoint.pt"; lt = rd / "s5_extract_z" / "latent.npz"
        with np.load(ds) as a:
            T = np.asarray(a["parameters"], float); x = np.asarray(a["x"], float); sid = np.asarray(a["subject_index"], int); counts = np.asarray(a["trial_counts"], int)
            y = np.asarray(a["y"] if "y" in a.files else a["choice"], float)
        with np.load(lt) as a:
            z = np.asarray(a["z"], float); splits = {k: np.asarray(a[k + "_ids"], int) for k in ("train", "validation", "test")}
            assert str(a["dataset_sha256"]) == sha256(ds) and str(a["checkpoint_sha256"]) == sha256(ck)
        st = torch.load(ck, map_location="cpu", weights_only=True)["extra_state"]["input_standardization"]
        starts = np.r_[0, np.cumsum(counts[:-1])]; bank = _detect_bank(x, counts); dom = toy_registry()["x_domain"][r["family"]]
        meta = dict(toy=tid(r), cls=r["cls"], checkpoint_file=str(ck), y_mean=float(st.get("y_mean", 0.0)), y_std=float(st.get("y_std", 1.0)),
                    x_domain=None if dom == "unbounded" else list(dom), hashes=dict(dataset=sha256(ds), checkpoint=sha256(ck), latent=sha256(lt)), run_id=r["run_id"])
        arr = dict(z=z, x=x, y=y, subject_index=sid, counts=counts, starts=starts, split_train=splits["train"], split_validation=splits["validation"], split_test=splits["test"],
                   x_mean=np.asarray(st["x_mean"], float), x_std=np.asarray(st["x_std"], float), meta=np.array(json.dumps(meta)))
        if bank is not None:
            arr["fixed_bank"] = bank
        PAY.mkdir(parents=True, exist_ok=True); np.savez(out, **arr)
        ev = dict(T=T, test=splits["test"], meta=np.array(json.dumps(dict(toy=tid(r), family=r["family"], variants=["s9", "s9_2"]))))
        for v, stage in (("s9", "s8_supervised_rotation"), ("s9_2", "s8_2_supervised_rotation")):
            with np.load(rd / stage / "rotation.npz") as f:
                zc = f["z_train_mean"]; zm = f["global_z_mean"] if v == "s9_2" else np.zeros(z.shape[1]); zs = f["global_z_std"] if v == "s9_2" else np.ones(z.shape[1])
                ev.update({f"{v}|R_star": f["R_star"], f"{v}|z_center": zc, f"{v}|t_mean": f["t_train_mean"], f"{v}|t_std": f["t_train_std"], f"{v}|z_mean": zm, f"{v}|z_std": zs})
        EV.mkdir(parents=True, exist_ok=True); np.savez(EV / f"{tid(r)}.npz", **ev); print("payload", tid(r), flush=True)


def load_conf_payload(t):
    from wave2.blind_inputs import BlindPayload
    with np.load(PAY / f"{t}.npz", allow_pickle=False) as a:
        m = json.loads(str(a["meta"]))
        return BlindPayload(m["toy"], m["cls"], a["z"], a["x"], a["y"], a["subject_index"], a["counts"], a["starts"], {k: a[f"split_{k}"] for k in ("train", "validation", "test")},
                            a["x_mean"], a["x_std"], float(m["y_mean"]), float(m["y_std"]), m["checkpoint_file"], a["fixed_bank"] if "fixed_bank" in a.files else None,
                            m["x_domain"], m["hashes"])


class ConfEval:
    def __init__(self, t):
        f = np.load(EV / f"{t}.npz", allow_pickle=False); self.toy = t; self.meta = json.loads(str(f["meta"])); self.T = f["T"]; self.test = f["test"]
        self.refs = {v: {k: f[f"{v}|{k}"] for k in ("R_star", "z_center", "t_mean", "t_std", "z_mean", "z_std")} for v in self.meta["variants"]}; self.family = self.meta["family"]


def lock_rule():
    return json.loads((RES / "algorithm_lock_W4.json").read_text())


def blind(t):
    L = lock_rule(); pl = load_conf_payload(t); D = L["deployment"]
    jobs = [("CONF-A", CAND, "Q05", "C", tuple(L["integration"]["iseeds"]), ("s9", "s9_2")),
            ("CONF-A-D", CAND, "Q05", "D", (D["d_seed"],), ("s9", "s9_2")),                        # search on the independent D cache (reported)
            ("CONF-A-CTRL", CAND, "Q01", "C", tuple(L["integration"]["iseeds"]), ("s9", "s9_2")),
            ("CONF-A-CTRL", E02, "Q01", "C", tuple(L["integration"]["iseeds"]), ("s9", "s9_2"))]
    for batch, cid, M, fid, iseeds, variants in jobs:
        SE.run_job(batch, t, cid, M, fid, iseeds, variants, pl=pl)
    deploy(t, pl)


def deploy(t, pl):
    """declared single-deployment rule (blind): the 9 selected Q05 C rotations of a variant ranked on the independent Q05 D cache"""
    L = lock_rule(); D = L["deployment"]; mask = prim_mask(pl); out = SE.BLIND / "CONF-A" / t.replace("#", "_") / "deployment.json"
    if out.is_file():
        return
    z = np.load(SE.job_path("CONF-A", t, CAND, "Q05")); diag = json.loads(str(z["diag"])); res = {}
    for v in ("s9", "s9_2"):
        LD = SE.objective(pl, CAND, "Q05", "D", D["d_seed"], mask, v)[0]; keys = sorted(k for k in diag if diag[k]["variant"] == v)
        sc = {k: float(LD(z[f"{k}|R"])) for k in keys}; best = min(keys, key=lambda k: (sc[k], k))
        fixed = f"{v}|Q05@C|{mask}|i{D['fixed_pair'][0]}|s{D['fixed_pair'][1]}|S0"
        res[v] = dict(D_ranked=best, D_scores=sc, fixed_pair=fixed)
    out.write_text(json.dumps(res, indent=1))


def stage_a(pl, ed):
    tr, te = np.sort(pl.splits["train"]), np.sort(ed.test); X = np.column_stack([np.ones(len(pl.z)), pl.z]); r2 = []
    for i in range(ed.T.shape[1]):
        b = np.linalg.lstsq(X[tr], ed.T[tr, i], rcond=None)[0]; e = ed.T[te, i] - X[te] @ b; r2.append(1 - e.var() / ed.T[te, i].var())
    return np.array(r2)


def evaluate():
    from wave2.evaluate_recovery import matched
    from wave2.blind_inputs import to_v
    from wave2.optimizers import rel_dist
    files = [f for f in sorted((SE.BLIND).glob("CONF-A*/*/*.npz")) if ".tmp" not in f.name] + sorted(SE.BLIND.glob("CONF-A/*/deployment.json"))
    pd.DataFrame([dict(file=str(f.relative_to(OUT)), sha256=sha256(f), bytes=f.stat().st_size) for f in files]).to_csv(RES / "confirmation_locks.csv", index=False)
    print("locked", len(files), "confirmation blind files (truth opened only now)", flush=True)
    rows, runs_rows = [], []
    for r in runs():
        t = tid(r); base = dict(run_id=r["run_id"], toy_id=t, toy=r["toy"], cls=r["cls"], family=r["family"], data_seed=r["data_seed"], train_seed=r["train_seed"])
        if not (PAY / f"{t}.npz").is_file() or not SE.job_path("CONF-A", t, CAND, "Q05").is_file():
            runs_rows.append(dict(base, status="missing (not trained / not searched)")); continue
        pl = load_conf_payload(t); ed = ConfEval(t); thr = leak_threshold(r["toy"]); sa = stage_a(pl, ed)
        sup = {}
        for v in ed.refs:
            V = to_v(pl, v, pl.z[ed.test]); sup[v] = matched(ed.T[ed.test], (V - ed.refs[v]["z_center"]) @ ed.refs[v]["R_star"].T)
        dep = json.loads((SE.BLIND / "CONF-A" / t.replace("#", "_") / "deployment.json").read_text())
        for f in (SE.job_path("CONF-A", t, CAND, "Q05"), SE.job_path("CONF-A-D", t, CAND, "Q05"), SE.job_path("CONF-A-CTRL", t, CAND, "Q01"), SE.job_path("CONF-A-CTRL", t, E02, "Q01")):
            if not f.is_file():
                continue
            z = np.load(f); diag = json.loads(str(z["diag"]))
            for key, d in diag.items():
                v = d["variant"]; R = z[f"{key}|R"]; V = to_v(pl, v, pl.z[ed.test]); m = matched(ed.T[ed.test], V @ R.T)
                rows.append(dict(base, batch=d["batch"], cid=d["cid"], key=key, variant=v, Q=d["Q"], iseed=d["iseed"], sseed=d["sseed"], L_hat=d["f"], MCC=m["MCC"], MCC_min=m["min"],
                                 leak_max=m["leak_max"], leak_mean=m["leak_mean"], supervised_MCC=sup[v]["MCC"], supervised_leak_max=sup[v]["leak_max"], MCC_gap=sup[v]["MCC"] - m["MCC"],
                                 dist_to_Rstar=rel_dist(R, ed.refs[v]["R_star"]), pass_lab05=bool(m["MCC"] >= sup[v]["MCC"] - .05 and m["leak_max"] < thr),
                                 pass_gap02=bool(m["MCC"] >= sup[v]["MCC"] - .02 and m["leak_max"] < thr), abs_good=bool(m["MCC"] >= .9 and m["leak_max"] < thr),
                                 deploy_D=bool(d["batch"] == "CONF-A" and key == dep[v]["D_ranked"]),
                                 deploy_fixed=bool(d["batch"] == "CONF-A" and key == dep[v]["fixed_pair"])))
        runs_rows.append(dict(base, status="evaluated", stageA_R2_mean=float(sa.mean()), stageA_R2_min=float(sa.min()), sup_MCC_s9=sup["s9"]["MCC"], sup_leak_s9=sup["s9"]["leak_max"],
                              sup_MCC_s9_2=sup["s9_2"]["MCC"]))
        print("  evaluated", t, flush=True)
    pd.DataFrame(rows).to_csv(RES / "confirmation_recovery.csv", index=False); pd.DataFrame(runs_rows).to_csv(RES / "confirmation_runs.csv", index=False)
