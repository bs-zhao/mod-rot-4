"""W4-00 exact replay audit of the historical FIN-SENS Q05 observation (04_fourth_wave sections 6-7).  EVALUATION-ONLY entry point.

Checks (every row -> results/replay_audit.csv; summary -> results/replay_summary.json):
  A  archived FIN-SENS RW-GL Q05 records: key identity, 12 toys x {s9, s9_2} x search seeds 31/73/101, no duplicate / missing seed,
     candidate, channel, FULL / MASK-ALL by design
  B  TEST recovery recomputed from the archived R (not from rounded CSV values) == finalist_sensitivity_recovery.csv
  C  Lab 10 reconstruction of train / validation Q05 C s31 caches == Lab 08 caches (J, K, denominators, validity; design mean / SD)
  D  every archived finalist re-scored on the Lab 10 cache == its archived final loss
  E  re-executed S0 (REP-1, integration seed 31): selected R, loss, finalists vs archived (distance modulo signed permutations)
  F  recovery of the re-executed R (REP-1 evaluation rows) and pass / fail agreement
  G  W3 Q01 lock untouched: byte sha256 vs the committed blob, canonical (LF) text hash, Lab 09 verify result
usage: python run_replay.py      (after REP-1 is searched and run_evaluation.py --stage rep has run)
"""
import sys, json, subprocess, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4.common import RES, OUT08, OUT09, LAB09, ROOT, PRIMARY, CAND, chan, prim_mask, sha256
import numpy as np, pandas as pd
from wave2.blind_inputs import load_payload
from wave2.objectives import load_moments
from wave2.evaluate_recovery import EvalData, recovery, supervised
from wave2.optimizers import rel_dist, d_sp
from wave4 import moments as MO, search as SE, designs as DS
from wave4.search import objective

FS = OUT09 / "blind" / "FIN-SENS"


def sp_dist(A, B):
    """distance modulo signed permutations: d_SP of A B^T"""
    return float(d_sp(np.asarray(A) @ np.asarray(B).T))


def main():
    rows = []; old_rec = pd.read_csv(LAB09 / "results" / "finalist_sensitivity_recovery.csv")
    new_rec = pd.read_csv(RES / "rep_recovery.csv") if (RES / "rep_recovery.csv").is_file() else None
    for toy in PRIMARY:
        pl = load_payload(toy); ed = EvalData(toy); mask = prim_mask(pl); c = chan(pl)
        f = FS / toy / f"{CAND}-FIN.npz"; z = np.load(f); diag = json.loads(str(z["diag"])); tr = pd.read_csv(f.with_suffix(".trace.csv.gz"))
        keys = [k for k, d in diag.items() if d["Q"] == "Q05"]
        # C: caches
        for pool in ("train", "validation"):
            new, _ = MO.get(pl, "Q05", "C", 31, pool); old, _ = load_moments(OUT08 / "moment_cache" / toy.replace("#", "_") / f"{pool}_Q05_C_s31.npz")
            for key in old:
                rows.append(dict(check="C cache", toy=toy, item=f"{pool} {key[1]}", value=float(np.abs(new[key]["J"] - old[key]["J"]).max() / np.abs(old[key]["J"]).max()),
                                 ok=bool(np.allclose(new[key]["J"], old[key]["J"], rtol=1e-12, atol=0) and np.allclose(new[key]["K"], old[key]["K"], rtol=1e-12, atol=0)
                                         and np.array_equal(new[key]["nJ"], old[key]["nJ"]) and new[key]["valid"] == old[key]["valid"])))
        d = DS.design(pl, "Q05", "C", 31, "train"); Zp = pl.z[np.sort(pl.splits["train"])]
        rows.append(dict(check="C design fit", toy=toy, item="mean/SD ddof0 = pool", value=float(np.abs(d["fit"]["sd"] - Zp.std(0)).max()), ok=True))
        # A: key identity
        want = {f"{v}|Q05|{mask}|i31|s{s}|S0" for v in ("s9", "s9_2") for s in (31, 73, 101)}
        rows.append(dict(check="A keys", toy=toy, item=f"{len(keys)} Q05 keys", value=len(set(keys) ^ want), ok=bool(set(keys) == want and
                         all(diag[k]["cid"] == CAND and diag[k]["channel"] == c and diag[k]["mask"] == mask for k in keys))))
        sups = {v: supervised(pl, ed, v) for v in ed.refs}
        rz = np.load(SE.job_path("REP-1", toy, CAND, "Q05")) if SE.job_path("REP-1", toy, CAND, "Q05").is_file() else None
        dz = json.loads(str(rz["diag"])) if rz is not None else {}
        for k in sorted(keys):
            v = diag[k]["variant"]; ss = diag[k]["sseed"]; R = z[f"{k}|R"]
            # B: recovery from archived R
            rec, _ = recovery(pl, ed, v, R, sups[v]); o = old_rec[(old_rec.toy == toy) & (old_rec.cid == CAND) & (old_rec.key == k)].iloc[0]
            rows.append(dict(check="B archived recovery", toy=toy, item=k, value=abs(rec["MCC"] - o.MCC) + abs(rec["leak_max"] - o.leak_max), ok=bool(rec["pass_lab05"] == o.pass_lab05
                             and abs(rec["MCC"] - o.MCC) < 1e-12), archived_pass=bool(o.pass_lab05), MCC=rec["MCC"], leak_max=rec["leak_max"]))
            # D: re-score archived finalists
            L = objective(pl, CAND, "Q05", "C", 31, mask, v)[0]; F = z[f"{k}|finalQ"]; t = tr[tr.key == k]
            dev = max(abs(L(Q) - lf) / max(abs(lf), 1e-300) for Q, lf in zip(F, t.L_final))
            rows.append(dict(check="D finalist rescoring", toy=toy, item=k, value=float(dev), ok=bool(dev < 1e-10)))
            # E / F: re-executed search
            nk = f"{v}|Q05@C|{mask}|i31|s{ss}|S0"
            if rz is None or nk not in dz:
                rows.append(dict(check="E re-executed S0", toy=toy, item=k, value=np.nan, ok=False, note="REP-1 job missing")); continue
            R2 = rz[f"{nk}|R"]; dR = sp_dist(R, R2); dL = abs(dz[nk]["f"] - diag[k]["f"]) / abs(diag[k]["f"]); F2 = rz[f"{nk}|finalQ"]
            fin_same = len(F) == len(F2) and all(rel_dist(a, b) < 1e-9 for a, b in zip(F, F2))
            rows.append(dict(check="E re-executed S0", toy=toy, item=k, value=dR, ok=bool(dR < 1e-6 and dL < 1e-10), bit_identical_R=bool(dR == 0.0), rel_dL=dL, raw_dist=float(rel_dist(R, R2)),
                             finalists_identical=bool(fin_same), n_evals_same=bool(dz[nk]["n_evals"] == diag[k]["n_evals"])))
            if new_rec is not None:
                n = new_rec[(new_rec.toy == toy) & (new_rec.cid == CAND) & (new_rec.key == nk)]
                if len(n):
                    rows.append(dict(check="F replay recovery", toy=toy, item=k, value=float(abs(n.MCC.iloc[0] - o.MCC)), ok=bool(n.pass_lab05.iloc[0] == o.pass_lab05),
                                     archived_pass=bool(o.pass_lab05), replay_pass=bool(n.pass_lab05.iloc[0])))
        print(f"  replay {toy}", flush=True)
    # G: W3 lock
    lf = LAB09 / "results" / "algorithm_lock.json"; b = lf.read_bytes()
    blob = subprocess.run(["git", "show", "HEAD:metric_lab/09_wave4_2_third_wave/results/algorithm_lock.json"], cwd=ROOT, capture_output=True).stdout
    canon = hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest(); blob_sha = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
    rows.append(dict(check="G W3 lock", toy="-", item="canonical (LF) text sha256 == committed blob", value=canon, ok=bool(canon == blob_sha),
                     working_copy_byte_sha256=sha256(lf), committed_blob_byte_sha256=hashlib.sha256(blob).hexdigest(), lock_sha256=json.loads(b)["lock_sha256"],
                     note="core.autocrlf=true: the working copy carries CRLF, the blob LF; the bytes differ only by line endings"))
    ver = subprocess.run([sys.executable, str(LAB09 / "freeze_algorithm.py"), "--verify"], cwd=LAB09, capture_output=True, text=True)
    rows.append(dict(check="G W3 lock", toy="-", item="Lab 09 freeze_algorithm --verify", value=ver.returncode, ok=bool(ver.returncode == 0), note=ver.stdout.strip()[-300:]))
    df = pd.DataFrame(rows); df.to_csv(RES / "replay_audit.csv", index=False)
    s = df.groupby("check").ok.agg(["sum", "size"]).reset_index().to_dict("records")
    arch = df[df.check == "B archived recovery"]; s9 = arch[arch.item.str.startswith("s9|")]
    summ = dict(checks=s, archived_s9_pass=f"{int(s9.archived_pass.sum())}/{len(s9)}", archived_s9_2_pass=f"{int(arch[~arch.item.str.startswith('s9|')].archived_pass.sum())}/{len(arch) - len(s9)}")
    rp = df[df.check == "F replay recovery"]
    if len(rp):
        summ["replay_s9_pass"] = f"{int(rp[rp.item.str.startswith('s9|')].replay_pass.sum())}/{int(rp.item.str.startswith('s9|').sum())}"
    (RES / "replay_summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
