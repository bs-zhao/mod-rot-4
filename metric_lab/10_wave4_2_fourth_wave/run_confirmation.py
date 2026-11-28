"""W4-05 CONF-A (04_fourth_wave section 19).

usage
  python run_confirmation.py --register                              planned datasets / NP runs -> OUT/confirmation/confirmation_manifest.json (once)
  python run_confirmation.py --generate --toy CR4                     S1 + S2 of the registered data seed (isolated root)
  python run_confirmation.py --train-jobs                             "toy train_seed" lines, longest historical training first
  python run_confirmation.py --train --toy CR4 --train-seed 2026112811   S3 / S5 / S8 / S8_2 (GPU)
  python run_confirmation.py --payload                                truth-free blind payloads + separate eval files of every finished run (REQUIRES the W4 lock)
  python run_confirmation.py --blind --toy CONF-CR4-D2026112801-T2026112811   locked rule + controls (blind)
  python run_confirmation.py --evaluate                               hash every blind output, THEN T / R* -> results/confirmation_*.csv
Generation and training read nothing produced by development and are read by nothing in development.  --payload / --blind / --evaluate
refuse to run unless results/algorithm_lock_W4.json verifies.
"""
import sys, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4.common import OUT, RES, PRIMARY, cfg, toy_entry, git_commit

CONF = OUT / "confirmation"
MAN = CONF / "confirmation_manifest.json"
HIST_EPOCHS = dict(C1=284, C2=486, C3=500, CR3=338, CR4=310, CR5=248, C1L=188, C2L=283, C3L=515, CR3L=458, CR4L=764, CR5L=318)


def register():
    if MAN.is_file():
        print("manifest exists (immutable):", MAN); return json.loads(MAN.read_text())
    P = cfg("confirmation_protocol.json")["CONF_A"]; runs = []
    for ds in P["data_seeds"]:
        for toy in PRIMARY:
            e = toy_entry(toy)
            for ts in P["train_seeds"]:
                runs.append(dict(run_id=f"CONF-{toy}-D{ds}-T{ts}", toy=toy, model=toy, cls=e["class"], family=e["family"], k=e["k"], beta=e["beta"],
                                 data_seed=ds, train_seed=ts, split_seed=31, root=str((CONF / f"D{ds}").relative_to(OUT))))
    m = dict(version="W4-CONF-A-1", registered=time.strftime("%Y-%m-%d %H:%M:%S"), git_commit=git_commit(), protocol=P, runs=runs)
    CONF.mkdir(parents=True, exist_ok=True); MAN.write_text(json.dumps(m, indent=1)); print(len(runs), "runs registered ->", MAN); return m


def manifest():
    return json.loads(MAN.read_text())


def run_of(run_id):
    return next(r for r in manifest()["runs"] if r["run_id"] == run_id)


def generate(toy):
    from wave4 import conf_pipeline as CP
    for ds in sorted({r["data_seed"] for r in manifest()["runs"] if r["toy"] == toy}):
        t0 = time.time(); p = CP.generate(OUT / f"confirmation/D{ds}", toy, ds); print(f"[CONF] {toy} D{ds}: {p} [{time.time()-t0:.0f}s]", flush=True)


def train(toy, ts):
    from wave4 import conf_pipeline as CP
    for r in manifest()["runs"]:
        if r["toy"] == toy and r["train_seed"] == ts:
            t0 = time.time(); rd = CP.train_and_extract(OUT / r["root"], r["model"], r["k"], r["beta"], ts)
            (CONF / "runs").mkdir(exist_ok=True)
            (CONF / "runs" / f"{r['run_id']}.json").write_text(json.dumps(dict(r, run_dir=str(rd), seconds=time.time() - t0), indent=1))
            print(f"[CONF] trained {r['run_id']} -> {rd} [{time.time()-t0:.0f}s]", flush=True)


def require_lock():
    import freeze_algorithm as FA
    if not (RES / "algorithm_lock_W4.json").is_file() or not FA.verify():
        raise SystemExit("no verified W4 algorithm lock: nothing done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--register", action="store_true"); ap.add_argument("--generate", action="store_true"); ap.add_argument("--toy")
    ap.add_argument("--train", action="store_true"); ap.add_argument("--train-seed", type=int); ap.add_argument("--train-jobs", action="store_true")
    ap.add_argument("--payload", action="store_true"); ap.add_argument("--blind", action="store_true"); ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    if a.register:
        register()
    elif a.generate:
        generate(a.toy)
    elif a.train_jobs:
        # GPU-bound queue: training seed 1 for all 12 toys first, shortest historical training first, then training seed 2
        ts = manifest()["protocol"]["train_seeds"]
        rs = sorted(manifest()["runs"], key=lambda r: (ts.index(r["train_seed"]), HIST_EPOCHS[r["toy"]] * (3 if r["toy"].startswith("CR") else 1)))
        for r in rs:
            print(r["toy"], r["train_seed"])
    elif a.train:
        train(a.toy, a.train_seed)
    else:
        require_lock()
        import wave4.confirmation as CF
        if a.payload:
            CF.payloads()
        elif a.blind:
            CF.blind(a.toy)
        elif a.evaluate:
            CF.evaluate()
