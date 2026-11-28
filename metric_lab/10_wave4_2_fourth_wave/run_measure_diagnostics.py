"""W4-02 diagnostics (04_fourth_wave sections 10-11).  Blind parts first, then evaluation-only parts (they open R*; run after the
measure-stage lock, i.e. after run_evaluation.py --stage measures).
usage: python run_measure_diagnostics.py --toy CR4 [--part blind|eval|all]     then  --merge
outputs -> results/support_diagnostics.csv, tail_energy.csv, predictive_by_tail.csv, loss_decomposition.csv, loss_decomposition_rows.csv.gz,
           cr4_geodesics.csv.gz (all toys run; file name kept from the protocol), block_bootstrap.csv
"""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4.common import OUT, RES, PRIMARY
import pandas as pd

DD = OUT / "eval_only" / "measure_diag"
PARTS = dict(blind=("support", "tail_energy", "predictive"), eval=("decomposition", "geodesics", "block_bootstrap"))
FILES = dict(support="support_diagnostics.csv", tail_energy="tail_energy.csv", predictive="predictive_by_tail.csv", decomposition="loss_decomposition.csv",
             decomposition_rows="loss_decomposition_rows.csv.gz", geodesics="cr4_geodesics.csv.gz", block_bootstrap="block_bootstrap.csv")
EVAL_TOYS = ("CR4", "CR4L", "CR3", "CR3L", "CR5", "CR5L", "C1", "C1L")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--toy"); ap.add_argument("--part", default="all"); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
    if a.merge:
        for name, fn in FILES.items():
            fs = sorted(DD.glob(f"*__{name}.csv.gz"))
            if fs:
                pd.concat([pd.read_csv(f) for f in fs]).to_csv(RES / fn, index=False); print(fn, len(fs))
        sys.exit()
    from wave4.eval_only import diagnostics as DG
    t0 = time.time(); DD.mkdir(parents=True, exist_ok=True); t = a.toy.replace("#", "_")
    names = PARTS["blind"] + PARTS["eval"] if a.part == "all" else PARTS[a.part]
    for n in names:
        if n in PARTS["eval"] and a.toy not in EVAL_TOYS and n != "block_bootstrap":
            continue
        f = DD / f"{t}__{n}.csv.gz"
        if f.is_file():
            continue
        if n == "decomposition":
            parts = [DG.decomposition(a.toy, v) for v in ("s9", "s9_2")]
            pd.concat([p[0] for p in parts]).to_csv(f, index=False); pd.concat([p[1] for p in parts]).to_csv(DD / f"{t}__decomposition_rows.csv.gz", index=False)
        elif n in ("geodesics", "block_bootstrap"):
            pd.concat([getattr(DG, n)(a.toy, v) for v in ("s9", "s9_2")]).to_csv(f, index=False)
        else:
            getattr(DG, n)(a.toy).to_csv(f, index=False)
        print(f"  [{a.toy}] {n} [{time.time()-t0:.0f}s]", flush=True)
