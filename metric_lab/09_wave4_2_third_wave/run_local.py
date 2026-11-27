"""EVALUATION-ONLY local Gap 1 / Gap 2 sweeps (RW-B for HE-based repair recipes, ST-A for gradient / contrast recipes).
usage: python run_local.py --toy CR3 [--kind he|struct|both]      then  python run_local.py --merge  -> results/local_flags.csv.gz"""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import RES, PRIMARY
import pandas as pd
from wave3.eval_only import local as LO

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--toy"); ap.add_argument("--kind", default="both"); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
    if a.merge:
        fs = sorted((LO.LOCAL / "he").glob("*.csv.gz")) + sorted((LO.LOCAL / "struct").glob("*.csv.gz"))
        df = pd.concat([pd.read_csv(f) for f in fs]); df.to_csv(RES / "local_flags.csv.gz", index=False); print(len(df), "rows"); sys.exit()
    t0 = time.time()
    if a.kind in ("he", "both"):
        print(len(LO.he_local(a.toy)), f"he rows [{time.time()-t0:.0f}s]", flush=True)
    if a.kind in ("struct", "both"):
        print(len(LO.struct_local(a.toy)), f"struct rows [{time.time()-t0:.0f}s]", flush=True)
