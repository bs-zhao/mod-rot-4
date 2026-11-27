"""Blind preparation (truth-free): TRAIN / VALIDATION product-grid gradient caches (W3-03) for one toy.  usage: python prepare.py --toy CR3"""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import SEEDS
from wave3 import structure as ST
from wave2.blind_inputs import load_payload

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--toy", required=True); ap.add_argument("--seeds", default="31"); a = ap.parse_args()
    pl = load_payload(a.toy); t0 = time.time()
    for s in [int(x) for x in a.seeds.split(",")]:
        for pool in ("train", "validation"):
            print(ST.build_grad_cache(pl, pool, s), f"[{time.time()-t0:.0f}s]", flush=True)
