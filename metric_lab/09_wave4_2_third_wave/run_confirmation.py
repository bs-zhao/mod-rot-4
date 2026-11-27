"""W3-06 CONF-A runner guard.  usage: python run_confirmation.py --batch CONF-A
Refuses to run unless results/algorithm_lock.json verifies against the current code / configs AND the confirmation datasets / NP runs listed
in a confirmation_manifest.json exist.  Wave 3 ships the lock and the protocol only; the fresh-data generation and NP training are a separate
compute task (configs/confirmation_protocol.json)."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import RES, OUT
import freeze_algorithm as FA

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", default="CONF-A"); a = ap.parse_args()
    if not (RES / "algorithm_lock.json").is_file() or not FA.verify():
        raise SystemExit("no verified algorithm lock: run freeze_algorithm.py first; nothing evaluated")
    man = OUT / "confirmation" / "confirmation_manifest.json"
    if not man.is_file():
        raise SystemExit(f"{a.batch}: no confirmation datasets / NP runs ({man} missing) - generate and train them first (configs/confirmation_protocol.json)")
    raise SystemExit("confirmation evaluation is implemented only after the fresh panel exists (blocked)")
