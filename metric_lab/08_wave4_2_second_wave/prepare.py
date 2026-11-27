"""Orchestrator step (runs before any blind job): blind payload (T dropped) and the separate evaluation-only file of one toy.
usage: python prepare.py --toy C1"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave2.blind_inputs import extract_payload
from wave2.evaluate_recovery import extract_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--toy", required=True); a = ap.parse_args()
    print(extract_payload(a.toy)); print(extract_eval(a.toy))
