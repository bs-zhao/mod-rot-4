"""W1-02 entry point: the measure / integration batches B-E of run_baseline.py (same code, same no-overwrite contract).
usage: python run_measures.py --toys all --batch B|C|D|E [--resume]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline import main

if __name__ == "__main__":
    if "--batch" in sys.argv and sys.argv[sys.argv.index("--batch") + 1] == "A":
        sys.exit("batch A is the W1-01 baseline: use run_baseline.py")
    main()
