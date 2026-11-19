"""S2: import T/X/choice and apply identical diagnostics to every model."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from modrot4.cli import parse_args
from modrot4.core import load
from modrot4.model_specs import get_spec
from modrot4.pipeline import dataset_path, precheck_path
from modrot4.unified_precheck import run_precheck


def main(args):
    model = get_spec(args.model)
    path = dataset_path(args.model)
    if not path.is_file():
        raise FileNotFoundError(f"Run Generate first: {path}")
    print(f"[PRECHECK] model={args.model} phase=start", flush=True)
    report = run_precheck(model, load(path), seed=args.precheck_seed, data_path=path)
    if not report["qualified"]:
        raise RuntimeError(f"Precheck failed: see {precheck_path(args.model)}")
    return report


if __name__ == "__main__":
    main(parse_args("S2: common T/X/choice diagnostics"))
