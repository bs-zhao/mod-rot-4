"""S2: one precheck workflow with model-specific functions supplied by ModelSpec."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from modrot4.cli import parse_args
from modrot4.model_specs import get_spec
from modrot4.pipeline import dataset_path,precheck_path
from modrot4.unified_precheck import run_unified_precheck
from modlearn_rot.generators.dataset import load_dataset


def precheck(args):
    spec=get_spec(args.model)
    target=dataset_path(spec.model_id)
    if not target.is_file():
        raise FileNotFoundError(f"Generate first: {target}")
    print(f"[PRECHECK] model={spec.model_id} phase=start",flush=True)
    data=load_dataset(target)
    provenance=spec.validate_design(data)
    report=run_unified_precheck(spec,data,seed=args.precheck_seed,
                                provenance=provenance,data_path=target)
    if not report["qualified"]:
        raise RuntimeError("Precheck failed; inspect "+str(precheck_path(spec.model_id)))
    return report


if __name__=="__main__":
    precheck(parse_args("S2: unified choice-data precheck"))
