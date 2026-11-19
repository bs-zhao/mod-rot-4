"""S1: the same Generate workflow for every choice model."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from modrot4.cli import parse_args
from modrot4.core import generate, save
from modrot4.model_specs import get_spec
from modrot4.pipeline import dataset_path, sha256
from modlearn_rot.utils.io import save_json


def main(args):
    spec = get_spec(args.model)
    target = dataset_path(args.model)
    if target.exists():
        raise FileExistsError(
            f"Dataset already exists: {target}. Archive this model's Generate "
            "and Precheck output folders before regenerating; nothing was overwritten.")
    print(f"[GENERATE] model={spec.model_id} subjects={args.subjects} phase=start",
          flush=True)
    data = generate(spec, args.subjects, args.data_seed, args.design_seed)
    save(data, target)
    save_json({
        "model": spec.model_id,
        "subjects": args.subjects,
        "data_seed": args.data_seed,
        "design_seed": args.design_seed,
        "parameters": list(spec.parameter_names),
        "dataset_fields": list(data.__dataclass_fields__),
        "dataset_sha256": sha256(target),
    }, target.with_name("s1_generate__manifest.json"))
    print(f"[GENERATE] model={spec.model_id} phase=done "
          f"subjects={len(data.parameters)} trials={len(data.choice)} "
          f"T_dim={spec.n_parameters} X_dim={spec.x_dim}", flush=True)
    print(f"[GENERATE] dataset={target}", flush=True)


if __name__ == "__main__":
    main(parse_args("S1: generate T, X, logit, p and choice"))
