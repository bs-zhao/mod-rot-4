"""S1: a single generate/save/report pipeline for all six model specifications."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from modrot4.cli import parse_args
from modrot4.model_specs import get_spec
from modrot4.pipeline import dataset_path
from modlearn_rot.generators.dataset import save_dataset, summarize_dataset
from modlearn_rot.utils.io import save_json


def generate(args):
    spec=get_spec(args.model)
    target=dataset_path(spec.model_id)
    print(f"[GENERATE] model={spec.model_id} subjects={args.subjects} phase=start",flush=True)
    if target.exists():
        if args.overwrite:
            raise RuntimeError("Archive existing data instead of overwriting: "+str(target))
        print(f"[GENERATE] model={spec.model_id} phase=done status=EXISTS",flush=True)
        print(f"[GENERATE] dataset={target}",flush=True)
        return
    data=spec.generate(args)
    save_dataset(data,target)
    save_json(spec.manifest(args,data,target),target.with_name("s1_generate__manifest.json"))
    save_json(dict(model_id=spec.model_id,**summarize_dataset(data)),
              target.with_name("s1_generate__summary.json"))
    print(f"[GENERATE] model={spec.model_id} phase=done status=CREATED "
          f"subjects={len(data.parameters)} trials={len(data.subject_index)} "
          f"parameters={len(spec.parameter_names)} x_dim={spec.x_dim}",flush=True)
    print(f"[GENERATE] dataset={target}",flush=True)


if __name__=="__main__":
    generate(parse_args("S1: unified choice-data generation"))
