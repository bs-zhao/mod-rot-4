"""S1: a single generate/save/report pipeline for all six model specifications."""
from pathlib import Path
import sys
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from modrot4.cli import parse_args
from modrot4.model_specs import get_spec, CR4_X_POLICY
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
        if spec.model_id == "CR4":
            manifest_path = target.with_name("s1_generate__manifest.json")
            if not manifest_path.is_file():
                raise RuntimeError(
                    "Old CR4 dataset has no sampling manifest. Archive "
                    "outputs/models/generate/CR4 and outputs/models/precheck/CR4 "
                    "before regenerating CR4 with the aligned X sampler.")
            manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("x_sampling_policy") != CR4_X_POLICY:
                raise RuntimeError(
                    "CR4 dataset uses OLD two-stream X generation. "
                    "Archive outputs/models/generate/CR4 and "
                    "outputs/models/precheck/CR4, then rerun Generate for CR4. "
                    "Existing data will not be overwritten.")
        with np.load(target,allow_pickle=False) as saved:
            existing_subjects=int(saved["parameters"].shape[0])
        if existing_subjects!=args.subjects:
            raise RuntimeError(
                f"[GENERATE] model={spec.model_id} status=EXISTS "
                f"existing_subjects={existing_subjects} requested_subjects={args.subjects}. "
                "Archive the old dataset and regenerate; never silently reuse wrong N.")
        print(f"[GENERATE] model={spec.model_id} phase=done status=EXISTS "
              f"subjects={existing_subjects}",flush=True)
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
