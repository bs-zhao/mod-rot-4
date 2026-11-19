"""S5: posterior-mean Z using every observed trial per subject."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

import numpy as np
import torch
from modrot4_nn.cli import parse_args
from modrot4_nn.pipeline import dataset_path,train_path,latent_path,sha256,_discrete_data
from modrot4_nn.dataset import load_choice_dataset
from modrot4_nn.net.discrete import DiscreteNeuralProcess,DiscreteNPConfig
from modrot4_nn.training_helpers import load_checkpoint


def extract(args):
    ckpt=train_path(args)
    if not ckpt.exists():
        raise FileNotFoundError(f"Train NP first: {ckpt}")
    state=load_checkpoint(ckpt)
    source=dataset_path(args.model)
    extra=state["extra_state"]
    if sha256(source)!=extra["dataset_sha256"]:
        raise ValueError("Checkpoint trained on another dataset")
    target=latent_path(args)
    if target.exists():
        raise FileExistsError(f"Z already exists; do not overwrite: {target}")
    generated=load_choice_dataset(source)
    data=_discrete_data(args,generated,
                        preprocessing=extra["input_standardization"])
    model=DiscreteNeuralProcess(DiscreteNPConfig(**state["model_config"]))
    model.load_state_dict(state["model_state_dict"])
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available()
                        else "cpu" if args.device=="auto" else args.device)
    model=model.to(device).eval()
    enc_mean=state["preprocessing"]["encoder_mean"].to(device).reshape(-1)
    enc_std=state["preprocessing"]["encoder_std"].to(device).reshape(-1)
    z=np.zeros((data.n_subjects,model.config.latent_dim),dtype=np.float32)
    z_logvar=np.zeros_like(z)
    ids={k:np.asarray(v,dtype=np.int64)
         for k,v in state["subject_splits"].items()}
    with torch.no_grad():
        for group in ("train","validation","test"):
            subj=ids[group]
            for start in range(0,len(subj),args.inference_batch):
                batch=subj[start:start+args.inference_batch]
                x=data.x[batch].to(device)
                y=data.choice[batch].float().to(device).unsqueeze(-1)
                mask=data.trial_mask[batch].to(device)
                obs=(torch.cat([x,y],dim=-1)-enc_mean)/enc_std
                mu,lv=model.encode_context(obs[...,:data.x_dim],
                                            obs[...,data.x_dim:],mask)
                z[batch]=mu.cpu().numpy()
                z_logvar[batch]=lv.cpu().numpy()
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,z=z,z_logvar=z_logvar,
                        train_ids=ids["train"],validation_ids=ids["validation"],
                        test_ids=ids["test"],
                        checkpoint_sha256=np.array(sha256(ckpt)),
                        dataset_sha256=np.array(sha256(source)),
                        context_mode=np.array("all_observed_trials"),
                        context_trials_per_subject=np.asarray(generated.trial_counts))
    print(f"[NN Z] saved {target} shape={z.shape}",flush=True)


if __name__=="__main__":
    extract(parse_args("S5: Extract frozen posterior-mean Z"))
