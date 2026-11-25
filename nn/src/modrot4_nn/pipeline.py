"""Single run tree and dataset provenance; no model-specific preprocessing."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .training_helpers import training_condition_tag

ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def dataset_path(model):
    return ROOT/"outputs"/"models"/"generate"/model/"s1_generate__dataset.npz"


def precheck_path(model):
    return ROOT/"outputs"/"models"/"precheck"/model/"s2_precheck__report.json"


def _condition_payload(args):
    return dict(
        beta=float(args.beta), lambda_sparse=float(args.lambda_sparse),
        context_min=float(args.context_min), context_max=float(args.context_max),
        latent_dim=int(args.latent_dim), batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay),
        max_epochs=int(args.max_epochs),patience=int(args.patience),
        min_delta=float(args.min_delta),train_seed=int(args.train_seed),
        split_seed=int(args.split_seed))


def condition(args):
    if args.run_id is not None:
        if not args.run_id or not all(c.isalnum() or c in "-_" for c in args.run_id):
            raise ValueError("--run-id must use alphanumeric, - and _ only")
        return args.run_id
    base=training_condition_tag(
        args.beta,args.lambda_sparse,args.context_min,args.context_max,args.latent_dim)
    digest=hashlib.sha256(json.dumps(_condition_payload(args),sort_keys=True).encode()).hexdigest()[:10]
    return f"{base}_{digest}"


def run_dir(args):
    return ROOT/"outputs"/"nn"/args.model/condition(args)/f"repeat{args.repeat:02d}"


def stage_folder(args, stage):
    return run_dir(args)/stage


def train_path(args):
    return stage_folder(args,"s3_train")/"checkpoint.pt"


def latent_path(args):
    return stage_folder(args,"s5_extract_z")/"latent.npz"


def _require_qualified(args):
    path=precheck_path(args.model)
    if not path.exists():
        raise FileNotFoundError(f"Run models/s2_precheck.py first: {path}")
    report=json.loads(path.read_text(encoding="utf-8"))
    if not report.get("qualified") or report.get("dataset_sha256")!=sha256(dataset_path(args.model)):
        raise RuntimeError("S2 precheck did not qualify this exact model dataset")


def _discrete_data(args, generated, *, preprocessing=None):
    """Standardize every X coordinate BEFORE any NN sees it.

    One TRAIN-subject-only scaler is fitted in S3. Its exact statistics are
    stored in the checkpoint and reused in S4/S5. All encoder and decoder
    X tensors are the standardized tensors. Raw X is never passed to the NP.
    """
    from .trainer import DiscreteData
    counts=np.asarray(generated.trial_counts,int)
    n=len(counts); total=int(counts.sum())
    if not np.array_equal(np.asarray(generated.subject_index),
                          np.repeat(np.arange(n),counts)):
        raise ValueError("Input dataset must have grouped subject trials")
    xx=np.asarray(generated.x,np.float32)
    continuous=not hasattr(generated,"choice")          # <MODEL>L: noisy-logit response
    y=(np.asarray(generated.y,np.float32).reshape(-1) if continuous
       else np.asarray(generated.choice,np.int64).reshape(-1))
    if xx.shape[0]!=total or len(y)!=total:
        raise ValueError("X/choice mismatch with subject trial counts")
    maximum=int(counts.max())
    x=torch.zeros((n,maximum,xx.shape[1]),dtype=torch.float32)
    choice=torch.zeros((n,maximum),dtype=torch.float32 if continuous else torch.long)   # response channel
    mask=torch.zeros((n,maximum),dtype=torch.bool)
    off=np.r_[0,counts.cumsum()]
    for i in range(n):
        t=int(counts[i]);left,right=int(off[i]),int(off[i+1])
        x[i,:t]=torch.from_numpy(xx[left:right].copy())
        choice[i,:t]=torch.from_numpy(y[left:right].copy())
        mask[i,:t]=True
    if preprocessing is not None:
        mu=torch.as_tensor(preprocessing["x_mean"],dtype=torch.float32)
        sd=torch.as_tensor(preprocessing["x_std"],dtype=torch.float32)
        if mu.shape!=(xx.shape[1],) or sd.shape!=(xx.shape[1],) or (sd<=0).any():
            raise ValueError("Invalid X normalizer in checkpoint")
        x=(x-mu)/sd
        x[~mask]=0
        if continuous:
            choice=(choice-float(preprocessing["y_mean"]))/float(preprocessing["y_std"])
            choice[~mask]=0
    return DiscreteData(x=x,choice=choice,trial_mask=mask)


def fit_x_scaler(data, train_ids):
    """TRAIN rows only; one μ and σ per raw X coordinate."""
    ids=torch.as_tensor(train_ids,dtype=torch.long)
    x=data.x[ids].numpy()
    mask=data.trial_mask[ids].numpy()
    vals=x[mask]
    mu=vals.mean(axis=0,dtype=np.float64)
    sd=vals.std(axis=0,dtype=np.float64)
    if np.any(sd<=1e-8):
        raise ValueError("Cannot standardize near-constant X dimension")
    return dict(x_mean=mu.tolist(),x_std=sd.tolist(),
                policy="train_only_dimensionwise_zscore_v1")


def fit_y_scaler(data, train_ids):
    """<MODEL>L only: TRAIN-subject mean/SD of the continuous response (data must hold the raw y)."""
    ids=torch.as_tensor(train_ids,dtype=torch.long)
    vals=data.choice[ids].numpy()[data.trial_mask[ids].numpy()]
    mu=float(vals.mean(dtype=np.float64));sd=float(vals.std(dtype=np.float64))
    if sd<=1e-8:
        raise ValueError("Cannot standardize a near-constant response")
    return dict(y_mean=mu,y_std=sd)
