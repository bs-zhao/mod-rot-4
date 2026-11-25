"""S3: train the original discrete NP on uniformly standardized X."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

import numpy as np
import torch

from modrot4_nn.cli import parse_args
from modrot4_nn.pipeline import (
    dataset_path,precheck_path,train_path,run_dir,condition,sha256,
    _require_qualified,_discrete_data,fit_x_scaler,fit_y_scaler,_condition_payload)
from modrot4_nn.specs import is_logit_model
from modrot4_nn.net.discrete import DiscreteNPConfig
from modrot4_nn.trainer import DiscreteNPTrainer,ContinuousNPTrainer,DiscreteNPTrainingConfig
from modrot4_nn.training_helpers import (
    split_subject_indices,make_fixed_context_masks,stable_run_seed,
    save_checkpoint,save_history_csv)
from modrot4_nn.files import save_json


def train(args):
    # Imported scientific data come only from the models module's S1 output.
    from modrot4_nn.dataset import load_choice_dataset
    _require_qualified(args)
    ckpt=train_path(args)
    if ckpt.exists():
        raise FileExistsError(
            f"Checkpoint already exists: {ckpt}. Choose another repeat/run configuration.")
    generated=load_choice_dataset(dataset_path(args.model))
    raw=_discrete_data(args,generated)
    splits=split_subject_indices(raw.n_subjects,.70,.15,args.split_seed)
    normalizer=fit_x_scaler(raw,splits["train"])
    continuous=is_logit_model(args.model)
    if continuous:
        normalizer.update(fit_y_scaler(raw,splits["train"]))
    data=_discrete_data(args,generated,preprocessing=normalizer)
    # X is normalized once prior to any neural-network computation.
    # Encoder and decoder receive only this same standardized X.
    model_cfg=DiscreteNPConfig(x_dim=data.x_dim,latent_dim=args.latent_dim,num_choices=1 if continuous else 2)
    cfg=DiscreteNPTrainingConfig(
        beta=args.beta,lambda_sparse=args.lambda_sparse,
        learning_rate=args.learning_rate,weight_decay=args.weight_decay,
        batch_size=args.batch_size,max_epochs=args.max_epochs,patience=args.patience,
        min_delta=args.min_delta,context_min_fraction=args.context_min,
        context_max_fraction=args.context_max,
        seed=stable_run_seed(args.train_seed,args.model,args.beta,args.repeat))
    masks=make_fixed_context_masks(data,splits,args.context_min,args.context_max,args.split_seed)
    trainer=(ContinuousNPTrainer if continuous else DiscreteNPTrainer)(model_cfg,cfg,args.device)

    def progress(epoch,tr,val):
        if continuous:
            if epoch==0 or (epoch+1)%args.log_every==0:
                print(f"[NN TRAIN] {args.model} epoch={epoch+1} "
                      f"trainMSE={tr['reconstruction']:.5f} "
                      f"valMSE={val['reconstruction']:.5f} "
                      f"valR2={val['r2']:.4f} betaKL={args.beta*tr['kl']:.5f}",
                      flush=True)
        elif epoch==0 or (epoch+1)%args.log_every==0:
            print(f"[NN TRAIN] {args.model} epoch={epoch+1} "
                  f"trainCE={tr['reconstruction']:.5f} "
                  f"valCE={val['reconstruction']:.5f} "
                  f"valACC={val['accuracy']:.4f} betaKL={args.beta*tr['kl']:.5f}",
                  flush=True)

    result=trainer.fit(data,subject_splits=splits,
                       fixed_context_masks=masks,epoch_callback=progress)
    extra=dict(model_id=args.model,repeat=args.repeat,
               dataset_sha256=sha256(dataset_path(args.model)),
               precheck_sha256=sha256(precheck_path(args.model)),
               input_standardization=normalizer,condition=condition(args),
               full_training_config=_condition_payload(args))
    save_checkpoint(ckpt,result,model_cfg,cfg,splits,masks,extra_state=extra)
    save_history_csv(result["history"],ckpt.parent/"history.csv")
    folder=run_dir(args)
    save_json(dict(model=args.model,condition=condition(args),repeat=args.repeat,
                   **_condition_payload(args)),folder/"config.json")
    save_json(dict(model=args.model,dataset_sha256=extra["dataset_sha256"],
                   checkpoint_sha256=sha256(ckpt),
                   x_standardization=normalizer,
                   n_subjects=raw.n_subjects),folder/"manifest.json")
    np.savez_compressed(folder/"splits.npz",**{
        k:np.asarray(v,dtype=np.int64) for k,v in splits.items()})
    save_json(dict(best_epoch=int(result["best_epoch"])+1,
                   validation_metrics=result["validation_metrics"],
                   test_metrics=result["test_metrics"],
                   training_config=cfg.to_dict(),
                   model_config=model_cfg.to_dict()),
              ckpt.parent/"summary.json")
    print(f"[NN TRAIN] saved checkpoint={ckpt}",flush=True)


if __name__=="__main__":
    train(parse_args("S3: Train discrete NP with standardized X"))
