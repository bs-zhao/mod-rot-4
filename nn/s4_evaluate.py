"""S4: checkpoint evaluation metrics and training curves; no retraining."""
from pathlib import Path
import csv
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modrot4_nn.cli import parse_args
from modrot4_nn.pipeline import train_path,stage_folder,dataset_path,sha256
from modrot4_nn.training_helpers import load_checkpoint
from modrot4_nn.files import save_json


def evaluate(args):
    checkpoint=train_path(args)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing S3 checkpoint: {checkpoint}")
    state=load_checkpoint(checkpoint)
    if state["extra_state"]["dataset_sha256"]!=sha256(dataset_path(args.model)):
        raise ValueError("Checkpoint/dataset mismatch")
    folder=stage_folder(args,"s4_evaluate")
    folder.mkdir(parents=True,exist_ok=True)
    history=state["history"]
    summary=dict(
        model=args.model,checkpoint_sha256=sha256(checkpoint),
        best_epoch=int(state["best_epoch"])+1,
        epochs=len(next(iter(history.values()))),
        validation=state["validation_metrics"],
        test=state["test_metrics"],
        metrics_protocol="fixed context; deterministic posterior mean")
    save_json(summary,folder/"metrics.json")
    with (folder/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["split","metric","value"])
        for group in ("validation","test"):
            for key,value in state[group+"_metrics"].items():
                w.writerow([group,key,value])
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    continuous="train_r2" in history          # <MODEL>L: continuous response
    second="r2" if continuous else "accuracy"
    for group in ("train","validation"):
        ax[0].plot(history[f"{group}_reconstruction"],label=group)
        ax[1].plot(history[f"{group}_{second}"],label=group)
    if continuous:
        ax[0].set(title="Mean squared error (standardized y)",xlabel="Epoch",ylabel="MSE")
        ax[1].set(title="R^2 on observed y",xlabel="Epoch",ylabel="R^2")
    else:
        ax[0].set(title="Cross-entropy",xlabel="Epoch",ylabel="CE")
        ax[1].set(title="Choice accuracy",xlabel="Epoch",ylabel="Accuracy")
    for axis in ax: axis.legend()
    fig.tight_layout();fig.savefig(folder/"learning_curves.png",dpi=160);plt.close(fig)
    print(f"[NN EVAL] TEST {state['test_metrics']} | {folder}",flush=True)


if __name__=="__main__":
    evaluate(parse_args("S4: Evaluate saved NN checkpoint"))
