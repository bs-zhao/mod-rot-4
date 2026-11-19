"""Exact split/context/checkpoint helpers required by the NP trainer."""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from .net.context import sample_context_keep_mask_torch


def split_subject_indices(n_subjects, train_fraction, validation_fraction, seed):
    if n_subjects < 3:
        raise ValueError("NP requires at least three subjects")
    rng = torch.Generator().manual_seed(int(seed))
    ids = torch.randperm(n_subjects, generator=rng)
    n_train = max(1, round(n_subjects * train_fraction))
    n_val = max(1, round(n_subjects * validation_fraction))
    if n_train + n_val >= n_subjects:
        n_train, n_val = n_subjects - 2, 1
    return {"train": ids[:n_train], "validation": ids[n_train:n_train+n_val],
            "test": ids[n_train+n_val:]}


def sample_context_mask(valid_trial_mask, minimum_fraction, maximum_fraction, generator):
    return sample_context_keep_mask_torch(
        valid_trial_mask, minimum_fraction, maximum_fraction, generator=generator)


def make_fixed_context_masks(data, subject_splits, minimum_fraction, maximum_fraction, seed):
    out = {}
    for offset, name in enumerate(("validation", "test")):
        generator = torch.Generator().manual_seed(int(seed) + 10000 + offset)
        sampled = sample_context_mask(
            data.trial_mask[subject_splits[name]],
            minimum_fraction, maximum_fraction, generator)
        full = torch.zeros_like(data.trial_mask)
        full[subject_splits[name]] = sampled
        out[name] = full
    return out


def stable_run_seed(base_seed, model_id, beta, repeat):
    payload = f"{int(base_seed)}|{model_id}|{float(beta):.12g}|{int(repeat)}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def number_tag(value):
    return format(float(value), ".9g").replace("-", "m").replace(".", "p")


def training_condition_tag(beta, lambda_sparse, context_min_fraction,
                           context_max_fraction, latent_dim):
    return (f"beta{number_tag(beta)}_lambda{number_tag(lambda_sparse)}_"
            f"ctx{number_tag(context_min_fraction)}to{number_tag(context_max_fraction)}_"
            f"L{latent_dim}")


def save_history_csv(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *keys])
        for i in range(len(next(iter(history.values())))):
            writer.writerow([i+1, *(history[k][i] for k in keys)])


def save_checkpoint(path, result, model_config, training_config, splits, masks, extra_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in result["model"].state_dict().items()},
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "history": result["history"],
        "best_epoch": result["best_epoch"],
        "validation_metrics": result["validation_metrics"],
        "test_metrics": result["test_metrics"],
        "subject_splits": {k: torch.as_tensor(v).cpu() for k, v in splits.items()},
        "fixed_context_masks": {k: torch.as_tensor(v).cpu() for k, v in masks.items()},
        "preprocessing": result["preprocessing"],
        "extra_state": extra_state or {},
    }, path)


def load_checkpoint(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True)
