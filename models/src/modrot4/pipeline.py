"""File locations for the two common model stages."""
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS = ("C1", "C2", "C3", "CR3", "CR4", "CR5")
# "<MODEL>L": same T, trial counts, X and true logit as <MODEL>; the observation is a noisy logit instead of a choice.
LOGIT_MODELS = tuple(m + "L" for m in MODELS)
ALL_MODELS = MODELS + LOGIT_MODELS


def base_model(model):
    return model[:-1] if model in LOGIT_MODELS else model


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def out_dir(model):
    if model not in ALL_MODELS:
        raise ValueError(f"Unknown model: {model}")
    return ROOT / "outputs" / "models" / "generate" / model


def dataset_path(model):
    return out_dir(model) / "s1_generate__dataset.npz"


def precheck_path(model):
    if model not in ALL_MODELS:
        raise ValueError(f"Unknown model: {model}")
    return (ROOT / "outputs" / "models" / "precheck" / model /
            "s2_precheck__report.json")


def _cr3_bank():
    return out_dir("CR3") / "s0_design" / "CR3_4P_fixed500.npz"


def _cr5_bank():
    return out_dir("CR5") / "s0_design" / "CR5_3P_cr3bank500.npz"
