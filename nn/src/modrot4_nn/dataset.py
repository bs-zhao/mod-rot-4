"""Read the new S1 NPZ fields without importing simulator code."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ChoiceDataset:
    parameters: np.ndarray
    trial_counts: np.ndarray
    subject_index: np.ndarray
    trial_index: np.ndarray
    x: np.ndarray
    logit: np.ndarray
    p: np.ndarray
    choice: np.ndarray


def load_choice_dataset(path):
    with np.load(Path(path),allow_pickle=False) as stored:
        fields=set(ChoiceDataset.__dataclass_fields__)
        if not fields.issubset(stored.files):
            raise ValueError(f"NN requires new S1 choice dataset fields {sorted(fields)}")
        return ChoiceDataset(**{k:stored[k] for k in fields})
