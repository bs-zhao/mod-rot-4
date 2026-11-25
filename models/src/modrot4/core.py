"""The single generation and serialization pipeline for all six models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit
from modlearn_rot.utils.seeds import make_rng_streams
from modrot4.model_specs import LOGIT_R2

FIELDS = ("parameters", "trial_counts", "subject_index", "trial_index",
          "x", "logit", "p", "choice")
RNG_NAMES = ("parameters", "trial_counts", "inputs", "observations")
LOGIT_FIELDS = ("parameters", "trial_counts", "subject_index", "trial_index",
                "x", "logit", "y")


@dataclass
class ChoiceData:
    parameters: np.ndarray
    trial_counts: np.ndarray
    subject_index: np.ndarray
    trial_index: np.ndarray
    x: np.ndarray
    logit: np.ndarray
    p: np.ndarray
    choice: np.ndarray


@dataclass
class NoisyLogitData:
    """<MODEL>L: no p / choice; the only observed response is y = logit + noise."""
    parameters: np.ndarray
    trial_counts: np.ndarray
    subject_index: np.ndarray
    trial_index: np.ndarray
    x: np.ndarray
    logit: np.ndarray
    y: np.ndarray


def noise_sd(logit, r2) -> float:
    return float(np.sqrt(np.var(logit) * (1 - r2) / r2))


def generate(spec, subjects: int, data_seed: int, design_seed: int):
    """T, then X by subject, then logit, p and Bernoulli choice: same for all."""
    spec.prepare(design_seed)
    rng = make_rng_streams(data_seed, RNG_NAMES)
    if spec.is_cpt:
        # Preserve the CPT subject priors and CR4/CR5 paired random streams.
        rng["parameters"] = np.random.default_rng(data_seed + 4)
        rng["observations"] = np.random.default_rng(data_seed + 1004)
    theta = np.asarray(spec.sample_parameters(subjects, rng["parameters"]))
    counts = rng["trial_counts"].integers(
        spec.trial_min, spec.trial_max + 1, size=subjects)
    ids, trials, xs, logits, ps, choices = [], [], [], [], [], []
    for sid, (true_t, trial_count) in enumerate(zip(theta, counts, strict=True)):
        n = int(trial_count)
        x = np.asarray(spec.sample_inputs(n, rng["inputs"]), dtype=np.float64)
        if x.shape != (n, spec.x_dim):
            raise ValueError(f"{spec.model_id}: X shape {x.shape} is invalid")
        eta = np.asarray(spec.logit(x, true_t), dtype=np.float64).reshape(-1)
        if eta.shape != (n,) or not np.isfinite(eta).all():
            raise ValueError(f"{spec.model_id}: invalid logit")
        ids.append(np.full(n, sid, dtype=np.int32))
        trials.append(np.arange(n, dtype=np.int32))
        xs.append(x)
        logits.append(eta)
        if spec.response == "noisy_logit":
            continue                      # same T / trial-count / X streams as the choice model
        prob = expit(eta)
        if not np.all((prob > 0) & (prob < 1)):
            raise ValueError(f"{spec.model_id}: invalid Bernoulli probabilities")
        y = spec.draw_choice(prob, rng["observations"])
        ps.append(prob)
        choices.append(np.asarray(y, dtype=np.int8))
    if spec.response == "noisy_logit":
        eta = np.concatenate(logits)
        sd = noise_sd(eta, LOGIT_R2)      # one noise SD per dataset, from the variance of all true logits
        y = eta + np.random.default_rng(spec.noise_seed).normal(0.0, sd, size=eta.shape)
        return NoisyLogitData(
            parameters=theta, trial_counts=counts.astype(np.int32),
            subject_index=np.concatenate(ids), trial_index=np.concatenate(trials),
            x=np.concatenate(xs), logit=eta, y=y)
    return ChoiceData(
        parameters=theta, trial_counts=counts.astype(np.int32),
        subject_index=np.concatenate(ids), trial_index=np.concatenate(trials),
        x=np.concatenate(xs), logit=np.concatenate(logits),
        p=np.concatenate(ps), choice=np.concatenate(choices))


def save(data: ChoiceData, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = LOGIT_FIELDS if isinstance(data, NoisyLogitData) else FIELDS
    np.savez_compressed(path, **{field: getattr(data, field) for field in fields})


def load(path: Path):
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) == set(LOGIT_FIELDS):
            return NoisyLogitData(**{field: stored[field] for field in LOGIT_FIELDS})
        missing = set(FIELDS) - set(stored.files)
        if missing:
            raise ValueError(
                f"Dataset uses an old/incomplete schema (missing {sorted(missing)}). "
                "Archive its model-specific generate and precheck folders, "
                "then rerun Generate; old data will not be overwritten.")
        return ChoiceData(**{field: stored[field] for field in FIELDS})
