"""Six mathematical model specifications; shared S1 and S2 never branch by model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from modlearn_rot.generators.registry import get_model
from modlearn_rot.generators.uniform_inputs import sample_independent_uniform_x
from modrot4.pipeline import ROOT
from modrot4.fixed_bank import ensure_local_bank, inspect_local_bank

ref = ROOT / "models" / "reference"
for folder in (ref, ref / "cr3", ref / "cr4"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from s0_cr2_design_recovery import (  # noqa: E402
    BOUNDS, NAMES, fit_one as cpt_fit, logits as cpt_logit, sample_prior)
from s0_s1_cr4 import (  # noqa: E402
    logits as cpt3_logit, sample_theta)
from s2_oracle_cr4 import (  # noqa: E402
    BOUNDS3, fit_subject as cpt3_fit, gradient as cpt3_gradient)

MODELS = ("C1", "C2", "C3", "CR3", "CR4", "CR5")
CPT3_NAMES = ("lambda_loss", "gamma", "tau")
CR4_LOW = np.array([5., 5., .05, 5., 5., .05])
CR4_HIGH = np.array([200., 200., .95, 200., 200., .95])


class ModelSpec:
    """The only model-dependent interface: T, X, logit and MLE/Fisher math."""

    def __init__(self, model_id: str):
        if model_id not in MODELS:
            raise ValueError(f"Unknown model {model_id}")
        self.model_id = model_id
        self.is_cpt = model_id.startswith("CR")
        self.classic = get_model(model_id) if not self.is_cpt else None
        self.n_parameters = (self.classic.n_factors if self.classic else
                             4 if model_id == "CR3" else 3)
        self.parameter_names = (tuple(self.classic.parameter_names) if self.classic else
                                tuple(NAMES[4]) if model_id == "CR3" else CPT3_NAMES)
        self.x_dim = self.classic.n_input_features if self.classic else 6
        self.trial_min = self.classic.trial_count_min if self.classic else 500
        self.trial_max = self.classic.trial_count_max if self.classic else 500
        self.starts = 4 if self.is_cpt else 1
        self.bank = None

    def prepare(self, design_seed):
        if self.model_id in ("CR3", "CR5"):
            ensure_local_bank(self.model_id, design_seed)
            self.bank = inspect_local_bank(self.model_id)[0]

    def sample_parameters(self, count, rng):
        if self.classic:
            return self.classic.sample_parameters(count, rng)
        return (sample_prior(count, 4, rng) if self.model_id == "CR3"
                else sample_theta(count, rng))

    def sample_inputs(self, count, rng):
        if self.bank is not None:
            if count != len(self.bank):
                raise ValueError("Fixed 500-trial bank size does not match trial count")
            return self.bank
        if self.classic:
            return self.classic.sample_inputs(count, rng)
        return sample_independent_uniform_x(
            count, rng, low=CR4_LOW, high=CR4_HIGH, n_features=6)

    def logit(self, x, theta):
        if self.classic:
            return np.asarray(self.classic.linear_predictor(x, theta))
        return (cpt_logit(x, theta) if self.model_id == "CR3"
                else cpt3_logit(x, theta))

    def draw_choice(self, p, rng):
        if self.is_cpt:
            return (rng.random(p.shape) < p).astype(np.int8)
        return rng.binomial(1, p, size=p.shape).astype(np.int8)

    def validate(self, theta, x, trial_counts):
        if (theta.ndim != 2 or theta.shape[1] != self.n_parameters or
            x.ndim != 2 or x.shape[1] != self.x_dim or
            np.any(trial_counts < self.trial_min) or
            np.any(trial_counts > self.trial_max)):
            raise ValueError(f"{self.model_id}: T/X dimensions or trial counts invalid")
        if self.classic:
            if not self.classic.validate_parameters(theta):
                raise ValueError("True T outside allowed model support")
            if not np.all((-1 <= x) & (x <= 1)):
                raise ValueError("X outside uniform(-1,1) range")
            return
        bounds = BOUNDS[4] if self.model_id == "CR3" else BOUNDS3
        if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
            raise ValueError("CPT T outside allowed parameter support")
        if self.model_id == "CR4":
            ok = np.all((x >= CR4_LOW) & (x <= CR4_HIGH))
        else:
            amounts = x[:, [0, 1, 3, 4]]
            probs = x[:, [2, 5]]
            ok = bool(np.all((amounts >= 0) & (amounts <= 200)) and
                      np.all((probs >= 0) & (probs <= 1)))
        if not ok:
            raise ValueError("CPT X outside allowed model support")

    def fit(self, x, choice, rng):
        if self.classic:
            return self.classic.oracle_fit(x, choice)
        if self.model_id == "CR3":
            return cpt_fit(x, choice, 4, rng, starts=self.starts)
        return cpt3_fit(x, choice, self.starts, rng)

    def fisher(self, x, theta, p):
        if self.classic:
            return self.classic.fisher_information(x, theta, average=False)
        if self.model_id != "CR3":
            jac = cpt3_gradient(x, theta)
        else:
            columns = []
            for j in range(4):
                step = 1e-5 * max(1., abs(float(theta[j])))
                left, right = theta.copy(), theta.copy()
                left[j] -= step
                right[j] += step
                columns.append(
                    (cpt_logit(x, right) - cpt_logit(x, left)) / (2*step))
            jac = np.column_stack(columns)
        return jac.T @ ((p*(1-p))[:, None] * jac)


def get_spec(model_id):
    return ModelSpec(model_id)
