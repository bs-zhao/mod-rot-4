"""S2 for the <MODEL>L datasets: the same four checks and thresholds as unified_precheck, with a Gaussian observation model.

Only T, X and the observed y are used; the true logit is recomputed from T and X (saved logit is not an input).
  response_balance   : residual y - logit(T, X) is centred (|z| <= 5) and a perfect predictor explains R^2 = 0.8 +- 0.01 of y
  parameter_recovery : maximum likelihood = bounded nonlinear least squares of logit(X, theta) on y
  fisher_information : J' J / s^2
"""
from __future__ import annotations

import csv

import numpy as np
from scipy.optimize import least_squares

from modlearn_rot.utils.io import save_json
from modrot4.model_specs import BOUNDS, BOUNDS3, LOGIT_R2
from modrot4.pipeline import precheck_path, sha256
from modrot4.unified_precheck import (
    RECOVERY_SUBJECTS, MIN_RECOVERY_SUBJECTS, MIN_PEARSON_R, MIN_FULL_RANK, _pearson)

R2_TOLERANCE = 0.01


def _jacobian(spec, x, theta):
    columns = []
    for j in range(len(theta)):
        step = 1e-6 * max(1., abs(float(theta[j])))
        left, right = theta.copy(), theta.copy()
        left[j] -= step
        right[j] += step
        columns.append((np.asarray(spec.logit(x, right)) - np.asarray(spec.logit(x, left))).reshape(-1) / (2 * step))
    return np.column_stack(columns)


def fit_subject(spec, x, y, rng):
    """Gaussian MLE. C models are linear in T (closed form); CPT models use bounded least squares with several starts."""
    if not spec.is_cpt:
        zero = np.zeros(spec.n_parameters)
        design = _jacobian(spec, x, zero)
        offset = np.asarray(spec.logit(x, zero)).reshape(-1)
        return np.linalg.lstsq(design, y - offset, rcond=None)[0]
    bounds = np.asarray(BOUNDS[4] if spec.base_id == "CR3" else BOUNDS3, dtype=float)
    low, high = bounds[:, 0], bounds[:, 1]
    best = None
    for start in range(spec.starts):
        initial = (low + high) / 2 if start == 0 else rng.uniform(low, high)
        result = least_squares(lambda t: np.asarray(spec.logit(x, t)).reshape(-1) - y,
                               initial, bounds=(low, high))
        if best is None or result.cost < best.cost:
            best = result
    return best.x


def run_noisy_logit_precheck(spec, data, *, seed, data_path):
    theta = np.asarray(data.parameters, dtype=np.float64)
    x = np.asarray(data.x, dtype=np.float64)
    y = np.asarray(data.y, dtype=np.float64).reshape(-1)
    counts = np.asarray(data.trial_counts, dtype=np.int64)
    n = len(counts)
    if n < 1 or theta.shape != (n, spec.n_parameters):
        raise ValueError("Invalid subject/parameter dimensions")
    total = int(counts.sum())
    if x.shape != (total, spec.x_dim) or y.shape != (total,):
        raise ValueError("Invalid X or y dimensions")
    if not (np.isfinite(theta).all() and np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Non-finite T/X/y")
    spec.validate(theta, x, counts)
    ids, trials = np.asarray(data.subject_index), np.asarray(data.trial_index)
    if ids.shape != (total,) or trials.shape != (total,):
        raise ValueError("Invalid subject/trial indices")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    eta = np.empty(total, dtype=np.float64)
    for sid in range(n):
        first, last = offsets[sid:sid+2]
        if not np.all(ids[first:last] == sid):
            raise ValueError(f"Subject index mismatch for subject {sid}")
        if not np.array_equal(trials[first:last], np.arange(last-first)):
            raise ValueError(f"Trial index mismatch for subject {sid}")
        eta[first:last] = np.asarray(spec.logit(x[first:last], theta[sid])).reshape(-1)

    residual = y - eta
    variance_eta = float(np.var(eta))
    sd_noise = float(np.sqrt(variance_eta * (1 - LOGIT_R2) / LOGIT_R2))
    z_score = float(abs(residual.mean()) / max(sd_noise / np.sqrt(total), 1e-15))
    realised_r2 = float(1 - np.mean(residual**2) / np.var(y))
    tolerance = max(R2_TOLERANCE, 5 * np.sqrt(2 / total))       # wider only for tiny test datasets
    response = {
        "passed": bool(z_score <= 5.0 and abs(realised_r2 - LOGIT_R2) <= tolerance),
        "r2_target": LOGIT_R2,
        "r2_perfect_predictor": realised_r2,
        "logit_variance": variance_eta,
        "noise_sd_expected": sd_noise,
        "noise_sd_observed": float(residual.std()),
        "residual_mean_z_score": z_score,
    }

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n, min(n, RECOVERY_SUBJECTS), replace=False))
    true = theta[chosen]
    fit = np.empty_like(true)
    information_eigs = np.empty(len(chosen))
    sd = (true.std(axis=0, ddof=1) if len(chosen) > 1 else np.ones(spec.n_parameters))
    sd = np.maximum(sd, 1e-12)
    fitting_rng = np.random.default_rng(seed + 202)
    for pos, sid in enumerate(chosen):
        first, last = offsets[sid:sid+2]
        fit[pos] = fit_subject(spec, x[first:last], y[first:last], fitting_rng)
        if not np.isfinite(fit[pos]).all():
            raise ValueError(f"MLE failed on subject {sid}")
        jac = _jacobian(spec, x[first:last], true[pos])
        scaled = (jac.T @ jac / sd_noise**2) * sd[:, None] * sd[None, :]
        information_eigs[pos] = np.linalg.eigvalsh((scaled + scaled.T) / 2)[0]
        if (pos+1) % 25 == 0 or pos+1 == len(chosen):
            print(f"[PRECHECK] model={spec.model_id} phase=MLE progress={pos+1}/{len(chosen)}", flush=True)

    error = fit - true
    r = np.array([_pearson(true[:, j], fit[:, j]) for j in range(spec.n_parameters)])
    variance = np.sum((true - true.mean(axis=0))**2, axis=0)
    r2 = 1 - np.sum(error**2, axis=0) / np.maximum(variance, 1e-15)
    recovery_ok = (len(chosen) >= MIN_RECOVERY_SUBJECTS and
                   np.isfinite(r).all() and (r >= MIN_PEARSON_R).all())
    full_rank_fraction = float(np.mean(information_eigs > 1e-8))
    checks = {
        "structure": {"passed": True, "subjects": n, "trials": total},
        "response_balance": response,
        "fisher_information": {
            "passed": bool(full_rank_fraction >= MIN_FULL_RANK),
            "full_rank_fraction": full_rank_fraction,
            "median_min_eigenvalue": float(np.median(information_eigs)),
        },
        "parameter_recovery": {
            "passed": bool(recovery_ok),
            "conclusive": bool(len(chosen) >= MIN_RECOVERY_SUBJECTS),
            "sample_size": len(chosen),
            "parameter_names": list(spec.parameter_names),
            "pearson_r": r.tolist(),
            "r2": r2.tolist(),
            "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "bias": error.mean(axis=0).tolist(),
            "threshold_r": MIN_PEARSON_R,
        },
    }
    qualified = bool(all(check["passed"] for check in checks.values()))
    report = {
        "model_id": spec.model_id, "response": "noisy_logit", "qualified": qualified,
        "dataset_sha256": sha256(data_path), "precheck_seed": seed, "checks": checks,
    }
    target = precheck_path(spec.model_id)
    save_json(report, target)
    with target.with_name("s2_precheck__recovery.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id"] + [item for name in spec.parameter_names for item in (name + "_true", name + "_mle")])
        for pos, sid in enumerate(chosen):
            writer.writerow([int(sid)] + [item for k in range(spec.n_parameters)
                                          for item in (float(true[pos, k]), float(fit[pos, k]))])
    print(f"[PRECHECK] model={spec.model_id} phase=done qualified={qualified} r={np.round(r,4)}", flush=True)
    print(f"[PRECHECK] report={target}", flush=True)
    return report
