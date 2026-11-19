"""One model-independent diagnostic protocol, using saved T, X, choice only."""
from __future__ import annotations

import csv
import numpy as np
from scipy.special import expit

from modlearn_rot.utils.io import save_json
from modrot4.pipeline import precheck_path, sha256

RECOVERY_SUBJECTS = 100
MIN_RECOVERY_SUBJECTS = 30
MIN_PEARSON_R = 0.75
MIN_FULL_RANK = 0.95


def _pearson(a, b):
    aa, bb = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(aa @ bb / denominator) if denominator > 0 else float("nan")


def run_precheck(spec, data, *, seed, data_path):
    # Only T, X and observed Y (choice) are input to scientific diagnostics.
    theta = np.asarray(data.parameters, dtype=np.float64)
    x = np.asarray(data.x, dtype=np.float64)
    y = np.asarray(data.choice).reshape(-1)
    counts = np.asarray(data.trial_counts, dtype=np.int64)
    n = len(counts)
    if n < 1 or theta.shape != (n, spec.n_parameters):
        raise ValueError("Invalid subject/parameter dimensions")
    total = int(counts.sum())
    if x.shape != (total, spec.x_dim) or y.shape != (total,):
        raise ValueError("Invalid X or choice dimensions")
    if not (np.isfinite(theta).all() and np.isfinite(x).all()
            and np.isin(y, (0, 1)).all()):
        raise ValueError("Non-finite T/X or non-binary choice")
    spec.validate(theta, x, counts)
    ids, trials = np.asarray(data.subject_index), np.asarray(data.trial_index)
    if ids.shape != (total,) or trials.shape != (total,):
        raise ValueError("Invalid subject/trial indices")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for sid in range(n):
        first, last = offsets[sid:sid+2]
        if not np.all(ids[first:last] == sid):
            raise ValueError(f"Subject index mismatch for subject {sid}")
        if not np.array_equal(trials[first:last], np.arange(last-first)):
            raise ValueError(f"Trial index mismatch for subject {sid}")

    # Recompute probabilities from T and X; do NOT use the saved S1 p/logit
    # arrays to evaluate recovery or response quality.
    prob = np.empty(total, dtype=np.float64)
    for sid in range(n):
        first, last = offsets[sid:sid+2]
        prob[first:last] = expit(spec.logit(x[first:last], theta[sid]))
    if not np.all((prob > 0) & (prob < 1)):
        raise ValueError("Invalid probabilities from T and X")
    rate = float(y.mean())
    expected_rate = float(prob.mean())
    se = float(np.sqrt(np.sum(prob*(1-prob))) / total)
    z_score = float(abs(rate-expected_rate) / max(se, 1e-15))
    response = {
        "passed": bool(z_score <= 5.0),
        "observed_choice_rate": rate,
        "expected_choice_rate": expected_rate,
        "bernoulli_z_score": z_score,
        "mean_bayes_accuracy": float(np.mean(np.maximum(prob, 1-prob))),
    }

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n, min(n, RECOVERY_SUBJECTS), replace=False))
    true = theta[chosen]
    fit = np.empty_like(true)
    information_eigs = np.empty(len(chosen))
    sd = (true.std(axis=0, ddof=1) if len(chosen) > 1
          else np.ones(spec.n_parameters))
    sd = np.maximum(sd, 1e-12)
    fitting_rng = np.random.default_rng(seed + 202)
    for pos, sid in enumerate(chosen):
        first, last = offsets[sid:sid+2]
        xx, yy, pp = x[first:last], y[first:last], prob[first:last]
        fit[pos] = spec.fit(xx, yy, fitting_rng)
        if not np.isfinite(fit[pos]).all():
            raise ValueError(f"MLE failed on subject {sid}")
        info = spec.fisher(xx, true[pos], pp)
        scaled = info * sd[:, None] * sd[None, :]
        information_eigs[pos] = np.linalg.eigvalsh(
            (scaled + scaled.T) / 2)[0]
        if (pos+1) % 25 == 0 or pos+1 == len(chosen):
            print(f"[PRECHECK] model={spec.model_id} "
                  f"phase=MLE progress={pos+1}/{len(chosen)}", flush=True)

    error = fit - true
    r = np.array([_pearson(true[:, j], fit[:, j])
                  for j in range(spec.n_parameters)])
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
        "model_id": spec.model_id, "qualified": qualified,
        "dataset_sha256": sha256(data_path),
        "precheck_seed": seed, "checks": checks,
    }
    target = precheck_path(spec.model_id)
    save_json(report, target)
    with target.with_name("s2_precheck__recovery.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id"] + [
            item for name in spec.parameter_names
            for item in (name + "_true", name + "_mle")])
        for pos, sid in enumerate(chosen):
            writer.writerow([int(sid)] + [
                item for k in range(spec.n_parameters)
                for item in (float(true[pos,k]), float(fit[pos,k]))])
    print(f"[PRECHECK] model={spec.model_id} phase=done "
          f"qualified={qualified} r={np.round(r,4)}", flush=True)
    print(f"[PRECHECK] report={target}", flush=True)
    return report
