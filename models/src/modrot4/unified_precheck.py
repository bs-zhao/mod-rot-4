"""Common S2 checks for all six simulated choice datasets.

Recovery fits the ACTUAL S1 X and Y, never a newly synthesized cohort.
"""
from __future__ import annotations
import csv
import sys
import numpy as np
from modlearn_rot.utils.io import save_json
from modrot4.pipeline import precheck_path, sha256

def pearson(a, b):
    aa, bb = a-a.mean(), b-b.mean()
    d = np.sqrt(np.dot(aa, aa)*np.dot(bb, bb))
    return float(np.dot(aa, bb)/d) if d > 0 else float("nan")


def run_unified_precheck(spec, data, *, seed, provenance, data_path):
    model_id = spec.model_id
    theta = np.asarray(data.parameters, dtype=float)
    counts = np.asarray(data.trial_counts, dtype=np.int64)
    x = np.asarray(data.x, dtype=float)
    y = np.asarray(data.y).reshape(-1)
    p = np.asarray(data.predictive_true, dtype=float).reshape(-1)
    names = spec.parameter_names
    if theta.ndim != 2 or len(counts) != len(theta) or len(theta) < 1:
        raise ValueError("Invalid parameter or subject counts")
    n, dim = theta.shape
    total = int(counts.sum())
    xdim = spec.x_dim
    if dim != len(names) or x.shape != (total, xdim) or y.shape != (total,) or p.shape != (total,):
        raise ValueError("Invalid T/X/Y/P dimensions")
    if np.any(counts < 1) or np.asarray(data.subject_index).shape != (total,) or np.asarray(data.trial_index).shape != (total,):
        raise ValueError("Invalid subject/trial index geometry")
    if not (np.isfinite(theta).all() and np.isfinite(x).all()
            and np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Non-finite T/X/Y/P")
    if not (np.isin(y, [0, 1]).all() and ((p > 0) & (p < 1)).all()):
        raise ValueError("Invalid Bernoulli Y/P")
    spec.validate_parameters_inputs(theta,x,counts)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    subjects = np.asarray(data.subject_index)
    trials = np.asarray(data.trial_index)
    for sid in range(n):
        lo, hi = int(offsets[sid]), int(offsets[sid+1])
        if not np.all(subjects[lo:hi] == sid) or not np.array_equal(trials[lo:hi], np.arange(hi-lo)):
            raise ValueError(f"Subject/trial ordering corrupted: {sid}")
    structure = dict(passed=True, total_subjects=n, total_trials=total,
                     min_trials=int(counts.min()), max_trials=int(counts.max()),
                     parameter_names=list(names))
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(n, size=min(RECOVERY_SUBJECTS,n), replace=False))
    formula_subjects = np.sort(rng.choice(n, size=min(32,n), replace=False))
    max_error = 0.
    for sid in formula_subjects:
        lo, hi = offsets[sid:sid+2]
        max_error = max(max_error, float(np.max(np.abs(
            spec.predict(x[lo:hi], theta[sid]) - p[lo:hi]))))
    generator = dict(passed=bool(max_error <= PROB_TOL),
                     subjects_checked=len(formula_subjects),
                     max_abs_error=max_error, tolerance=PROB_TOL)
    design = dict(passed=True, provenance=provenance)
    expected_rate = float(p.mean())
    actual_rate = float(y.mean())
    se = float(np.sqrt(np.sum(p*(1-p)))/total)
    z = abs(actual_rate - expected_rate)/max(se, 1e-15)
    response = dict(passed=bool(z <= 5),
                    observed_choice_rate=actual_rate, expected_choice_rate=expected_rate,
                    bernoulli_z_score=float(z), z_limit=5.,
                    mean_bayes_accuracy=float(np.maximum(p,1-p).mean()))
    truth = theta[selected]
    estimated = np.empty_like(truth)
    eigmin = np.empty(len(selected))
    condition = np.empty(len(selected))
    sd = np.std(truth, axis=0, ddof=1) if len(selected) > 1 else np.ones(dim)
    sd = np.maximum(sd, 1e-12)
    fit_rng = np.random.default_rng(seed+202)
    for pos, sid in enumerate(selected):
        lo, hi = offsets[sid:sid+2]
        xx, yy, pp = x[lo:hi], y[lo:hi], p[lo:hi]
        estimated[pos] = spec.fit(xx, yy, fit_rng)
        if not np.isfinite(estimated[pos]).all():
            raise ValueError(f"MLE returned non-finite result for subject {sid}")
        info = spec.fisher(xx, truth[pos], pp)
        scaled = info * sd[:,None] * sd[None,:]
        ev = np.linalg.eigvalsh((scaled+scaled.T)/2)
        eigmin[pos] = ev[0]
        condition[pos] = ev[-1]/ev[0] if ev[0] > 1e-12 else np.inf
        if (pos+1)%25 == 0 or pos+1 == len(selected):
            print(f"[PRECHECK] model={model_id} phase=MLE progress={pos+1}/{len(selected)}",flush=True)
    err = estimated-truth
    r = np.asarray([pearson(truth[:,k], estimated[:,k]) for k in range(dim)])
    denom = np.sum((truth-truth.mean(axis=0))**2, axis=0)
    r2 = 1-np.sum(err**2, axis=0)/np.maximum(denom, 1e-15)
    passed = np.isfinite(r) & (r>=MIN_PEARSON_R)
    enough = bool(len(selected)>=MIN_RECOVERY_SUBJECTS)
    recovery = dict(passed=bool(enough and passed.all()), conclusive=enough,
                    sample_size=len(selected), original_dataset_subjects=n,
                    cohort="actual S1 subjects sampled without replacement",
                    minimum_sample_size=MIN_RECOVERY_SUBJECTS,
                    initializations=spec.n_starts,
                    threshold_metric="each_parameter_pearson_r",
                    threshold=MIN_PEARSON_R, names=list(names),
                    pearson_r=r.tolist(), r2=r2.tolist(),
                    rmse=np.sqrt(np.mean(err**2,axis=0)).tolist(),
                    bias=np.mean(err,axis=0).tolist(), each_parameter_passed=passed.tolist())
    fullrank = float(np.mean(eigmin > 1e-8))
    fisher_check = dict(passed=bool(fullrank >= MIN_FISHER_FULL_RANK_FRACTION),
                        full_rank_fraction=fullrank, threshold=MIN_FISHER_FULL_RANK_FRACTION,
                        min_eigenvalue_p01=float(np.percentile(eigmin,1)),
                        median_min_eigenvalue=float(np.median(eigmin)),
                        median_condition=float(np.median(condition)))
    checks = dict(structure=structure, generator_fidelity=generator,
                  stimulus_design=design, response_balance=response,
                  fisher_information=fisher_check, parameter_recovery=recovery)
    qualified = bool(all(v["passed"] for v in checks.values()))
    report = dict(model_id=model_id, qualified=qualified, subjects=n,
                  trials=total, precheck_seed=seed,
                  dataset_sha256=sha256(data_path), checks=checks)
    output = precheck_path(model_id)
    save_json(report, output)
    with output.with_name("s2_precheck__summary.csv").open("w", newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["check","passed"])
        writer.writerows((key,val["passed"]) for key,val in checks.items())
        writer.writerow(["OVERALL",qualified])
    with output.with_name("s2_precheck__oracle_recovery.csv").open("w", newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["subject_id"]+[column for name in names for column in
                        (name+"_true", name+"_estimate")])
        for pos,sid in enumerate(selected):
            writer.writerow([int(sid)]+[value for k in range(dim) for value in
                            (float(truth[pos,k]),float(estimated[pos,k]))])
    np.savez_compressed(output.with_name("s2_precheck__oracle_subjects.npz"),
                        subject_ids=selected, theta_true=truth, theta_mle=estimated,
                        fisher_min_eigenvalue=eigmin, fisher_condition=condition)
    print(f"[PRECHECK] model={model_id} phase=done qualified={qualified} "
          f"sample={len(selected)} r={np.round(r,4)}",flush=True)
    print(f"[PRECHECK] report={output}",flush=True)
    return report
