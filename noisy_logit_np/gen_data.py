"""Generate the NOISY-LOGIT datasets (no binary choices).

y = eta(T, X) + eps,  eps ~ N(0, s^2),  s^2 = Var(eta) * (1-R2)/R2   with R2 = 0.8

so that a perfect predictor (one that knows the true logit) explains exactly 80 % of the
variance of the observed target y and the remaining 20 % is irreducible noise.
T, trial counts and X use the same random streams as the original binary-choice S1
generator, so each subject has the same true parameters and stimuli as before; only the
observation changes (continuous noisy logit instead of a Bernoulli draw).

usage: python gen_data.py [MODEL ...]
"""
import json, sys
import numpy as np
from common import *
from modrot4.model_specs import get_spec
from modlearn_rot.utils.seeds import make_rng_streams

RNG_NAMES = ("parameters", "trial_counts", "inputs", "observations")


from common import data_seed as data_seed_, noise_seed as noise_seed_


def generate(model, subjects=SUBJECTS, r2=R2_TARGET, noise_seed=NOISE_SEED):
    spec = get_spec(base(model))
    if base(model) == "CR3":   # fixed 500-trial stimulus bank: read the existing file, never rebuild/write
        bank_file = PROJECT / "outputs" / "models" / "generate" / "CR3" / "s0_design" / "CR3_4P_fixed500.npz"
        spec.bank = np.load(bank_file)["X"]
    data_seed = data_seed_(model)
    rng = make_rng_streams(data_seed, RNG_NAMES)
    if spec.is_cpt:
        rng["parameters"] = np.random.default_rng(data_seed + 4)
    theta = np.asarray(spec.sample_parameters(subjects, rng["parameters"]))
    counts = rng["trial_counts"].integers(spec.trial_min, spec.trial_max + 1, size=subjects)
    ids, xs, etas = [], [], []
    for sid, (t, n) in enumerate(zip(theta, counts)):
        n = int(n)
        x = np.asarray(spec.sample_inputs(n, rng["inputs"]), dtype=np.float64)
        eta = np.asarray(spec.logit(x, t), dtype=np.float64).reshape(-1)
        assert x.shape == (n, spec.x_dim) and eta.shape == (n,) and np.isfinite(eta).all()
        ids.append(np.full(n, sid, dtype=np.int32)); xs.append(x); etas.append(eta)
    eta = np.concatenate(etas)
    var_eta = float(np.var(eta)); sd = float(np.sqrt(var_eta * (1 - r2) / r2))
    y = eta + np.random.default_rng(noise_seed_(model)).normal(0.0, sd, size=eta.shape)
    out = dict(parameters=theta, trial_counts=counts.astype(np.int32), subject_index=np.concatenate(ids),
               x=np.concatenate(xs), logit=eta, y=y)
    np.savez_compressed(data_path(model), **out)
    meta = dict(model=model, subjects=int(subjects), trials=int(len(eta)), parameters=list(spec.parameter_names),
                x_dim=int(spec.x_dim), var_logit=var_eta, noise_sd=sd, r2_target=r2,
                realised_r2_perfect_predictor=float(np.corrcoef(eta, y)[0, 1] ** 2),
                data_seed=data_seed, noise_seed=noise_seed_(model))
    (DATA / f"{model}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    for m in (sys.argv[1:] or MODELS + MODELS_R2):
        meta = generate(m)
        print(f"{m}: subjects {meta['subjects']} trials {meta['trials']} T_dim {len(meta['parameters'])} X_dim {meta['x_dim']} "
              f"Var(logit) {meta['var_logit']:.3f} noise sd {meta['noise_sd']:.3f} realised R2(perfect) {meta['realised_r2_perfect_predictor']:.4f}", flush=True)
