"""Shared interface for six choice models, with preserved original formulas and seeds.

The driver is identical for every model. Only the T, X, P/Y sampling routines,
the original MLE/Fisher operators, and design provenance differ by ModelSpec.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit

from modlearn_rot.generators.dataset import GeneratedDataset, generate_dataset
from modlearn_rot.generators.choice.base import BinaryChoiceModel
from modlearn_rot.generators.uniform_inputs import sample_independent_uniform_x
from modlearn_rot.utils.seeds import make_rng_streams
from modlearn_rot.generators.registry import get_model
from modlearn_rot.task_specs import get_task_spec_for_model
from modrot4.pipeline import (ROOT, dataset_path, sha256, _cr3_bank, _cr5_bank)
from modrot4.fixed_bank import ensure_local_bank, inspect_local_bank
REFERENCE = ROOT / "models" / "reference"
for folder in (REFERENCE, REFERENCE / "cr3", REFERENCE / "cr4", REFERENCE / "cr5"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from s0_cr2_design_recovery import (BOUNDS, NAMES, fit_one as fit_cpt,  # noqa: E402
                                    logits as logits_cpt, sample_prior)
from s0_s1_cr3 import dominance_audit, condition_counts, COUNTS  # noqa: E402
from s0_s1_cr4 import (sample_theta, sample_subject_x,  # noqa: E402
                       logits as logits_3p, SAMPLING_POLICY,
                       DEFAULT_AMOUNT_RANGE, DEFAULT_PROB_RANGE)
from s2_oracle_cr4 import (BOUNDS3, fit_subject as fit_3p,  # noqa: E402
                           gradient as gradient_3p)

# Preserved model-specific checks from the previous S2 are now adapter internals.
# These verify the actual generated X distributions and immutable bank provenance.
def _verify_cr3(data):
    bank = _cr3_bank()
    if not bank.is_file():
        raise FileNotFoundError(f'Missing original CR3 fixed-X bank: {bank}')
    with np.load(bank, allow_pickle=False) as d:
        x = np.asarray(d['X'], dtype=np.float64)
    if x.shape != (500, 6):
        raise ValueError(f'CR3 bank must be (500,6), got {x.shape}')
    # Import the exact FOSD and categorization functions from frozen reference.
    sys.path.insert(0, str(ROOT / 'models' / 'reference' / 'cr3'))
    from s0_s1_cr3 import dominance_audit, condition_counts, COUNTS
    if dominance_audit(x)['total']:
        raise ValueError('CR3 stimulus bank contains strictly dominant trials')
    observed = condition_counts(x)
    expected = dict(gain_only=COUNTS['gain_only'], loss_only=COUNTS['loss_only'],
                    gamma_neutral=COUNTS['gamma_neutral'])
    if any(observed[k] != v for k, v in expected.items()):
        raise ValueError(f'CR3 bank composition changed: {observed}')
    manifest_path = dataset_path('CR3').with_name('s1_generate__manifest.json')
    if not manifest_path.exists():
        raise FileNotFoundError('Missing CR3 dataset provenance manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['bank_sha256'] != sha256(bank):
        raise RuntimeError('CR3 dataset originated from a different stimulus bank')
    if manifest['dataset_sha256'] != sha256(dataset_path('CR3')):
        raise RuntimeError('CR3 dataset checksum mismatch')
    n = len(data.parameters)
    if data.parameters.shape != (n, 4) or not np.all(data.trial_counts == 500):
        raise ValueError('CR3 must have four parameters and 500 trials per subject')
    if data.x.shape != (500 * n, 6):
        raise ValueError('CR3 observed X geometry changed')
    if not np.array_equal(data.x.reshape(n, 500, 6)[0], x):
        raise ValueError('CR3 trial order no longer matches frozen bank')
    if not np.array_equal(data.x.reshape(n, 500, 6), np.broadcast_to(x, (n, 500, 6))):
        raise ValueError('CR3 subjects do not share the frozen 500-trial bank')
    return x


def _verify_cr4(data):
    """Verify independent per-subject/trial X, continuous support and CPT."""
    manifest_file = dataset_path('CR4').with_name('s1_generate__manifest.json')
    if not manifest_file.is_file():
        raise FileNotFoundError('Missing CR4 generator manifest')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest.get('x_sampling_policy') != CR4_X_POLICY:
        raise RuntimeError('Old CR4 X sampling stream: archive its generate/CR4 and precheck/CR4 folders, then regenerate CR4 with the shared C1/C2/C3 RNG-based sampler')
    if manifest.get('x_shared') is not False:
        raise RuntimeError('CR4 manifest does not declare subject-specific X')
    if manifest['dataset_sha256'] != sha256(dataset_path('CR4')):
        raise RuntimeError('CR4 IID-X dataset checksum mismatch')
    n = len(data.parameters)
    if data.parameters.shape != (n, 3) or not np.all(data.trial_counts == 500):
        raise ValueError('CR4 requires three parameters and 500 trials per subject')
    if data.x.shape != (n * 500, 6):
        raise ValueError('CR4 X geometry mismatch')
    x = np.asarray(data.x, dtype=np.float64).reshape(n, 500, 6)
    sys.path.insert(0, str(ROOT / 'models' / 'reference' / 'cr4'))
    from s0_s1_cr4 import verify_x, logits, sample_subject_x
    if not verify_x(x, manifest['amount_range'], manifest['probability_range']):
        raise ValueError('CR4 X contains amounts/probabilities outside sampling support')
    if manifest.get('subjects') != n:
        raise RuntimeError('CR4 manifest subject count disagrees with dataset')
    audit_n = min(n, 2)
    audit_rng = make_rng_streams(manifest["data_seed"], TRIALWISE_STREAM_NAMES)["inputs"]
    iid_sampler = CR4TrialwiseModel(audit_n)
    audit_x = np.stack([iid_sampler.sample_inputs(500, audit_rng)
                        for _ in range(audit_n)])
    if not np.array_equal(x[:audit_n], audit_x):
        raise RuntimeError('CR4 X differs from the C1/C2/C3 single-RNG trialwise sampling stream')
    if n > 1 and np.array_equal(x[0], x[1]):
        raise ValueError('CR4 first two subjects share identical 500-trial X')
    if np.array_equal(x[0, 0], x[0, 1]):
        raise ValueError('CR4 trial X is repeated instead of independently sampled')
    from scipy.special import expit
    ids = sorted({0, n // 2, n - 1})
    true_p = np.asarray(data.predictive_true).reshape(n, 500)
    error = max(float(np.max(np.abs(
        expit(logits(x[i], data.parameters[i])) - true_p[i]
    ))) for i in ids)
    if error > 1e-12:
        raise RuntimeError(f'CR4 IID-X generator fidelity error {error}')
    return x, manifest, error


def _verify_cr5(data):
    """Audit CR5 against its *own* immutable bank; no CR3 files are needed."""
    bank,codes,names,design=inspect_local_bank("CR5")
    n=len(data.parameters)
    if data.parameters.shape != (n,3) or not np.all(np.asarray(data.trial_counts)==500):
        raise ValueError("CR5 must have exactly 3 parameters and 500 trials per subject")
    if data.x.shape != (n*500,6):
        raise ValueError("CR5 X shape mismatch")
    if not np.array_equal(np.asarray(data.x).reshape(n,500,6),
                          np.broadcast_to(bank,(n,500,6))):
        raise ValueError("CR5 X differs from its own local 500-trial bank")
    manifest_path=dataset_path("CR5").with_name("s1_generate__manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError("Missing CR5 subject dataset manifest")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["bank_sha256"]!=sha256(_cr5_bank()) or
        manifest["dataset_sha256"]!=sha256(dataset_path("CR5"))):
        raise RuntimeError("CR5 local bank or subject dataset hash changed")
    p=np.asarray(data.predictive_true).reshape(n,500)
    check_ids=sorted({0,n//2,n-1})
    error=max(float(np.max(np.abs(
        expit(logits_3p(bank[None],data.parameters[s:s+1]))-p[s:s+1]
    ))) for s in check_ids)
    if error>1e-12:
        raise RuntimeError(f"CR5 generator fidelity error {error}")
    return bank,codes,names,design,error


# The old two-stream CR4 X generator is retained only in reference/cr4.
# The active standard S1 policy uses the same SeedSequence/one-RNG
# per-participant sampler as C1/C2/C3. Old CR4 X must be regenerated.
CR4_X_POLICY = "iid_uniform_single_inputs_seedsequence_v2"
CR4_X_LOW = np.array([5.0, 5.0, 0.05, 5.0, 5.0, 0.05])
CR4_X_HIGH = np.array([200.0, 200.0, 0.95, 200.0, 200.0, 0.95])
TRIALWISE_STREAM_NAMES = ("parameters", "trial_counts", "inputs", "observations")


class CR4TrialwiseModel(BinaryChoiceModel):
    """CR4 X via precisely the C1/C2/C3 trialwise generator contract."""
    model_id = "CR4"
    model_version = "x-v2"
    n_factors = 3
    parameter_names = ("lambda_loss", "gamma", "tau")
    n_input_features = 6
    trial_count_min = 500
    trial_count_max = 500
    n_subjects = 8000

    def __init__(self, n_subjects):
        self.n_subjects = int(n_subjects)

    def sample_parameters(self, n_subjects, rng):
        # Keep exact CR4/CR5 paired theta draws (only X-policy changes).
        return sample_theta(n_subjects, rng)

    def sample_inputs(self, n_trials, rng):
        return sample_independent_uniform_x(
            n_trials, rng, low=CR4_X_LOW, high=CR4_X_HIGH,
            n_features=self.n_input_features)

    def choice_probability(self, x, theta):
        return expit(logits_3p(np.asarray(x, dtype=float), theta))

    def sample_observations(self, x, theta, rng):
        # Preserve CR4/CR5 uniform-draw Bernoulli coupling. This is a
        # model response method; the X generation code is fully shared.
        p = self.choice_probability(x, theta)
        return (rng.random(p.shape) < p).astype(np.int64)


def cr4_rng_streams(seed):
    streams = make_rng_streams(seed, TRIALWISE_STREAM_NAMES)
    # Preserve CR4/CR5 theta coupling and observation stream. The X "inputs"
    # stream remains exactly the standard C1/C2/C3 SeedSequence child.
    streams["parameters"] = np.random.default_rng(seed + 4)
    streams["observations"] = np.random.default_rng(seed + 1004)
    return streams


class ModelSpec:
    def __init__(self, model_id):
        self.model_id = model_id
        self.classic = model_id in ("C1", "C2", "C3")
        self.model = get_model(model_id) if self.classic else None
        self.parameter_names = (tuple(self.model.parameter_names) if self.classic
                                else tuple(NAMES[4]) if model_id == "CR3"
                                else ("lambda_loss", "gamma", "tau"))
        self.x_dim = (int(self.model.n_input_features) if self.classic else 6)
        self.n_starts = 1 if self.classic else 4
        self.trial_range = ((300,500) if self.classic else (500,500))

    def _prepare_stimuli(self, args):
        # CR3 and CR5 independently generate their own banks in their own
        # directories using the same deterministic source recipe.
        # No dependency on CR2/CR3/CR5 output artifacts of another model.
        if self.model_id in ("CR3", "CR5"):
            ensure_local_bank(self.model_id, args.design_seed)

    def generate(self, args):
        """Return the one standard GeneratedDataset, without saving or printing."""
        if self.classic:
            model = get_model(self.model_id)
            model.n_subjects = args.subjects
            return generate_dataset(model, args.data_seed)
        if self.model_id == "CR4":
            model = CR4TrialwiseModel(args.subjects)
            return generate_dataset(
                model, args.data_seed,
                rng_streams=cr4_rng_streams(args.data_seed))
        self._prepare_stimuli(args)
        n = args.subjects
        rng_theta = np.random.default_rng(args.data_seed + 4)
        rng_choice = np.random.default_rng(args.data_seed + 1004)
        theta = (sample_prior(n,4,rng_theta) if self.model_id == "CR3"
                 else sample_theta(n,rng_theta))
        bank = _cr3_bank() if self.model_id == "CR3" else _cr5_bank()
        with np.load(bank,allow_pickle=False) as source:
            fixed = np.asarray(source["X"], dtype=np.float64)
        if fixed.shape != (500,6):
            raise ValueError("Expected a 500-by-6 immutable CPT stimulus bank")
        x3 = np.broadcast_to(fixed, (n,500,6))
        eta = (logits_cpt(x3,theta[:,None,:]) if self.model_id == "CR3"
               else logits_3p(x3,theta[:,None,:]))
        p = expit(eta)
        if not np.isfinite(p).all() or not ((p>0)&(p<1)).all():
            raise ValueError("Original model produced invalid probabilities")
        y = (rng_choice.random(p.shape) < p).astype(np.int64)
        counts = np.full(n,500,dtype=np.int64)
        return GeneratedDataset(
            parameters=np.asarray(theta),
            trial_counts=counts,
            subject_index=np.repeat(np.arange(n,dtype=np.int64),500),
            trial_index=np.tile(np.arange(500,dtype=np.int64),n),
            x=np.asarray(x3).reshape(-1,6).copy(),
            y=y.reshape(-1,1), predictive_true=p.reshape(-1,1))

    def manifest(self, args, data, saved_dataset):
        """Same named manifest file for all six; old provenance keys retained."""
        if self.classic:
            model = get_model(self.model_id)
            task = get_task_spec_for_model(model)
            return dict(model=model.metadata(), generator=model.manifest_extras(),
                        task_spec=task.__dict__, data_seed=args.data_seed,
                        subjects=len(data.parameters), dataset_sha256=sha256(saved_dataset))
        if self.model_id == "CR3":
            return dict(model="CR3_4P",subjects=args.subjects,trials_per_subject=500,
                        parameter_names=list(self.parameter_names),
                        data_seed=args.data_seed,bank_sha256=sha256(_cr3_bank()),
                        dataset_sha256=sha256(saved_dataset),
                        cr2_formula_and_priors_unchanged=True,
                        neural_network_trained=False)
        if self.model_id == "CR4":
            return dict(model="CR4",n_parameters=3,fixed_alpha=1.0,
                        subjects=args.subjects,trials_per_subject=500,
                        x_dim=6,x_shared=False,x_sampling_policy=CR4_X_POLICY,
                        x_rng_stream="SeedSequence(data_seed).spawn(4)[2]",
                        x_sampling_driver="generate_trialwise_dataset",
                        parameter_names=list(self.parameter_names),
                        data_seed=args.data_seed,
                        amount_range=list(DEFAULT_AMOUNT_RANGE),
                        probability_range=list(DEFAULT_PROB_RANGE),
                        dataset_sha256=sha256(saved_dataset),
                        formula="CR2/CR3 logits with alpha=1 exactly",
                        lambda_gamma_tau_priors="unchanged CR2/CR3 marginals",
                        decoder_x_preprocessing="theoretical uniform X moments")
        _,_,_,info = inspect_local_bank("CR5")
        return dict(model="CR5",fixed_alpha=1.0,subjects=args.subjects,
                    trials_per_subject=500,parameter_names=list(self.parameter_names),
                    data_seed=args.data_seed,bank_sha256=sha256(_cr5_bank()),
                    dataset_sha256=sha256(saved_dataset),
                    design_seed=info.get("seed",args.design_seed),
                    design_recipe=info.get("design_recipe","legacy_local_CR5_bank"),
                    independent_bank=True,
                    source_type_counts=info["source_condition_counts"],
                    generator_equals_CR4_3P=True,
                    same_subject_prior_seed_policy_as_CR4=True,
                    same_X_design_recipe_as_CR3=True,
                    trained_network=False)

    def validate_design(self, data):
        if self.classic:
            return {"sampling":"independent_per_trial", "passed":True}
        if self.model_id == "CR3":
            _verify_cr3(data)
            return {"sampling":"shared_immutable_CR3_500",
                    "bank_sha256":sha256(_cr3_bank()),"passed":True}
        if self.model_id == "CR4":
            _, manifest, error = _verify_cr4(data)
            return {"sampling":"iid_per_subject_per_trial",
                    "x_rng_stream":manifest["x_rng_stream"],"x_shared":False,
                    "max_formula_error":error,"passed":True}
        _, codes, names, info, error = _verify_cr5(data)
        return {"sampling":"independent_local_CR5_frozen_500","passed":True,
                "bank_sha256":sha256(_cr5_bank()),
                "condition_counts":info["source_condition_counts"],
                "max_formula_error":error}

    def validate_parameters_inputs(self, theta, x, counts):
        if not (np.all(counts>=self.trial_range[0]) and
                np.all(counts<=self.trial_range[1])):
            raise ValueError("Trial counts outside original model range")
        if self.classic:
            if not self.model.validate_parameters(theta):
                raise ValueError("True parameters violate model support")
            if not self.model.validate_inputs(x):
                raise ValueError("X violates model support")
        else:
            bounds = BOUNDS[4] if self.model_id=="CR3" else BOUNDS3
            if np.any(theta<bounds[:,0]) or np.any(theta>bounds[:,1]):
                raise ValueError("CPT parameters outside the prior support")

    def predict(self, x, theta):
        if self.classic:
            return np.asarray(self.model.choice_probability(x,theta)).reshape(-1)
        h = logits_cpt(x,theta) if self.model_id=="CR3" else logits_3p(x,theta)
        return expit(h).reshape(-1)

    def fit(self, x, y, rng):
        if self.classic:
            return np.asarray(self.model.oracle_fit(x,y),dtype=float)
        if self.model_id=="CR3":
            return np.asarray(fit_cpt(x,y,4,rng,starts=4),dtype=float)
        return np.asarray(fit_3p(x,y,4,rng),dtype=float)

    def fisher(self, x, theta, p):
        if self.classic:
            return np.asarray(self.model.fisher_information(x,theta,average=False))
        if self.model_id=="CR3":
            cols=[]
            for k in range(4):
                h=1e-5*max(1.,abs(float(theta[k])))
                hi,lo=theta.copy(),theta.copy()
                hi[k]+=h; lo[k]-=h
                cols.append((logits_cpt(x,hi)-logits_cpt(x,lo))/(2*h))
            jac=np.column_stack(cols)
        else:
            jac=gradient_3p(x,theta)
        return jac.T@((p*(1-p))[:,None]*jac)


def get_spec(model_id):
    if model_id not in ("C1","C2","C3","CR3","CR4","CR5"):
        raise ValueError(f"Unknown model {model_id}")
    return ModelSpec(model_id)
