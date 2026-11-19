from pathlib import Path
import sys
# Support direct invocation as: python np/sN_*.py ...
MODELS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELS_ROOT / 'src'))
"""S2: Common validity, functional fidelity and actual-S1-subject recovery."""
import json
import numpy as np
from modrot4.cli import parse_args
from modrot4.pipeline import (ROOT, dataset_path, precheck_path, sha256, _cr3_bank, _cr5_bank)
from modlearn_rot.generators.registry import get_model
from modlearn_rot.generators.dataset import load_dataset
from modrot4.unified_precheck import run_unified_precheck

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
    if manifest.get('x_sampling_policy') != 'iid_subject_trial_uniform_v1':
        raise RuntimeError('Legacy CR4 shared-bank dataset found; archive and regenerate IID-X CR4')
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
    audit_x = sample_subject_x(audit_n, seed=manifest['x_seed'],
                               amount_range=manifest['amount_range'],
                               prob_range=manifest['probability_range'])
    if not np.array_equal(x[:audit_n], audit_x):
        raise RuntimeError('CR4 stored X does not match deterministic IID RNG stream')
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
    """Verify exact CR3 trial bank/codes, 3P CR4 generator and strong provenance."""
    sys.path.insert(0,str(ROOT/'models'/'reference'/'cr5'))
    from s0_s1_cr5 import validate_immutable_bank,logits
    bank,codes,names,design=validate_immutable_bank()
    n=len(data.parameters)
    if data.parameters.shape != (n,3) or not np.all(np.asarray(data.trial_counts)==500):
        raise ValueError('CR5 must have exactly 3 parameters and 500 trials per subject')
    if data.x.shape != (n*500,6):
        raise ValueError('CR5 observed X dimension mismatch')
    if not np.array_equal(
        np.asarray(data.x).reshape(n,500,6),
        np.broadcast_to(bank,(n,500,6))
    ):
        raise RuntimeError('CR5 raw X does not equal CR3 bank in every subject')
    manifest_path=dataset_path('CR5').with_name('s1_generate__manifest.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['bank_sha256']!=sha256(_cr5_bank()) or
        manifest['dataset_sha256']!=sha256(dataset_path('CR5')) or
        manifest['cr3_source_sha256']!=design['cr3_source_sha256']):
        raise RuntimeError('CR5 dataset or CR3 bank provenance mismatch')
    from scipy.special import expit
    p=np.asarray(data.predictive_true).reshape(n,500)
    check_ids=sorted({0,n//2,n-1})
    error=max(float(np.max(np.abs(
        expit(logits(bank[None],data.parameters[s:s+1]))-p[s:s+1]
    ))) for s in check_ids)
    if error>1e-12:
        raise RuntimeError(f'CR5 3P CPT generator fidelity error {error}')
    return bank,codes,names,design,error


def precheck(args):
    path = dataset_path(args.model)
    if not path.is_file():
        raise FileNotFoundError(f"Generate first: {path}")
    data = load_dataset(path)
    provenance = {"stimulus_scheme": "iid_trialwise_uniform"}
    if args.model == "CR3":
        _verify_cr3(data)
        provenance = {"stimulus_scheme": "shared_frozen_CR3",
                      "bank_sha256": sha256(_cr3_bank()), "strict_dominance": 0}
    elif args.model == "CR4":
        _x, manifest, error = _verify_cr4(data)
        provenance = {"stimulus_scheme": "iid_subject_trial_uniform",
                      "x_shared": False, "x_seed": manifest["x_seed"],
                      "max_generator_error": error}
    elif args.model == "CR5":
        _x, codes, names, design, error = _verify_cr5(data)
        provenance = {"stimulus_scheme": "exact_shared_CR3",
                      "bank_sha256": sha256(_cr5_bank()),
                      "condition_counts": design["source_condition_counts"],
                      "max_generator_error": error}
    report = run_unified_precheck(
        args.model, data, seed=args.precheck_seed,
        provenance=provenance, data_path=path)
    if not report["qualified"]:
        raise RuntimeError(
            "Precheck failed: see " + str(precheck_path(args.model)) +
            ". Fewer than 30 S1 subjects means statistically inconclusive recovery.")
    return report


if __name__ == "__main__":
    precheck(parse_args("S2: unified model precheck"))
