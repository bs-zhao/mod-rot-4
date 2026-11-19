
"""Stand-alone, per-model fixed-X design builder for CR3 and CR5.

Each model owns its own output folder and constructs its bank from the exact
original CR2-4P optimization + CR3 conditional-bank recipe. No reads or writes
in another model's output tree. With the same design seed, CR3 and CR5 use
the same deterministic recipe; they need not be run in any particular order.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

from modrot4.pipeline import ROOT, _cr3_bank, _cr5_bank, sha256
from modlearn_rot.utils.io import save_json

REF=ROOT/"models"/"reference"
for folder in (REF,REF/"cr3",REF/"cr4"):
    if str(folder) not in sys.path:
        sys.path.insert(0,str(folder))

from s0_cr2_design_recovery import (  # noqa: E402
    FAMILIES, NAMES, make_candidates, sample_prior, fisher_and_effects,
    design as cr2_design, refine, metrics, logits
)
from s0_s1_cr3 import (  # noqa: E402
    COUNTS, make_bank, condition_counts, dominance_audit
)
from s0_s1_cr4 import X_NAMES, PARAMETER_NAMES  # noqa: E402

CR2_RECIPE = dict(candidate_count=6000,reference_count=48,
                  shortlist=96,swaps=600,balance_weight=0.18)
RECIPE_ID = "independent_cr2_4p_optimizer_plus_cr3_500_conditioning_v1"


def design_path(model_id):
    if model_id=="CR3":
        return _cr3_bank().parent/"CR3_4P_design.json"
    if model_id=="CR5":
        return _cr5_bank().parent/"CR5_3P_design.json"
    raise ValueError("Only CR3 and CR5 require this fixed-X design")


def bank_path(model_id):
    if model_id=="CR3":
        return _cr3_bank()
    if model_id=="CR5":
        return _cr5_bank()
    raise ValueError(model_id)


def _sha_of_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _optimise_4p_source(seed):
    """Exactly the upstream CR2-4P default design recipe, *in memory*."""
    rng=np.random.default_rng(int(seed)+4)
    candidates,families=make_candidates(CR2_RECIPE["candidate_count"],rng)
    references=sample_prior(CR2_RECIPE["reference_count"],4,rng)
    info,effects,_=fisher_and_effects(candidates,references)
    selected=cr2_design(candidates,families,info,effects,500,rng,
                        CR2_RECIPE["shortlist"],CR2_RECIPE["balance_weight"])
    selected,score,accepted=refine(selected,info,effects,rng,
                                  CR2_RECIPE["swaps"],CR2_RECIPE["balance_weight"])
    return candidates[selected], dict(
        objective=float(score),swaps_accepted=int(accepted),
        family_counts={family:int(np.sum(families[selected]==j))
                       for j,family in enumerate(FAMILIES)})


def inspect_local_bank(model_id):
    """Validate only this model's own bank and metadata, never other outputs."""
    path=bank_path(model_id)
    info_path=design_path(model_id)
    if not path.exists():
        raise FileNotFoundError(f"{model_id} bank missing: {path}")
    if not info_path.exists():
        raise FileNotFoundError(f"{model_id} local bank manifest missing: {info_path}")
    with np.load(path,allow_pickle=False) as stored:
        x=np.asarray(stored["X"],np.float64)
        codes=np.asarray(stored["type_codes"],np.int64)
        names=np.asarray(stored["type_names"])
        params=np.asarray(stored["parameter_names"])
    if x.shape!=(500,6) or codes.shape!=(500,) or len(names)!=len(COUNTS):
        raise ValueError(f"{model_id} local bank geometry is invalid")
    if not np.array_equal(names,np.asarray(list(COUNTS))):
        raise ValueError(f"{model_id} bank trial-type labels differ")
    if not np.array_equal(np.bincount(codes,minlength=len(names)),
                          np.asarray(list(COUNTS.values()))):
        raise ValueError(f"{model_id} trial-type counts differ")
    if model_id=="CR3":
        expected_params=np.asarray(NAMES[4])
    else:
        expected_params=np.asarray(PARAMETER_NAMES)
    if not np.array_equal(params,expected_params):
        raise ValueError(f"{model_id} bank parameter labels differ")
    if dominance_audit(x)["total"]!=0:
        raise ValueError(f"{model_id} bank includes strictly dominant trials")
    counts=condition_counts(x)
    expected=dict(gain_only=60,gamma_neutral=60,loss_only=40,general=340)
    if any(counts[k]!=v for k,v in expected.items()):
        raise ValueError(f"{model_id} stimulus composition changed: {counts}")
    info=json.loads(info_path.read_text(encoding="utf-8"))
    own_hash=info.get("new_bank_sha256") if model_id=="CR3" else info.get("cr5_bank_sha256")
    if own_hash!=sha256(path):
        raise RuntimeError(f"{model_id} bank has changed since its local design")
    # Historical CR5 design metadata may mention CR3 as provenance,
    # but NEVER read CR3 assets or require them at runtime.
    return x,codes,names,info


def ensure_local_bank(model_id,seed):
    """Prepare the requested model's own bank without another model's outputs."""
    path=bank_path(model_id)
    meta=design_path(model_id)
    if path.exists():
        inspect_local_bank(model_id)
        return path
    if meta.exists():
        raise RuntimeError(f"{model_id} metadata exists without bank: {meta}. "
                           "Archive the incomplete model folder before retrying.")
    # Expensive CR2 design optimization runs once per model, locally/in-memory.
    source,source_meta=_optimise_4p_source(seed)
    x,codes=make_bank(source,seed)
    if dominance_audit(x)["total"]!=0:
        raise AssertionError("Original CR3 fixed-X design violates FOSD requirement")
    # Original CR3 probability model sanity check.
    rng=np.random.default_rng(int(seed)+111)
    theta=sample_prior(8,4,rng)
    probabilities=expit(logits(x[None,:,:],theta[:,None,:]))
    if not np.isfinite(probabilities).all() or not ((probabilities>0)&(probabilities<1)).all():
        raise AssertionError("Designed bank produces invalid CPT probabilities")
    path.parent.mkdir(parents=True,exist_ok=True)
    contents=dict(X=x,type_codes=codes,
                  type_names=np.asarray(list(COUNTS)),
                  parameter_names=np.asarray(
                      NAMES[4] if model_id=="CR3" else PARAMETER_NAMES))
    if model_id=="CR5":
        contents["x_names"]=np.asarray(X_NAMES)
    np.savez_compressed(path,**contents)
    check=condition_counts(x)
    common=dict(
        model=model_id,seed=int(seed),trial_count=500,
        design_recipe=RECIPE_ID,
        source_cr2_design_algorithm="s0_cr2_design_recovery.py",
        source_cr3_conditioning_algorithm="s0_s1_cr3.make_bank",
        independent_output=True,
        other_model_outputs_used=False,
        cr2_optimizer_settings=CR2_RECIPE,
        cr2_optimizer_result=source_meta,
        optimized_source_x_sha256=_sha_of_array(source),
        x_sha256=_sha_of_array(x),
        source_condition_counts={
            name:int(np.sum(codes==i)) for i,name in enumerate(COUNTS)},
        measured_counts=check,
        strict_first_order_dominance_forbidden=True,
        same_algorithm_and_seed_as_CR3=True,
    )
    if model_id=="CR3":
        common.update(
            parameter_names=list(NAMES[4]),
            new_bank_sha256=sha256(path))
    else:
        common.update(
            fixed_alpha=1.0,parameter_names=list(PARAMETER_NAMES),
            cr5_bank_sha256=sha256(path),
            original_bank_X_and_types_exact=True,
            formula="CR3 4P CPT with alpha=1",
            source_model="independent_CR3_formula_not_CR3_output")
    save_json(common,meta)
    inspect_local_bank(model_id)
    return path
