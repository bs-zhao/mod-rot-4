"""W3-06 section 24: freeze ONE fixed blind algorithm before any independent confirmation truth is opened.

usage: python freeze_algorithm.py --candidate <finalist base id> --variant s9_2      -> results/algorithm_lock.json (hash-signed)
       python freeze_algorithm.py --verify                                            re-hash code / configs and compare with the lock
The lock records: the objective (metric / estimator / measure / variant / exact hyperparameters), observation channel rule, TRAIN-fit
preprocessing, integration design and search budgets, seed rule, tie handling, validity policy, the planned confirmation panels
(configs/confirmation_protocol.json), the recovery thresholds, the no-per-toy-selection and failure policies, and code / config / dependency
hashes.  Changing anything after a lock requires a new development version and a new confirmation panel.
"""
import sys, json, hashlib, argparse, platform, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import RES, CONFIGS, LAB, LAB08, LAB07, cfg, git_commit
from wave3 import jobs as JB


def file_hashes():
    fs = sorted((LAB / "wave3").glob("*.py")) + sorted((LAB08 / "wave2").glob("*.py")) + sorted((LAB07 / "wave4_2").glob("*.py")) + \
        [LAB / "run_search.py", LAB08 / "run_blind.py", LAB / "freeze_algorithm.py"] + sorted(CONFIGS.glob("*.json"))
    return {str(f.relative_to(LAB.parents[1])): hashlib.sha256(f.read_bytes()).hexdigest() for f in fs if f.name != "finalists.json" or True}


def lock(candidate, variant):
    reg = JB.registry(); import run_search as RS
    if candidate in RS.FUSIONS:
        obj = dict(type="fusion", fusion=RS.FUSIONS[candidate], components=cfg("fusion_candidates.json")["fusion"])
    else:
        e = reg[candidate]; obj = dict(type=e["spec"]["type"], spec=e["spec"], surface=e["surface"])
    import torch, numpy, scipy
    L = dict(version="W3-LOCK-1", created=time.strftime("%Y-%m-%d %H:%M:%S"), candidate=candidate, variant=variant, objective=obj,
             estimator="frozen S3 decoder, autograd Hessian (float64 CPU)" if obj.get("surface", "AD") == "AD" else obj.get("surface"),
             channel_rule=dict(B="class1 - class0 logit (Y-L)", C="decoder output in original Y units (Y-C)"),
             frame=dict(s9="raw Z", s9_2="marginal standardisation; confirmation uses the TRAIN-fit scaler applied unchanged to VALIDATION / TEST"),
             measure="Q01: TRAIN subjects x own observed trials, 512 subjects x 64 trials (fixed bank: primary mask MASK-ALL; random X: FULL)",
             integration_seed_rule="three independent designs [seed,1]/[seed,2,subject] with seeds 31 / 73 / 101 (all reported; no selection among them)",
             search=dict(budget="S0", k3_k4=[400, 8], k6=[2000, 30], identity_start=True, spans=[45, 15, 5, 1.5, 0.5], seeds=[31, 73, 101],
                         selection="lowest valid TRAIN loss; relative tie 1e-9 broken by the VALIDATION-cache loss; never by truth"),
             validity="non-finite / zero-energy objectives are invalid (NaN), never replaced", thresholds=dict(MCC_gap=.05, leak_max=dict(classic=.2, CR=.3)),
             policies=dict(per_toy_selection="forbidden: one fixed rule for every dataset", failures="every run reported; abstention counts as failure in coverage",
                           truth="confirmation T / R* opened only after every blind output of the panel is hashed into confirmation_locks.csv"),
             confirmation_protocol=cfg("confirmation_protocol.json"), git_commit=git_commit(), code_and_config_sha256=file_hashes(),
             dependencies=dict(python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__, scipy=scipy.__version__))
    body = json.dumps(L, indent=1, sort_keys=True); L["lock_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    (RES / "algorithm_lock.json").write_text(json.dumps(L, indent=1, sort_keys=True)); print("locked", L["lock_sha256"])


def select_final():
    """Section 23 final rule on the FIN 3 x 3 runs (development selection with T, recorded): one fixed (candidate, variant); rank by toys
    passing in all 9 runs, then worst normalised recovery margin min((.05 - gap)/.05, (thr - leak)/thr) over all runs (higher is better),
    then lexical id (compute / complexity: every finalist uses the same S0 search budget).  Support jobs of MX-SEL are not candidates."""
    import pandas as pd
    from wave3.common import leak_threshold
    fins = {f["base"]: f for f in json.loads((CONFIGS / "finalists.json").read_text())["finalists"]}
    br = pd.read_csv(RES / "finalist_recovery.csv"); br = br[(br.batch == "FIN") & (br.panel == "primary") & br.cid.isin([b for b, f in fins.items() if not f["role"].startswith("support")])].copy()
    br["thr"] = br.toy.map(leak_threshold); br["margin"] = np.minimum((.05 - br.MCC_gap) / .05, (br.thr - br.leak_max) / br.thr)
    g = br.groupby(["cid", "variant", "toy"]).agg(ok=("pass_lab05", "all"), n=("pass_lab05", "size")).reset_index()
    r = g.groupby(["cid", "variant"]).agg(toys=("ok", "sum"), runs_min=("n", "min")).join(br.groupby(["cid", "variant"]).margin.min().rename("worst_margin")).reset_index()
    r = r.sort_values(["toys", "worst_margin", "cid"], ascending=[False, False, True]); r.to_csv(RES / "final_selection.csv", index=False)
    print(r.head(10).to_string(index=False)); best = r.iloc[0]
    return best.cid, best.variant


def verify():
    L = json.loads((RES / "algorithm_lock.json").read_text()); now = file_hashes()
    diff = [f for f, h in L["code_and_config_sha256"].items() if now.get(f) != h]
    print("lock", L["lock_sha256"], "candidate", L["candidate"], L["variant"]); print("changed files:", diff or "none")
    return not diff


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--candidate"); ap.add_argument("--variant", default="s9_2"); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--auto", action="store_true", help="apply the section 23 final rule to results/finalist_recovery.csv"); a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify() else 1)
    if a.auto:
        a.candidate, a.variant = select_final(); print("selected", a.candidate, a.variant)
    lock(a.candidate, a.variant)
