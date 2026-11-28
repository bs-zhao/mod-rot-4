"""W4-05 section 18: measure-aware algorithm lock W4-Q05-v1 (new schema; the W3 Q01 lock in Lab 09 is never touched).

usage: python freeze_algorithm.py --candidate W4-Q05-v1 --deployment D-ranked|fixed-pair  -> results/algorithm_lock_W4.json (refuses to overwrite)
       python freeze_algorithm.py --verify      re-hash code / configs (canonical LF text) and the W3 lock; exit 1 on any change
Hashes cover wave4/*.py, the reused Lab 09 wave3 / Lab 08 wave2 / Lab 07 wave4_2 modules, Lab 08 run_blind.py, the Lab 10 runners used by the
locked path and every config.  Canonical text hashes (CRLF -> LF) are used so a Git line-ending conversion cannot fake a change; the raw
byte hash of the W3 lock is recorded separately.
"""
import sys, json, hashlib, argparse, platform, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4.common import RES, CONFIGS, LAB, LAB08, LAB09, ROOT, CAND, E02, code_files, git_commit, spec_of, sha256, cfg, OUT

LOCKF = RES / "algorithm_lock_W4.json"


def canon(f):
    return hashlib.sha256(Path(f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


ALGO_W4 = ("common.py", "designs.py", "moments.py", "search.py")


def files(kind="algorithm"):
    """algorithm: everything the locked blind rule executes (a change fails --verify); orchestration: confirmation adapters, evaluation and
    reporting code (recorded; a later change is listed by --verify as a logged deviation, it cannot alter a blind output)"""
    cf = code_files(); w4 = LAB / "wave4"
    algo = [f for f in cf if f.parent != w4 or f.name in ALGO_W4] + [LAB08 / "run_blind.py", LAB / "run_replication.py"] +         [CONFIGS / n for n in ("primary_candidate.json", "replication_protocol.json", "confirmation_protocol.json")]
    if kind == "orchestration":
        fs = [f for f in cf if f.parent == w4 and f.name not in ALGO_W4] + sorted((w4 / "eval_only").glob("*.py")) +             [LAB / n for n in ("run_confirmation.py", "freeze_algorithm.py", "run_evaluation.py")]
    else:
        fs = algo
    return {str(f.relative_to(ROOT)).replace("\\", "/"): canon(f) for f in fs}


def lock(candidate, deployment):
    if LOCKF.is_file():
        raise SystemExit(f"{LOCKF} exists: a lock is immutable (new development version needed)")
    import torch, numpy, scipy
    man = OUT / "confirmation" / "confirmation_manifest.json"
    L = dict(version=candidate, schema="W4-LOCK-1", created=time.strftime("%Y-%m-%d %H:%M:%S"),
             development_selection=("Q05 was chosen after inspecting the W3 FIN-SENS recovery (post-hoc, integration seed 31) and the W4 development "
                                    "replications REP-1/2/3 on the same 12 toys; this lock is a DEVELOPMENT selection, not a confirmed result"),
             objective=dict(cid=CAND, spec=spec_of(CAND), formula="L = .75 sum_{i in A} (r_i / sum_A r) s_i + .25 (1/kp) sum_ij v_ij / (v_ij + c/(kp)); s_i = (1/p) sum_j q_ij / (q_ij + c/p); c = .03, a = 1",
                            active_rule="A = rows with r_i > 1e-15 E (TOL 1e-12 x ACT_CUT 1e-3); E <= 1e-12 or non-finite -> NaN (invalid, never replaced)", dtype="float64",
                            block="UX only: B_ij(R) = e_i' R S J^(j) S R' e_i, J^(j) = E[h_j h_j'], h_j = d2 g / dv dx_j (frozen S3 decoder, torch.func AD)"),
             channel=dict(B="class-1 minus class-0 logit (YL)", C="decoder output * y_std + y_mean (YC, original Y units)"),
             frame=dict(primary="s9 raw Z (V = Z)", secondary="s9_2 marginal standardisation (all-subject mean / SD, ddof 0) - reported separately, not the primary rule"),
             measure=dict(id="Q05", rule=("independent marginal Gaussian fitted to the TRAIN pool Z of the dataset itself (mean, population SD ddof 0; recomputed for every "
                                          "dataset - not the historical numbers), standard normals RNG [iseed,3,5] (MAXDRAW 4096), X: complete observed TRAIN rows, "
                                          "permutation RNG [iseed,4,5] (random designs) or the full fixed bank for every draw; the SAME base points for every R"),
                          mask="FULL for random X; MASK-ALL (every X column interior: amount > 0, 0 < prob < 1) for declared-domain fixed banks",
                          fidelity="C: 512 draws x 64 rows (random) / 128 draws x full bank", validation="same law on the VALIDATION pool (numerical tie-break only)"),
             integration=dict(iseeds=[31, 73, 101], note="all-run assessment = 3 integration x 3 search seeds, every run reported"),
             search=dict(optimizer="Lab 08 wave2.optimizers.search", budget="S0", k3_k4=[400, 8], k6=[2000, 30], identity_start=True, spans=[45, 15, 5, 1.5, 0.5],
                         sseeds=[31, 73, 101], tie="lowest valid TRAIN loss; relative 1e-9 tie broken by the VALIDATION loss; never by truth",
                         S1="5x budget only as a reported blind escalation; never replaces the S0 selection"),
             deployment=dict(primary=deployment, d_seed=211, fixed_pair=[31, 31],
                             rules={"D-ranked": "among the 9 selected S0 rotations of the variant, the one with the lowest loss on an independent Q05 D cache (1024 x 128 / 512 x full bank, TRAIN pool, iseed 211); ties -> lexical key",
                                    "fixed-pair": "the S0 selection of integration seed 31 x search seed 31"},
                             note="both rules declared before confirmation; the primary is the one named in `primary`; the all-run 3 x 3 assessment is reported in addition"),
             controls=dict(W3_Q01=dict(cid=CAND, measure="Q01"), E02=dict(cid=E02, measure="Q01", variant="s9_2")),
             evaluation=dict(criterion="MCC >= supervised MCC - .05 and leak_max < .2 (classic) / .3 (CR); Hungarian on |corr| on TEST",
                             supervised="S8 / S8_2 TRAIN-fit Procrustes R* (TRAIN-centred Z, TRAIN-standardised T), evaluated on TEST",
                             sensitivity=["gap .02", "common leak .2 / .3", "absolute MCC >= .9"], stage_A="linear R2 of T on Z (TRAIN OLS, TEST), descriptive only"),
             policies=dict(per_toy_selection="forbidden", failures="every attempted run in the denominator; missing / failed training = failure; no regeneration",
                           truth="confirmation T / R* opened only after every blind CONF-A file is hashed into results/confirmation_locks.csv"),
             w3_lock=dict(path="metric_lab/09_wave4_2_third_wave/results/algorithm_lock.json", byte_sha256=sha256(LAB09 / "results" / "algorithm_lock.json"),
                          canonical_sha256=canon(LAB09 / "results" / "algorithm_lock.json"), status="untouched"),
             confirmation_manifest=dict(path=str(man.relative_to(ROOT)).replace("\\", "/"), sha256=canon(man) if man.is_file() else None),
             data=dict(primary_payloads="outputs/metric_lab/08_wave4_2_second_wave/W2r1/blind_payload (development)", toy_registry="metric_lab/07_unified_benchmark/configs/toy_registry.json"),
             git_commit=git_commit(), code_and_config_sha256_canonical=files(), orchestration_sha256_canonical=files("orchestration"),
             dependencies=dict(python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__, scipy=scipy.__version__))
    body = json.dumps(L, indent=1, sort_keys=True); L["lock_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    LOCKF.write_text(json.dumps(L, indent=1, sort_keys=True)); print("locked", candidate, L["lock_sha256"])


def verify():
    L = json.loads(LOCKF.read_text()); now = files()
    diff = [f for f, h in L["code_and_config_sha256_canonical"].items() if now.get(f) != h]
    w3 = canon(LAB09 / "results" / "algorithm_lock.json") == L["w3_lock"]["canonical_sha256"]
    oc = files("orchestration"); od = [f for f, h in L.get("orchestration_sha256_canonical", {}).items() if oc.get(f) != h]
    print("orchestration / evaluation files changed since the lock (logged deviation, not part of the blind rule):", od or "none")
    print("lock", L["lock_sha256"], L["version"], "deployment", L["deployment"]["primary"]); print("changed files:", diff or "none", "| W3 lock unchanged:", w3)
    return not diff and w3


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--candidate", default="W4-Q05-v1"); ap.add_argument("--deployment", choices=["D-ranked", "fixed-pair"])
    ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify() else 1)
    lock(a.candidate, a.deployment)
