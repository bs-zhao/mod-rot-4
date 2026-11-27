"""Paths, frozen constants and read-only adapters for Lab 09 (wave 4-2 third wave, 03_third_wave_README.md).

Lab 09 reuses Lab 08's validated blind payload, designs, moment caches, objectives, optimizer and recovery evaluator READ-ONLY
(imported as the `wave2` package) and, through Lab 08, the Lab 07 / Lab 06 / Lab 03 numerical modules.  New arrays go to
outputs/metric_lab/09_wave4_2_third_wave/<RUN_ID>/ (never overwritten; every runner resumes by skipping finished files).
Truth-enabled modules live in wave3/eval_only/ and are imported only by evaluation entry points (tests/test_w3.py enforces this).
"""
import json, os, sys, hashlib, subprocess
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[1]
LAB08 = ROOT / "metric_lab" / "08_wave4_2_second_wave"
LAB07 = ROOT / "metric_lab" / "07_unified_benchmark"
OUT07 = ROOT / "outputs" / "metric_lab" / "07_unified_benchmark"
OUT08 = ROOT / "outputs" / "metric_lab" / "08_wave4_2_second_wave" / "W2r1"
RUN_ID = "W3r1"
OUT = ROOT / "outputs" / "metric_lab" / "09_wave4_2_third_wave" / RUN_ID
RES = LAB / "results"
CONFIGS = LAB / "configs"
if str(LAB08) not in sys.path:
    sys.path.insert(0, str(LAB08))
import wave2.common as W2C                              # noqa: E402  (sets the Lab 07 / 06 / 03 import paths)

PRIMARY = W2C.PRIMARY
SECONDARY = W2C.SECONDARY
VARIANTS = W2C.VARIANTS
SEEDS = (31, 73, 101)
FIDELITY = W2C.FIDELITY
toy_entry, family_group, leak_threshold, sha256, git_commit = W2C.toy_entry, W2C.family_group, W2C.leak_threshold, W2C.sha256, W2C.git_commit


def cfg(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def chan(pl):
    return "YC" if pl.cls == "C" else "YL"


def prim_mask(pl):
    from wave2.objectives import masks_of
    return "MASK-ALL" if masks_of(pl) == ("MASK-ALL", "MASK-COL") else "FULL"


def code_hash(extra=()):
    h = hashlib.sha256()
    files = sorted((LAB / "wave3").glob("*.py")) + sorted((LAB08 / "wave2").glob("*.py")) + sorted((LAB07 / "wave4_2").glob("*.py")) + list(extra)
    for f in files:
        h.update(f.name.encode()); h.update(Path(f).read_bytes())
    return h.hexdigest()[:16]


def atomic_savez(path, **arr):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp.npz")
    import numpy as np
    np.savez_compressed(tmp, **arr); tmp.replace(path)


def jdump(obj):
    import numpy as np
    return np.array(json.dumps(obj, default=float))
