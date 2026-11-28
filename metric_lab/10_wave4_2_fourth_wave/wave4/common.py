"""Paths, frozen constants and read-only adapters for Lab 10 (wave 4-2 fourth wave, 04_fourth_wave_README.md).

Lab 10 reuses READ-ONLY: Lab 09 `wave3` (RW-GL row metric, objective builder, job plumbing, evaluator), Lab 08 `wave2` (blind payload,
historical designs Q01-Q05, AD moment accumulation, S0 / S1 optimizer, T-U recovery) and, through them, Lab 07 `wave4_2`.
Nothing in Lab 01-09 code, outputs or locks is written.  New arrays: outputs/metric_lab/10_wave4_2_fourth_wave/<RUN_ID>/.
Truth-enabled modules live in wave4/eval_only/ and are imported only by evaluation entry points (tests/test_w4.py enforces this).
"""
import json, os, sys, hashlib
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[1]
LAB09 = ROOT / "metric_lab" / "09_wave4_2_third_wave"
if str(LAB09) not in sys.path:
    sys.path.insert(0, str(LAB09))
import wave3.common as W3C                                  # noqa: E402  (sets the Lab 08 / 07 / 06 / 03 import paths)

LAB08, LAB07 = W3C.LAB08, W3C.LAB07
OUT08, OUT09 = W3C.OUT08, W3C.OUT
RUN_ID = "W4r1"
OUT = ROOT / "outputs" / "metric_lab" / "10_wave4_2_fourth_wave" / RUN_ID
RES = LAB / "results"
CONFIGS = LAB / "configs"
PRIMARY, SECONDARY, VARIANTS = W3C.PRIMARY, W3C.SECONDARY, W3C.VARIANTS
FIDELITY = W3C.FIDELITY
toy_entry, family_group, leak_threshold, sha256, git_commit, chan, prim_mask = (W3C.toy_entry, W3C.family_group, W3C.leak_threshold, W3C.sha256,
                                                                                W3C.git_commit, W3C.chan, W3C.prim_mask)
atomic_savez = W3C.atomic_savez

CAND = "RW-GL-l0.75-a1.0"                      # W4-Q05 objective (the W3 implementation, unchanged)
E02 = "MX-B08c03@AD"                            # historical E02 = L06|B08.M03.c0.03
B01 = "MX-B01c03@AD"                            # global-support control
SSEEDS = (31, 73, 101)


def cfg(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def spec_of(cid):
    from wave3 import jobs as JB
    return JB.registry()[cid]["spec"]


def code_files():
    return sorted((LAB / "wave4").glob("*.py")) + sorted((LAB09 / "wave3").glob("*.py")) + sorted((LAB08 / "wave2").glob("*.py")) + \
        sorted((LAB07 / "wave4_2").glob("*.py"))


def code_hash():
    h = hashlib.sha256()
    for f in code_files():
        h.update(f.name.encode()); h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def jdump(obj):
    import numpy as np
    return np.array(json.dumps(obj, default=float))
