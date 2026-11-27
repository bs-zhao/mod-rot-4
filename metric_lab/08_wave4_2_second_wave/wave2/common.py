"""Paths, frozen constants and versioned read-only adapters to Lab 07 (wave 4-2 second wave).

Lab 08 imports the validated Lab 07 numerical modules read-only (moments / metrics / derivatives / measures / surfaces), as
02_second_wave_README.md section 6 asks ("reuse Lab 07's validated contraction and metric adapters read-only").  Truth-enabled
Lab 07 modules (truth, bridge, inputs.load, global_oracle, run_global_oracle) are imported ONLY by the evaluation-side modules
(evaluate_recovery, oracle_diagnostics, the analytic arms of conditional / inference); tests/test_w2.py enforces this split.
"""
import json, os, sys, hashlib, subprocess
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[1]
LAB07 = ROOT / "metric_lab" / "07_unified_benchmark"
LAB03 = ROOT / "metric_lab" / "03_s9_metric_curves"
LAB06 = ROOT / "metric_lab" / "06_wave1_screen"
NNOUT = ROOT / "outputs" / "nn"
OUT07 = ROOT / "outputs" / "metric_lab" / "07_unified_benchmark"
RUN_ID = "W2r1"                                      # immutable run id of this second wave
OUT = ROOT / "outputs" / "metric_lab" / "08_wave4_2_second_wave" / RUN_ID
RES = LAB / "results"
CONFIGS = LAB / "configs"
for _p in (LAB07, LAB03, LAB06, ROOT / "models" / "src", ROOT / "nn" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FIDELITY = {"pilot": (256, 32, 32), "C": (512, 64, 128), "D": (1024, 128, 512)}   # identical to Lab 07
VARIANTS = ("s9", "s9_2")
PRIMARY = ("C1", "C2", "C3", "CR3", "CR4", "CR5", "C1L", "C2L", "C3L", "CR3L", "CR4L", "CR5L")
SECONDARY = ("C1#r2", "C2#r2", "C3#r2", "CR3#b5e-5", "CR4#r2")


def cfg(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def toy_registry():
    return json.loads((LAB07 / "configs" / "toy_registry.json").read_text(encoding="utf-8"))


def toy_entry(toy):
    reg = toy_registry()
    for panel in ("primary", "secondary"):
        if toy in reg[panel]:
            e = dict(reg[panel][toy]); e["panel"] = panel; e["toy"] = toy; e["run_dir"] = NNOUT / e["run"]; return e
    raise KeyError(toy)


def family_group(toy):
    return "CR" if toy.startswith("CR") else "C"


def leak_threshold(toy):
    return 0.3 if family_group(toy) == "CR" else 0.2


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for d in iter(lambda: f.read(1 << 20), b""):
            h.update(d)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def code_hash(extra=()):
    h = hashlib.sha256()
    files = sorted((LAB / "wave2").glob("*.py")) + sorted((LAB07 / "wave4_2").glob("*.py")) + \
        [LAB03 / "s9curves" / "metrics152.py", LAB06 / "wave1" / "library.py"] + list(extra)
    for f in files:
        h.update(f.name.encode()); h.update(Path(f).read_bytes())
    return h.hexdigest()[:16]
