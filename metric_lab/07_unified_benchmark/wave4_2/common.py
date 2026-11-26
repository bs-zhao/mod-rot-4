"""Paths, registries and import shims shared by every W1 module (no truth / no data here)."""
import json, os, sys, hashlib, subprocess
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[1]
LAB03 = ROOT / "metric_lab" / "03_s9_metric_curves"
LAB06 = ROOT / "metric_lab" / "06_wave1_screen"
NNOUT = ROOT / "outputs" / "nn"
OUT = ROOT / "outputs" / "metric_lab" / "07_unified_benchmark"
RES = LAB / "results"
CONFIGS = LAB / "configs"
# read-only reuse of the validated Lab 03 loaders / metric code and the Lab 06 metric library (no copies)
for _p in (LAB03, LAB06, ROOT / "models" / "src", ROOT / "nn" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

VARIANTS = ("s9", "s9_2")
FIDELITY = {           # (subjects, trials per subject, latent draws for a full fixed bank)
    "pilot": (256, 32, 32),
    "C": (512, 64, 128),
    "D": (1024, 128, 512),
}
SEEDS = (31, 73, 101)
MEASURES = ("Q01", "Q02", "Q03", "Q04", "Q05")


def registry():
    return json.loads((CONFIGS / "toy_registry.json").read_text(encoding="utf-8"))


def libraries(panel="primary"):
    """-> list of dict(toy, model, class, family, run_dir, beta, k, p); one entry per NP run (both variants inside)."""
    reg = registry(); out = []
    for name, e in reg[panel].items():
        out.append(dict(toy=name, model=e.get("model", name), cls=e["class"], family=e["family"], run_dir=NNOUT / e["run"],
                        run=e["run"], beta=e["beta"], k=e["k"], p=e["p"], panel=panel))
    return out


def library(toy):
    for panel in ("primary", "secondary"):
        for e in libraries(panel):
            if e["toy"] == toy:
                return e
    raise KeyError(toy)


def channels(cls):
    return tuple(registry()["channels"][cls])


def x_domain(family):
    d = registry()["x_domain"][family]
    return None if d == "unbounded" else list(d)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def code_hash():
    """hash of every .py file of this lab plus the reused Lab 03/06 metric code (versioning of the registries)."""
    h = hashlib.sha256()
    files = sorted((LAB / "wave4_2").glob("*.py")) + [LAB03 / "s9curves" / "metrics152.py", LAB06 / "wave1" / "library.py"]
    for f in files:
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()[:16]
