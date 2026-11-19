"""S1/S2 paths and subprocess helpers, independent of NP/torch.

All generated outputs use the new model-stage-first layout:
outputs/models/generate/<MODEL>/ and outputs/models/precheck/<MODEL>/.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
MODELS = ("C1", "C2", "C3", "CR3", "CR4", "CR5")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def out_dir(model: str) -> Path:
    if model not in MODELS and model != "CR2":
        raise ValueError(f"Unknown model: {model}")
    return ROOT / "outputs" / "models" / "generate" / model


def dataset_path(model: str) -> Path:
    return out_dir(model) / "s1_generate__dataset.npz"


def precheck_path(model: str) -> Path:
    return ROOT / "outputs" / "models" / "precheck" / model / "s2_precheck__report.json"


def _run_legacy(script: Path, argv: list[str]) -> None:
    if not script.is_file():
        raise FileNotFoundError(f"Missing frozen reference script: {script}")
    cmd = [sys.executable, str(script), *argv]
    print("> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _cr3_refs() -> tuple[Path, Path]:
    reference = ROOT / "models" / "reference"
    return (reference / "s0_cr2_design_recovery.py",
            reference / "cr3" / "s0_s1_cr3.py")


def _cr2_bank() -> Path:
    return out_dir("CR2") / "s0_design_recovery" / "CR2_4P_seed20261023_fixed500.npz"


def _cr3_bank() -> Path:
    return out_dir("CR3") / "s0_design" / "CR3_4P_fixed500.npz"


def _cr4_bank() -> Path:
    # Legacy name retained for compatibility; current CR4 has no shared X bank.
    return out_dir("CR4") / "s0_design" / "CR4_3P_random500.npz"


def _cr4_ref() -> Path:
    return ROOT / "models" / "reference" / "cr4" / "s0_s1_cr4.py"


def _cr4_oracle_ref() -> Path:
    return ROOT / "models" / "reference" / "cr4" / "s2_oracle_cr4.py"


def _cr5_bank() -> Path:
    return out_dir("CR5") / "s0_design" / "CR5_3P_cr3bank500.npz"


def _cr5_ref() -> Path:
    return ROOT / "models" / "reference" / "cr5" / "s0_s1_cr5.py"


def _cr5_oracle_ref() -> Path:
    return ROOT / "models" / "reference" / "cr5" / "s2_oracle_cr5.py"
