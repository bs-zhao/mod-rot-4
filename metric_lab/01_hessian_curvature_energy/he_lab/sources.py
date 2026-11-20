"""Read-only adapters to mod-rot-4 S1/S3/S5/S8/S9 artifacts."""
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .coordinates import Rotation

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "models" / "src", ROOT / "nn" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from modrot4.model_specs import get_spec  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


@dataclass
class Sources:
    model: str
    run_dir: Path
    dataset_file: Path
    checkpoint_file: Path
    latent_file: Path
    parameters: np.ndarray
    x: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    counts: np.ndarray
    true_choice: np.ndarray
    z: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    splits: dict

    @property
    def x_dim(self):
        return self.x.shape[1]

    @property
    def latent_dim(self):
        return self.z.shape[1]

    @property
    def fixed_x(self):
        if self.model not in ("CR3", "CR5"):
            return None
        bank = self.x[:500]
        if not np.array_equal(self.trial_ids[:500], np.arange(500)):
            raise ValueError("Fixed bank trial indexing differs from S1 contract")
        # Spot-check that other subjects share exactly this 500-by-6 design.
        for s in (0, len(self.counts)//2, len(self.counts)-1):
            offset = int(self.counts[:s].sum())
            if not np.array_equal(self.x[offset:offset+500], bank):
                raise ValueError("S1 fixed-bank subjects have different X")
        return bank


def load_sources(model, run_dir):
    import torch
    run_dir = Path(run_dir).resolve()
    dataset = ROOT / "outputs" / "models" / "generate" / model / "s1_generate__dataset.npz"
    ckpt = run_dir / "s3_train" / "checkpoint.pt"
    latent = run_dir / "s5_extract_z" / "latent.npz"
    for f in (dataset, ckpt, latent):
        if not f.is_file():
            raise FileNotFoundError(f"Required upstream artifact missing: {f}")
    with np.load(dataset, allow_pickle=False) as arr:
        t = np.asarray(arr["parameters"], dtype=np.float64)
        x = np.asarray(arr["x"], dtype=np.float64)
        sid = np.asarray(arr["subject_index"], dtype=np.int64)
        tid = np.asarray(arr["trial_index"], dtype=np.int64)
        counts = np.asarray(arr["trial_counts"], dtype=np.int64)
        y = np.asarray(arr["choice"], dtype=np.int8)
    with np.load(latent, allow_pickle=False) as arr:
        z = np.asarray(arr["z"], dtype=np.float64)
        if str(arr["dataset_sha256"]) != sha256(dataset):
            raise ValueError("S5 dataset SHA differs from S1")
        if str(arr["checkpoint_sha256"]) != sha256(ckpt):
            raise ValueError("S5 checkpoint SHA differs from S3")
        if str(arr["context_mode"]) != "all_observed_trials":
            raise ValueError("S5 z must derive from all observed trials")
        splits = {k: np.asarray(arr[k + "_ids"], dtype=int)
                  for k in ("train", "validation", "test")}
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    extra = state.get("extra_state", {})
    standardization = extra.get("input_standardization")
    if not isinstance(standardization, dict):
        raise ValueError("S3 checkpoint missing input_standardization")
    xm = np.asarray(standardization["x_mean"], dtype=float)
    xs = np.asarray(standardization["x_std"], dtype=float)
    if not np.all(xs > 0) or x.shape[1] != len(xs):
        raise ValueError("Invalid S3 X scaler")
    if z.shape[0] != len(t) or len(counts) != len(t) or z.ndim != 2:
        raise ValueError("Subject dimensions do not match")
    if not np.array_equal(sid, np.repeat(np.arange(len(counts)), counts)):
        raise ValueError("S1 subject IDs are not grouped")
    return Sources(model, run_dir, dataset, ckpt, latent,
                   t, x, sid, tid, counts, y, z, xm, xs, splits)


def load_rotations(src, variant):
    stage = "s9_rotation_sweep" if variant == "s9" else "s9_2_rotation_sweep"
    base = src.run_dir / stage
    index = base / "rotation_index.csv"
    if not index.exists():
        raise FileNotFoundError(f"Run NN {stage} first: {index}")
    s8 = ("s8_supervised_rotation" if variant == "s9"
          else "s8_2_supervised_rotation")
    upstream = src.run_dir / s8 / "rotation.npz"
    rotations = []
    with np.load(upstream, allow_pickle=False) as f:
        t_mean = np.asarray(f["t_train_mean"], dtype=float)
        t_std = np.asarray(f["t_train_std"], dtype=float)
        R_star = np.asarray(f["R_star"], dtype=float)
    with open(index, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        path = base / row["rotation_file"]
        with np.load(path, allow_pickle=False) as data:
            R = np.asarray(data["R"], dtype=float)
            G = np.asarray(data["delta_R"], dtype=float)
            zm = np.asarray(data["global_z_mean"], dtype=float)
            zs = np.asarray(data["global_z_std"], dtype=float)
        if R.shape != R_star.shape or not np.allclose(G @ R_star, R, atol=1e-7):
            raise ValueError(f"S9 rotation inconsistent with S8: {path}")
        if str(row["candidate_type"]) == "baseline_unswept":
            rotation_id = "baseline_unswept"
        else:
            rotation_id = path.stem
        axes = tuple(int(t)-1 for t in str(row["axes_1based"]).replace(",", " ").split()
                     if t.strip().isdigit())
        rot = Rotation(rotation_id, variant, row["candidate_type"], axes,
                       float(row["angle_degrees"] or 0), R, R_star, G, zm, zs,
                       t_mean, t_std)
        rot.validate()
        if rot.dim != src.latent_dim or rot.dim != src.parameters.shape[1]:
            raise ValueError("Metric Lab 01 currently requires L == dim(T)")
        rotations.append(rot)
    return rotations


def output_base(src):
    # Use the existing complete NN run-relative path; no re-computation of conditions.
    return ROOT / "outputs" / "metric_lab" / "01_hessian_curvature_energy" / src.model / src.run_dir.parent.name / src.run_dir.name
