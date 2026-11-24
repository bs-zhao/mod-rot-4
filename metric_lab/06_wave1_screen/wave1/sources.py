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


def _load_rotation_normalizer(s8_data, source, variant):
    """Resolve physical Z -> rotation coordinates without guessing scales.

    Original S9 always operates on raw Z (identity). S9_2 must use the
    *saved* all-subject normalizer from S8_2, not an ad hoc z-score.
    """
    dim = source.latent_dim
    if variant == "s9":
        return np.zeros(dim, dtype=float), np.ones(dim, dtype=float)
    if variant != "s9_2":
        raise ValueError(f"Unknown rotation variant: {variant}")
    if "global_z_mean" not in s8_data or "global_z_std" not in s8_data:
        raise ValueError(
            "S9_2 requires S8_2 global_z_mean/global_z_std in rotation.npz. "
            "Do not substitute an identity normalizer; regenerate S8_2."
        )
    mu = np.asarray(s8_data["global_z_mean"], dtype=float)
    sd = np.asarray(s8_data["global_z_std"], dtype=float)
    if mu.shape != (dim,) or sd.shape != (dim,) or (
        not np.isfinite(mu).all() or not np.isfinite(sd).all() or
        (sd <= 1e-12).any()
    ):
        raise ValueError("Invalid S8_2 global Z normalizer")
    # Validate the normalization convention against original S5 Z.
    z = np.asarray(source.z, dtype=float)
    if not (np.allclose(z.mean(axis=0), mu, rtol=1e-8, atol=1e-10)
            and np.allclose(z.std(axis=0, ddof=0), sd,
                            rtol=1e-8, atol=1e-10)):
        raise ValueError(
            "S8_2 Z normalizer does not match current S5 latent file"
        )
    return mu, sd


def _index_candidate_kind(row, data, filename):
    """Historic S9 indices omitted 'candidate_type' and unswept baseline."""
    indicated = str(row.get("candidate_type") or "").strip()
    if not indicated and "candidate_type" in data:
        indicated = str(data["candidate_type"])
    if not indicated:
        indicated = ("baseline_unswept" if filename.name == "unswept.npz"
                     else "sweep")
    if indicated not in ("baseline_unswept", "sweep"):
        raise ValueError(f"Unsupported candidate_type: {indicated!r}")
    return indicated


def load_rotations(src, variant):
    """Read S9/S9_2 outputs from all historical mod-rot-4 producer versions.

    S9 producers before Nov 19 wrote index CSVs without candidate_type,
    rotation NPZs without Z normalization and no indexed unswept R*. The
    unswept row is synthesized in memory, never written into upstream S9.
    S9_2 normalization is always taken from the actual S8_2 artifact.
    """
    if variant not in ("s9", "s9_2"):
        raise ValueError(f"Unknown rotation variant: {variant}")
    stage = "s9_rotation_sweep" if variant == "s9" else "s9_2_rotation_sweep"
    base = src.run_dir / stage
    index = base / "rotation_index.csv"
    if not index.is_file():
        raise FileNotFoundError(f"Run NN {stage} first: {index}")
    s8 = "s8_supervised_rotation" if variant == "s9" else "s8_2_supervised_rotation"
    upstream = src.run_dir / s8 / "rotation.npz"
    if not upstream.is_file():
        raise FileNotFoundError(f"Missing upstream {s8}: {upstream}")
    with np.load(upstream, allow_pickle=False) as f:
        t_mean = np.asarray(f["t_train_mean"], dtype=float)
        t_std = np.asarray(f["t_train_std"], dtype=float)
        R_star = np.asarray(f["R_star"], dtype=float)
        zm, zs = _load_rotation_normalizer(f, src, variant)
    with index.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "rotation_file" not in reader.fieldnames:
            raise ValueError(f"Invalid rotation_index.csv: {index}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty rotation index: {index}")

    rotations = []
    legacy_missing_kind = 0
    legacy_missing_zstats = 0
    for row in rows:
        relative = str(row["rotation_file"] or "").strip().replace("\\", "/")
        if not relative:
            raise ValueError(f"Missing rotation filename in {index}")
        path = base / relative
        if not path.is_file():
            raise FileNotFoundError(f"Rotation listed but not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            R = np.asarray(data["R"], dtype=float)
            if "R_star" in data and not np.allclose(
                np.asarray(data["R_star"], dtype=float), R_star, atol=1e-7
            ):
                raise ValueError(f"Rotation R_star differs from {s8}: {path}")
            G = (np.asarray(data["delta_R"], dtype=float)
                 if "delta_R" in data else R @ R_star.T)
            # S9 original uses identity regardless of archive vintage.
            # S9_2 uses the S8_2 normalizer even if an old R NPZ omits it.
            for key, expected in (("global_z_mean", zm), ("global_z_std", zs)):
                if key in data:
                    if not np.allclose(np.asarray(data[key], dtype=float),
                                       expected, rtol=1e-8, atol=1e-10):
                        raise ValueError(
                            f"Rotation {key} inconsistent with {s8}: {path}"
                        )
                else:
                    legacy_missing_zstats += 1
            kind = _index_candidate_kind(row, data, path)
            if "candidate_type" not in row:
                legacy_missing_kind += 1
            axes_text = str(row.get("axes_1based") or "").replace(",", " ").strip()
            if axes_text:
                axes = tuple(int(x) - 1 for x in axes_text.split())
            elif "axes_1based" in data:
                axes = tuple(int(x) - 1 for x in data["axes_1based"])
            else:
                axes = ()
            angle_text = str(row.get("angle_degrees") or "").strip()
            angle = (float(angle_text) if angle_text else
                     float(data["angle_degrees"]) if "angle_degrees" in data
                     else 0.0)
        if R.shape != R_star.shape or not np.allclose(
            G @ R_star, R, atol=1e-7
        ):
            raise ValueError(f"S9 rotation inconsistent with S8: {path}")
        rotation_id = "baseline_unswept" if kind == "baseline_unswept" else path.stem
        rot = Rotation(rotation_id, variant, kind, axes, angle,
                       R, R_star, G, zm.copy(), zs.copy(),
                       t_mean.copy(), t_std.copy())
        rot.validate()
        if rot.dim != src.latent_dim or rot.dim != src.parameters.shape[1]:
            raise ValueError("Metric Lab HE requires L == dim(T)")
        rotations.append(rot)

    baselines = [r for r in rotations if r.candidate_type == "baseline_unswept"]
    if len(baselines) > 1:
        raise ValueError(f"Multiple unswept baselines in {index}")
    if not baselines:
        # Older S9 runs never indexed the independent unswept R*. Preserve
        # the first-class candidate without regenerating/mutating S9 files.
        baseline = Rotation(
            "baseline_unswept", variant, "baseline_unswept", (), 0.0,
            R_star.copy(), R_star.copy(), np.eye(len(R_star)),
            zm.copy(), zs.copy(), t_mean.copy(), t_std.copy()
        )
        baseline.validate()
        rotations.insert(0, baseline)
    elif rotations[0].candidate_type != "baseline_unswept":
        # Consistent first-class baseline order across legacy/modern producers.
        rotations = baselines + [r for r in rotations
                                 if r.candidate_type != "baseline_unswept"]

    names = [r.name for r in rotations]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate rotation IDs in {index}")
    if legacy_missing_kind or legacy_missing_zstats or not baselines:
        print(f"[ROTATION COMPAT] {variant}: {len(rotations)} candidates, "
              f"legacy_index={bool(legacy_missing_kind)}, "
              f"missing_npz_normalizer={bool(legacy_missing_zstats)}, "
              f"synthesized_baseline={not bool(baselines)}", flush=True)
    return rotations


def output_base(src):
    # Use the existing complete NN run-relative path; no re-computation of conditions.
    return ROOT / "outputs" / "metric_lab" / "06_wave1_screen" / src.model / src.run_dir.parent.name / src.run_dir.name
