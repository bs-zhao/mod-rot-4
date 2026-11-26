"""W1-00 frozen inputs, split into two separate interfaces.

BlindInputs  : Z, X, Y, subject bookkeeping, splits, saved X / Y scalers, frozen checkpoint path, observed-design metadata.
               This is the ONLY object that blind surfaces / metric scoring may receive.
EvalOnly     : T, the analytic family, parameter prior and bounds, S8/S8_2 references (R*, centres, T scalers).
               Used for analytic benchmarks, bridges and after-the-fact evaluation only.
Rotations    : the saved S9 / S9_2 candidates.  The blind scorer receives only (R, z_mean, z_std); labels (axes, angle, R*, G)
               stay in the harness manifest.
All files are read through the validated Lab 03 adapter (read-only); historical manifests are normalised in memory only.
"""
from dataclasses import dataclass, field
import numpy as np
from .common import library, x_domain, sha256
from s9curves.sources import load_sources, load_rotations          # Lab 03 adapter (read-only)


@dataclass(frozen=True)
class BlindInputs:
    toy: str
    cls: str                      # observation class B / C (a legitimate method input)
    z: np.ndarray                 # (S, k) S5 posterior means, all observed trials
    x: np.ndarray                 # (N, p) raw X
    y: np.ndarray                 # (N,) observed choice (B) or continuous y (C)
    subject_index: np.ndarray
    counts: np.ndarray
    starts: np.ndarray
    splits: dict
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    checkpoint_file: object
    fixed_bank: object            # (rows, p) if every subject saw the same rows in the same order (detected from data) else None
    x_domain: object              # declared column domains (masks only) or None
    hashes: dict = field(default_factory=dict)

    @property
    def k(self): return self.z.shape[1]

    @property
    def p(self): return self.x.shape[1]


@dataclass(frozen=True)
class EvalOnly:
    toy: str
    family: str
    T: np.ndarray                 # (S, k) true parameters
    refs: dict                    # variant -> dict(R_star, z_center (in V coords), t_mean, t_std, z_mean, z_std)


@dataclass(frozen=True)
class BlindRotation:
    """what a blind scorer may see of a candidate: the matrix and the saved coordinate convention, nothing else"""
    R: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray


def _detect_bank(x, counts):
    c = int(counts[0])
    if not np.all(counts == c):
        return None
    xs = x.reshape(len(counts), c, x.shape[1])
    return xs[0].copy() if np.array_equal(xs, np.broadcast_to(xs[:1], xs.shape)) else None


def load(toy):
    lib = library(toy)
    src = load_sources(lib["model"], lib["run_dir"])
    starts = np.r_[0, np.cumsum(src.counts[:-1])]
    bank = _detect_bank(src.x, src.counts)
    hashes = dict(dataset=sha256(src.dataset_file), checkpoint=sha256(src.checkpoint_file), latent=sha256(src.latent_file))
    blind = BlindInputs(toy, lib["cls"], src.z, src.x, src.true_choice, src.subject_ids, src.counts, starts, src.splits,
                        src.x_mean, src.x_std, float(src.y_mean), float(src.y_std), src.checkpoint_file, bank,
                        x_domain(lib["family"]), hashes)
    refs, rots = {}, {}
    for v in ("s9", "s9_2"):
        try:
            rl = load_rotations(src, v)
        except FileNotFoundError:
            continue
        r0 = rl[0]
        stage = "s8_supervised_rotation" if v == "s9" else "s8_2_supervised_rotation"
        with np.load(src.run_dir / stage / "rotation.npz") as f:
            zc = np.asarray(f["z_train_mean"], float)          # centre of the Procrustes fit, in the V coordinates of the variant
            ids = {k: np.asarray(f[k + "_ids"], int) for k in ("train", "validation", "test")}
        for k_, v_ in ids.items():
            if not np.array_equal(np.sort(v_), np.sort(src.splits[k_])):
                raise ValueError(f"{toy} {v}: S8 {k_} ids differ from S5 splits")
        refs[v] = dict(R_star=r0.R_star.copy(), z_center=zc, t_mean=r0.t_mean.copy(), t_std=r0.t_std.copy(),
                       z_mean=r0.z_mean.copy(), z_std=r0.z_std.copy())
        rots[v] = rl
    ev = EvalOnly(toy, lib["family"], src.parameters, refs)
    return blind, ev, rots, src


def blind_rotation(rot):
    return BlindRotation(rot.R.copy(), rot.z_mean.copy(), rot.z_std.copy())
