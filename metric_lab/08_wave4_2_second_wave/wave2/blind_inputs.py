"""Minimal blind payload (02_second_wave section 3.2).

`extract_payload` is the ORCHESTRATOR step: it reads the S1 dataset / S5 latent / S3 checkpoint through the validated Lab 03
adapter, DROPS the true parameters, and writes a blind-only file outputs/.../blind_payload/<toy>.npz.  It never opens S8/S9
rotation files.  `load_payload` reads only that file; the blind objective / optimizer / surrogates receive only a BlindPayload
(plus the frozen checkpoint path for the decoder).  tests/test_w2.py runs the blind objective and a short search under a Python
audit hook that raises on any attempt to open the S1 dataset, the S8/S8_2/S9/S9_2 directories or Lab 07 tensors / losses.

Coordinate variants (blind reconstruction, checked against the saved S9 / S9_2 normalisers in W2-00):
  s9    V = Z                     (raw)
  s9_2  V = (Z - mean) / sd       mean / sd over ALL subjects' Z, ddof 0 (marginal standardisation, not whitening)
"""
from dataclasses import dataclass, field
import json
import numpy as np
from .common import OUT, toy_entry, toy_registry, sha256

PAYLOAD_DIR = OUT / "blind_payload"


@dataclass(frozen=True)
class BlindPayload:
    toy: str
    cls: str
    z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    subject_index: np.ndarray
    counts: np.ndarray
    starts: np.ndarray
    splits: dict
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    checkpoint_file: str
    fixed_bank: object
    x_domain: object
    hashes: dict = field(default_factory=dict)

    @property
    def k(self): return self.z.shape[1]

    @property
    def p(self): return self.x.shape[1]


def _detect_bank(x, counts):
    c = int(counts[0])
    if not np.all(counts == c):
        return None
    xs = x.reshape(len(counts), c, x.shape[1])
    return xs[0].copy() if np.array_equal(xs, np.broadcast_to(xs[:1], xs.shape)) else None


def payload_path(toy):
    return PAYLOAD_DIR / f"{toy.replace('#', '_')}.npz"


def extract_payload(toy, overwrite=False):
    """orchestrator only (reads the dataset, discards T).  Returns the payload path."""
    from s9curves.sources import load_sources
    out = payload_path(toy)
    if out.is_file() and not overwrite:
        return out
    e = toy_entry(toy); src = load_sources(e.get("model", toy), e["run_dir"])
    starts = np.r_[0, np.cumsum(src.counts[:-1])]
    bank = _detect_bank(src.x, src.counts)
    dom = toy_registry()["x_domain"][e["family"]]
    meta = dict(toy=toy, cls=e["class"], checkpoint_file=str(src.checkpoint_file), y_mean=float(src.y_mean), y_std=float(src.y_std),
                x_domain=None if dom == "unbounded" else list(dom),
                hashes=dict(dataset=sha256(src.dataset_file), checkpoint=sha256(src.checkpoint_file), latent=sha256(src.latent_file)))
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = dict(z=src.z, x=src.x, y=src.true_choice, subject_index=src.subject_ids, counts=src.counts, starts=starts,
               split_train=src.splits["train"], split_validation=src.splits["validation"], split_test=src.splits["test"],
               x_mean=src.x_mean, x_std=src.x_std, meta=np.array(json.dumps(meta)))
    if bank is not None:
        arr["fixed_bank"] = bank
    tmp = out.with_suffix(".tmp.npz"); np.savez(tmp, **arr); tmp.replace(out)
    return out


def load_payload(toy):
    f = payload_path(toy)
    with np.load(f, allow_pickle=False) as a:
        meta = json.loads(str(a["meta"]))
        return BlindPayload(meta["toy"], meta["cls"], a["z"], a["x"], a["y"], a["subject_index"], a["counts"], a["starts"],
                            {k: a[f"split_{k}"] for k in ("train", "validation", "test")}, a["x_mean"], a["x_std"],
                            float(meta["y_mean"]), float(meta["y_std"]), meta["checkpoint_file"],
                            a["fixed_bank"] if "fixed_bank" in a.files else None, meta["x_domain"], meta["hashes"])


def variant_frame(pl, variant):
    """(centre, scale) of V = (Z - centre) / scale, reconstructed from Z only"""
    if variant == "s9":
        return np.zeros(pl.k), np.ones(pl.k)
    if variant == "s9_2":
        return pl.z.mean(0), pl.z.std(0)
    raise ValueError(variant)


def to_v(pl, variant, z=None):
    c, s = variant_frame(pl, variant)
    return ((pl.z if z is None else z) - c) / s
