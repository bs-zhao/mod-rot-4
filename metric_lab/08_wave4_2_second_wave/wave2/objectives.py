"""Blind moment caches and frozen objectives (02_second_wave sections 5-6).

Base frame: raw Z (latent) and X standardised with the saved S3 scaler.  K_abcd = E[H_ab H_cd] (latent block), J^(j)_ab = E[H_a,xj H_b,xj].
A variant V = (Z - c) / s and a rotation U = R V give  d raw / d U = diag(s) R^T;  B_ij(R) = (R S J^(j) S R^T)_ii, A from K.  Squared
energies are never rotated.  Metric values come from the unchanged Lab 03 / Lab 06 registries through the Lab 07 adapters
(fast single-metric B01/B02/B05-B10 paths, tested equal to the full library).  Fz / CAND fusions use the historical Haar null of the
same configuration (200 Haar rotations, scipy random_state 4), never a supervised reference.
"""
import json
import numpy as np
from .common import OUT
from wave4_2.derivatives import ad_chunks, channel_hessian            # Lab 07 (truth-free numerical modules)
from wave4_2.moments import Moments, contract, jac
from wave4_2.measures import col_mask
from wave4_2 import metrics as MT

CACHE = OUT / "moment_cache"


def masks_of(pl):
    return ("MASK-ALL", "MASK-COL") if (pl.x_domain is not None and pl.fixed_bank is not None) else ("FULL",)


def compute_moments(pl, eta, d, chans, masks, chunk=1024):
    """-> {(channel, mask): finished moment dict}.  d = a design dict (lat, x)."""
    xs = (d["x"] - pl.x_mean) / pl.x_std; cm = col_mask(d["x"], pl.x_domain)
    P = np.column_stack((d["lat"], xs)); acc = {(c, m): Moments(pl.k, pl.p, m) for c in chans for m in masks}
    for sl, e, g, H in ad_chunks(eta, P, chunk):
        for c in chans:
            Hc = channel_hessian(e, g, H, c)
            for m in masks:
                acc[(c, m)].add(Hc, cm[sl])
    return {key: a.finish() for key, a in acc.items()}


def save_moments(path, res, meta):
    arr = {}
    for (c, m), r in res.items():
        for kk in ("K", "J", "nJ", "coverage"):
            arr[f"{c}|{m}|{kk}"] = np.asarray(r[kk])
        arr[f"{c}|{m}|scalars"] = np.array([r["nK"], r["n_points"], r["nonfinite"], float(r["valid"])])
    arr["meta"] = np.array(json.dumps(meta, default=float))
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp.npz"); np.savez_compressed(tmp, **arr); tmp.replace(path)


def load_moments(path):
    f = np.load(path, allow_pickle=False); res = {}
    for key in f.files:
        if key == "meta":
            continue
        c, m, kk = key.split("|"); r = res.setdefault((c, m), {})
        if kk == "scalars":
            r["nK"], r["n_points"], r["nonfinite"], r["valid"] = int(f[key][0]), int(f[key][1]), int(f[key][2]), bool(f[key][3])
        else:
            r[kk] = f[key]
    return res, json.loads(str(f["meta"]))


def cached_moments(pl, eta_factory, design_fn, tag, chans, masks):
    """compute once per toy / pool / measure / fidelity / seed (tag) and reuse"""
    path = CACHE / pl.toy.replace("#", "_") / f"{tag}.npz"
    if path.is_file():
        return load_moments(path)
    d = design_fn()
    res = compute_moments(pl, eta_factory(), d, chans, masks)
    meta = dict(design=d["meta"], n_points=int(len(d["x"])), subjects=None if d.get("sid") is None else int(len(np.unique(d["sid"]))))
    save_moments(path, res, meta)
    return res, meta


# ---------------------------------------------------------------- neutral contraction (adapted from Lab 07; tested vs moments.contract)
def contract_factory(K, J, scale):
    k = K.shape[0]; K2 = K.reshape(k * k, k * k); s = np.asarray(scale, float)

    def AB(R):
        Jt = R * s[None, :]                                    # = (d raw / d U)^T, rows = rotated axes
        W = np.kron(Jt, Jt); A = np.einsum("ij,ij->i", W @ K2, W).reshape(k, k)
        B = np.einsum("ai,xik,ak->ax", Jt, J, Jt)
        return np.maximum((A + A.T) / 2, 0), np.maximum(B, 0)
    return AB


def _needs_null(mid):
    reg, name = mid.split("|", 1)
    return reg == "L03" and name.split(".")[0] in ("Fz", "CAND")


class Objective:
    """frozen L(R) for one metric id on one moment cache and coordinate variant"""
    def __init__(self, mo, scale, mid):
        self.mid = mid; self.k = mo["K"].shape[0]; self.valid = bool(mo["valid"])
        self.AB = contract_factory(mo["K"], mo["J"], scale)
        A0, B0 = self.AB(np.eye(self.k)); self.zero, self.EA, self.EB = MT.zero_flags(A0, B0)
        reg, name = mid.split("|", 1)
        if reg == "L03":
            st = MT.l03_null(mo["K"], mo["J"], scale, contract, jac) if _needs_null(mid) else dict(null={})
            f = MT.L03[name]; self._f = lambda R: float(f(*self._BA(R), st))
            self.null = st["null"]
        else:
            g = MT.single_l06(name); z = self.zero; self._f = lambda R: float(g(*self.AB(R), z)); self.null = None
        self.n_evals = 0

    def _BA(self, R):
        A, B = self.AB(R); return B, A

    def __call__(self, R):
        self.n_evals += 1
        if not self.valid:
            return np.nan
        return self._f(R)
