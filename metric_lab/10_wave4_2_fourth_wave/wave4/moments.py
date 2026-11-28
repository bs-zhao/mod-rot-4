"""Frozen-decoder AD moment caches with latent-draw block resolution (04_fourth_wave sections 3, 8.1, 11).

Global moments are accumulated by the UNCHANGED Lab 07 `Moments` class (same masks, denominators, validity) and saved in the Lab 08
cache format (wave2.objectives.save_moments / load_moments).  In the same AD pass the UX block is also summed per latent-draw block
(rows sharing one latent draw are not iid):  SJ[b, j] = sum_{rows of block b, valid for column j} h_j h_j^T,  NJ[b, j] = count.
sum_b SJ / sum_b NJ == the global J (tested).  The block file feeds the latent-block bootstrap, the tail-strata energy and the J-mixture
diagnostics.  Points: [raw latent, X standardised with the saved S3 scaler]; float64 CPU torch.func AD.
"""
import json, hashlib, os
import numpy as np
from wave4_2.derivatives import ad_chunks, channel_hessian
from wave4_2.moments import Moments
from wave4_2.measures import col_mask
from wave2.objectives import save_moments, load_moments
from .common import OUT, chan
from . import designs as DS

CACHE = OUT / "moment_cache"


def masks_for(pl):
    return ("MASK-ALL", "MASK-COL", "FULL") if (pl.x_domain is not None and pl.fixed_bank is not None) else ("FULL",)


def block_ids(d):
    n = len(d["x"])
    if d.get("sid") is not None:
        s = np.asarray(d["sid"]); new = np.r_[True, s[1:] != s[:-1]]; return np.cumsum(new) - 1
    return np.arange(n) // int(d["rows_per_latent"])


def compute(pl, eta, d, masks, chunk=1024):
    c = chan(pl); xs = (d["x"] - pl.x_mean) / pl.x_std; cm = col_mask(d["x"], pl.x_domain); P = np.column_stack((d["lat"], xs))
    k, p = pl.k, pl.p; blk = block_ids(d); nb = int(blk.max()) + 1
    acc = {m: Moments(k, p, m) for m in masks}; SJ = {m: np.zeros((nb, p, k, k)) for m in masks}; NJ = {m: np.zeros((nb, p)) for m in masks}
    for sl, e, g, H in ad_chunks(eta, P, chunk):
        Hc = channel_hessian(e, g, H, c); b = blk[sl]
        for m in masks:
            acc[m].add(Hc, cm[sl])
            for j in range(p):
                ok = cm[sl].all(1) if m == "MASK-ALL" else (cm[sl, j] if m == "MASK-COL" else np.ones(len(b), bool))
                h = Hc[ok, :k, k + j]
                np.add.at(SJ[m][:, j], b[ok], h[:, :, None] * h[:, None, :]); np.add.at(NJ[m][:, j], b[ok], 1)
    res = {(c, m): a.finish() for m, a in acc.items()}
    return res, dict(SJ=SJ, NJ=NJ)


def tag_of(pool, M, fid, seed):
    return f"{pool}_{M}_{fid}_s{seed}"


def path_of(pl, tag):
    return CACHE / pl.toy.replace("#", "_") / f"{tag}.npz"


_ETA = {}


def get(pl, M, fid, seed, pool="train", blocks=False):
    """-> (moment dict {(channel, mask): mo}, meta[, blocks]) ; computed once, never overwritten"""
    tag = tag_of(pool, M, fid, seed); f = path_of(pl, tag); fb = f.with_name(tag + ".blocks.npz")
    if not f.is_file():
        d = DS.design(pl, M, fid, seed, pool=pool)
        if d is None:
            return None
        if pl.toy not in _ETA:
            import run_blind as RB8
            _ETA[pl.toy] = RB8.eta_factory(pl)()
        res, bl = compute(pl, _ETA[pl.toy], d, masks_for(pl))
        fit = {kk: np.asarray(v).tolist() for kk, v in d["fit"].items()}
        meta = dict(design=d["meta"], n_points=int(len(d["x"])), fit=fit, x_rows_sha=None if d.get("rows") is None else hashlib.sha256(np.ascontiguousarray(d["rows"]).tobytes()).hexdigest(),
                    n_blocks=int(block_ids(d).max() + 1))
        arr = {}
        for m in bl["SJ"]:
            arr[f"{m}|SJ"] = bl["SJ"][m]; arr[f"{m}|NJ"] = bl["NJ"][m]
        lat0 = d["lat"][np.r_[True, block_ids(d)[1:] != block_ids(d)[:-1]]]
        arr["lat_blocks"] = lat0; arr["meta"] = np.array(json.dumps(meta, default=float))
        f.parent.mkdir(parents=True, exist_ok=True); tmp = fb.with_name(f"{tag}.blocks.{os.getpid()}.tmp.npz"); np.savez_compressed(tmp, **arr); tmp.replace(fb)
        fp = f.with_name(f"{tag}.{os.getpid()}.npz"); save_moments(fp, res, meta); fp.replace(f)
    res, meta = load_moments(f)
    if blocks:
        z = np.load(fb, allow_pickle=False); return res, meta, {kk: z[kk] for kk in z.files}
    return res, meta


def mixture(moE, moD, alpha):
    """J_alpha = (1-alpha) J_E + alpha J_D on the SAME X panel / mask (equal denominators by construction): HE under a mixture of measures"""
    if not np.array_equal(moE["nJ"], moD["nJ"]):
        raise ValueError("mixture needs identical X panel / mask denominators")
    out = dict(moD); out["J"] = (1 - alpha) * moE["J"] + alpha * moD["J"]; out["K"] = (1 - alpha) * moE["K"] + alpha * moD["K"]
    out["valid"] = bool(moE["valid"] and moD["valid"]); return out
