"""EVALUATION-ONLY: T-U recovery of a LOCKED rotation (02_second_wave section 8).  Never imported by blind modules.

For TEST subjects, U = R V (V = blind variant coordinates).  C_ij = |corr(T_i, U_j)|, Hungarian matching on C, MCC = mean matched |r|,
leakage = unmatched |r| (mean / max).  Supervised reference U* = R* (V - c*) from the saved S8 / S8_2 Procrustes fit.
Lab 05 operational criterion: MCC >= supervised MCC - .05 and max leakage < .2 (C families) / < .3 (CR families).
"""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from .common import OUT, toy_entry, leak_threshold
from .optimizers import rel_dist

EVAL_DIR = OUT / "eval_only"


def eval_path(toy):
    return EVAL_DIR / f"{toy.replace('#', '_')}.npz"


def extract_eval(toy, overwrite=False):
    """T, splits and S8 / S8_2 references (through the validated Lab 07 loader)"""
    out = eval_path(toy)
    if out.is_file() and not overwrite:
        return out
    from wave4_2.inputs import load
    b, ev, rots, src = load(toy)
    arr = dict(T=ev.T, test=b.splits["test"], meta=np.array(json.dumps(dict(toy=toy, family=ev.family, variants=list(ev.refs)))))
    for v, r in ev.refs.items():
        for kk, vv in r.items():
            arr[f"{v}|{kk}"] = np.asarray(vv)
    out.parent.mkdir(parents=True, exist_ok=True); np.savez(out, **arr)
    return out


class EvalData:
    def __init__(self, toy):
        f = np.load(eval_path(toy), allow_pickle=False); self.toy = toy
        self.meta = json.loads(str(f["meta"])); self.T = f["T"]; self.test = f["test"]
        self.refs = {v: {kk: f[f"{v}|{kk}"] for kk in ("R_star", "z_center", "t_mean", "t_std", "z_mean", "z_std")} for v in self.meta["variants"]}
        self.family = self.meta["family"]


def matched(T, U):
    k = T.shape[1]; C = np.corrcoef(T.T, U.T)[:k, k:]; A = np.abs(C); r, c = linear_sum_assignment(-A); m = A[r, c]
    M = A.copy(); M[r, c] = np.nan; off = M[~np.isnan(M)]
    return dict(signed=C, perm=c, matched=m, MCC=float(m.mean()), min=float(m.min()),
                leak_mean=float(off.mean()) if off.size else 0.0, leak_max=float(off.max()) if off.size else 0.0)


def v_test(pl, ed, variant):
    from .blind_inputs import to_v
    return to_v(pl, variant, pl.z[ed.test])


def supervised(pl, ed, variant):
    ref = ed.refs[variant]; V = v_test(pl, ed, variant)
    return matched(ed.T[ed.test], (V - ref["z_center"]) @ ref["R_star"].T)


def recovery(pl, ed, variant, R, sup=None):
    V = v_test(pl, ed, variant); res = matched(ed.T[ed.test], V @ np.asarray(R).T)
    sup = sup or supervised(pl, ed, variant); thr = leak_threshold(pl.toy)
    out = dict(MCC=res["MCC"], MCC_min=res["min"], leak_mean=res["leak_mean"], leak_max=res["leak_max"], supervised_MCC=sup["MCC"],
               supervised_leak_max=sup["leak_max"], MCC_gap=sup["MCC"] - res["MCC"],
               dist_to_Rstar=rel_dist(np.asarray(R), ed.refs[variant]["R_star"]),
               pass_lab05=bool(res["MCC"] >= sup["MCC"] - .05 and res["leak_max"] < thr),
               pass_gap02=bool(res["MCC"] >= sup["MCC"] - .02 and res["leak_max"] < thr),
               pass_leak20=bool(res["MCC"] >= sup["MCC"] - .05 and res["leak_max"] < .2),
               pass_leak30=bool(res["MCC"] >= sup["MCC"] - .05 and res["leak_max"] < .3))
    for i, v in enumerate(res["matched"]):
        out[f"matched_{i}"] = float(v)
    return out, res
