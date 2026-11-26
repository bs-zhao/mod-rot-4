"""The two historical metric registries, preserved and versioned (no formula changes).

L03 : Lab 03 (`s9curves/metrics152.py`) 152 metrics + the pre-registered candidate CAND.rowsp.75+UXcol.hoy = 153.
      Signature f(UX, UU, st); Fz / CAND fusions are z-scored with a null of 200 Haar rotations of the SAME configuration
      (scipy special_ortho_group, random_state 4, as in Lab 03).
L06 : Lab 06 (`wave1/library.py`) metrics(A=M^UU, B=M^UX, zero) -> 262 entries (incl. controls / diagnostic counts).
      `zero` = whole-landscape zero-block flags decided once per configuration (EA or EB <= 1e-9 (EA+EB)), never per angle.
Registry IDs are 'L03|<name>' and 'L06|<name>'.  Every value: lower = simpler.
"""
import numpy as np
from scipy.stats import special_ortho_group
from s9curves import metrics152 as MET
from wave1.library import metrics as l06_metrics, metric_names as l06_names, scalars as l06_scalars, ids as l06_ids, _norm01, _vech, _offv, TOL as L06TOL

L06_CONTROLS = {"CTRL.EA", "CTRL.EB", "CTRL.EB_colsum_cv", "B03.CTRL.diag_share", "B04.CTRL.off_share", "B05.n_active", "B07.n_active"}


def l03_build():
    M = MET.build(); comp = M["_comp"]
    comp["A75"] = lambda UX, UU: MET.rowsp(UX, 0.75, 0.1)
    comp["K0"] = lambda UX, UU: float(np.mean([MET.hoy(UX[:, j]) for j in range(UX.shape[1])]))
    M["CAND.rowsp.75+UXcol.hoy"] = lambda UX, UU, st: sum((comp[k](UX, UU) - st["null"][k][0]) / st["null"][k][1] for k in ("A75", "K0"))
    return M


L03 = l03_build()
L03_NAMES = [k for k in L03 if k != "_comp"]
L06_NAMES = l06_names()
ALL_IDS = [f"L03|{n}" for n in L03_NAMES] + [f"L06|{n}" for n in L06_NAMES]


def zero_flags(A, B):
    EA, EB = float(A.sum()), float(B.sum())
    return dict(A=bool(EA <= 1e-9 * (EA + EB)), B=bool(EB <= 1e-9 * (EA + EB))), EA, EB


def haar(k, n=200, seed=4):
    return special_ortho_group.rvs(k, size=n, random_state=seed)


def l03_null(K, J, scale, contract, jac):
    """null statistics of the fusion components from 200 Haar rotations of the same configuration (blind: no reference)"""
    vals = {c: [] for c in L03["_comp"]}
    for Q in haar(K.shape[0]):
        A, B = contract(K, J, jac(scale, Q))
        for c, f in L03["_comp"].items():
            vals[c].append(f(B, A))
    return dict(null={c: (float(np.median(v)), float(np.std(v)) + 1e-12) for c, v in vals.items()})


def evaluate_all(A, B, st, zero):
    """-> np.array over ALL_IDS for one rotation"""
    out = [float(L03[n](B, A, st)) for n in L03_NAMES]
    d = l06_metrics(A, B, zero)
    return np.array(out + [float(d[n]) for n in L06_NAMES])


def evaluate_ids(A, B, st, zero, ids):
    """subset evaluation (priority panel / global search); identical values to evaluate_all"""
    out = []; d = None
    for i in ids:
        reg, name = i.split("|", 1)
        if reg == "L03":
            out.append(float(L03[name](B, A, st)))
        else:
            out.append(float(single_l06(name)(A, B, zero)))
    return np.array(out)


# ---------------------------------------------------------------- fast single L06 metric (validated against l06_metrics in tests)
_CACHE = {}


def single_l06(name):
    if name in _CACHE:
        return _CACHE[name]
    parts = name.split(".")
    rec = parts[0]

    def full(A, B, zero):
        return l06_metrics(A, B, zero)[name]
    fn = full
    if rec in ("B01", "B02") and name not in L06_CONTROLS:
        w = name.split(".", 1)[1]
        def fn(A, B, zero, w=w, rec=rec):
            k, p = B.shape
            if rec == "B01":
                EB = float(B.sum())
                if not (EB > L06TOL and not zero["B"]):
                    return np.nan
                return l06_scalars(B.ravel() / EB, k * p, EB, [w])[w]
            EA = float(A.sum())
            if not (EA > L06TOL and not zero["A"]):
                return np.nan
            return l06_scalars(_vech(A) / EA, k * (k + 1) // 2, EA, [w])[w]
    elif rec == "B09":
        wn, w = parts[1], ".".join(parts[2:])
        wB, wA = {"w50": (.5, .5), "w75": (.75, .25), "w25": (.25, .75)}[wn]
        def fn(A, B, zero, w=w, wB=wB, wA=wA):
            k, p = B.shape; EA, EB = float(A.sum()), float(B.sum()); nA, nB = k * (k + 1) // 2, k * p
            b = B.ravel() / EB if EB > L06TOL and not zero["B"] else None
            a = _vech(A) / EA if EA > L06TOL and not zero["A"] else None
            sb = _norm01(w, l06_scalars(b, nB, EB, [w])[w], nB, nB) if b is not None else np.nan
            sa = _norm01(w, l06_scalars(a, nA, EA, [w])[w], nA, nA) if a is not None else np.nan
            if b is not None and a is not None: return wB * sb + wA * sa
            if b is not None and zero["A"]: return sb
            if a is not None and zero["B"]: return sa
            return np.nan
    elif rec == "B10":
        blk, w = parts[1], ".".join(parts[2:])
        def fn(A, B, zero, w=w, blk=blk):
            k, p = B.shape; EA, EB = float(A.sum()), float(B.sum()); nA, nB = k * (k + 1) // 2, k * p
            b = B.ravel() / EB if EB > L06TOL and not zero["B"] else None
            comps = [_norm01(w, l06_scalars(b, nB, EB, [w])[w], nB, nB)] if b is not None else []
            if EA > L06TOL and not zero["A"]:
                if blk == "3blk":
                    sd = l06_scalars(np.diag(A) / EA, nA, EA, [w])[w]; so = l06_scalars(_offv(A) / EA, nA, EA, [w])[w]
                    return float(np.mean(comps + [_norm01(w, sd, nA, nA), _norm01(w, so, nA, nA)])) if comps else np.nan
                sa = l06_scalars(_vech(A) / EA, nA, EA, [w])[w]
                return float(np.mean(comps + [_norm01(w, sa, nA, nA)])) if comps else np.nan
            if zero["A"]:
                return comps[0] if comps else np.nan
            return np.nan
    elif name == "CTRL.EB_colsum_cv":
        def fn(A, B, zero):
            return float(np.std(B.sum(0)) / (np.mean(B.sum(0)) + 1e-300))
    elif name in ("CTRL.EA", "CTRL.EB"):
        def fn(A, B, zero, name=name):
            return float(A.sum() if name == "CTRL.EA" else B.sum())
    elif rec in ("B05", "B06", "B07", "B08") and name not in L06_CONTROLS:
        w = name.split(".", 1)[1]
        def fn(A, B, zero, w=w, rec=rec):
            k, p = B.shape; EB = float(B.sum())
            axis = 0 if rec in ("B05", "B06") else 1; n0 = k if axis == 0 else p
            vecs = [B[:, j] for j in range(p)] if axis == 0 else [B[i, :] for i in range(k)]
            tot = np.array([x.sum() for x in vecs]); act = tot > L06TOL * max(EB, 1e-300) * 1e-3
            if (not act.any()) or zero["B"]:
                return np.nan
            res = np.array([l06_scalars(vecs[j] / tot[j], n0, tot[j], [w])[w] for j in np.where(act)[0]])
            if rec in ("B05", "B07"):
                return float(np.nanmean(res)) if np.isfinite(res).any() else np.nan
            wts = tot[act] / tot[act].sum(); ok = np.isfinite(res)
            return float(np.nansum(res * wts) / wts[ok].sum()) if ok.any() else np.nan
    _CACHE[name] = fn
    return fn


# ---------------------------------------------------------------- registry description (registry_mapping.csv)
def describe(rid):
    reg, name = rid.split("|", 1); fam = name.split(".")[0]
    if reg == "L03":
        d = {"UX": "concentration of the flattened UX block (block-normalised)", "UUtri": "concentration of the unique UU entries",
             "POOL": "concentration of UU-unique + UX entries pooled", "UXrow": "row-wise (latent axis) concentration of UX, rows weighted by share^a",
             "UXcol": "column-wise (stimulus) concentration of UX, weighted by share^a", "UUrow": "row-wise concentration of UU",
             "AXrow": "row-wise concentration of [UU | UX]", "UXrowtot": "concentration of UX row totals", "UUdiag": "concentration of the UU diagonal",
             "UUoff": "concentration of UU off-diagonal", "UXrms": "concentration of sqrt(UX)", "POOLrms": "concentration of sqrt(pooled)",
             "S": "spectral / share statistics", "rowsp": "row-support: share^a weighted mean of (row-normalised UX + 1e-4)^q",
             "F1": "plain sum rowsp(a=.5,q=.1) + UX Hoyer", "Fz": "sum of Haar-null z-scored components (A rowsp, B UX Hoyer, C UU diag share, K UXcol Hoyer, G row-total Hoyer, S self tail)",
             "CAND": "z-scored rowsp(.75,.1) + mean UX column Hoyer", "SL0": "soft-L0 sum m/(m+eps^2) of block-normalised energies",
             "SL0m": "multiscale soft-L0 (log ratio)"}.get(fam, fam)
        norm = "Haar-null z-score" if fam in ("Fz", "CAND") else "block-normalised energies"
    else:
        d = {"B01": "UX flattened, scalar M..", "B02": "UU unique entries, scalar M..", "B03": "UU diagonal / full UU energy",
             "B04": "UU off-diagonal / full UU energy", "B05": "UX columns, unweighted mean over active columns", "B06": "UX columns, energy-weighted",
             "B07": "UX rows, unweighted mean over active rows", "B08": "UX rows, row-energy-weighted", "B09": "joint UX + UU, weights w..",
             "B10": "UX + UU diag + UU off (3blk) / UX + UU (2blk)", "B11": "UX row totals", "B12": "spectral concentration (effective / stable rank)",
             "CTRL": "rotation-invariant control"}.get(fam, fam)
        norm = "delta = c / n0 on block-normalised energy; zero blocks decided per configuration"
    control = (reg == "L06" and name in L06_CONTROLS)
    return dict(id=rid, registry=reg, name=name, recipe=fam, description=d, normalisation=norm, control_or_diagnostic=control)
