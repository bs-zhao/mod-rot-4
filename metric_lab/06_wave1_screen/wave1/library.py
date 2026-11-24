"""Block 1 of 04.3: scalar library M01-M13 and structural recipes B01-B12 (all values: lower = simpler).

Inputs: A = M^UU (k x k, symmetric), B = M^UX (k x p), clipped at 0.  `zero` = dict(A=bool, B=bool): block identically zero for the WHOLE sweep
(rotation-invariant energies, decided once per configuration, never per angle).  Undefined results are NaN (never silently 0).
Names are  <recipe>.<scalar>.<setting>, e.g. B05.M03.c0.1 ; B09.w50.M10 ; CTRL.EA .
"""
import numpy as np

CS = (0.01, 0.03, 0.1, 0.3, 1.0)           # delta = c / n0 on block-normalised energy
RS = (0.25, 0.5, 0.75)                      # fractional powers
FS = (0.10, 0.25, 0.50)                     # tail fractions
TOL = 1e-12


def _mfrac(f, n): return max(1, int(np.ceil(f * n)))


def scalars(v, n0, E, want):
    """v: nonnegative vector, already divided by the denominator energy E (support-like quantities act on v; concentration on q = v/sum v).
    Returns {id: value}.  `want` is a list of ids like 'M03.c0.1'; unknown/undefined -> NaN."""
    v = np.asarray(v, float).ravel(); n = v.size; s = float(v.sum()); out = {}
    q = v / s if s > TOL else None
    srt = np.sort(q)[::-1] if q is not None else None
    for w in want:
        m, _, par = w.partition("."); val = np.nan
        if n < 2:  # singleton vectors are non-informative, not passes
            out[w] = np.nan; continue
        if m == "M01": val = float(np.sum(v > 1e-9)) / n
        elif m in ("M02", "M03", "M04", "M06"):
            c = float(par[1:]); d = c / n0
            if m == "M02": val = float(np.mean(v > d))
            elif m == "M03": val = float(np.mean(v / (v + d)))
            elif m == "M04": val = float(np.mean(1 - np.exp(-v / d)))
            else: val = float(np.mean(np.log1p(v / d)))
        elif m == "M05": val = float(np.mean([np.mean(v / (v + c / n0)) for c in CS]))
        elif m == "M07": val = float(np.mean(v ** float(par[1:])))
        elif q is None: val = np.nan
        elif m == "M08": val = float(-(q * np.log(q + 1e-300)).sum() / np.log(n))
        elif m == "M09": val = float(((1.0 / (n * (q * q).sum())) - 1.0 / n) / (1 - 1.0 / n))
        elif m == "M10": val = float(((q.sum() / np.sqrt((q * q).sum())) - 1) / (np.sqrt(n) - 1))
        elif m == "M11":
            x = np.sort(q); val = float(1 - (2 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x)) / (2 * (n - 1) * q.sum()))
        elif m in ("M12", "M13"):
            mm = _mfrac(float(par[1:]), n)
            if mm >= n: val = np.nan
            else: val = float(srt[mm:].sum()) if m == "M12" else float(np.sqrt((srt[mm:] ** 2).sum()))
        out[w] = val
    return out


def ids(spec):
    out = []
    for t in spec.split(","):
        t = t.strip()
        if t in ("M02", "M03", "M04", "M06"): out += [f"{t}.c{c}" for c in CS]
        elif t == "M07": out += [f"M07.r{r}" for r in RS]
        elif t in ("M12", "M13"): out += [f"{t}.f{f}" for f in FS]
        else: out.append(t)
    return out


ALL = "M01,M02,M03,M04,M05,M06,M07,M08,M09,M10,M11,M12,M13"
SUP = "M02,M03,M04,M05,M06,M07"


def _vech(A): return A[np.triu_indices(A.shape[0])]
def _offv(A): return A[np.triu_indices(A.shape[0], 1)]


def _norm01(name, val, n, n0):
    """comparable [0,1] version for the joint recipes (B09/B10): M06 and M07 divided by their fixed attainable upper bound."""
    m, _, par = name.partition(".")
    if m == "M06": return val / np.log1p(n0 / (n * float(par[1:]))) if n > 0 else np.nan
    if m == "M07": return val / (n ** (-float(par[1:])))
    return val


def metrics(A, B, zero):
    """All first-wave metrics for one rotation -> dict name -> float."""
    k, p = B.shape; EA = float(A.sum()); EB = float(B.sum()); out = {}
    nA = k * (k + 1) // 2; nB = k * p
    a = _vech(A) / EA if EA > TOL and not zero["A"] else None
    b = B.ravel() / EB if EB > TOL and not zero["B"] else None
    out["CTRL.EA"] = EA; out["CTRL.EB"] = EB; out["CTRL.EB_colsum_cv"] = float(np.std(B.sum(0)) / (np.mean(B.sum(0)) + 1e-300))

    def put(prefix, res):
        for kk, vv in res.items(): out[f"{prefix}.{kk}"] = vv
    def nanset(prefix, wl):
        for w in wl: out[f"{prefix}.{w}"] = np.nan
    # B01 / B02
    w = ids(ALL)
    put("B01", scalars(b, nB, EB, w)) if b is not None else nanset("B01", w)
    put("B02", scalars(a, nA, EA, w)) if a is not None else nanset("B02", w)
    # B03 / B04 (normalised by the FULL UU energy)
    w = ids(SUP)
    if EA > TOL and not zero["A"]:
        d = np.diag(A) / EA; o = _offv(A) / EA
        put("B03", scalars(d, nA, EA, w)); put("B04", scalars(o, nA, EA, w)); out["B03.CTRL.diag_share"] = float(np.trace(A) / EA); out["B04.CTRL.off_share"] = float(_offv(A).sum() * 2 / EA)
    else: nanset("B03", w); nanset("B04", w); out["B03.CTRL.diag_share"] = out["B04.CTRL.off_share"] = np.nan
    # B05/B06 columns of UX (n0 = k) ; B07/B08 rows (n0 = p)
    wc = ids("M03,M05,M07,M08,M09,M10,M11,M12"); wr = ids("M03,M05,M08,M09,M10,M11,M12")
    for tag, M_, wl, n0, axis in (("B05", B, wc, k, 0), ("B07", B, wr, p, 1)):
        vecs = [B[:, j] for j in range(p)] if axis == 0 else [B[i, :] for i in range(k)]
        tot = np.array([x.sum() for x in vecs]); act = tot > TOL * max(EB, 1e-300) * 1e-3
        if (not act.any()) or zero["B"]:
            nanset(tag, wl); nanset(tag.replace("B05", "B06").replace("B07", "B08"), wl); continue
        res = [scalars(vecs[j] / tot[j], n0, tot[j], wl) for j in np.where(act)[0]]; wts = tot[act] / tot[act].sum()
        tag2 = "B06" if tag == "B05" else "B08"
        for w_ in wl:
            vals = np.array([r[w_] for r in res])
            out[f"{tag}.{w_}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            ok = np.isfinite(vals); out[f"{tag2}.{w_}"] = float(np.nansum(vals * wts) / wts[ok].sum()) if ok.any() else np.nan
        out[f"{tag}.n_active"] = float(act.sum())
    # B09 joint UX + unique UU, separate normalisation, three weightings
    wl = ids("M03,M05,M06,M07,M08,M09,M10,M11")
    for wn, (wB, wA) in (("w50", (.5, .5)), ("w75", (.75, .25)), ("w25", (.25, .75))):
        for w_ in wl:
            sb = scalars(b, nB, EB, [w_])[w_] if b is not None else np.nan; sa = scalars(a, nA, EA, [w_])[w_] if a is not None else np.nan
            if b is not None: sb = _norm01(w_, sb, nB, nB)
            if a is not None: sa = _norm01(w_, sa, nA, nA)
            if b is not None and a is not None: v = wB * sb + wA * sa
            elif b is not None and zero["A"]: v = sb            # block identically zero for the whole sweep: ignored
            elif a is not None and zero["B"]: v = sa
            else: v = np.nan
            out[f"B09.{wn}.{w_}"] = v
    # B10 UX + UU diagonal + UU off-diagonal (equal weights); and A as a single block (equal 1/2,1/2) as sensitivity
    wl = ids("M03,M05,M07")
    for w_ in wl:
        comps = []
        sb = scalars(b, nB, EB, [w_])[w_] if b is not None else np.nan
        if b is not None: comps.append(_norm01(w_, sb, nB, nB))
        if EA > TOL and not zero["A"]:
            sd = scalars(np.diag(A) / EA, nA, EA, [w_])[w_]; so = scalars(_offv(A) / EA, nA, EA, [w_])[w_]; sa = scalars(_vech(A) / EA, nA, EA, [w_])[w_]
            out[f"B10.3blk.{w_}"] = float(np.mean(comps + [_norm01(w_, sd, nA, nA), _norm01(w_, so, nA, nA)])) if comps else np.nan
            out[f"B10.2blk.{w_}"] = float(np.mean(comps + [_norm01(w_, sa, nA, nA)])) if comps else np.nan
        elif zero["A"]:
            out[f"B10.3blk.{w_}"] = out[f"B10.2blk.{w_}"] = comps[0] if comps else np.nan
        else: out[f"B10.3blk.{w_}"] = out[f"B10.2blk.{w_}"] = np.nan
    # B11 UX row totals
    rt = B.sum(1); wl = ids("M08,M09,M10")
    put("B11", scalars(rt / max(EB, 1e-300), k, EB, wl)) if (EB > TOL and not zero["B"]) else nanset("B11", wl)
    # B12 spectral concentration
    def spec(M_, den):
        s = np.linalg.svd(M_, compute_uv=False); s2 = s * s
        if s.sum() < TOL: return np.nan, np.nan
        pp = s / s.sum(); er = float(np.exp(-(pp * np.log(pp + 1e-300)).sum())); sr = float(s2.sum() / s2[0])
        return er / den, sr / den
    if EB > TOL and not zero["B"]: out["B12.B.effrank"], out["B12.B.stablerank"] = spec(B, min(k, p))
    else: out["B12.B.effrank"] = out["B12.B.stablerank"] = np.nan
    if EA > TOL and not zero["A"]: out["B12.A.effrank"], out["B12.A.stablerank"] = spec(A, k)
    else: out["B12.A.effrank"] = out["B12.A.stablerank"] = np.nan
    return out


def metric_names():
    A = np.eye(3) + 0.1; B = np.ones((3, 4)) + np.arange(12).reshape(3, 4); return list(metrics(A, B, dict(A=False, B=False)).keys())
