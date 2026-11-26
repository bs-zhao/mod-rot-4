"""Evaluations 1 and 2 of 04.2, legacy and strict views (W1-05 section 7.1), vectorised over metrics.

Curve = one axis subset, angles 0,15,...,90.  S = {theta : L(theta) <= L_min + tau}.
  legacy tolerance      L03: tau = 1e-9 (max|L| + 1e-12) ;  L06: tau = 1e-9 (max|L| + 1e-300)
  harmonised tolerance  tau = 1e-9 (max|L| + 1e-12) for both registries
  perfect  : S intersects {0, 90}          marginal : S intersects {0, 15, 75, 90}
  strict-* : S is a non-empty SUBSET of the accepted angles
  invalid  : non-finite or constant (range <= tau) curve -> never passes
  margin m = (min interior(15..75) - min endpoint(0,90)) / range
A library passes only if EVERY sweep passes and the axis-subset coverage is complete (2^k - k - 1 subsets).
"""
import numpy as np
from scipy.stats import rankdata

ANGLES = np.array([0, 15, 30, 45, 60, 75, 90])
PERF = np.isin(ANGLES, [0, 90]); MARG = np.isin(ANGLES, [0, 15, 75, 90]); INTER = (ANGLES >= 15) & (ANGLES <= 75)


def sweeps(manifest):
    """manifest: list of dict(name, candidate_type, axes(tuple), angle) -> list of (axes, index array ordered by angle)"""
    by = {}
    for i, r in enumerate(manifest):
        if r["candidate_type"] == "sweep":
            by.setdefault(tuple(r["axes"]), []).append((r["angle"], i))
    out = []
    for ax, lst in sorted(by.items()):
        lst = sorted(lst); angs = np.array([a for a, _ in lst])
        if not np.array_equal(angs, ANGLES):
            raise ValueError(f"sweep {ax} has angles {angs}")
        out.append((ax, np.array([i for _, i in lst])))
    return out


def curve_stats(L, sw, rule="harmonised", registry_of=None):
    """L (n_rot, nm).  -> dict of arrays (n_sweeps, nm)"""
    X = np.stack([L[idx] for _, idx in sw])                    # (ns, 7, nm)
    finite = np.isfinite(X).all(1)
    Xf = np.where(np.isfinite(X), X, 0.0)
    mx = np.abs(Xf).max(1)
    if rule == "harmonised":
        tau = 1e-9 * (mx + 1e-12)
    else:                                                      # legacy per registry
        add = np.where(np.asarray(registry_of) == "L03", 1e-12, 1e-300)[None, :]
        tau = 1e-9 * (mx + add)
    mn = Xf.min(1); rng = Xf.max(1) - mn
    valid = finite & (rng > tau)
    S = Xf <= (mn + tau)[:, None, :]                           # (ns, 7, nm) minimum set
    perf = valid & (S & PERF[None, :, None]).any(1); marg = valid & (S & MARG[None, :, None]).any(1)
    sperf = valid & ~(S & ~PERF[None, :, None]).any(1); smarg = valid & ~(S & ~MARG[None, :, None]).any(1)
    inter_min = Xf[:, INTER, :].min(1); end_min = Xf[:, PERF, :].min(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        margin = np.where(valid, (inter_min - end_min) / rng, np.nan)
        end_diff = np.where(valid, np.abs(Xf[:, 0, :] - Xf[:, -1, :]) / rng, np.nan)
    itie = valid & (S & INTER[None, :, None]).any(1) & (S & PERF[None, :, None]).any(1)
    minset = (S * (1 << np.arange(7))[None, :, None]).sum(1)
    return dict(valid=valid, constant=finite & ~(rng > tau), nonfinite=~finite, perfect=perf, marginal=marg, strict_perfect=sperf,
                strict_marginal=smarg, margin=margin, interior_tie=itie, endpoint_diff=end_diff, range=rng, minset=minset)


def library_flags(cs, n_expected):
    """all-sweep library pass flags from curve_stats output"""
    ns = cs["valid"].shape[0]; complete = ns == n_expected
    out = {k: complete & cs[k].all(0) for k in ("perfect", "marginal", "strict_perfect", "strict_marginal")}
    out["n_sweeps"] = ns; out["complete"] = complete
    out["n_valid"] = cs["valid"].sum(0); out["n_perfect"] = cs["perfect"].sum(0); out["n_marginal"] = cs["marginal"].sum(0)
    out["any_constant"] = cs["constant"].any(0); out["any_nonfinite"] = cs["nonfinite"].any(0)
    with np.errstate(invalid="ignore"):
        out["min_margin"] = np.where(cs["valid"].any(0), np.nanmin(np.where(cs["valid"], cs["margin"], np.inf), 0), np.nan)
    out["n_interior_tie"] = cs["interior_tie"].sum(0)
    return out


def unique_rows(manifest, mats=None):
    """baseline + every sweep rotation with angle > 0 (the 0-degree entries duplicate R*); optional exact-matrix dedup"""
    keep = np.array([(r["candidate_type"] == "baseline_unswept") or (r["angle"] > 0) for r in manifest])
    if mats is None:
        return keep
    seen, ded = [], keep.copy()
    for i in np.where(keep)[0]:
        if any(np.allclose(mats[i], mats[j], atol=1e-12) for j in seen):
            ded[i] = False
        else:
            seen.append(i)
    return keep, ded


def _corr(a, b):
    a = a - a.mean(0); b = b - b.mean(0); den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(0) / den, np.nan)


def eval1(Lo, Li, manifest, sw, rows=None):
    """pooled Pearson / Spearman over unique rows, mean and min per-curve Pearson; constant / non-finite -> NA"""
    rows = unique_rows(manifest) if rows is None else rows
    o, i = Lo[rows], Li[rows]
    ok = np.isfinite(o).all(0) & np.isfinite(i).all(0)
    ok &= (np.ptp(np.where(np.isfinite(o), o, 0), 0) > 1e-9 * (np.abs(np.nan_to_num(o)).max(0) + 1e-12))
    ok &= (np.ptp(np.where(np.isfinite(i), i, 0), 0) > 1e-9 * (np.abs(np.nan_to_num(i)).max(0) + 1e-12))
    pe = np.where(ok, _corr(np.nan_to_num(o), np.nan_to_num(i)), np.nan)
    sp = np.where(ok, _corr(rankdata(np.nan_to_num(o), axis=0), rankdata(np.nan_to_num(i), axis=0)), np.nan)
    cc = np.stack([_corr(np.nan_to_num(Lo[idx]), np.nan_to_num(Li[idx])) for _, idx in sw])
    cfin = np.stack([np.isfinite(Lo[idx]).all(0) & np.isfinite(Li[idx]).all(0) for _, idx in sw])
    cc = np.where(cfin, cc, np.nan)
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmean = np.nanmean(cc, 0); cmin = np.nanmin(cc, 0)
        ro = np.nanmax(o, 0) - np.nanmin(o, 0); ri = np.nanmax(i, 0) - np.nanmin(i, 0)
    return dict(n=int(rows.sum()), pearson=pe, spearman=sp, curve_mean=cmean, curve_min=cmin, frac_curve_na=np.isnan(cc).mean(0), range_o=ro, range_i=ri)


def matrix_errors(Ao, Bo, Ai, Bi, zero_tol=1e-9):
    """per rotation: relative Frobenius errors of the energy-normalised blocks and inferred energy share in oracle-zero cells"""
    def nrm(A, B):
        E = A.sum() + B.sum(); return (A / E, B / E) if E > 0 else (A * np.nan, B * np.nan)
    ao, bo = nrm(Ao, Bo); ai, bi = nrm(Ai, Bi)
    eB = np.linalg.norm(bi - bo) / (np.linalg.norm(bo) + 1e-300)
    eA = np.linalg.norm(ai - ao) / (np.linalg.norm(ao) + 1e-300) if np.linalg.norm(ao) > 0 else np.nan
    zB = bo <= zero_tol; zA = ao <= zero_tol
    fp = float(bi[zB].sum() + ai[zA].sum())
    return eA, eB, fp
