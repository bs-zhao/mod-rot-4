"""W3-01 row-weight repair family (03_third_wave sections 5-6).  Pure functions of the UX energy matrix B (k x p, >= 0).

Notation: r_i = sum_j B_ij, E = sum_i r_i, e_i = r_i / E, q_ij = B_ij / r_i (rows with r_i > 0),
          s_i(c) = (1/p) sum_j q_ij / (q_ij + c/p)       (M03 row support; Lab 06 B07/B08 row score).
Recipes (all lower = simpler, all invariant to signed permutations of the rows):
  POW(a, c)     sum_{i in A} r_i^a s_i / sum_{i in A} r_i^a          A = historical active rows r_i > 1e-12 * 1e-3 * E
                (a = 1, c = .03 is E02 = L06|B08.M03.c0.03 exactly; a = 0 is B07)
  SM(a, t)      w_i = (e_i + d)^a A_d(e_i) / sum,  A_d(e) = e / (e + d),  d = t / k,  c = .03; a truly zero row has weight 0
  BL(rho)       w_i = rho w_i^(0, d) + (1 - rho) e_i,  d = .01 / k,  c = .03
  GL(lam, a)    lam POW(a, .03) + (1 - lam) G_.03,   G_c = (1/n) sum_ij v_ij / (v_ij + c/n),  v = B / E,  n = k p   (= B01.M03.c0.03)
  ACT(beta)     POW(1, .03) + beta (1/k) sum_i e_i / (e_i + d),  d = .01 / k
Undefined (NaN): zero / non-finite block energy.  `terms` returns the per-row (weight, score) used by the exact decomposition
  Delta L = sum_i wbar_i Delta s_i + sum_i sbar_i Delta w_i  (+ Delta of the non-row component for GL / ACT, reported separately).
"""
import numpy as np

TOL = 1e-12
ACT_CUT = TOL * 1e-3          # historical Lab 06 active-row cutoff (relative to the block energy)


def _rows(B, c):
    B = np.asarray(B, float); k, p = B.shape; r = B.sum(1); E = float(r.sum())
    s = np.zeros(k); pos = r > 0
    if pos.any():
        q = B[pos] / r[pos, None]; d = c / p
        s[pos] = np.mean(q / (q + d), axis=1)
    return r, E, s


def g_support(B, c=0.03):
    B = np.asarray(B, float); E = float(B.sum()); n = B.size
    if not (E > TOL):
        return np.nan
    v = B.ravel() / E; return float(np.mean(v / (v + c / n)))


def weights(spec, r, E, k):
    """normalised row weights (sum 1) and the activity mask under the recipe's own convention"""
    kind = spec["kind"]
    if kind in ("POW", "GL", "ACT") or kind == "E":
        a = spec.get("a", 1.0); act = r > ACT_CUT * E
        w = np.zeros_like(r); w[act] = r[act] ** a if a != 0 else 1.0
        return w / w.sum(), act
    e = r / E
    if kind == "SM":
        d = spec["t"] / k; A = e / (e + d); w = (e + d) ** spec["a"] * A
        return w / w.sum(), r > 0
    if kind == "BL":
        d = 0.01 / k; A = e / (e + d); w0 = A / A.sum(); w = spec["rho"] * w0 + (1 - spec["rho"]) * e
        return w / w.sum(), r > 0
    raise ValueError(kind)


def terms(spec, B):
    """-> dict(w, s, act, L_row, extra, L) for one rotation's B under `spec` (NaN-safe)"""
    B = np.asarray(B, float); k, p = B.shape
    c = spec.get("c", 0.03); r, E, s = _rows(B, c)
    if not (E > TOL) or not np.isfinite(E):
        return dict(w=np.full(k, np.nan), s=np.full(k, np.nan), act=np.zeros(k, bool), L_row=np.nan, extra=np.nan, L=np.nan, r=r, E=E)
    w, act = weights(spec, r, E, k); s = np.where(w > 0, s, 0.0); Lr = float((w * s).sum()); extra = 0.0
    if spec["kind"] == "GL":
        extra = (1 - spec["lam"]) * g_support(B, 0.03); L = spec["lam"] * Lr + extra
    elif spec["kind"] == "ACT":
        d = 0.01 / k; e = r / E; extra = spec["beta"] * float(np.mean(e / (e + d))); L = Lr + extra
    else:
        L = Lr
    return dict(w=w, s=s, act=act, L_row=Lr, extra=extra, L=float(L), r=r, E=E)


def value(spec, B):
    return terms(spec, B)["L"]


def decompose(spec, B_a, B_b, perm=None):
    """exact symmetric decomposition of L(b) - L(a); `perm[i]` = row of b matched to row i of a (evaluation-only matching).
    Rows absent from a weight set carry weight 0 and score 0 (declared convention)."""
    ta, tb = terms(spec, B_a), terms(spec, B_b)
    if perm is None:
        perm = np.arange(len(ta["w"]))
    wa, sa = ta["w"], ta["s"]; wb, sb = tb["w"][perm], tb["s"][perm]
    wbar, sbar = (wa + wb) / 2, (sa + sb) / 2
    score_term = wbar * (sb - sa); weight_term = sbar * (wb - wa)
    dL_row = tb["L_row"] - ta["L_row"]
    if spec["kind"] == "GL":
        dL_row = spec["lam"] * dL_row; score_term = spec["lam"] * score_term; weight_term = spec["lam"] * weight_term
    return dict(dL=tb["L"] - ta["L"], dL_row=dL_row, score_term=score_term, weight_term=weight_term, d_extra=tb["extra"] - ta["extra"],
                residual=float(dL_row - score_term.sum() - weight_term.sum()), act_a=ta["act"], act_b=tb["act"][perm],
                w_a=wa, w_b=wb, s_a=sa, s_b=sb, support_change=int((ta["act"] != tb["act"][perm]).sum()))


def registry(rc):
    """expand configs/repair_candidates.json into {cid: spec}"""
    out = {}
    for fam in rc["families"]:
        for p in fam["grid"]:
            spec = dict(kind=fam["kind"], **p); cid = fam["id_format"].format(**p); spec["cid"] = cid; spec["family"] = fam["family"]; out[cid] = spec
    return out
