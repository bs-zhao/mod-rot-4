"""Truth-free numerical search on O(k) (02_second_wave section 6, protocol S0 / S1).

Haar starts (Gaussian QR with sign correction, both determinant components) -> the best distinct starts (distance between
starts measured modulo signed permutations, since every registry metric is signed-permutation invariant) -> Givens coordinate
descent with a DECREASING angular span: each plane is scanned on a 13-point grid over [-span, span] and refined by an 8-step
golden section in the best bracket; when a whole sweep improves the loss by less than the relative tolerance the span shrinks
(45 -> 15 -> 5 -> 1.5 -> 0.5 degrees); convergence = no improvement at the smallest span.  Function evaluations are capped.
The search RNG ([seed, 77, k]) is distinct from every integration RNG.  Selection = lowest valid loss; nothing here sees T.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

SPANS = (45.0, 15.0, 5.0, 1.5, 0.5)
BUDGET = {"S0": {3: (400, 8), 4: (400, 8), 6: (2000, 30)}, "S1": {3: (2000, 40), 4: (2000, 40), 6: (10000, 150)}}


def haar(k, n, rng):
    X = rng.standard_normal((n, k, k)); Q, R = np.linalg.qr(X)
    return Q * np.sign(np.diagonal(R, axis1=1, axis2=2))[:, None, :]


def d_sp(Q):
    """normalised distance to the nearest signed permutation: sqrt(2k - 2 max_pi sum |Q_i,pi(i)|) / sqrt(2k)"""
    k = Q.shape[0]; A = np.abs(Q); r, c = linear_sum_assignment(-A)
    return float(np.sqrt(max(2 * k - 2 * A[r, c].sum(), 0.0)) / np.sqrt(2 * k))


def rel_dist(Q1, Q2):
    """distance between two rotations modulo signed permutations of the rotated axes"""
    return d_sp(Q1 @ Q2.T)


def givens(k, i, j, t):
    G = np.eye(k); c, s = np.cos(t), np.sin(t); G[i, i] = c; G[j, j] = c; G[i, j] = -s; G[j, i] = s; return G


def refine(L, Q0, spans=SPANS, grid=13, golden=8, max_sweeps=30, max_evals=60000, rtol=1e-9):
    k = Q0.shape[0]; Q = Q0.copy(); f = L(Q); n = 1; si = 0; conv = False; trace = [(0, float(f), spans[0])]
    f = f if np.isfinite(f) else np.inf; gr = (np.sqrt(5) - 1) / 2; sw = 0
    for sw in range(1, max_sweeps + 1):
        f_start = f; span = np.deg2rad(spans[si]); angs = np.linspace(-span, span, grid); step = angs[1] - angs[0]
        for i in range(k):
            for j in range(i + 1, k):
                vals = np.array([L(givens(k, i, j, t) @ Q) for t in angs]); vals[~np.isfinite(vals)] = np.inf; n += grid
                b = int(np.argmin(vals)); lo, hi = angs[b] - step, angs[b] + step
                x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo); f1, f2 = L(givens(k, i, j, x1) @ Q), L(givens(k, i, j, x2) @ Q); n += 2
                f1 = f1 if np.isfinite(f1) else np.inf; f2 = f2 if np.isfinite(f2) else np.inf
                for _ in range(golden):
                    if f1 <= f2:
                        hi, x2, f2 = x2, x1, f1; x1 = hi - gr * (hi - lo); f1 = L(givens(k, i, j, x1) @ Q); f1 = f1 if np.isfinite(f1) else np.inf
                    else:
                        lo, x1, f1 = x1, x2, f2; x2 = lo + gr * (hi - lo); f2 = L(givens(k, i, j, x2) @ Q); f2 = f2 if np.isfinite(f2) else np.inf
                    n += 1
                fv, tv = min([(vals[b], angs[b]), (f1, x1), (f2, x2)], key=lambda z: z[0])
                if fv < f - rtol * (abs(f) + 1e-12):
                    Q = givens(k, i, j, tv) @ Q; f = fv
        trace.append((sw, float(f), spans[si]))
        if not (f < f_start - rtol * (abs(f_start) + 1e-12)):
            if si == len(spans) - 1:
                conv = True; break
            si += 1
        if n >= max_evals:
            break
    return dict(Q=Q, f=float(f), n_evals=int(n), converged=bool(conv), sweeps=int(sw), final_span=spans[si], trace=trace)


def distinct_best(Qs, vals, m, min_dist=0.05):
    order = np.argsort(np.where(np.isfinite(vals), vals, np.inf)); keep = []
    for i in order:
        if not np.isfinite(vals[i]):
            continue
        if all(rel_dist(Qs[i], Qs[j]) > min_dist for j in keep):
            keep.append(int(i))
        if len(keep) == m:
            break
    return keep


def search(L, k, seed, budget="S0", identity_start=True, tie_breaker=None, tie_rtol=1e-9):
    """-> dict(R, f, finalists=[...], starts=(vals), diagnostics).  `tie_breaker(Q)`: optional blind validation loss used ONLY among
    finalists whose optimisation loss is within tie_rtol of the best (predeclared numerical tie rule)."""
    n_starts, n_ref = BUDGET[budget][k]
    Qs = haar(k, n_starts, np.random.default_rng([seed, 77, k]))
    if identity_start:
        Qs = np.concatenate([np.eye(k)[None], Qs])
    vals = np.array([L(Q) for Q in Qs], float)
    keep = distinct_best(Qs, vals, n_ref)
    fin = []
    for rank, i in enumerate(keep):
        r = refine(L, Qs[i]); r.update(start=int(i), rank=rank, L_initial=float(vals[i]), identity_start=bool(identity_start and i == 0)); fin.append(r)
    if not fin:
        return dict(R=None, f=np.nan, finalists=[], start_vals=vals, status="no valid start")
    fs = np.array([r["f"] for r in fin]); best = int(np.argmin(fs)); tied = np.where(fs <= fs[best] + tie_rtol * (abs(fs[best]) + 1e-12))[0]
    tie_used = False
    if tie_breaker is not None and len(tied) > 1:
        tv = np.array([tie_breaker(fin[i]["Q"]) for i in tied]); best = int(tied[np.nanargmin(tv)]); tie_used = True
    near = [i for i in range(len(fin)) if fs[i] <= fs[best] + 1e-3 * (abs(fs[best]) + 1e-12)]
    sep = max([rel_dist(fin[i]["Q"], fin[best]["Q"]) for i in near] + [0.0])
    return dict(R=fin[best]["Q"], f=float(fs[best]), best_index=best, finalists=fin, start_vals=vals, status="ok", n_tied=int(len(tied)),
                tie_breaker_used=tie_used, near_minima_max_sep=float(sep), n_near_minima=len(near),
                best_converged=bool(fin[best]["converged"]), n_evals=int(sum(r["n_evals"] for r in fin) + len(vals)))
