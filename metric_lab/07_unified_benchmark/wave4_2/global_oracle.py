"""EVALUATION-ONLY (W1-04): global analytic preference on O(k).  Not a blind search, not Gap 3.

Q rotates standardised T (U = Q T_s), so d raw / d U = diag(t_std) Q^T.  L(Q) uses the same contraction and registry code as W1-01.
Search: Haar starts (Gaussian QR with sign correction: both determinant components), best distinct starts refined by Givens
coordinate descent (per plane: 13-point grid over [-45, 45] degrees + 8-step golden section), fixed budgets.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import metrics as MT


def kron_contract_factory(K, J, t_std):
    k = K.shape[0]; K2 = K.reshape(k * k, k * k); ts = np.asarray(t_std)

    def AB(Q):
        Jt = Q * ts[None, :]                                   # = Jz^T, rows = rotated axes
        W = np.kron(Jt, Jt); A = np.einsum("ij,ij->i", W @ K2, W).reshape(k, k)
        B = np.einsum("ai,xik,ak->ax", Jt, J, Jt)
        return np.maximum((A + A.T) / 2, 0), np.maximum(B, 0)
    return AB


def metric_fn(mid, AB, st, zero):
    reg, name = mid.split("|", 1)
    if reg == "L03":
        f = MT.L03[name]; return lambda Q: _l03(f, AB, st, Q)
    g = MT.single_l06(name); return lambda Q: float(g(*AB(Q), zero))


def _l03(f, AB, st, Q):
    A, B = AB(Q); return float(f(B, A, st))


def haar(k, n, rng):
    X = rng.standard_normal((n, k, k)); Q, R = np.linalg.qr(X)
    return Q * np.sign(np.diagonal(R, axis1=1, axis2=2))[:, None, :]


def d_sp(Q):
    """distance to the nearest signed permutation: d^2 = 2k - 2 max_pi sum |Q_i,pi(i)|; returned normalised by sqrt(2k)"""
    k = Q.shape[0]; A = np.abs(Q); r, c = linear_sum_assignment(-A)
    return float(np.sqrt(max(2 * k - 2 * A[r, c].sum(), 0.0)) / np.sqrt(2 * k))


def nearest_signed_perm(Q):
    A = np.abs(Q); r, c = linear_sum_assignment(-A); P = np.zeros_like(Q); P[r, c] = np.sign(Q[r, c]); return P


def givens(k, i, j, t):
    G = np.eye(k); c, s = np.cos(t), np.sin(t); G[i, i] = c; G[j, j] = c; G[i, j] = -s; G[j, i] = s; return G


def refine(L, Q0, max_sweeps=8, grid=13, golden=8):
    k = Q0.shape[0]; Q = Q0.copy(); f = L(Q); n = 1; conv = False
    angs = np.deg2rad(np.linspace(-45, 45, grid)); step = angs[1] - angs[0]; gr = (np.sqrt(5) - 1) / 2
    for sw in range(max_sweeps):
        f_start = f
        for i in range(k):
            for j in range(i + 1, k):
                vals = []
                for t in angs:
                    v = L(givens(k, i, j, t) @ Q); vals.append(v if np.isfinite(v) else np.inf)
                n += grid; b = int(np.argmin(vals)); lo, hi = angs[b] - step, angs[b] + step
                x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo); f1, f2 = L(givens(k, i, j, x1) @ Q), L(givens(k, i, j, x2) @ Q); n += 2
                for _ in range(golden):
                    if f1 <= f2: hi, x2, f2 = x2, x1, f1; x1 = hi - gr * (hi - lo); f1 = L(givens(k, i, j, x1) @ Q)
                    else: lo, x1, f1 = x1, x2, f2; x2 = lo + gr * (hi - lo); f2 = L(givens(k, i, j, x2) @ Q)
                    n += 1
                cand = [(vals[b], angs[b]), (f1, x1), (f2, x2)]; fv, tv = min(cand, key=lambda z: z[0])
                if fv < f - 1e-12 * (abs(f) + 1e-12):
                    Q = givens(k, i, j, tv) @ Q; f = fv
        if f >= f_start - 1e-10 * (abs(f_start) + 1e-12):
            conv = True; break
    return Q, f, n, conv, sw + 1


def distinct_best(Qs, vals, m, min_dist=0.05):
    order = np.argsort(vals); keep = []
    for i in order:
        if not np.isfinite(vals[i]):
            continue
        if all(d_sp(Qs[i] @ Qs[j].T) > min_dist for j in keep):
            keep.append(i)
        if len(keep) == m:
            break
    return keep
