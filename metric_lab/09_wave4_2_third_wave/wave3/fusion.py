"""W3-02 same-frame fusion of two frozen component objectives (03_third_wave section 9).  Truth-free.

Calibration: a blind Haar reference sample of 512 rotations (RNG [20261127, 9, k], independent of every search start stream) evaluated on
the TRAIN cache of each component; z_m(R) = (L_m(R) - median_H L_m) / IQR_H L_m.  A component with IQR <= 1e-9 (|median| + 1e-12) is
flagged invariant / uninformative and every fusion that needs it is invalid (NaN), never silently a different method.
Fusions (lower = better):  F_w = w z_E + (1 - w) z_X (w = .25/.5/.75);  F_max = max(z_E, z_X);
  epsilon-constraint  z_X + P max(0, z_E - (zE_min + delta)),  P = 1000, delta = .1/.25/.5 Haar-IQR units, zE_min = the blind TRAIN minimum
  of the E component (lowest selected loss of its three S0 search seeds on the same cache).  Violations are reported.
Selector (section 9.2): finalists of both component searches and the three F_w searches (TRAIN); on the VALIDATION caches evaluate both
z-scores with the TRAIN calibration; choose min max(z_E, z_X), ties (1e-12) by the sum, then by the lexical finalist key.
"""
import numpy as np
from wave2.optimizers import haar

CAL_SEED = 20261127
N_CAL = 512
PENALTY = 1000.0


def calibrate(L, k, kind="iqr"):
    Hs = haar(k, N_CAL, np.random.default_rng([CAL_SEED, 9, k])); v = np.array([L(Q) for Q in Hs], float); v = v[np.isfinite(v)]
    if v.size < N_CAL // 2:
        return dict(center=np.nan, scale=np.nan, invariant=True, n=int(v.size), kind=kind)
    med = float(np.median(v))
    if kind == "iqr":
        sc = float(np.subtract(*np.percentile(v, [75, 25])))
    elif kind == "sd":
        med = float(v.mean()); sc = float(v.std())
    elif kind == "q9505":
        sc = float(np.subtract(*np.percentile(v, [95, 5])))
    else:
        raise ValueError(kind)
    inv = not (sc > 1e-9 * (abs(med) + 1e-12))
    return dict(center=med, scale=sc, invariant=bool(inv), n=int(v.size), kind=kind, q=np.percentile(v, [5, 25, 50, 75, 95]).tolist())


class Z:
    def __init__(self, L, cal):
        self.L, self.cal = L, cal

    def __call__(self, R):
        if self.cal["invariant"]:
            return np.nan
        return (self.L(R) - self.cal["center"]) / self.cal["scale"]


class Fused:
    def __init__(self, zE, zX, kind, w=None, delta=None, zE_min=None):
        self.zE, self.zX, self.kind, self.w, self.delta, self.zE_min = zE, zX, kind, w, delta, zE_min; self.n_evals = 0

    def parts(self, R):
        return self.zE(R), self.zX(R)

    def __call__(self, R):
        self.n_evals += 1; e, x = self.parts(R)
        if not (np.isfinite(e) and np.isfinite(x)):
            return np.nan
        if self.kind == "w":
            return self.w * e + (1 - self.w) * x
        if self.kind == "max":
            return max(e, x)
        if self.kind == "eps":
            return x + PENALTY * max(0.0, e - (self.zE_min + self.delta))
        raise ValueError(self.kind)


def pareto(points):
    """points (n, 2) lower-better -> boolean mask of the non-dominated set"""
    P = np.asarray(points, float); n = len(P); keep = np.isfinite(P).all(1)
    for i in range(n):
        if not keep[i]:
            continue
        dom = (P[:, 0] <= P[i, 0]) & (P[:, 1] <= P[i, 1]) & ((P[:, 0] < P[i, 0]) | (P[:, 1] < P[i, 1])) & np.isfinite(P).all(1)
        if dom.any():
            keep[i] = False
    return keep


def select(cands):
    """cands: list of dict(key, zE_val, zX_val) -> chosen key (min max, ties by sum then key)"""
    ok = [c for c in cands if np.isfinite(c["zE_val"]) and np.isfinite(c["zX_val"])]
    if not ok:
        return None
    best = min(max(c["zE_val"], c["zX_val"]) for c in ok)
    tied = [c for c in ok if max(c["zE_val"], c["zX_val"]) <= best + 1e-12 * (abs(best) + 1)]
    bs = min(c["zE_val"] + c["zX_val"] for c in tied)
    tied = [c for c in tied if c["zE_val"] + c["zX_val"] <= bs + 1e-12 * (abs(bs) + 1)]
    return sorted(tied, key=lambda c: c["key"])[0]["key"]
