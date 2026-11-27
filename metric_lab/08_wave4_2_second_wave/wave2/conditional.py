"""W2-03 conditional Hessian energy: product designs, per-row / per-latent / per-group moment caches and the P0-P5 aggregators.

Truth-free numerics (the analytic arm feeds paired T into the same functions; the blind arm feeds Z).  02_second_wave sections 12-14.
  P0  Phi(E_{V,X} H^2)                     pooled, on the SAME product points as P1 (primary comparison)
  P1  E_X[Phi(E_V H^2)]                    every complete X row is its own condition
  P2  E_V[Phi(E_X H^2)]                    latent conditioning
  P4  sum_s w_s Phi(E[H^2 | X in S_s])     k-means of standardised X rows (seed 5, K = 4 / 8 / 16, min 4 rows, deterministic merge)
  P5  lam P0 + (1 - lam) P1                lam in .25 / .5 / .75, only for the bounded [0, 1] soft-L0 scalars (M03 / M05 recipes)
  (P3 pointwise: not run, see README)
Normalisation: N1 = each group's own block-invariant energy (the unchanged legacy adapters, primary); for the pure support recipes
B01.M03 / B01.M04 also N0 (one pooled scale E0 for every group) and N2 (local normalisation, groups weighted by their invariant UX energy).
Weights: frequency (rows / valid points) primary; equal group weights as sensitivity.
Masks: FULL (random designs, CR4); fixed banks MASK-ALL = complete-case interior rows (primary) and MASK-COL = column-wise valid
subspace (P1: a row keeps only its valid UX columns; rows without any valid column are omitted, mass recorded; P0 / P4: per-column
valid denominators).  Group zero flags are decided once per configuration (invariant energies), never per rotation.
"""
import numpy as np
from scipy.stats import special_ortho_group
from wave4_2.derivatives import ad_chunks
from wave4_2.measures import col_mask
from wave4_2 import metrics as MT
from wave1.library import scalars as l06_scalars, TOL as L06TOL

SOFT = ("L06|B01.M03.c0.01", "L06|B01.M03.c0.03", "L06|B08.M03.c0.01", "L06|B08.M03.c0.03", "L06|B08.M05", "L06|B09.w75.M03.c0.01",
        "L06|B09.w50.M03.c0.03", "L06|B10.2blk.M05")
NORM_ALT = ("L06|B01.M03.c0.01", "L06|B01.M03.c0.03", "L06|B01.M04.c0.01")


def product_design(pl, fid_sizes, seed, pool="train"):
    """(n_rows complete empirical X rows of the pool) x (n_lat pool subjects), or the full bank x n_lat subjects"""
    n_rows, n_lat = fid_sizes; P = np.sort(pl.splits[pool]); rng = np.random.default_rng([seed, 31, 5])
    sub = rng.permutation(P)[:n_lat]
    if pl.fixed_bank is not None:
        X = pl.fixed_bank.copy(); kind = "full bank"
    else:
        rows = np.concatenate([pl.starts[s] + np.arange(pl.counts[s]) for s in P])
        X = pl.x[rng.permutation(rows)[:n_rows]]; kind = "empirical complete X rows"
    return dict(sub=sub, X=X, meta=dict(pool=pool.upper(), n_lat=len(sub), n_rows=len(X), x_kind=kind, seed=seed))


def product_moments(eta, lat, X, x_mean, x_std, domain, chunk_rows=None):
    """per-row moments (n_rows, k,k,k,k) / (n_rows, p, k, k) over the latents, per-latent moments for each mask mode, column validity"""
    k = lat.shape[1]; p = X.shape[1]; nR, nL = len(X), len(lat); xs = (X - x_mean) / x_std; cm = col_mask(X, domain)
    K_r = np.zeros((nR, k, k, k, k)); J_r = np.zeros((nR, p, k, k))
    HL_uu = np.zeros((nL, nR, k, k)); HL_ux = np.zeros((nL, nR, k, p))
    for l in range(nL):
        P = np.column_stack((np.repeat(lat[l][None], nR, 0), xs))
        H = np.concatenate([h for _, _, _, h in ad_chunks(eta, P, 2048)])
        Huu, Hux = H[:, :k, :k], H[:, :k, k:]
        HL_uu[l], HL_ux[l] = Huu, Hux
    # per-row: average over latents
    K_r = np.einsum("lrab,lrcd->rabcd", HL_uu, HL_uu, optimize=True) / nL
    J_r = np.einsum("lrax,lrbx->rxab", HL_ux, HL_ux, optimize=True) / nL
    finite = np.isfinite(HL_uu).all(axis=(0, 2, 3)) & np.isfinite(HL_ux).all(axis=(0, 2, 3))
    per_lat = {}
    for mode in ("FULL", "MASK-ALL", "MASK-COL"):
        if mode == "FULL":
            ruu = np.ones(nR, bool); rux = np.ones((nR, p), bool)
        elif mode == "MASK-ALL":
            ruu = cm.all(1); rux = np.repeat(ruu[:, None], p, 1)
        else:
            ruu = np.ones(nR, bool); rux = cm
        Kl = np.einsum("lrab,lrcd->labcd", HL_uu[:, ruu], HL_uu[:, ruu], optimize=True) / max(ruu.sum(), 1)
        Jl = []
        for j in range(p):
            Hj = HL_ux[:, :, :, j][:, rux[:, j]]                  # (nL, n_valid_rows, k): basic slice first, then the row mask
            Jl.append(np.einsum("lra,lrb->lab", Hj, Hj) / max(rux[:, j].sum(), 1))
        Jl = np.stack(Jl, 1)                                       # (nL, p, k, k)
        per_lat[mode] = (Kl, Jl, ruu, rux)
    return dict(K_r=K_r, J_r=J_r, cm=cm, finite_rows=finite, per_lat=per_lat, n_lat=nL, n_rows=nR)


def kmeans_groups(Xs, K, seed=5, min_group=4, iters=100):
    """deterministic k-means (k-means++ init with a fixed seed) on standardised complete X rows; groups smaller than min_group are merged
    into the group with the nearest centroid (repeated until all groups are large enough)"""
    rng = np.random.default_rng([seed, K]); n = len(Xs)
    C = [Xs[rng.integers(n)]]
    for _ in range(1, K):
        d = np.min(((Xs[:, None] - np.array(C)[None]) ** 2).sum(-1), 1); C.append(Xs[rng.choice(n, p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((Xs[:, None] - C[None]) ** 2).sum(-1), 1)
        Cn = np.array([Xs[lab == g].mean(0) if (lab == g).any() else C[g] for g in range(K)])
        if np.allclose(Cn, C):
            break
        C = Cn
    while True:
        sizes = np.bincount(lab, minlength=len(C)); small = [g for g in range(len(C)) if 0 < sizes[g] < min_group]
        if not small:
            break
        g = small[0]; others = [h for h in range(len(C)) if h != g and sizes[h] > 0]
        tgt = others[int(np.argmin(((C[others] - C[g]) ** 2).sum(-1)))]; lab[lab == g] = tgt
    uniq = np.unique(lab); return np.searchsorted(uniq, lab)


def group_moments(pm, labels, mode):
    """frequency-weighted group moments with the mode's valid rows / per-column denominators.  -> list of (K, J, w_uu, w_ux_cols)"""
    K_r, J_r, cm = pm["K_r"], pm["J_r"], pm["cm"]; nR, p = J_r.shape[0], J_r.shape[1]
    if mode == "FULL":
        ruu = np.ones(nR, bool); rux = np.ones((nR, p), bool)
    elif mode == "MASK-ALL":
        ruu = cm.all(1); rux = np.repeat(ruu[:, None], p, 1)
    else:
        ruu = np.ones(nR, bool); rux = cm
    out = []
    for g in np.unique(labels):
        m = labels == g; mu = m & ruu
        if not mu.any():
            continue
        K = K_r[mu].mean(0); J = np.stack([J_r[m & rux[:, j], j].mean(0) if (m & rux[:, j]).any() else np.full(J_r.shape[2:], np.nan) for j in range(p)])
        out.append((K, J, int(mu.sum()), (m[:, None] & rux).sum(0)))
    return out


def contract_batch(K, J, Jt):
    """K (n, k,k,k,k), J (n, p, k, k), Jt (k, k) rows = rotated axes -> A (n, k, k), B (n, k, p)"""
    k = Jt.shape[0]; W = np.kron(Jt, Jt); K2 = K.reshape(len(K), k * k, k * k)
    A = np.einsum("ij,njl,il->ni", W, K2, W, optimize=True).reshape(len(K), k, k)
    B = np.einsum("ai,nxik,ak->nax", Jt, J, Jt, optimize=True)
    return np.maximum((A + A.transpose(0, 2, 1)) / 2, 0), np.maximum(B, 0)


class Scorer:
    """metric value of one (A, B) block with a frozen zero flag; Fz handled by component aggregation (see aggregate)"""
    def __init__(self, mid):
        self.mid = mid; reg, name = mid.split("|", 1); self.reg = reg; self.name = name
        self.fz = reg == "L03" and name.startswith("Fz.")
        self.keys = name.split(".", 1)[1].split("+") if self.fz else None
        if reg == "L06":
            self.g = MT.single_l06(name)
        elif not self.fz:
            self.f = MT.L03[name]

    def value(self, A, B, zero):
        if self.reg == "L06":
            return float(self.g(A, B, zero))
        if self.fz:
            return np.array([MT.L03["_comp"][c](B, A) for c in self.keys], float)
        return float(self.f(B, A, dict(null={})))


def soft_fixed_scale(B, E0, c, kind="M03"):
    """B01-type separable support score with a FIXED scale E0 (N0):  (1/n) sum_j  (B_j/E0) / (B_j/E0 + c/n)"""
    v = B.ravel() / E0; n = v.size; d = c / n
    return float(np.mean(v / (v + d))) if kind == "M03" else float(np.mean(1 - np.exp(-v / d)))
