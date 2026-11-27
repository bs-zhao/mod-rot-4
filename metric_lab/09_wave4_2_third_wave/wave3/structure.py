"""W3-03 first-order structure: gradient (G) and observed-row contrast (D) targets (03_third_wave sections 11-13, 16).  Truth-free.

Product design (TRAIN / VALIDATION pools for blind search; TEST in the evaluation-only local arm):
  random X designs: n_V pool-subject latents (seeded permutation) x n_X empirical complete X rows of the same pool (independent of V)
  fixed banks:      n_V pool-subject latents x the full bank (500 rows)
  medium (search) 256 x 256 / 128 x 500;  pilot (local Gap 1/2) 128 x 128 / 32 x 500.
At every grid point the scalar eta (class B logit difference, class C mean in Y units) is differentiated ONCE (AD, float64): a(v, x) = d eta
/ d raw-latent (k).  All targets are built from the stored grid of gradients by averaging / subtraction (no further decoder calls):
  C_G = E_{V,X}[a a^T],  abar(v) = E_X a,  C_0 = E_V[abar abar^T],  C_D = E_{V,X}[(a - abar)(a - abar)^T]   (C_G = C_0 + C_D exactly)
  C_x = E_V[a(V,x) a(V,x)^T] (one per observed row);  centred C^c_x = E_V[(a - abar)(a - abar)^T] (D-C, one column per row)
  C_pair(x, x') = E_V[b b^T],  b = a(v,x) - a(v,x')  for frozen observed-row pairs (D-U / D-N / D-F / D-A).
Rotation: grad_U = R diag(s) a (V = (Z - c)/s, U = R V); a matrix C becomes R S C S R^T.  Squared quantities are never rotated.
Gradient / contrast targets take NO X derivative, so they use every observed row (valid zero amounts included); the ST-C control restricts
the rows to the interior (every bounded X column strictly inside its domain) on identical latents.  The same grid also yields the pooled HE
moments (Hessian from the same AD pass) as the matched-support control.
"""
import json
import numpy as np
from .common import OUT, chan
from wave2.objectives import save_moments, load_moments
from wave4_2.moments import Moments
from wave4_2.measures import col_mask
from wave4_2.derivatives import ad_chunks
from . import rowmetrics as RM

GRAD = OUT / "grad_cache"
SIZES = {"pilot": dict(rand=(128, 128), bank=(32, 500), pairs=256), "C": dict(rand=(256, 256), bank=(128, 500), pairs=1024)}
PAIR_RULES = ("U", "N", "F", "A")


# ---------------------------------------------------------------- design + pair designs (complete observed X rows, no response, no truth)
def grid_design(pl, pool, fid, seed):
    sz = SIZES[fid]; P = np.sort(pl.splits[pool]); rng = np.random.default_rng([seed, 101])
    if pl.fixed_bank is not None:
        nV, _ = sz["bank"]; X = pl.fixed_bank.copy(); xrows = None
    else:
        nV, nX = sz["rand"]
        rows = np.concatenate([pl.starts[s] + np.arange(pl.counts[s]) for s in P])
        xrows = rows[np.random.default_rng([seed, 102]).permutation(len(rows))[:nX]]; X = pl.x[xrows]
    sub = rng.permutation(P)[:nV]
    return dict(sid=sub, lat=pl.z[sub], x=X, xrows=xrows, meta=dict(pool=pool.upper(), fid=fid, seed=seed, n_V=int(len(sub)), n_X=int(len(X)),
                                                                     design="pool latents x full bank" if xrows is None else "pool latents x empirical rows (product)"))


def pair_designs(xs, n_pairs, seed, rows_ok=None, quant=0.9):
    """frozen pair index arrays per rule on TRAIN-standardised complete X rows xs (n_X, p); identical rows excluded"""
    n = len(xs); idx = np.arange(n) if rows_ok is None else np.where(rows_ok)[0]
    I, J = np.triu_indices(len(idx), 1); I, J = idx[I], idx[J]; D = xs[I] - xs[J]; d2 = (D ** 2).sum(1); nz = d2 > 1e-24
    I, J, D, d2 = I[nz], J[nz], D[nz], d2[nz]; out, audit = {}, {}
    rng = lambda r: np.random.default_rng([seed, 103, r])
    m = min(n_pairs, len(I))
    out["U"] = np.stack([I, J], 1)[rng(1).choice(len(I), m, replace=False)]
    # D-N: 5-nearest-neighbour graph edges (undirected, deduplicated), uniform sample
    Xs = xs[idx]; dd = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1); np.fill_diagonal(dd, np.inf); dd[dd <= 1e-24] = np.inf
    nn = np.argsort(dd, 1)[:, :5]; E = {(min(a, b), max(a, b)) for a in range(len(idx)) for b in nn[a] if np.isfinite(dd[a, b])}
    E = np.array(sorted(E)); E = idx[E]; out["N"] = E[rng(2).choice(len(E), min(n_pairs, len(E)), replace=False)]
    far = d2 >= np.percentile(d2, 75); Fi = np.where(far)[0]; out["F"] = np.stack([I, J], 1)[Fi[rng(3).choice(len(Fi), min(n_pairs, len(Fi)), replace=False)]]
    # D-A: per coordinate, pairs whose share of squared distance in that coordinate is in its top decile; n_pairs / p per coordinate
    p = xs.shape[1]; per = max(1, n_pairs // p); sel = []; short = {}
    ratio = D ** 2 / d2[:, None]
    for j in range(p):
        cand = np.where((ratio[:, j] >= np.quantile(ratio[:, j], quant)) & (D[:, j] ** 2 > 0))[0]
        take = cand[rng(10 + j).choice(len(cand), min(per, len(cand)), replace=False)] if len(cand) else np.array([], int)
        short[j] = int(per - len(take)); sel.append(take)
    sel = np.concatenate(sel); out["A"] = np.stack([I, J], 1)[sel]
    for r, pr in out.items():
        dist = np.sqrt(((xs[pr[:, 0]] - xs[pr[:, 1]]) ** 2).sum(1))
        share = ((xs[pr[:, 0]] - xs[pr[:, 1]]) ** 2) / np.maximum(dist[:, None] ** 2, 1e-300)
        audit[r] = dict(n_pairs=int(len(pr)), dist_q=np.percentile(dist, [5, 25, 50, 75, 95]).tolist(), max_coord_share_mean=float(share.max(1).mean()),
                        coord_share_mean=share.mean(0).tolist(), candidate_pairs=int(len(I)))
    audit["A"]["per_coordinate_shortage"] = short
    return out, audit


# ---------------------------------------------------------------- base matrices from a gradient grid
def base_matrices(G, pairs, rows=None):
    """G (n_V, n_X, k) raw-frame gradients -> dict of raw-frame matrices (rows: optional subset of X rows)"""
    if rows is not None:
        G = G[:, rows]
    nV, nX, k = G.shape; ab = G.mean(1); Dc = G - ab[:, None, :]
    out = dict(CG=np.einsum("vxa,vxb->ab", G, G) / (nV * nX), C0=ab.T @ ab / nV, CD=np.einsum("vxa,vxb->ab", Dc, Dc) / (nV * nX),
               Cx=np.einsum("vxa,vxb->xab", G, G) / nV, Cc=np.einsum("vxa,vxb->xab", Dc, Dc) / nV)
    for r, pr in pairs.items():
        b = G[:, pr[:, 0]] - G[:, pr[:, 1]]; out[f"P{r}"] = np.einsum("vla,vlb->lab", b, b) / nV
    return out


def _he_moments(pl, P, cm, masks, eta, chunk=1024):
    """gradients of eta wrt raw latent AND pooled HE moments from one AD pass"""
    k = pl.k; acc = {m: Moments(k, pl.p, m) for m in masks}; G = np.zeros((len(P), k)); E = np.zeros(len(P))
    for sl, e, g, H in ad_chunks(eta, P, chunk):
        G[sl] = g[:, :k]; E[sl] = e
        for m in masks:
            acc[m].add(H, cm[sl])
    return G, E, {m: a.finish() for m, a in acc.items()}


def compute_grid(pl, d, eta, lat=None, he=True):
    """gradient grid (n_V, n_X, k) and (optionally) the pooled HE moments of the same points.  `lat` overrides the latent rows (oracle arm)."""
    from wave2.objectives import masks_of
    lat = d["lat"] if lat is None else lat; nV, nX = len(lat), len(d["x"])
    xs = (d["x"] - pl.x_mean) / pl.x_std; Lr = np.repeat(lat, nX, 0); Xr = np.tile(xs, (nV, 1)); P = np.column_stack((Lr, Xr))
    cm = col_mask(np.tile(d["x"], (nV, 1)), pl.x_domain)
    G, E, mo = _he_moments(pl, P, cm, masks_of(pl) if he else (), eta)
    return G.reshape(nV, nX, -1), E.reshape(nV, nX), mo


def interior_rows(pl, X):
    return col_mask(X, pl.x_domain).all(1)


def grad_cache_path(pl, pool, seed, fid="C", tag=""):
    return GRAD / pl.toy.replace("#", "_") / f"{pool}_{fid}_s{seed}{tag}.npz"


def build_grad_cache(pl, pool, seed, fid="C", eta=None):
    f = grad_cache_path(pl, pool, seed, fid)
    if f.is_file():
        return f
    import run_blind as RB8
    d = grid_design(pl, pool, fid, seed); eta = eta or RB8.eta_factory(pl)()
    G, E, mo = compute_grid(pl, d, eta)
    xs = (d["x"] - pl.x_mean) / pl.x_std; inter = interior_rows(pl, d["x"])
    pairs, audit = pair_designs(xs, SIZES[fid]["pairs"], seed)
    arr = dict(G=G, eta=E, x=d["x"], sid=d["sid"], interior=inter, meta=np.array(json.dumps(dict(d["meta"], audit=audit), default=float)))
    for r, pr in pairs.items():
        arr[f"pairs_{r}"] = pr
    if pl.fixed_bank is not None and pl.x_domain is not None:
        pairs_i, audit_i = pair_designs(xs, SIZES[fid]["pairs"], seed, rows_ok=inter)
        for r, pr in pairs_i.items():
            arr[f"pairsI_{r}"] = pr
        arr["meta_interior"] = np.array(json.dumps(audit_i, default=float))
    for m, r in mo.items():
        for kk in ("K", "J", "nJ", "coverage"):
            arr[f"HE|{m}|{kk}"] = np.asarray(r[kk])
        arr[f"HE|{m}|scalars"] = np.array([r["nK"], r["n_points"], r["nonfinite"], float(r["valid"])])
    f.parent.mkdir(parents=True, exist_ok=True); tmp = f.with_suffix(".tmp.npz"); np.savez_compressed(tmp, **arr); tmp.replace(f)
    return f


_GC = {}


def load_grad(path):
    path = str(path)
    if path in _GC:
        return _GC[path]
    z = np.load(path, allow_pickle=False); G = z["G"]
    pairs = {r: z[f"pairs_{r}"] for r in PAIR_RULES}
    c = dict(G=G, x=z["x"], interior=z["interior"], meta=json.loads(str(z["meta"])), mats=base_matrices(G, pairs), valid=bool(np.isfinite(G).all()))
    if "pairsI_U" in z.files:
        pI = {r: z[f"pairsI_{r}"] for r in PAIR_RULES}; rows = np.where(z["interior"])[0]
        # interior control: same latents, only interior rows; pair indices are global row ids -> remap
        remap = -np.ones(len(z["interior"]), int); remap[rows] = np.arange(len(rows))
        c["mats_int"] = base_matrices(G, {r: remap[p] for r, p in pI.items()}, rows=rows)
    he = {}
    for key in z.files:
        if key.startswith("HE|"):
            _, m, kk = key.split("|"); r = he.setdefault(m, {})
            if kk == "scalars":
                r["nK"], r["n_points"], r["nonfinite"], r["valid"] = int(z[key][0]), int(z[key][1]), int(z[key][2]), bool(z[key][3])
            else:
                r[kk] = z[key]
    c["HE"] = he
    _GC[path] = c
    return c


def grad_cache(pl, pool, seed, mask=None, fid="C"):
    return load_grad(build_grad_cache(pl, pool, seed, fid))


# ---------------------------------------------------------------- scores
def sup_vec(v, c):
    s = v.sum()
    if not (s > 1e-300):
        return np.nan
    q = v / s; return float(np.mean(q / (q + c / len(q))))


def pow_vec(v, r):
    s = v.sum()
    if not (s > 1e-300):
        return np.nan
    return float(np.mean((v / s) ** r))


def offdiag(M, den):
    if not (den > 1e-300):
        return np.nan
    o = M - np.diag(np.diag(M)); return float((o ** 2).sum() / den)


def glob_sup(Bd, c=0.03):
    E = Bd.sum()
    if not (E > 1e-300):
        return np.nan
    v = Bd.ravel() / E; return float(np.mean(v / (v + c / v.size)))


def col_sup(Bd, c=0.03):
    """column-wise support over the k axes, column-energy weighted (B06-type on observed contrasts)"""
    tot = Bd.sum(0); act = tot > 1e-15 * max(tot.sum(), 1e-300)
    if not act.any():
        return np.nan
    q = Bd[:, act] / tot[act]; k = Bd.shape[0]; s = np.mean(q / (q + c / k), 0); w = tot[act] / tot[act].sum()
    return float((w * s).sum())


ROW_E02 = dict(kind="POW", a=1.0, c=0.03)


class StructObjective:
    """spec: target in {G, G0, GD, GJ, GX, GXJ, GXJt, P<rule>, PC} (+ rows='all'|'interior'), recipe, params; or MEANFUS / HECTRL."""
    def __init__(self, cache, scale, spec):
        self.spec = spec; s = np.asarray(scale, float); self.k = len(s); self.n_evals = 0
        if spec.get("rows") == "interior" and "mats_int" not in cache:
            self.valid = False; return                                               # interior control defined for bounded banks only
        mats = cache["mats_int"] if spec.get("rows") == "interior" else cache["mats"]
        self.valid = cache["valid"]
        S = lambda M: M * s[..., :, None] * s[..., None, :]                      # raw -> V frame: S M S
        t = spec["target"]
        if t == "HECTRL":
            from wave2.objectives import Objective
            m = spec.get("mask", "primary")
            if m == "primary":
                m = "MASK-ALL" if "MASK-ALL" in cache["HE"] else "FULL"
            self._he = Objective(cache["HE"][m], s, spec["metric"]); self.valid = self._he.valid; return
        if t == "MEANFUS":
            self._pair = StructObjective(cache, scale, spec["pair"]); self._mean = StructObjective(cache, scale, dict(target="G0", recipe="M03", c=0.03))
            self.valid = self._pair.valid and self._mean.valid; return
        key = {"G": "CG", "G0": "C0", "GD": "CD", "GX": "Cx", "GXJ": "Cx", "GXJt": "Cx", "PC": "Cc"}.get(t, t)
        if t == "GJ":
            self.M = np.stack([S(mats["C0"]), S(mats["CD"])])
        else:
            self.M = S(mats[key])
        if self.M.ndim == 2:
            self.M = self.M[None]
        tr = np.trace(self.M, axis1=1, axis2=2); self.fro = (self.M ** 2).sum((1, 2)); self.tr = tr
        if not (tr.sum() > 1e-300):
            self.valid = False                                                       # rotation-invariant zero block: undefined

    def diag(self, R):
        return np.einsum("ia,lab,ib->li", R, self.M, R, optimize=True)            # (n_mats, k)

    def __call__(self, R):
        self.n_evals += 1
        if not self.valid:
            return np.nan
        sp = self.spec; t = sp["target"]
        if t == "HECTRL":
            return self._he(R)
        if t == "MEANFUS":
            return sp["lam"] * self._pair(R) + (1 - sp["lam"]) * self._mean(R)
        rec = sp.get("recipe")
        if t in ("G", "G0", "GD"):
            if rec == "OFF":
                Mr = R @ self.M[0] @ R.T; return offdiag(Mr, self.fro[0])
            v = np.maximum(self.diag(R)[0], 0)
            return sup_vec(v, sp["c"]) if rec == "M03" else pow_vec(v, sp["r"])
        if t == "GJ":
            vals = [offdiag(R @ (self.M[i] / self.tr[i]) @ R.T, self.fro[i] / self.tr[i] ** 2) for i in range(2) if self.tr[i] > 1e-300]
            return float(np.mean(vals)) if vals else np.nan
        if t == "GX":
            D = np.maximum(self.diag(R), 0); ok = self.tr > 1e-300 * max(self.tr.max(), 1e-300)
            q = D[ok] / self.tr[ok, None]; c = sp["c"]; return float(np.mean(np.mean(q / (q + c / self.k), 1)))
        if t in ("GXJ", "GXJt"):
            Mr = np.einsum("ia,lab,jb->lij", R, self.M, R, optimize=True); off = (Mr ** 2).sum((1, 2)) - (np.einsum("lii->li", Mr) ** 2).sum(1)
            ok = self.fro > 1e-300 * max(self.fro.max(), 1e-300); r = off[ok] / self.fro[ok]
            if t == "GXJ":
                return float(r.mean())
            w = self.tr[ok] / self.tr[ok].sum(); return float((w * r).sum())
        Bd = np.maximum(self.diag(R), 0).T                                               # (k, n_cols)
        if rec == "glob":
            return glob_sup(Bd, sp.get("c", 0.03))
        if rec == "row":
            return RM.value(ROW_E02, Bd)
        if rec == "col":
            return col_sup(Bd, sp.get("c", 0.03))
        raise ValueError(rec)
