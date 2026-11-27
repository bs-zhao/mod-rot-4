"""W3-03 ST-E reduced-axis predictive structure, PILOT (03_third_wave section 15).  Truth-free in the blind arm.

Reference: isotropic Gaussian over the variant coordinates V (so U = R V has the same law for every R; for S9 raw Z this is an extrapolation
of the latent law and is labelled as such).  Single-axis deletion (total-effect variance, paired replacement identity):
    D_i(x, R) = 1/2 E[(g(U, x) - g(U^(i), x))^2],   U^(i) = U with coordinate i replaced by an independent N(0, 1) draw
    i.e. V' = V + (xi - r_i . V) r_i,  r_i = row i of R.
Common random numbers (fixed V draws, antithetic xi draws, X rows) make L(R) deterministic.  Per column (64 observed X rows + the stimulus-averaged mean
response):  q_i = D_i / sum_j D_j,  score = (1/k) sum_i q_i / (q_i + c/k), c = .03;  L = mean over columns with sum_j D_j > 0 (columns
without latent effect recorded separately).  Pilot draws: 64 base x 16 replacement x 64 rows.  Decoder in float32 on the GPU (equivalence
to CPU float64 is tested in validate / tests); capped search budget RA (declared; not S0): 64 Haar starts + identity, 2 refinements,
spans 45 / 15 / 5 / 1.5 degrees, <= 3000 evaluations per refinement.
"""
import numpy as np, torch
from wave2.optimizers import haar, distinct_best, refine

NB, NR, NX, C = 64, 16, 64, 0.03
RA_BUDGET = dict(starts=64, refined=2, spans=(45.0, 15.0, 5.0, 1.5), max_evals=3000)


def xrows(pl, pool, seed, n=NX):
    if pl.fixed_bank is not None:
        X = pl.fixed_bank[np.random.default_rng([seed, 203]).permutation(len(pl.fixed_bank))[:n]]
    else:
        P = np.sort(pl.splits[pool]); rows = np.concatenate([pl.starts[s] + np.arange(pl.counts[s]) for s in P])
        X = pl.x[rows[np.random.default_rng([seed, 203]).permutation(len(rows))[:n]]]
    return (X - pl.x_mean) / pl.x_std


class DeletionLoss:
    """L(R) for a scalar response g(V, x) given as a batched torch function gfun(V (m, k), Xs (m, p)) -> (m,)"""
    def __init__(self, gfun, k, Xs, seed=31, device="cuda", dtype=torch.float32, c=C):
        rng = np.random.default_rng([seed, 201]); self.k = k; self.c = c; self.dev, self.dt = device, dtype
        self.V0 = torch.as_tensor(rng.standard_normal((NB, k)), dtype=dtype, device=device)
        half = np.random.default_rng([seed, 202]).standard_normal((NB, NR // 2))          # antithetic: exact invariance under sign flips of r_i
        self.xi = torch.as_tensor(np.concatenate([half, -half], 1), dtype=dtype, device=device)
        self.X = torch.as_tensor(Xs, dtype=dtype, device=device); self.g = gfun; self.n_evals = 0; self.valid = True
        nx = len(Xs)
        with torch.no_grad():
            self.g0 = self._eval(self.V0[:, None, :].expand(NB, nx, k))                                   # (NB, nX)

    def _eval(self, V):
        """V (..., nX, k) -> g (..., nX)"""
        sh = V.shape[:-1]; nx = self.X.shape[0]
        Xb = self.X.expand(*sh[:-1], nx, self.X.shape[1]) if len(sh) > 1 else self.X
        return self.g(V.reshape(-1, self.k), Xb.reshape(-1, self.X.shape[1])).reshape(sh)

    @torch.no_grad()
    def effects(self, R):
        R = torch.as_tensor(np.asarray(R), dtype=self.dt, device=self.dev); nx = self.X.shape[0]; D = []
        for i in range(self.k):
            r = R[i]; u = self.V0 @ r                                                                     # (NB,)
            Vp = self.V0[:, None, :] + (self.xi - u[:, None])[..., None] * r                              # (NB, NR, k)
            gp = self._eval(Vp[:, :, None, :].expand(NB, NR, nx, self.k))                                 # (NB, NR, nX)
            d = 0.5 * (self.g0[:, None, :] - gp) ** 2; dm = 0.5 * (self.g0.mean(-1)[:, None] - gp.mean(-1)) ** 2
            D.append(torch.cat([d.mean((0, 1)), dm.mean()[None]]))
        return torch.stack(D).double().cpu().numpy()                                                       # (k, nX + 1)

    def __call__(self, R):
        self.n_evals += 1; D = self.effects(R); tot = D.sum(0); ok = tot > 1e-12 * max(tot.max(), 1e-300)
        if not ok.any():
            return np.nan
        q = D[:, ok] / tot[ok]; return float(np.mean(np.mean(q / (q + self.c / self.k), 0)))


def decoder_gfun(pl, center, scale, device="cuda", dtype=torch.float32):
    from wave4_2.surfaces import DecoderSurface
    ds = DecoderSurface(pl); net = ds.full_net.decoder.to(device=device, dtype=dtype)
    c = torch.as_tensor(center, dtype=dtype, device=device); s = torch.as_tensor(scale, dtype=dtype, device=device)
    if ds.continuous:
        ym, ys = ds.ym, ds.ys
        return lambda V, X: net(torch.cat((X, c + s * V), 1))[:, 0] * ys + ym
    return lambda V, X: (lambda o: o[:, 1] - o[:, 0])(net(torch.cat((X, c + s * V), 1)))


def capped_search(L, k, seed):
    Qs = np.concatenate([np.eye(k)[None], haar(k, RA_BUDGET["starts"], np.random.default_rng([seed, 77, k]))])
    vals = np.array([L(Q) for Q in Qs]); keep = distinct_best(Qs, vals, RA_BUDGET["refined"]); fin = []
    for rank, i in enumerate(keep):
        r = refine(L, Qs[i], spans=RA_BUDGET["spans"], max_evals=RA_BUDGET["max_evals"]); r.update(start=int(i), rank=rank, L_initial=float(vals[i]), identity_start=bool(i == 0)); fin.append(r)
    fs = np.array([r["f"] for r in fin]); b = int(np.argmin(fs))
    return dict(R=fin[b]["Q"], f=float(fs[b]), best_index=b, finalists=fin, start_vals=vals, status="ok", n_tied=1, tie_breaker_used=False,
                near_minima_max_sep=0.0, n_near_minima=1, best_converged=bool(fin[b]["converged"]), n_evals=int(sum(r["n_evals"] for r in fin) + len(vals)))
