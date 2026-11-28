"""Blind integration designs for Lab 10 (04_fourth_wave sections 4 and 10).  No T, no prior draws: every law is fitted to the pool's Z.

Historical ids Q01-Q05 reproduce Lab 08 wave2.designs.design bit-for-bit (they ARE that function).  Controlled ids share ONE X panel and
ONE standard-normal stream wherever the law allows it, so a contrast changes one factor:

  common X panel   random design: complete observed rows of the pool, permutation RNG [seed,4,5], first n*nt rows (= historical Q05 rows);
                   fixed bank: the full bank for every latent draw (= historical Q05 / Q04)
  common normals   eps = RNG [seed,3,5].standard_normal((MAXDRAW, k)) (= historical Q05 normals)
  M-E    empirical Z of the first n subjects of the pool permutation [seed,1]                         (product, empirical latent)
  M-J    mu + eps Sigma^(1/2)   (symmetric principal square root, population covariance ddof 0)    (joint Gaussian)
  M-D    mu + eps D             (D = population SD)  == historical Q05 (tested)                     (independent marginal Gaussian)
  M-I    column i: Z_i of the first n subjects of permutation [seed,6,i]   (independent empirical marginals, Z columns only)
  RHO-r  mu + eps Sigma_r^(1/2),  Sigma_r = (1-r) D^2 + r Sigma                                       (covariance interpolation)
  KAP-x  mu + x eps D                                                                                (scale)
  TRN-a  mu + e D, e_i ~ N(0,1) | |e_i| <= a by rejection from RNG [seed,7,10a] (fixed budget 64 MAXDRAW per column; not clipping)
  SUB-f-u-X   M-D with mu / SD fitted on a TRAIN subpool (fraction f, RNG [u,8,100f]); X panel fixed (XF: full pool rows) or from the
              subpool's own rows (XC)
Rows per latent are consecutive (block = one latent draw).  Validation (tie-break) caches use the same id on the VALIDATION pool
(SUB-* -> M-D on the full VALIDATION pool).
"""
import numpy as np
from wave2.designs import design as hist_design, available as hist_available, QNUM, MAXDRAW
from .common import FIDELITY

HIST = ("Q01", "Q02", "Q03", "Q04", "Q05")


def available(pl, M):
    return hist_available(pl, M) if M in HIST else True


def sqrtm_psd(S):
    w, V = np.linalg.eigh((S + S.T) / 2)
    if w.min() < -1e-12 * max(w.max(), 1e-300):
        raise ValueError(f"covariance not PSD (min eig {w.min():.3g})")
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def pool_fit(Zp):
    mu = Zp.mean(0); sd = Zp.std(0); S = np.cov(Zp, rowvar=False, ddof=0)
    return dict(mu=mu, sd=sd, cov=S, n=len(Zp), corr=S / np.outer(sd, sd), eig=np.linalg.eigvalsh(S)[::-1])


def val_recipe(M):
    return "M-D" if M.startswith("SUB-") else M


def design(pl, M, fidelity, seed, pool="train", sizes=None):
    if M in HIST:
        d = hist_design(pl, M, fidelity, seed, pool=pool, sizes=sizes)
        if d is not None:
            Zp = pl.z[np.sort(pl.splits[pool])]; d["fit"] = pool_fit(Zp)
        return d
    ns, nt, nb = sizes or FIDELITY[fidelity]; P = np.sort(pl.splits[pool]); bank = pl.fixed_bank; k = pl.k
    n = nb if bank is not None else ns; rep = len(bank) if bank is not None else nt
    meta = dict(Q=M, fidelity=fidelity, seed=seed, pool=pool.upper(), pool_size=len(P))
    Pfit, Prows = P, P
    if M.startswith("SUB-"):
        _, f, u, xm = M.split("-"); f = float(f); u = int(u)
        Pfit = np.sort(np.random.default_rng([u, 8, int(round(100 * f))]).permutation(P)[:int(round(f * len(P)))])
        Prows = Pfit if xm == "XC" else P; meta.update(subpool_fraction=f, subpool_seed=u, x_panel="subpool rows" if xm == "XC" else "full pool rows",
                                                      subpool_size=len(Pfit))
    fit = pool_fit(pl.z[Pfit]); mu, sd, S = fit["mu"], fit["sd"], fit["cov"]
    eps = np.random.default_rng([seed, 3, 5]).standard_normal((MAXDRAW, k))[:n]
    sid = None
    if M == "M-E":
        sub = np.random.default_rng([seed, 1]).permutation(P)[:min(n, len(P))]; lat0 = pl.z[sub]; sid = sub; kind = "empirical Z (pool subjects)"
    elif M == "M-I":
        lat0 = np.column_stack([pl.z[np.random.default_rng([seed, 6, i]).permutation(P)[:n], i] for i in range(k)]); kind = "independent empirical marginals"
    elif M == "M-J":
        lat0 = mu + eps @ sqrtm_psd(S); kind = "joint Gaussian (symmetric sqrt)"
    elif M == "M-D" or M.startswith("SUB-"):
        lat0 = mu + eps * sd; kind = "independent marginal Gaussian"
    elif M.startswith("RHO-"):
        r = float(M[4:]); Sr = (1 - r) * np.diag(sd ** 2) + r * S; lat0 = mu + eps @ sqrtm_psd(Sr); kind = f"Gaussian, covariance interpolation rho={r}"
    elif M.startswith("KAP-"):
        x = float(M[4:]); lat0 = mu + x * eps * sd; kind = f"marginal Gaussian, scale kappa={x}"
    elif M.startswith("TRN-"):
        a = float(M[4:]); g = np.random.default_rng([seed, 7, int(round(10 * a))]).standard_normal((64 * MAXDRAW, k)); e = np.empty((n, k))
        for i in range(k):
            acc = g[np.abs(g[:, i]) <= a, i]
            if len(acc) < n:
                raise RuntimeError("truncation budget exhausted")
            e[:, i] = acc[:n]
        lat0 = mu + e * sd; kind = f"componentwise truncated marginal Gaussian |e|<={a}"; meta["realised_var_e"] = e.var(0).tolist()
    else:
        raise ValueError(M)
    if bank is not None:
        x = np.tile(bank, (n, 1)); rows = None; xd = "full bank"
    else:
        allrows = np.concatenate([pl.starts[s] + np.arange(pl.counts[s]) for s in Prows])
        rows = allrows[np.random.default_rng([seed, 4, QNUM["Q05"]]).permutation(len(allrows))[:n * nt]]; x = pl.x[rows]; xd = "empirical X rows (Q05 stream)"
    lat = np.repeat(lat0, rep, 0)
    meta.update(latent=kind, latent_draws=n, rows_per_latent=rep, design=f"{kind} x {xd}")
    return dict(lat=lat, x=x, sid=None if sid is None else np.repeat(sid, rep), rows=rows, n_latent=n, rows_per_latent=rep, meta=meta, fit=fit, lat0=lat0)
