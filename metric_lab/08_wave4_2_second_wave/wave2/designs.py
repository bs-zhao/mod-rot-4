"""Blind integration designs Q01-Q05 with an explicit subject pool (02_second_wave section 3.2-3.3).

`design(pl, Q, fidelity, seed, pool="test")` reproduces Lab 07 `measures.blind_design` bit-for-bit (historical development replay;
tested).  pool="train" / "validation" gives the split-safe caches; `exclude` removes subjects (independent high-fidelity
re-evaluation caches use TRAIN subjects not used by the medium caches).  Complete X rows only; banks are never resampled column-wise.
No T, no prior draws: Gaussian measures are fitted to the pool's Z.
"""
import numpy as np
from .common import FIDELITY

QNUM = {"Q01": 1, "Q02": 2, "Q03": 3, "Q04": 4, "Q05": 5}
MAXDRAW = 4096


def available(pl, Q):
    return not (Q == "Q04" and pl.fixed_bank is None)


def design(pl, Q, fidelity, seed, pool="test", exclude=None, sizes=None):
    if not available(pl, Q):
        return None
    ns, nt, nb = sizes or FIDELITY[fidelity]; P = np.sort(pl.splits[pool]); bank = pl.fixed_bank
    if exclude is not None and len(exclude):
        P = np.setdiff1d(P, np.asarray(exclude))
    perm = np.random.default_rng([seed, 1]).permutation(P)
    meta = dict(Q=Q, fidelity=fidelity, seed=seed, pool=pool.upper(), pool_size=len(P), excluded=0 if exclude is None else int(len(exclude)))

    def trials_of(s, n):
        c = int(pl.counts[s]); order = np.random.default_rng([seed, 2, int(s)]).permutation(c)
        return pl.starts[s] + order[:min(n, c)]

    def emp_rows(n):
        rows = np.concatenate([pl.starts[s] + np.arange(pl.counts[s]) for s in P])
        return rows[np.random.default_rng([seed, 4, QNUM[Q]]).permutation(len(rows))[:n]]

    if Q == "Q01":
        sub = perm[:min(ns, len(P))]
        idx = np.concatenate([trials_of(s, nt) for s in sub]); sid = pl.subject_index[idx]
        meta.update(subjects=len(sub), trials_per_subject=nt, design="subject x own observed trials")
        return dict(lat=pl.z[sid], x=pl.x[idx], sid=sid, rows=idx, n_latent=len(sub), rows_per_latent=nt, meta=meta)
    if Q == "Q02":
        if bank is not None:
            sub = perm[:min(nb, len(P))]; sid = np.repeat(sub, len(bank))
            meta.update(subjects=len(sub), rows_per_latent=len(bank), design="subject x full bank")
            return dict(lat=pl.z[sid], x=np.tile(bank, (len(sub), 1)), sid=sid, rows=None, n_latent=len(sub), rows_per_latent=len(bank), meta=meta)
        sub = perm[:min(ns, len(P))]; sid = np.repeat(sub, nt); rows = emp_rows(len(sid))
        meta.update(subjects=len(sub), rows_per_latent=nt, design="subject x independent empirical X rows (product)")
        return dict(lat=pl.z[sid], x=pl.x[rows], sid=sid, rows=rows, n_latent=len(sub), rows_per_latent=nt, meta=meta)
    Zp = pl.z[P]; mu = Zp.mean(0); eps = np.random.default_rng([seed, 3, QNUM[Q]]).standard_normal((MAXDRAW, pl.k))
    if Q in ("Q03", "Q04"):
        L = np.linalg.cholesky(np.cov(Zp, rowvar=False, ddof=0)); kind = "joint Gaussian fitted to pool Z"
        draw = lambda n: mu + eps[:n] @ L.T
    else:
        sd = Zp.std(0); kind = "independent marginal Gaussian (pool Z mean/SD)"
        draw = lambda n: mu + eps[:n] * sd
    if Q == "Q04" or (Q == "Q05" and bank is not None):
        n = nb; lat = np.repeat(draw(n), len(bank), 0); x = np.tile(bank, (n, 1)); rep = len(bank); xd = "full bank"; rows = None
    else:
        n = ns; lat = np.repeat(draw(n), nt, 0); rows = emp_rows(n * nt); x = pl.x[rows]; rep = nt; xd = "empirical X rows"
    meta.update(latent=kind, latent_draws=n, rows_per_latent=rep, design=f"{kind} x {xd}")
    return dict(lat=lat, x=x, sid=None, rows=rows, n_latent=n, rows_per_latent=rep, meta=meta)


def used_subjects(d):
    return np.unique(d["sid"]) if d is not None and d.get("sid") is not None else np.array([], int)
