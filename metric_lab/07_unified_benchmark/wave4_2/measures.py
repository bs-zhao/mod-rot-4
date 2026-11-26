"""Evaluation measures Q01-Q05 and X masks.

Blind side (`blind_design`) builds the points from Z, observed X rows and declared design only.  The analytic side (`oracle_latent`,
in measures but fed by EvalOnly) supplies the paired true T (Q01/Q02) or true-prior draws (Q03-Q05) for the SAME X rows.
Complete X rows only; a fixed bank is never resampled column-wise.  Points are nested across fidelities within a seed
(subject order = one permutation of the TEST pool; each subject's trial order = one permutation; Gaussian draws = prefix of one stream).

  Q01  TEST subject Z  x  that subject's observed trials                                (oracle: paired actual T)
  Q02  TEST subject Z  x  empirical X rows (random design) / the full bank (bank)        (oracle: paired actual T)
  Q03  joint Gaussian fitted to TEST Z  x  empirical X rows                              (oracle: true-prior T draws)
  Q04  joint Gaussian TEST Z  x  full bank (bank designs only)                           (oracle: true-prior T draws x bank)
  Q05  independent marginal Gaussian (TEST Z mean/SD) x empirical rows / full bank       (oracle: true prior;  Q05leg: independent
       Gaussian T with TEST mean/SD, the legacy sensitivity, out-of-support draws flagged and excluded)
"""
import numpy as np
from .common import FIDELITY

QNUM = {"Q01": 1, "Q02": 2, "Q03": 3, "Q04": 4, "Q05": 5}
MAXDRAW = 4096


def col_mask(x, domain):
    """(N, p) True where the classical derivative in that X column is defined (interior of the declared column domain)"""
    m = np.ones(x.shape, bool)
    if domain is None:
        return m
    for j, d in enumerate(domain):
        if d == "amount":
            m[:, j] = x[:, j] > 0
        elif d == "prob":
            m[:, j] = (x[:, j] > 0) & (x[:, j] < 1)
    return m


def available(blind, Q):
    return not (Q == "Q04" and blind.fixed_bank is None)


def blind_design(blind, Q, fidelity, seed):
    """-> dict(lat (n,k) raw Z, x (n,p) raw X, sid (n,) TEST subject ids or None, n_latent, rows_per_latent, meta)"""
    if not available(blind, Q):
        return None
    ns, nt, nb = FIDELITY[fidelity]; pool = np.sort(blind.splits["test"]); bank = blind.fixed_bank
    perm = np.random.default_rng([seed, 1]).permutation(pool)                      # nested subject order (same for every Q)
    meta = dict(Q=Q, fidelity=fidelity, seed=seed, pool="TEST", pool_size=len(pool))

    def trials_of(s, n):
        c = int(blind.counts[s]); order = np.random.default_rng([seed, 2, int(s)]).permutation(c)
        return blind.starts[s] + order[:min(n, c)]

    def emp_rows(n):                                                                # empirical complete X rows of TEST subjects
        rows = np.concatenate([blind.starts[s] + np.arange(blind.counts[s]) for s in pool])
        return rows[np.random.default_rng([seed, 4, QNUM[Q]]).permutation(len(rows))[:n]]

    if Q == "Q01":
        sub = perm[:min(ns, len(pool))]
        idx = np.concatenate([trials_of(s, nt) for s in sub]); sid = blind.subject_index[idx]
        meta.update(subjects=len(sub), trials_per_subject=nt, with_replacement=False, design="subject x own observed trials")
        return dict(lat=blind.z[sid], x=blind.x[idx], sid=sid, n_latent=len(sub), rows_per_latent=nt, meta=meta)
    if Q == "Q02":
        if bank is not None:
            sub = perm[:min(nb, len(pool))]; sid = np.repeat(sub, len(bank))
            meta.update(subjects=len(sub), rows_per_latent=len(bank), design="subject x full bank")
            return dict(lat=blind.z[sid], x=np.tile(bank, (len(sub), 1)), sid=sid, n_latent=len(sub), rows_per_latent=len(bank), meta=meta)
        sub = perm[:min(ns, len(pool))]; sid = np.repeat(sub, nt); rows = emp_rows(len(sid))
        meta.update(subjects=len(sub), rows_per_latent=nt, design="subject x independent empirical X rows (product)")
        return dict(lat=blind.z[sid], x=blind.x[rows], sid=sid, n_latent=len(sub), rows_per_latent=nt, meta=meta)
    Zp = blind.z[pool]; mu = Zp.mean(0); eps = np.random.default_rng([seed, 3, QNUM[Q]]).standard_normal((MAXDRAW, blind.k))
    if Q in ("Q03", "Q04"):
        L = np.linalg.cholesky(np.cov(Zp, rowvar=False, ddof=0)); kind = "joint Gaussian fitted to TEST Z"
        draw = lambda n: mu + eps[:n] @ L.T
    else:
        sd = Zp.std(0); kind = "independent marginal Gaussian (TEST Z mean/SD)"
        draw = lambda n: mu + eps[:n] * sd
    if Q == "Q04" or (Q == "Q05" and bank is not None):
        n = nb; lat = np.repeat(draw(n), len(bank), 0); x = np.tile(bank, (n, 1)); rep = len(bank); xd = "full bank"
    else:
        n = ns; lat = np.repeat(draw(n), nt, 0); x = blind.x[emp_rows(n * nt)]; rep = nt; xd = "empirical X rows"
    meta.update(latent=kind, latent_draws=n, rows_per_latent=rep, design=f"{kind} x {xd}")
    return dict(lat=lat, x=x, sid=None, n_latent=n, rows_per_latent=rep, meta=meta)


def oracle_latent(design, Q, T, prior_draw, seed, legacy=False, test_ids=None):
    """paired true T (Q01/Q02) or true-prior draws for the same X rows (Q03-Q05).  prior_draw(n, rng) is evaluation-only."""
    if Q in ("Q01", "Q02"):
        return T[design["sid"]], np.ones(len(design["sid"]), bool)
    n, rep = design["n_latent"], design["rows_per_latent"]
    if legacy:                                                   # Q05leg: independent Gaussian T with TEST mean/SD
        Tp = T[np.sort(test_ids)]
        eps = np.random.default_rng([seed, 5, 9]).standard_normal((MAXDRAW, T.shape[1]))[:n]
        lat = Tp.mean(0) + eps * Tp.std(0)
    else:
        lat = prior_draw(MAXDRAW, np.random.default_rng([seed, 6, QNUM[Q]]))[:n]
    return np.repeat(lat, rep, 0), None
