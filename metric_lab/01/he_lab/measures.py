"""Independent choice of integration/empirical averaging locations."""
from dataclasses import dataclass
import numpy as np
from .sources import get_spec


@dataclass
class PointSet:
    latent: np.ndarray
    x: np.ndarray
    weights: np.ndarray
    name: str
    source: str
    meta: dict


def _choice_indices(src, n, seed):
    rng = np.random.default_rng(seed)
    if n == 0:
        return np.arange(len(src.subject_ids))
    # Subject-equal policy: each subject drawn equally, then a valid trial.
    s = rng.integers(0, len(src.counts), size=n)
    t = (rng.random(n) * src.counts[s]).astype(int)
    starts = np.r_[0, np.cumsum(src.counts[:-1])]
    return starts[s] + t


def training_indices(src, n, seed):
    return _choice_indices(src, n, seed)


def _theoretical_x(src, count, rng):
    spec = get_spec(src.model)
    return np.asarray(spec.sample_inputs(count, rng), dtype=float)


def evaluation_points(src, *, source, measure, n_eval, seed, latent_draws=16):
    """source='oracle': latent=T; source='nn': latent=raw S5 z.

    A fixed CPT X bank is evaluated in its entirety for each latent draw,
    without resampling/truncating the bank.
    """
    if source not in ("oracle", "nn"):
        raise ValueError(source)
    if measure not in ("observed_joint", "theory_x"):
        raise ValueError(measure)
    rng = np.random.default_rng(seed)
    if measure == "observed_joint":
        idx = _choice_indices(src, n_eval, seed)
        lat = src.parameters[src.subject_ids[idx]] if source == "oracle" else src.z[src.subject_ids[idx]]
        return PointSet(lat, src.x[idx], np.ones(len(idx)), measure, source,
                        {"row_count": len(idx), "design": "subject-equal observed joint"})
    if source == "oracle":
        lat = get_spec(src.model).sample_parameters(
            latent_draws if src.fixed_x is not None else n_eval, rng)
    else:
        # Joint Gaussian fit to SUBJECT-level Z, never independent per rotated coordinate.
        mu = src.z.mean(axis=0)
        cov = np.cov(src.z, rowvar=False, ddof=0)
        cov = np.atleast_2d(cov)
        lat = rng.multivariate_normal(mu, cov, size=(latent_draws if src.fixed_x is not None else n_eval))
    bank = src.fixed_x
    if bank is not None:
        n, k = len(lat), len(bank)
        return PointSet(np.repeat(lat, k, axis=0), np.tile(bank, (n, 1)),
                        np.ones(n*k), measure, source,
                        {"latent_draws": n, "fixed_bank_rows": k, "design": "full crossed latent x bank"})
    x = _theoretical_x(src, len(lat), rng)
    return PointSet(np.asarray(lat), x, np.ones(len(lat)), measure, source,
                    {"latent_draws": len(lat), "design": "joint latent x model-uniform"})
