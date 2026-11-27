"""Frozen blind objectives L(R) for Lab 09 (one interface for registry, repair, structural and fused objectives).

A candidate spec is a dict with a "type":
  registry   {"metric": "L06|..." | "L03|..."}         -> Lab 08 wave2.objectives.Objective on an HE moment cache (K, J)
  row        a rowmetrics spec (POW / SM / BL / GL / ACT)   -> computed on B(R) = M^UX of the same HE cache
  struct     a wave3.structure spec on a gradient / contrast cache
  fusion     wave3.fusion (built from two component objectives)
Coordinates: V = (Z - c) / s, U = R V, d raw / d U = diag(s) R^T (Lab 08 convention); squared energies are never rotated.
"""
import numpy as np
from wave2.objectives import Objective as RegistryObjective
from . import rowmetrics as RM


def ux_factory(J, scale):
    s = np.asarray(scale, float); J = np.asarray(J, float)

    def B(R):
        Jt = R * s[None, :]
        return np.maximum(np.einsum("ai,xik,ak->ax", Jt, J, Jt, optimize=True), 0)
    return B


class RowObjective:
    def __init__(self, mo, scale, spec):
        self.spec = dict(spec); self.k = mo["J"].shape[1]; self.valid = bool(mo["valid"]); self.B = ux_factory(mo["J"], scale)
        EB = float(self.B(np.eye(self.k)).sum()); self.zero_B = not (EB > 1e-300)
        self.n_evals = 0

    def __call__(self, R):
        self.n_evals += 1
        if not self.valid or self.zero_B:
            return np.nan
        return RM.value(self.spec, self.B(R))

    def terms(self, R):
        return RM.terms(self.spec, self.B(R))


def build(spec, mo, scale):
    t = spec["type"]
    if t == "registry":
        return RegistryObjective(mo, scale, spec["metric"])
    if t == "row":
        return RowObjective(mo, scale, spec)
    if t == "struct":
        from .structure import StructObjective
        return StructObjective(mo, scale, spec)
    raise ValueError(t)
