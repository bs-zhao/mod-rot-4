"""The single metric registry for all HE sources (oracle, W-Y, W-Y+MLP).

No knowledge of model families or inferred HE methods. New scalar metrics only
require a function and register_metric() or register_structure_metric().
"""
from . import energy,sparsity,structure
from .scopes import matrix_scopes
METRICS={
 "energy":energy.total_energy,
 "hhi":sparsity.hhi,
 "entropy":sparsity.entropy,
 "effective_n":sparsity.effective_n,
 "hoyer":sparsity.hoyer,
 "tail_l1":sparsity.tail_l1,
 "tail_l2":sparsity.tail_l2,
 "top_k_mass":sparsity.top_k_mass,
}
STRUCTURE_METRICS={
 "UU.offdiag_energy_double_count":structure.uu_offdiag_energy_double_count,
 "UU.self_tail_q1":structure.uu_self_tail_q1,
 "UX.active_axis_count":structure.ux_active_axis_count,
 "UX.mean_active_row_entropy":structure.ux_mean_active_row_entropy,
}
DIRECTIONS={
 "energy":"descriptive",
 "hhi":"maximize",
 "entropy":"minimize",
 "effective_n":"minimize",
 "hoyer":"maximize",
 "tail_l1":"minimize",
 "tail_l2":"minimize",
 "top_k_mass":"maximize",
 "UU.offdiag_energy_double_count":"descriptive",
 "UU.self_tail_q1":"minimize",
 "UX.active_axis_count":"descriptive",
 "UX.mean_active_row_entropy":"minimize",
}
def register_metric(name,fn,*,direction="descriptive"):
    if name in METRICS or name in STRUCTURE_METRICS or not callable(fn):
        raise ValueError("Unique callable metric required")
    if direction not in ("minimize","maximize","descriptive"):
        raise ValueError("Unsupported direction")
    METRICS[name]=fn
    DIRECTIONS[name]=direction
def register_structure_metric(name,fn,*,direction="descriptive"):
    if name in METRICS or name in STRUCTURE_METRICS or not callable(fn):
        raise ValueError("Unique callable structure metric required")
    if direction not in ("minimize","maximize","descriptive"):
        raise ValueError("Unsupported direction")
    STRUCTURE_METRICS[name]=fn
    DIRECTIONS[name]=direction
def compute_metrics(M,dim_u,*,top_k=1):
    result={}
    for scope,vector in matrix_scopes(M,dim_u).items():
        for metric,fn in METRICS.items():
            result[f"{scope}.{metric}"]=float(fn(vector,top_k=top_k))
    for name,fn in STRUCTURE_METRICS.items():
        result[name]=float(fn(M,dim_u,top_k=top_k))
    return result

def metric_direction(name):
    if name in DIRECTIONS:
        return DIRECTIONS[name]
    for suffix in sorted(METRICS,key=len,reverse=True):
        if name.endswith("."+suffix):
            return DIRECTIONS[suffix]
    return "descriptive"
