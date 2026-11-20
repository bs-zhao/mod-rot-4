"""Common matrix-only statistics; no oracle/estimator dependencies."""
import numpy as np


def _finite(v):
    v=np.asarray(v,dtype=float).ravel()
    return v if len(v)>0 and np.all(np.isfinite(v)) and np.all(v>=-1e-9) else None


def _calc(v, prefix, output, top_k=1):
    v=_finite(v)
    if v is None:
        for key in ("energy","hhi","entropy","effective_n","hoyer","tail_l1",
                    "tail_l2","top_k_mass"):
            output[f"{prefix}.{key}"]=float("nan")
        return
    v=np.maximum(v,0)
    total=float(v.sum())
    n=len(v)
    output[f"{prefix}.energy"]=total
    if total==0:
        # Sparsity is undefined for an all-zero vector; record zero energy only.
        for key in ("hhi","entropy","effective_n","hoyer","tail_l1","tail_l2","top_k_mass"):
            output[f"{prefix}.{key}"]=float("nan")
        return
    p=v/total
    nz=p[p>0]
    entropy=float(-(nz*np.log(nz)).sum())
    hhi=float(np.dot(p,p))
    norm2=float(np.linalg.norm(v))
    sorted_v=np.sort(v)[::-1]
    tail=sorted_v[max(0,int(top_k)):]
    output.update({
        f"{prefix}.hhi":hhi,
        f"{prefix}.entropy":entropy,
        f"{prefix}.effective_n":float(np.exp(entropy)),
        f"{prefix}.hoyer":float((np.sqrt(n)-total/norm2)/(np.sqrt(n)-1)) if n>1 else float("nan"),
        f"{prefix}.tail_l1":float(tail.sum()/total),
        f"{prefix}.tail_l2":float(np.linalg.norm(tail)),
        f"{prefix}.top_k_mass":float(sorted_v[:top_k].sum()/total),
    })


def compute_metrics(m, dim_u, *, top_k=1):
    """Uses UU diagonal, independent off-diagonal pairs, UX, and full blocks.

    No metric is silently computed on a partially valid (NaN) energy block.
    """
    m=np.asarray(m,dtype=float)
    d=m.shape[0]
    if m.shape!=(d,d) or dim_u<1 or dim_u>=d:
        raise ValueError("Invalid HE matrix or latent dimension")
    uu=m[:dim_u,:dim_u]
    ux=m[:dim_u,dim_u:]
    xx=m[dim_u:,dim_u:]
    out={}
    _calc(m.ravel(),"FULL",out,top_k)
    _calc(np.diag(uu),"UU.diagonal",out,top_k)
    _calc(uu[np.triu_indices(dim_u,k=1)],"UU.offdiag_pairs",out,top_k)
    _calc(uu.ravel(),"UU",out,top_k)
    _calc(ux.ravel(),"UX",out,top_k)
    _calc(xx.ravel(),"XX",out,top_k)
    row=ux.sum(axis=1)
    _calc(row,"UX.row_totals",out,top_k)
    if np.all(np.isfinite(ux)):
        row_entropies=[]
        for v in ux:
            if v.sum()>0:
                p=v/v.sum()
                p=p[p>0]
                row_entropies.append(-np.dot(p,np.log(p)))
        out["UX.active_axis_count"]=sum(v>1e-12 for v in row)
        out["UX.mean_active_row_entropy"]=float(np.mean(row_entropies)) if row_entropies else float("nan")
    else:
        out["UX.active_axis_count"]=float("nan")
        out["UX.mean_active_row_entropy"]=float("nan")
    if np.all(np.isfinite(uu)):
        out["UU.offdiag_energy_double_count"]=float(uu.sum()-np.trace(uu))
        out["UU.self_tail_q1"]=float(np.sort(np.diag(uu))[::-1][1:].sum())
    else:
        out["UU.offdiag_energy_double_count"]=float("nan")
        out["UU.self_tail_q1"]=float("nan")
    return out
