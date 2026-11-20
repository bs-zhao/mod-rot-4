"""Metrics depending on UU/UX matrix structure rather than a flat vector."""
import numpy as np

def uu_offdiag_energy_double_count(M,dim_u,*,top_k=1):
    uu=np.asarray(M[:dim_u,:dim_u],float)
    return (float(uu.sum()-np.trace(uu))
            if np.isfinite(uu).all() else float("nan"))

def uu_self_tail_q1(M,dim_u,*,top_k=1):
    uu=np.asarray(M[:dim_u,:dim_u],float)
    return (float(np.sort(np.diag(uu))[::-1][1:].sum())
            if np.isfinite(uu).all() else float("nan"))

def ux_active_axis_count(M,dim_u,*,top_k=1):
    ux=np.asarray(M[:dim_u,dim_u:],float)
    return (float((ux.sum(axis=1)>1e-12).sum())
            if np.isfinite(ux).all() else float("nan"))

def ux_mean_active_row_entropy(M,dim_u,*,top_k=1):
    ux=np.asarray(M[:dim_u,dim_u:],float)
    if not np.isfinite(ux).all():
        return float("nan")
    ent=[]
    for row in ux:
        if row.sum()>0:
            p=row/row.sum()
            q=p[p>0]
            ent.append(-np.dot(q,np.log(q)))
    return float(np.mean(ent)) if ent else float("nan")
