"""Elementary concentration/sparsity metrics; one named function per metric."""
import numpy as np
from .scopes import normalized

def hhi(v,*,top_k=1):
    p=normalized(v)
    return float(p@p) if p is not None else float("nan")

def entropy(v,*,top_k=1):
    p=normalized(v)
    return float(-np.sum(p[p>0]*np.log(p[p>0]))) if p is not None else float("nan")

def effective_n(v,*,top_k=1):
    e=entropy(v)
    return float(np.exp(e)) if np.isfinite(e) else float("nan")

def hoyer(v,*,top_k=1):
    p=normalized(v)
    if p is None or len(p)<=1:
        return float("nan")
    return float((np.sqrt(len(p))-1/np.linalg.norm(p))/(np.sqrt(len(p))-1))

def tail_l1(v,*,top_k=1):
    p=normalized(v)
    if p is None:return float("nan")
    return float(np.sort(p)[::-1][max(0,int(top_k)):].sum())

def tail_l2(v,*,top_k=1):
    # Absolute energy scale; unlike tail_l1 this is not normalized.
    p=normalized(v)
    if p is None:return float("nan")
    v=np.maximum(np.asarray(v,float).reshape(-1),0)
    return float(np.linalg.norm(np.sort(v)[::-1][max(0,int(top_k)):]))

def top_k_mass(v,*,top_k=1):
    p=normalized(v)
    if p is None:return float("nan")
    return float(np.sort(p)[::-1][:max(0,int(top_k))].sum())
