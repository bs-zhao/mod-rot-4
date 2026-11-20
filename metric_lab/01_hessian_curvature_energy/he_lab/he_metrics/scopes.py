"""Named, nonoverlapping semantics for HE matrix slices and axis grouping."""
import numpy as np

def matrix_scopes(M,dim_u):
    M=np.asarray(M,dtype=float)
    if M.ndim!=2 or M.shape[0]!=M.shape[1] or not 0<dim_u<len(M):
        raise ValueError("Expected full square HE matrix and valid dim_u")
    UU=M[:dim_u,:dim_u]
    UX=M[:dim_u,dim_u:]
    XX=M[dim_u:,dim_u:]
    return {
        "FULL":M.ravel(),
        "UU":UU.ravel(),
        "UU.diagonal":np.diag(UU),
        "UU.offdiag_pairs":UU[np.triu_indices(dim_u,k=1)],
        "UX":UX.ravel(),
        "UX.row_totals":UX.sum(axis=1),
        "XX":XX.ravel(),
    }

def valid_energy(v):
    v=np.asarray(v,float).reshape(-1)
    return len(v)>0 and np.isfinite(v).all() and (v>=-1e-9).all()

def normalized(v):
    if not valid_energy(v):
        return None
    v=np.maximum(np.asarray(v,float).reshape(-1),0)
    if v.sum()<=0:
        return None
    return v/v.sum()
