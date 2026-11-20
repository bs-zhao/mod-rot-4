"""Shared function contract for frozen-MLP-assisted HE inference."""
import numpy as np
from ..energy import HEResult,transform_moment

def validate(W,Y,mlp,eval_W,weights,dim_u,target):
    W=np.asarray(W,dtype=float)
    Y=np.asarray(Y,dtype=float).reshape(-1)
    eval_W=np.asarray(eval_W,dtype=float)
    if W.ndim!=2 or eval_W.ndim!=2 or len(W)==0 or len(eval_W)==0:
        raise ValueError("W and eval_W must be nonempty matrices")
    if len(W)!=len(Y) or W.shape[1]!=eval_W.shape[1] or not 0<dim_u<W.shape[1]:
        raise ValueError("W,Y/evaluation shape mismatch")
    if not (np.isfinite(W).all() and np.isfinite(Y).all() and np.isfinite(eval_W).all()):
        raise ValueError("W,Y,eval_W must be finite")
    if target not in ("logit","probability"):
        raise ValueError("Only smooth logit or probability targets are supported")
    if not callable(mlp):
        raise TypeError("Frozen MLP must implement a callable scalar function of W")
    weights=np.ones(len(eval_W)) if weights is None else np.asarray(weights,dtype=float)
    if weights.shape!=(len(eval_W),) or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError("Invalid evaluation weights")
    return W,Y,eval_W,weights

def make_result(moment,dim_u,method,target,rotation_id,measure,meta,
                *,reference_jacobian=None):
    # Without a Jacobian, moment is already in W=[U,X_raw] coordinates.
    m=(transform_moment(moment,reference_jacobian)
       if reference_jacobian is not None else np.diag(moment).reshape(dim_u+meta["dim_x"],-1))
    m=(m+m.T)/2
    return HEResult(m,dim_u,method,target,rotation_id,measure,
                    np.ones_like(m,dtype=bool),meta)
