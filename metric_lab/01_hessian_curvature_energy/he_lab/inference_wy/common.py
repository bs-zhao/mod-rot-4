"""W,Y-only contract; no generating model or frozen NN imports."""
import numpy as np
from ..energy import HEResult

def _standardize(train, evaluate):
    train = np.asarray(train, dtype=float)
    evaluate = np.asarray(evaluate, dtype=float)
    mean = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0)
    sd = np.maximum(sd, 1e-6)
    return (train-mean)/sd, (evaluate-mean)/sd, sd



def validate(W,Y,eval_W,weights,dim_u,binary,target):
    W=np.asarray(W,float);Y=np.asarray(Y,float).reshape(-1)
    eval_W=np.asarray(eval_W,float)
    if W.ndim!=2 or eval_W.ndim!=2 or not len(W) or not len(eval_W):
        raise ValueError("W/eval_W must be nonempty matrices")
    if W.shape[1]!=eval_W.shape[1] or len(W)!=len(Y) or not 0<dim_u<W.shape[1]:
        raise ValueError("Incompatible W,Y and latent dimension")
    if not all(np.isfinite(a).all() for a in (W,Y,eval_W)):
        raise ValueError("W,Y and eval_W must be finite")
    if target not in ("logit","probability"):
        raise ValueError("Unknown smooth target")
    if binary and (target!="probability" or not np.isin(Y,[0,1]).all()):
        raise ValueError("Binary choice Y supports probability target only")
    weights=np.ones(len(eval_W)) if weights is None else np.asarray(weights,float)
    if weights.shape!=(len(eval_W),) or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError("Invalid evaluation weights")
    return W,Y,eval_W,weights

def finish(H,weights,dim_u,method,target,rotation_id,measure,meta):
    H=np.asarray(H,float)
    if H.ndim!=3 or H.shape[1]!=H.shape[2] or not np.isfinite(H).all():
        raise ValueError("Invalid input Hessians")
    M=np.average(H*H,axis=0,weights=weights)
    M=(M+M.T)/2
    return HEResult(M,dim_u,method,target,rotation_id,measure,
                    np.ones(M.shape,bool),meta)
