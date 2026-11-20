"""Public W,Y + pretrained frozen MLP estimator registry.

Each method file exposes estimate_he(W,Y,mlp,*,eval_W,weights,dim_u,target,...)
returning an HEResult. Functions may use Y to fit a calibrated residual model;
current autograd/finite-difference algorithms intentionally do NOT refit Y.
"""
from . import autograd,finite_difference
METHODS={
 "nn_autograd":autograd.estimate_he,
 "nn_finite_difference":finite_difference.estimate_he,
}
def register_method(name,fn):
    if not name.startswith("nn_") or name in METHODS or not callable(fn):
        raise ValueError("Unique nn_* callable required")
    METHODS[name]=fn
def infer_he(method,W,Y,mlp,*,eval_W,weights=None,dim_u,target,**kwargs):
    if method not in METHODS:
        raise KeyError(f"Unknown frozen-MLP HE method {method!r}")
    return METHODS[method](W,Y,mlp,eval_W=eval_W,weights=weights,
                           dim_u=dim_u,target=target,**kwargs)
