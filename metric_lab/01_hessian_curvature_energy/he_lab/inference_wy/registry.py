"""Public functional API for blind HE: infer_he(method, W, Y, ...).

Algorithms live in separate files, each exposes estimate_he(W,Y,...)->HEResult.
Adding a method requires no changes in the experimental pipeline.
"""
from . import polynomial,kernel_ridge,local_quadratic
METHODS={
 "wy_polynomial":polynomial.estimate_he,
 "wy_kernel_ridge":kernel_ridge.estimate_he,
 "wy_local_quadratic":local_quadratic.estimate_he,
}
def register_method(name,fn):
    if not name.startswith("wy_") or name in METHODS or not callable(fn):
        raise ValueError("Unique wy_* callable required")
    METHODS[name]=fn
def infer_he(method,W,Y,*,eval_W,weights=None,dim_u,target,**kwargs):
    if method not in METHODS:
        raise KeyError(f"Unknown blind HE method {method!r}")
    return METHODS[method](W,Y,eval_W=eval_W,weights=weights,
                           dim_u=dim_u,target=target,**kwargs)
