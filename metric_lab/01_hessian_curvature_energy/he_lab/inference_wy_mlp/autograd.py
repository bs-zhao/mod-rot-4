"""Autograd method: one function estimate_he(W,Y,trained_mlp,...)->HEResult."""
import numpy as np
from ..derivatives import moment_hessians
from ..energy import HEResult,transform_moment
from .common import validate

def estimate_he(W,Y,mlp,*,eval_W,weights=None,dim_u,target="probability",
                chunk=32,reference_moment=None,reference_jacobian=None,
                rotation_id="",measure="",**kwargs):
    """Differentiate frozen MLP output in W coordinates.

    Y is accepted for the common W,Y,MLP contract, but is not used to retrain:
    this method probes the frozen network directly. A precomputed base-Z/X
    moment may be supplied for mathematically identical fast sweep evaluation.
    """
    W,Y,eval_W,weights=validate(W,Y,mlp,eval_W,weights,dim_u,target)
    if reference_moment is None:
        moment=moment_hessians(lambda w: mlp(w,target),eval_W,weights=weights,chunk=chunk)
        d=eval_W.shape[1]
        m=np.diag(moment).reshape(d,d)
    else:
        if reference_jacobian is None:
            raise ValueError("Reference moment requires a coordinate Jacobian")
        m=transform_moment(reference_moment,reference_jacobian)
    m=(m+m.T)/2
    return HEResult(m,dim_u,"nn_autograd",target,rotation_id,measure,
        np.ones_like(m,bool),{"frozen":True,"uses_Y_for_fit":False,
        "evaluation_rows":len(eval_W),"accelerated_reference_moment":reference_moment is not None})
