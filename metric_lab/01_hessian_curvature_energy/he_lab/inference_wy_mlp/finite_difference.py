"""Numerical decoder-forward Hessian, independent of PyTorch autograd."""
import numpy as np
from ..decoder import finite_difference_hessians
from ..energy import HEResult,transform_moment,moment_from_hessians
from .common import validate

def estimate_he(W,Y,mlp,*,eval_W,weights=None,dim_u,target="probability",
                relative_step=0.003,chunk=32,reference_moment=None,
                reference_jacobian=None,rotation_id="",measure="",**kwargs):
    """Only calls frozen model forward in W; no NN parameter gradients."""
    W,Y,eval_W,weights=validate(W,Y,mlp,eval_W,weights,dim_u,target)
    if reference_moment is not None:
        if reference_jacobian is None:
            raise ValueError("Reference moment requires coordinate Jacobian")
        m=transform_moment(reference_moment,reference_jacobian)
    else:
        scales=np.maximum(np.std(W,axis=0,ddof=0),1e-5)
        step=relative_step*scales
        d=eval_W.shape[1]
        cross=np.zeros((d*d,d*d))
        for start in range(0,len(eval_W),chunk):
            hs=finite_difference_hessians(lambda w: mlp(w,target),
                        eval_W[start:start+chunk],step)
            h=hs.reshape(len(hs),d*d)
            cross+=h.T@(weights[start:start+len(h),None]*h)
        m=np.diag(cross/weights.sum()).reshape(d,d)
    m=(m+m.T)/2
    return HEResult(m,dim_u,"nn_finite_difference",target,rotation_id,measure,
        np.ones_like(m,bool),{"frozen":True,"uses_Y_for_fit":False,
        "evaluation_rows":len(eval_W),"accelerated_reference_moment":reference_moment is not None})
