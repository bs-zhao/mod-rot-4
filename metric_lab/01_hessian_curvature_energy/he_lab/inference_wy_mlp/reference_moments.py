"""Optional exact all-R acceleration: reusable Z/X derivative second moment.

No generating formula or true T. The evaluator supplies frozen Decoder, original
subject Z and real physical X, never rotating an already-squared HE matrix.
"""
import numpy as np
from ..decoder import finite_difference_hessians
from ..derivatives import moment_hessians

def estimate_reference_moment(decoder,points,*,method,target,chunk=32,step=0.003,
                              z_scale=None,x_scale=None):
    base=np.column_stack((points.latent,points.x))
    scalar=lambda v:decoder.scalar(v,target=target)
    if method=="nn_autograd":
        return moment_hessians(scalar,base,weights=points.weights,chunk=chunk)
    if method!="nn_finite_difference":
        raise KeyError(method)
    if z_scale is None or x_scale is None:
        raise ValueError("Need original Z/X scales for finite differences")
    steps=step*np.maximum(np.r_[z_scale,x_scale],1e-5)
    d=base.shape[1]
    cross=np.zeros((d*d,d*d),dtype=float)
    for start in range(0,len(base),chunk):
        H=finite_difference_hessians(scalar,base[start:start+chunk],steps)
        hf=H.reshape(len(H),d*d)
        cross+=hf.T@(points.weights[start:start+len(H),None]*hf)
    return cross/points.weights.sum()
