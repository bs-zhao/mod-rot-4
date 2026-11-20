"""RBF kernel ridge surrogate from W,Y, followed by analytic kernel Hessian."""
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from .common import _standardize,validate,finish

def kernel_ridge_hessians(train_w, train_y, eval_w, *, binary,
                          alpha=1., max_centers=600, seed=123):
    if len(train_w)>max_centers:
        ix=np.random.default_rng(seed).choice(len(train_w),max_centers,replace=False)
        train_w=np.asarray(train_w)[ix]
        train_y=np.asarray(train_y)[ix]
    wt, we, sd = _standardize(train_w, eval_w)
    d = wt.shape[1]
    gamma = 1./d
    model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    model.fit(wt,train_y)
    coeff = np.asarray(model.dual_coef_, dtype=float)
    out = np.empty((len(we),d,d))
    eye=np.eye(d)
    for start in range(0,len(we),128):
        delta = we[start:start+128,None,:] - wt[None,:,:]
        kern = np.exp(-gamma*np.sum(delta**2,axis=-1))
        weighted = kern*coeff[None,:]
        h = (4*gamma**2*np.einsum("nm,nmi,nmj->nij",weighted,delta,delta)
             -2*gamma*weighted.sum(axis=1)[:,None,None]*eye)
        out[start:start+128] = h/(sd[None,:,None]*sd[None,None,:])
    # KRR on 0/1 directly estimates E[Y|W] in a least-squares sense.
    return out



def estimate_he(W,Y,*,eval_W,weights=None,dim_u,target="probability",binary=False,
                alpha=1.0,max_centers=500,seed=1234,rotation_id="",measure="",**kwargs):
    W,Y,eval_W,weights=validate(W,Y,eval_W,weights,dim_u,binary,target)
    H=kernel_ridge_hessians(W,Y,eval_W,binary=binary,alpha=alpha,max_centers=max_centers,seed=seed)
    return finish(H,weights,dim_u,"wy_kernel_ridge",target,rotation_id,measure,
                  {"binary":binary,"alpha":alpha,"max_centers":max_centers,"train_rows":len(W)})
