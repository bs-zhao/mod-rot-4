"""W,Y + pretrained frozen MLP: supervised polynomial correction then HE.

Unlike pure NN autograd, this method **uses Y**. The original MLP parameters
stay frozen. An estimated smooth correction is fitted from (W,Y).
"""
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LogisticRegression
from ..derivatives import moment_hessians
from ..energy import HEResult
from .common import validate

def _predict(mlp,W,target,chunk=1024):
    out=[]
    with torch.no_grad():
        for i in range(0,len(W),chunk):
            tensor=torch.as_tensor(W[i:i+chunk],dtype=torch.float64)
            out.append(torch.func.vmap(lambda w:mlp(w,target))(tensor).detach().numpy())
    return np.concatenate(out)

def estimate_he(W,Y,mlp,*,eval_W,weights=None,dim_u,target="probability",
                binary=False,degree=2,alpha=1.0,chunk=32,
                rotation_id="",measure="",**unused):
    """Fit calibration from W,Y,MLP then estimate E[(H_W f_corrected)^2].

    For binary choice: logistic regression on [MLP logit, polynomial(W)]
    yields a valid sigmoid probability. Continuous logit/probability outcomes:
    Ridge fits residuals on the chosen output scale. No ground-truth T.
    """
    W,Y,eval_W,weights=validate(W,Y,mlp,eval_W,weights,dim_u,target)
    if binary and (target!="probability" or not np.isin(Y,[0.,1.]).all()):
        raise ValueError("Binary Y requires smooth probability target and 0/1 choices")
    if not 1<=degree<=4:
        raise ValueError("Polynomial degree must be between 1 and 4")
    sd=np.maximum(W.std(axis=0,ddof=0),1e-6)
    mu=W.mean(axis=0)
    poly=PolynomialFeatures(degree=degree,include_bias=True)
    design=poly.fit_transform((W-mu)/sd)
    nn_target="logit" if binary else target
    baseline=_predict(mlp,W,nn_target)
    if binary:
        regression=LogisticRegression(C=1/max(float(alpha),1e-8),
                 fit_intercept=False,solver="lbfgs",max_iter=1000)
        regression.fit(np.column_stack((baseline,design)),Y.astype(int))
        coef_nn=float(regression.coef_[0,0])
        coef_poly=np.asarray(regression.coef_[0,1:],dtype=float)
    else:
        regression=Ridge(alpha=alpha,fit_intercept=False)
        regression.fit(design,Y-baseline)
        coef_nn=1.
        coef_poly=np.asarray(regression.coef_,dtype=float)
    powers=torch.as_tensor(poly.powers_,dtype=torch.int64)
    coefficients=torch.as_tensor(coef_poly,dtype=torch.float64)
    mean=torch.as_tensor(mu,dtype=torch.float64)
    scale=torch.as_tensor(sd,dtype=torch.float64)
    def corrected(w):
        w_std=(w-mean)/scale
        features=torch.prod(w_std[None,:]**powers,dim=1)
        pred=coef_nn*mlp(w,nn_target)+torch.dot(coefficients,features)
        return torch.sigmoid(pred) if binary else pred
    moment=moment_hessians(corrected,eval_W,weights=weights,chunk=chunk)
    d=eval_W.shape[1]
    M=np.diag(moment).reshape(d,d)
    M=(M+M.T)/2
    return HEResult(M,dim_u,"nn_residual_polynomial",target,rotation_id,measure,
                    np.ones_like(M,bool),{
                        "frozen":True,"uses_Y_for_fit":True,
                        "binary":bool(binary),"degree":degree,"alpha":float(alpha),
                        "train_rows":len(W),"evaluation_rows":len(eval_W),
                        "fit":"logistic NN logit calibration" if binary
                               else "ridge polynomial residual of NN output",
                    })
