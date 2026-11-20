"""Each inference algorithm is a standalone estimate_he function."""
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LogisticRegression
from .common import _standardize,validate,finish

def _poly_terms(q, coefficients, powers):
    """Value, gradient and Hessian of an arbitrary multivariate polynomial."""
    n, d = q.shape
    value = np.zeros(n)
    gradient = np.zeros((n,d))
    hessian = np.zeros((n,d,d))
    for coef, power in zip(coefficients, powers):
        if abs(coef) < 1e-16:
            continue
        power = np.asarray(power, dtype=int)
        value += coef*np.prod(q**power, axis=1)
        active = np.flatnonzero(power)
        for i in active:
            ep = power.copy(); ep[i]-=1
            gradient[:,i] += coef*power[i]*np.prod(q**ep, axis=1)
            for j in active:
                if i == j and power[i] < 2:
                    continue
                ex = ep.copy(); ex[j]-=1
                factor = power[i]*(power[j]-int(i==j))
                hessian[:,i,j] += coef*factor*np.prod(q**ex, axis=1)
    return value, gradient, hessian


def polynomial_hessians(train_w, train_y, eval_w, *, binary, degree=3, alpha=1.0):
    wt, we, sd = _standardize(train_w, eval_w)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    basis = poly.fit_transform(wt)
    if binary:
        model = LogisticRegression(C=1/max(alpha,1e-8),fit_intercept=False,
                                   max_iter=500, solver="lbfgs")
        model.fit(basis, np.asarray(train_y).astype(int))
        coefficients = model.coef_[0]
    else:
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(basis, train_y)
        coefficients = model.coef_
    value, grad, hess = _poly_terms(we, coefficients, poly.powers_)
    grad /= sd[None,:]
    hess /= sd[None,:,None]*sd[None,None,:]
    if binary:
        p = expit(value)
        return (p*(1-p))[:,None,None] * (
            hess + (1-2*p)[:,None,None]*np.einsum("ni,nj->nij",grad,grad))
    return hess



def estimate_he(W,Y,*,eval_W,weights=None,dim_u,target="probability",binary=False,
                degree=3,alpha=1.0,rotation_id="",measure="",**kwargs):
    """Only W,Y are used, with polynomial surrogate then analytic Hessian."""
    W,Y,eval_W,weights=validate(W,Y,eval_W,weights,dim_u,binary,target)
    H=polynomial_hessians(W,Y,eval_W,binary=binary,degree=degree,alpha=alpha)
    return finish(H,weights,dim_u,"wy_polynomial",target,rotation_id,measure,
                  {"binary":binary,"degree":degree,"alpha":alpha,"train_rows":len(W)})
