"""Blind W,Y function-fit inference; oracle T/formulas are never inputs.

WY methods fit EACH R separately, because regularization/basis choices need not
be exactly equivariant under rotation. No method imports the oracle.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import NearestNeighbors
from .energy import HEResult


def _standardize(train, evaluate):
    train = np.asarray(train, dtype=float)
    evaluate = np.asarray(evaluate, dtype=float)
    mean = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0)
    sd = np.maximum(sd, 1e-6)
    return (train-mean)/sd, (evaluate-mean)/sd, sd


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


def local_quadratic_hessians(train_w, train_y, eval_w, *,
                             binary, ridge=1e-3, neighbors=180):
    wt, we, sd = _standardize(train_w, eval_w)
    d = wt.shape[1]
    pairs = [(i,j) for i in range(d) for j in range(i+1,d)]
    q = 1+d+d+len(pairs)
    k = min(len(wt), max(int(neighbors),2*q))
    if k < q:
        raise ValueError("Too few training points for local quadratic Hessian")
    finder = NearestNeighbors(n_neighbors=k).fit(wt)
    distances, ids = finder.kneighbors(we)
    result = np.empty((len(we),d,d))
    yy=np.asarray(train_y, dtype=float)
    for n, loc in enumerate(we):
        dv = wt[ids[n]]-loc
        radius = max(float(distances[n,-1]),1e-8)
        w = np.exp(-0.5*(distances[n]/radius*2)**2)
        basis = [np.ones(k),*dv.T,*(0.5*dv[:,i]**2 for i in range(d)),
                 *(dv[:,i]*dv[:,j] for i,j in pairs)]
        a=np.column_stack(basis)
        reg=np.eye(q)*ridge
        reg[0,0]=0
        coef=np.linalg.solve(a.T@(w[:,None]*a)+reg,a.T@(w*yy[ids[n]]))
        h=np.diag(coef[1+d:1+2*d])
        for val,(i,j) in zip(coef[1+2*d:],pairs):
            h[i,j]=h[j,i]=val
        result[n]=h/(sd[:,None]*sd[None,:])
    # Choice: local WLS estimates smooth probability, not a Bernoulli logit.
    return result


REGISTRY = {
    "wy_polynomial": polynomial_hessians,
    "wy_kernel_ridge": kernel_ridge_hessians,
    "wy_local_quadratic": local_quadratic_hessians,
}


def estimate_wy(method, rotation, src, prepared, eval_points, y_name,
                measure_name, *, degree=3, max_centers=600, neighbors=180,
                seed=123):
    if method not in REGISTRY:
        raise KeyError(method)
    idx = prepared["row_index"]
    w_fit = np.column_stack((rotation.z_to_u(src.z[src.subject_ids[idx]]),
                             src.x[idx]))
    w_eval = np.column_stack((rotation.z_to_u(eval_points.latent),
                              eval_points.x))
    y = prepared[y_name]
    binary = y_name.endswith("choice")
    if method == "wy_polynomial":
        hh = polynomial_hessians(w_fit, y, w_eval, binary=binary, degree=degree)
    elif method == "wy_kernel_ridge":
        hh = kernel_ridge_hessians(w_fit, y, w_eval, binary=binary,
                                  max_centers=max_centers, seed=seed)
    else:
        hh = local_quadratic_hessians(w_fit, y, w_eval, binary=binary,
                                     neighbors=neighbors)
    m=np.average(hh**2, axis=0, weights=eval_points.weights)
    m=(m+m.T)/2
    target="logit" if y_name=="nn_logit" else "probability"
    return HEResult(m,src.latent_dim,method,target,rotation.name,measure_name,
                    np.ones(m.shape, bool), {
                        "training_rows":len(idx), "evaluation_rows":len(w_eval),
                        "binary":binary, "y_name":y_name,
                        "evaluation_meta":eval_points.meta,
                        "choice_fit": ("logistic polynomial" if binary and method=="wy_polynomial"
                                       else "smooth probability WLS/KRR" if binary else None)})
