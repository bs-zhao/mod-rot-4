"""Local W,Y weighted quadratic fits; no true parameter or trained NN required."""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .common import _standardize,validate,finish

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



def estimate_he(W,Y,*,eval_W,weights=None,dim_u,target="probability",binary=False,
                ridge=1e-3,neighbors=180,rotation_id="",measure="",**kwargs):
    W,Y,eval_W,weights=validate(W,Y,eval_W,weights,dim_u,binary,target)
    H=local_quadratic_hessians(W,Y,eval_W,binary=binary,ridge=ridge,neighbors=neighbors)
    return finish(H,weights,dim_u,"wy_local_quadratic",target,rotation_id,measure,
                  {"binary":binary,"ridge":ridge,"neighbors":neighbors,"train_rows":len(W)})
