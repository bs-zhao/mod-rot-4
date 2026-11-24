import sys, glob, itertools, warnings, json, time, os
import numpy as np
warnings.filterwarnings("ignore")
ROOT="D:/Projects/rotation/mod-rot-4"; HERE=os.path.dirname(os.path.abspath(__file__))
sys.path[:0]=[f"{ROOT}/metric_lab/01_hessian_curvature_energy",f"{ROOT}/models/src",f"{ROOT}/nn/src"]
from he_lab.sources import load_sources, load_rotations, get_spec
from he_lab.energy import transform_moment
from he_lab.decoder import Decoder
from he_lab.derivatives import moment_hessians
from scipy.stats import special_ortho_group
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import expm
class G:
    def __init__(s,model,run_glob=None,ndraw=None,N=4096,seed=2,lat_mode="empirical",repeat=1):
        s.model=model
        run=sorted(glob.glob(f"{ROOT}/outputs/nn/{model}/*/repeat{repeat:02d}"))[0] if run_glob is None else glob.glob(run_glob)[0]
        s.src=load_sources(model,run); src=s.src; s.d=src.latent_dim; s.p=src.x_dim
        s.base=[r for r in load_rotations(src,"s9") if r.candidate_type=="baseline_unswept"][0]
        s.xs=np.asarray(src.x_std,float)
        tag=f"{model}_{os.path.basename(os.path.dirname(run))}_{lat_mode}_{N}_{seed}"+("" if repeat==1 else f"_r{repeat}")
        f=f"{HERE}/cache_{tag}.npz"
        if not os.path.exists(f):
            dec=Decoder(src); rng=np.random.default_rng(seed)
            if src.fixed_x is not None:
                nd=ndraw or 24; lat=src.z[rng.choice(len(src.z),nd,replace=True)] if lat_mode=="empirical" else rng.multivariate_normal(src.z.mean(0),np.cov(src.z,rowvar=False,ddof=0),size=nd)
                bank=src.fixed_x; pts=np.column_stack((np.repeat(lat,len(bank),0),np.tile(bank,(nd,1))))
            else:
                lat=src.z[rng.choice(len(src.z),N,replace=True)] if lat_mode=="empirical" else rng.multivariate_normal(src.z.mean(0),np.cov(src.z,rowvar=False,ddof=0),size=N)
                x=np.asarray(get_spec(model).sample_inputs(N,rng),float); pts=np.column_stack((lat,x))
            out={}
            for t in ("logit","probability"):
                out[t]=moment_hessians(lambda v,t=t:dec.scalar(v,target=t),pts,weights=np.ones(len(pts)),chunk=64)
            np.savez(f,logit=out["logit"],probability=out["probability"],npts=len(pts))
        z=np.load(f); s.mom={"logit":z["logit"],"probability":z["probability"]}
        Z=np.asarray(src.z); T=np.asarray(src.parameters); s.Zs=(Z-Z.mean(0))/Z.std(0); s.Ts=(T-T.mean(0))/T.std(0)
    def J(s,Q,xstd=True):
        d,p=s.d,s.p; J=np.eye(d+p); J[:d,:d]=np.diag(s.base.z_std)@(Q@s.base.R_star).T
        if xstd: J[d:,d:]=np.diag(s.xs)
        return J
    def blocks(s,ch,Q,xstd=True):
        m=transform_moment(s.mom[ch],s.J(Q,xstd)); d=s.d; return m[:d,:d],m[:d,d:]
    def skew(s,x):
        d=s.d; A=np.zeros((d,d)); A[np.triu_indices(d,1)]=x; return A-A.T
    def err(s,Q):
        d=s.d; Cc=np.abs(Q); r,c=linear_sum_assignment(-Cc); P=np.zeros((d,d)); P[r,c]=np.sign(Q[r,c]); A=Q@P.T
        if np.linalg.det(A)<0: A=A@np.diag([-1]+[1]*(d-1))
        return float(np.degrees(np.max(np.abs(np.angle(np.linalg.eigvals(A))))))
    def mcc(s,Q):
        d=s.d; U=s.Zs@(Q@s.base.R_star).T; Cc=np.abs(np.corrcoef(np.c_[U,s.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-Cc)
        return float(Cc[r,c].mean()), float((Cc.sum()-Cc[r,c].sum())/(d*d-d))
    def opt(s,f,Q0,maxiter=300):
        d=s.d; g=lambda x:f(expm(s.skew(x))@Q0)
        r=minimize(g,np.zeros(d*(d-1)//2),method="Nelder-Mead",options=dict(maxiter=maxiter*(1 if d<=4 else 3),xatol=1e-4,fatol=1e-14))
        return r.fun,expm(s.skew(r.x))@Q0
    def search(s,f,nh=300,nstart=5,seed=4):
        Hh=special_ortho_group.rvs(s.d,size=nh,random_state=seed); v=np.array([f(Q) for Q in Hh])
        res=[s.opt(f,Hh[i]) for i in np.argsort(v)[:nstart]]; return min(res,key=lambda r:r[0])

import torch
def per_point(g,N=4096,seed=2,lat_mode="empirical",target="logit"):
    src=g.src; dec=Decoder(src); rng=np.random.default_rng(seed)
    if src.fixed_x is not None:
        nd=max(1,N//len(src.fixed_x)); lat=src.z[rng.choice(len(src.z),nd,replace=True)]; bank=src.fixed_x
        pts=np.column_stack((np.repeat(lat,len(bank),0),np.tile(bank,(nd,1))))
    else:
        lat=src.z[rng.choice(len(src.z),N,replace=True)]; x=np.asarray(get_spec(g.model).sample_inputs(N,rng),float); pts=np.column_stack((lat,x))
    fn=lambda v:dec.scalar(v,target=target)
    H=torch.func.vmap(torch.func.hessian(fn))(torch.as_tensor(pts)).numpy(); Gr=torch.func.vmap(torch.func.grad(fn))(torch.as_tensor(pts)).numpy()
    return pts,H,Gr

class PP:
    """Per-point Hessian/gradient store for NN (or oracle) with level-set (gradient-orthogonal) energies."""
    def __init__(s,g,N=4096,seed=2,target="logit",stab=0.05):
        s.g=g; s.pts,s.H,s.Gr=per_point(g,N=N,seed=seed,target=target); s.stab=stab
    def energies(s,Q,mode="perp",xstd=True,power=1.0):
        g=s.g; d=g.d; J=g.J(Q,xstd)
        HW=np.einsum("ia,nij,jb->nab",J,s.H,J,optimize=True); gW=np.einsum("ia,ni->na",J,s.Gr,optimize=True)
        if mode=="plain": E=HW**2; return E.mean(0)[:d,:d],E.mean(0)[:d,d:]
        nrm=np.linalg.norm(gW,axis=1,keepdims=True); gh=gW/(nrm+1e-30); D=HW.shape[1]
        Hp=HW-np.einsum("na,nb,nbc->nac",gh,gh,HW)  # P H
        Hp=Hp-np.einsum("nab,nc,nb->nac",Hp,gh,gh)  # P H P
        scale=1.0/(nrm[:,0]+s.stab*nrm.mean())**power
        E=(Hp**2)*(scale**2)[:,None,None]
        Em=E.mean(0); return Em[:d,:d],Em[:d,d:]

from he_lab.oracle import oracle_scalar
class OG:
    """oracle (analytic) stand-in: coordinates U=Q T_std"""
    def __init__(s,model,N=4096,seed=3,target="logit"):
        s.model=model; s.g0=None
        D=np.load(f"{ROOT}/outputs/models/generate/{model}/s1_generate__dataset.npz"); T=D["parameters"]; s.ts=T.std(0); s.d=T.shape[1]
        src=load_sources(model,sorted(glob.glob(f"{ROOT}/outputs/nn/{model}/*/repeat01"))[0]); s.p=src.x_dim; s.xs=np.asarray(src.x_std,float)
        rng=np.random.default_rng(seed)
        if model=="CR3":
            bank=D["x"][:500]; ok=(bank[:,[0,1,3,4]]>0).all(1)&(bank[:,[2,5]]>0).all(1)&(bank[:,[2,5]]<1).all(1); bank=bank[ok]
            nd=max(1,N//len(bank)); lat=T[rng.choice(len(T),nd,replace=True)]; pts=np.column_stack((np.repeat(lat,len(bank),0),np.tile(bank,(nd,1))))
        else:
            lat=T[rng.choice(len(T),N,replace=True)]; x=np.asarray(get_spec(model).sample_inputs(N,rng),float); pts=np.column_stack((lat,x))
        fn=lambda v:oracle_scalar(model,v,target=target)
        s.pts=pts; s.H=torch.func.vmap(torch.func.hessian(fn))(torch.as_tensor(pts)).numpy(); s.Gr=torch.func.vmap(torch.func.grad(fn))(torch.as_tensor(pts)).numpy()
    def J(s,Q,xstd=True):
        J=np.eye(s.d+s.p); J[:s.d,:s.d]=np.diag(s.ts)@Q.T
        if xstd: J[s.p*0+s.d:,s.d:]=np.diag(s.xs)
        return J
    def err(s,Q):
        d=s.d; Cc=np.abs(Q); r,c=linear_sum_assignment(-Cc); P=np.zeros((d,d)); P[r,c]=np.sign(Q[r,c]); A=Q@P.T
        if np.linalg.det(A)<0: A=A@np.diag([-1]+[1]*(d-1))
        return float(np.degrees(np.max(np.abs(np.angle(np.linalg.eigvals(A))))))
    def skew(s,x): return G.skew(s,x)
    def opt(s,f,Q0,maxiter=300): return G.opt(s,f,Q0,maxiter)
    def search(s,f,nh=300,nstart=5,seed=4): return G.search(s,f,nh,nstart,seed)
class PO(PP):
    def __init__(s,og,stab=0.05): s.g=og; s.pts,s.H,s.Gr=og.pts,og.H,og.Gr; s.stab=stab
