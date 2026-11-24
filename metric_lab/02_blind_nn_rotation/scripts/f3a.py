# Sensitivity of F3 = zA(a)+zB+zC to the row-weight exponent a  (E[H^2] moment path, I1). T only for evaluation.
# Goal: if F3 works for a in {0, .5, 1}, use a=1 (plain energy weighting, no free parameter) and drop the T-picked a=.5.
from lib2 import *
import bench, time
AG=[0.0,0.5,1.0]
for model in sys.argv[1].split(","):
    t0=time.time(); g=G(model,seed=2); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std); C=g.mom["logit"]
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def ux(Q):
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.maximum(np.einsum("ai,ij,aj->a",bs,C,bs,optimize=True).reshape(d,p),0)
    def uu(Q): m=transform_moment(C,J2(Q)); return m[:d,:d]
    hoy=lambda U:(lambda v:float(1-(np.sqrt(v.size)-np.abs(v).sum()/np.sqrt((v**2).sum()))/(np.sqrt(v.size)-1)))(U.ravel())
    mB=lambda Q:hoy(ux(Q)); mC=lambda Q:float(np.trace(uu(Q))/uu(Q).sum())
    Hz=special_ortho_group.rvs(d,size=200,random_state=4)
    def nul(m): v=[m(Q) for Q in Hz]; return float(np.median(v)),float(np.std(v))+1e-12
    nB,nC=nul(mB),nul(mC)
    def ev(Q):
        U=g.Zs@Q.T; Cc=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-Cc)
        m=Cc[r,c]; mask=np.ones_like(Cc,bool); mask[r,c]=False; off=Cc[mask]
        return f"MCC {m.mean():.3f} (min {m.min():.3f}) leak {off.mean():.3f}/{off.max():.3f}"
    def strong(f,nh=600,nst=20,seed=4,maxiter=1500):
        H6=special_ortho_group.rvs(d,size=nh,random_state=seed); v=np.array([f(Q) for Q in H6]); best=(1e18,None)
        for i in np.argsort(v)[:nst]:
            val,Q=g.opt(f,H6[i],maxiter=maxiter)
            if val<best[0]: best=(val,Q)
        return best
    print(f"\n## [F3 vs a] {model} d={d} p={p}",flush=True)
    for a in AG:
        mA=lambda Q,a=a:bench.rowsp(None,ux(Q),a=a,q=0.1,eps=1e-4); nA=nul(mA)
        f=lambda Q:(mA(Q)-nA[0])/nA[1]+(mB(Q)-nB[0])/nB[1]+(mC(Q)-nC[0])/nC[1]
        v,Q=strong(f); print(f"  F3 with a={a:<4}: {ev(Q)}",flush=True)
    print(f"DONE {model} [{time.time()-t0:.0f}s]",flush=True)
