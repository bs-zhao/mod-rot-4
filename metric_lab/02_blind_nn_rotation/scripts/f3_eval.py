# FROZEN pipeline evaluation on a given NN run. usage: f3_eval.py MODEL:REPEAT[,MODEL:REPEAT...]
# F3 = zA+zB+zC, I1 (logit autograd E[H^2], empirical Z x model X, std X), a=.5,q=.1,eps=1e-4, R*-free, strong search (600/20/1500). NOTHING tuned after seeing these results.
from lib2 import *
import bench, time
for spec in sys.argv[1].split(","):
    model,rep=spec.split(":"); rep=int(rep); t0=time.time(); g=G(model,seed=2,repeat=rep); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std); C=g.mom["logit"]
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def ux(Q):
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.maximum(np.einsum("ai,ij,aj->a",bs,C,bs,optimize=True).reshape(d,p),0)
    def uu(Q): m=transform_moment(C,J2(Q)); return m[:d,:d]
    hoy=lambda U:(lambda v:float(1-(np.sqrt(v.size)-np.abs(v).sum()/np.sqrt((v**2).sum()))/(np.sqrt(v.size)-1)))(U.ravel())
    mA=lambda Q:bench.rowsp(None,ux(Q),a=0.5,q=0.1,eps=1e-4); mB=lambda Q:hoy(ux(Q)); mC=lambda Q:float(np.trace(uu(Q))/uu(Q).sum())
    Hz=special_ortho_group.rvs(d,size=200,random_state=4); nul={}
    for k,m in (("A",mA),("B",mB),("C",mC)): v=[m(Q) for Q in Hz]; nul[k]=(float(np.median(v)),float(np.std(v))+1e-12)
    z=lambda k,m:(lambda Q:(m(Q)-nul[k][0])/nul[k][1]); zA,zB,zC=z("A",mA),z("B",mB),z("C",mC)
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
    print(f"\n## [FROZEN] {model} repeat{rep} d={d} p={p} ceiling: {ev(Rorig)}",flush=True)
    for name,f in {"A rowsp":mA,"B global UX 1-Hoyer":mB,"F3 = zA+zB+zC":lambda Q:zA(Q)+zB(Q)+zC(Q)}.items():
        v,Q=strong(f); print(f"  {name:22s}: {ev(Q)}",flush=True)
    print(f"DONE {model} r{rep} [{time.time()-t0:.0f}s]",flush=True)
