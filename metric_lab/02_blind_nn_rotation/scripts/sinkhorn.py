# Sinkhorn-balanced UX sparsity (parameter-free scaling: rows -> 1/d, cols -> 1/p), I1 (logit autograd E[H^2]), R*-free, strong search. T only for evaluation.
#  SK-H  = 1 - Hoyer(vec balanced UX)        SK-L0 = mean((balanced*d*p + 1e-4)^0.1)  (soft-L0 on the balanced matrix)
#  fusions (plain z-sums, null = random rotations):  zSK-H + zC ;  zSK-H + zSK-L0 ;  zSK-H + zSK-L0 + zC
from lib2 import *
import bench, time
for model in sys.argv[1].split(","):
    t0=time.time(); g=G(model,seed=2); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std); C=g.mom["logit"]
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def ux(Q):
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.maximum(np.einsum("ai,ij,aj->a",bs,C,bs,optimize=True).reshape(d,p),0)
    def uu(Q): m=transform_moment(C,J2(Q)); return m[:d,:d]
    def sink(U,it=60):
        M=U/U.sum()+1e-9; r=np.ones(d); c=np.ones(p)
        for _ in range(it): r=(1.0/d)/(M@c); c=(1.0/p)/(M.T@r)
        return M*r[:,None]*c[None,:]
    def hoyer(v):
        v=np.abs(np.asarray(v,float)).ravel(); n=v.size; s2=np.sqrt((v**2).sum()); return 0.0 if s2<=0 else float((np.sqrt(n)-v.sum()/s2)/(np.sqrt(n)-1))
    mH=lambda Q:1-hoyer(sink(ux(Q))); mL=lambda Q:float(np.mean((sink(ux(Q))*d*p+1e-4)**0.1)); mC=lambda Q:float(np.trace(uu(Q))/uu(Q).sum())
    Hz=special_ortho_group.rvs(d,size=200,random_state=4); nul={}
    for k,m in (("H",mH),("L",mL),("C",mC)): v=[m(Q) for Q in Hz]; nul[k]=(float(np.median(v)),float(np.std(v))+1e-12)
    z=lambda k,m:(lambda Q:(m(Q)-nul[k][0])/nul[k][1]); zH,zL,zC=z("H",mH),z("L",mL),z("C",mC)
    mets={"SK-H":mH,"SK-L0":mL,"zSK-H + zC":lambda Q:zH(Q)+zC(Q),"zSK-H + zSK-L0":lambda Q:zH(Q)+zL(Q),"zSK-H + zSK-L0 + zC":lambda Q:zH(Q)+zL(Q)+zC(Q)}
    def ev(Q):
        U=g.Zs@Q.T; Cc=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-Cc)
        m=Cc[r,c]; mask=np.ones_like(Cc,bool); mask[r,c]=False; off=Cc[mask]
        return f"MCC {m.mean():.3f} (min {m.min():.3f}) leak {off.mean():.3f}/{off.max():.3f}"
    def strong(f,nh=500,nst=12,seed=4,maxiter=1200):
        H6=special_ortho_group.rvs(d,size=nh,random_state=seed); v=np.array([f(Q) for Q in H6]); best=(1e18,None)
        for i in np.argsort(v)[:nst]:
            val,Q=g.opt(f,H6[i],maxiter=maxiter)
            if val<best[0]: best=(val,Q)
        return best
    print(f"\n## [sinkhorn] {model} d={d} p={p} ceiling: {ev(Rorig)}",flush=True)
    for name,f in mets.items(): v,Q=strong(f); print(f"  {name:22s}: {ev(Q)}",flush=True)
    print(f"DONE {model} [{time.time()-t0:.0f}s]",flush=True)
