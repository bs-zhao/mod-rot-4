# (clustered variant) PRE-REGISTERED: metrics whose null (random-rotation) values correlate >=0.8 are one family; within a family average the z-scores, across families sum.
# [orig header] full-coverage fusion: one representative of EVERY rotation-variant summary family listed in the doc (sec.17 table), z-scored vs random rotations, plain sum.
#  A  rowsp(a=.5,q=.1,eps=1e-4)   UX per-row support sparsity           (a=.5 is the T-picked value, flagged)
#  B  1-Hoyer(vec UX)             UX global concentration
#  G  1-Hoyer(r_i), r_i=row sums  UX row-energy concentration (G12)
#  K  colH                        UX column-wise concentration
#  C  trace(UU)/sum(UU)           UU diagonal share (E06)
#  S  tail1(diag UU)              self-curvature tail, q=1 (L01)
#  F6 = zA+zB+zG+zK+zC+zS ; leave-one-out F6\X.  I1 (E[H^2], logit autograd), R*-free, strong search. T only for evaluation.
from lib2 import *
import bench, time
for model in sys.argv[1].split(","):
    t0=time.time(); g=G(model,seed=2); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std); C=g.mom["logit"]
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def ux(Q):
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.maximum(np.einsum("ai,ij,aj->a",bs,C,bs,optimize=True).reshape(d,p),0)
    def uu(Q): m=transform_moment(C,J2(Q)); return m[:d,:d]
    def hoyer(v):
        v=np.abs(np.asarray(v,float)).ravel(); n=v.size; s2=np.sqrt((v**2).sum()); return 0.0 if s2<=0 or n<2 else float((np.sqrt(n)-v.sum()/s2)/(np.sqrt(n)-1))
    mA=lambda Q:bench.rowsp(None,ux(Q),a=0.5,q=0.1,eps=1e-4); mB=lambda Q:1-hoyer(ux(Q)); mG=lambda Q:1-hoyer(ux(Q).sum(1))
    mK=lambda Q:(lambda U:float(np.sum(U.sum(0)/U.sum()*np.array([1-hoyer(U[:,a]) for a in range(U.shape[1])]))))(ux(Q))
    mC=lambda Q:(lambda U:float(np.trace(U)/U.sum()))(uu(Q)); mS=lambda Q:(lambda U:(lambda s:float(s[1:].sum()/U.sum()))(np.sort(np.diag(U))[::-1]))(uu(Q))
    fam={"A":mA,"B":mB,"G":mG,"K":mK,"C":mC,"S":mS}
    Hz=special_ortho_group.rvs(d,size=400,random_state=4); nul={}
    for k,m in fam.items(): v=[m(Q) for Q in Hz]; nul[k]=(float(np.median(v)),float(np.std(v))+1e-12)
    from scipy.cluster.hierarchy import linkage,fcluster
    from scipy.spatial.distance import squareform
    vals={k:np.array([m(Q) for Q in Hz]) for k,m in fam.items()}; keys=list(fam)
    R=np.corrcoef(np.array([vals[k] for k in keys])); Dm=1-np.abs(R); np.fill_diagonal(Dm,0)
    lab=fcluster(linkage(squareform(Dm,checks=False),"average"),t=0.2,criterion="distance")
    groups={}; [groups.setdefault(l,[]).append(k) for k,l in zip(keys,lab)]
    print(f"## [famcluster] {model} d={d} p={p}  null |corr| matrix:",flush=True)
    print("      "+" ".join(f"{k:>5s}" for k in keys)); [print(f"  {keys[i]:>3s} "+" ".join(f"{abs(R[i,j]):5.2f}" for j in range(len(keys)))) for i in range(len(keys))]
    print("  families (|corr|>=0.8, average linkage):",list(groups.values()),flush=True)
    z={k:(lambda Q,k=k,m=m:(m(Q)-vals[k].mean())/(vals[k].std()+1e-12)) for k,m in fam.items()}
    fcl=lambda Q:sum(np.mean([z[k](Q) for k in g_]) for g_ in groups.values())
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
    v,Q=strong(fcl); print(f"  F_cluster (z-mean within family, sum across families): {ev(Q)}",flush=True)
    print(f"DONE {model} [{time.time()-t0:.0f}s]",flush=True)
