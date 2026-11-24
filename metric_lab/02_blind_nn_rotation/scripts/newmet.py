# New metric families on I1 (logit autograd, empirical Z, std X, R*-free), T only for evaluation. Parameter-free by construction.
#  colH  = sum_a (c_a/S) * (1 - Hoyer(UX[:,a]))          column-wise concentration (c_a = rotation-invariant column energy)
#  L07-A / L07-B = mean over K=3 unsupervised X-clusters (k-means on standardised X) of A / B computed on that cluster's UX
#  fusions: plain sums of [0,1]-bounded metrics, or z-sums (null = random rotation)
from lib2 import *
import bench, time
from bench import udshare
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def kmeans(X,K=3,seed=0,it=30):
    r=np.random.default_rng(seed); c=X[r.choice(len(X),K,replace=False)]
    for _ in range(it):
        l=np.argmin(((X[:,None]-c[None])**2).sum(-1),1); c=np.array([X[l==k].mean(0) if (l==k).any() else c[k] for k in range(K)])
    return l
for model in sys.argv[1].split(","):
    t0=time.time(); g=G(model,seed=2); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std); C=g.mom["logit"]
    pts,H,_=per_point(g,N=4096,seed=2,target="logit"); N=len(H); Hf=np.asarray(H).reshape(N,-1); lab=kmeans(pts[:,d:]/g.xs,3); Cs=[Hf[lab==k].T@Hf[lab==k]/max(1,(lab==k).sum()) for k in range(3)]
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def ux(Q,Cm=C):
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.maximum(np.einsum("ai,ij,aj->a",bs,Cm,bs,optimize=True).reshape(d,p),0)
    def uu(Q): m=transform_moment(C,J2(Q)); return m[:d,:d]
    def hoy(v):
        v=np.abs(v).ravel(); n=v.size; s2=np.sqrt((v**2).sum())
        return 0.0 if s2<=0 else float(1-(np.sqrt(n)-v.sum()/s2)/(np.sqrt(n)-1))
    rowA=lambda U:bench.rowsp(None,U,a=0.5,q=0.1,eps=1e-4)
    colH=lambda U:(lambda c:float(np.sum(c/c.sum()*np.array([1-(0 if U.shape[0]<2 else hoy_axis(U[:,a])) for a in range(U.shape[1])]))))(U.sum(0))
    def hoy_axis(v):
        v=np.abs(v); n=len(v); s2=np.sqrt((v**2).sum()); return 0.0 if s2<=0 else float((np.sqrt(n)-v.sum()/s2)/(np.sqrt(n)-1))
    mA=lambda Q:rowA(ux(Q)); mB=lambda Q:hoy(ux(Q)); mcol=lambda Q:colH(ux(Q)); mC=lambda Q:float(np.trace(uu(Q))/uu(Q).sum())
    l07A=lambda Q:float(np.mean([rowA(ux(Q,Cm)) for Cm in Cs])); l07B=lambda Q:float(np.mean([hoy(ux(Q,Cm)) for Cm in Cs]))
    Hs=special_ortho_group.rvs(d,size=200,random_state=4)
    nul={}
    for k,m in (("A",mA),("B",mB),("col",mcol),("C",mC)):
        v=[m(Q) for Q in Hs]; nul[k]=(float(np.median(v)),float(np.std(v))+1e-12)
    z=lambda k,m:(lambda Q:(m(Q)-nul[k][0])/nul[k][1]); zA,zB,zc,zC=z("A",mA),z("B",mB),z("col",mcol),z("C",mC)
    mets={"colH (column concentration)":mcol,"L07-A (3 X-regions)":l07A,"L07-B (3 X-regions)":l07B,
          "A+colH":lambda Q:mA(Q)+mcol(Q),"B+colH":lambda Q:mB(Q)+mcol(Q),"A+B+colH":lambda Q:mA(Q)+mB(Q)+mcol(Q),
          "zA+zB+zcol":lambda Q:zA(Q)+zB(Q)+zc(Q),"L07-A+L07-B":lambda Q:l07A(Q)+l07B(Q)}
    def strong(f,nh=600,nst=20,seed=4,maxiter=1500):
        H6=special_ortho_group.rvs(d,size=nh,random_state=seed); v=np.array([f(Q) for Q in H6]); best=(1e18,None)
        for i in np.argsort(v)[:nst]:
            val,Q=g.opt(f,H6[i],maxiter=maxiter)
            if val<best[0]: best=(val,Q)
        return best
    def ev(Q):
        U=g.Zs@Q.T; Cc=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-Cc)
        m=Cc[r,c]; mask=np.ones_like(Cc,bool); mask[r,c]=False; off=Cc[mask]
        return dict(mcc=float(m.mean()),mn=float(m.min()),lm=float(off.mean()),lx=float(off.max()),M=Cc[[r[list(c).index(j)] for j in range(d)]])
    f=lambda e:f"MCC {e['mcc']:.3f} (min {e['mn']:.3f}) leak {e['lm']:.3f}/{e['lx']:.3f}"
    ec=ev(Rorig); print(f"\n## [new] {model} d={d} p={p} ceiling {f(ec)}",flush=True); R={"ceiling":ec}
    for name,m in mets.items():
        v,Q=strong(m); R[name]=ev(Q); print(f"  {name:28s}: {f(R[name])}",flush=True)
    names=list(R); fig,axs=plt.subplots(3,3,figsize=(15,14))
    for ax,name in zip(axs.ravel(),names):
        e=R[name]; M=e['M']; im=ax.imshow(M,vmin=0,vmax=1,cmap="viridis")
        for i in range(d):
            for j in range(d): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="w" if M[i,j]<0.6 else "k",fontsize=8)
        ax.set_xticks(range(d)); ax.set_xticklabels([f"T{j+1}" for j in range(d)]); ax.set_yticks(range(d)); ax.set_yticklabels([f"U{i+1}" for i in range(d)])
        ax.set_title(f"{name}"+chr(10)+f"MCC {e['mcc']:.3f} leak {e['lm']:.3f}/{e['lx']:.3f}",fontsize=9)
    fig.suptitle(f"{model}: new metric families |corr(U,T)|",fontsize=12); fig.colorbar(im,ax=axs,shrink=0.5)
    fig.savefig(f"heatN2_{model}.png",dpi=95,bbox_inches="tight"); plt.close(fig); print(f"HEAT new {model} [{time.time()-t0:.0f}s]",flush=True)
