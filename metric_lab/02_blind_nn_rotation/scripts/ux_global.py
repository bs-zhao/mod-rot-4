# Global UX-concentration metrics on the NN (R*-free, logit, std X). Hypothesis from the ANALYTIC oracle table (five/res_*.json):
# global UX entropy works on C1/C2/C3 but not CR3/CR4, while rowsp(a=.5) works on CR3/CR4/CR5/C1 but not C2/C3.
from lib2 import *
import bench, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for model in sys.argv[1].split(","):
    t0=time.time(); g=G(model,seed=2); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std); Cfull=g.mom["logit"]
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def ux_fast(C,Q):
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.einsum("ai,ij,aj->a",bs,C,bs,optimize=True).reshape(d,p)
    def strong(f,nh=600,nst=20,seed=4,maxiter=1500):
        Hs=special_ortho_group.rvs(d,size=nh,random_state=seed); v=np.array([f(Q) for Q in Hs]); best=(1e18,None)
        for i in np.argsort(v)[:nst]:
            val,Q=g.opt(f,Hs[i],maxiter=maxiter)
            if val<best[0]: best=(val,Q)
        return best,float(np.median(v))
    def evalQ(Q):
        U=g.Zs@Q.T; Cc=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-Cc)
        m=Cc[r,c]; mask=np.ones_like(Cc,bool); mask[r,c]=False; off=Cc[mask]
        return dict(mcc=m.mean(),mn=m.min(),lm=off.mean(),lx=off.max(),err=g.err(Q@Rorig.T),M=Cc[[r[list(c).index(j)] for j in range(d)]])
    fmt=lambda e:f"MCC {e['mcc']:.3f} (min {e['mn']:.3f}) | leak mean {e['lm']:.3f} max {e['lx']:.3f} | err {e['err']:.1f}deg"
    ceil=evalQ(Rorig); print(f"\n## {model} d={d} p={p}   ceiling: {fmt(ceil)}",flush=True)
    ent=lambda U:(lambda q:float(-(q*np.log(q+1e-12)).sum()))(U.ravel()/U.sum())
    hoy=lambda U:(lambda v:float(1-(np.sqrt(v.size)-np.abs(v).sum()/np.sqrt((v**2).sum()))/(np.sqrt(v.size)-1)))(U.ravel())
    mets={"rowsp a=.5 (ref)":lambda U:bench.rowsp(None,U,a=0.5,q=0.1,eps=1e-4),"UX entropy (global)":ent,"UX 1-Hoyer (global)":hoy}
    QS={"ceiling":ceil}; 
    for name,m in mets.items():
        (v,Q),med=strong(lambda Q,m=m:m(ux_fast(Cfull,Q))); e=evalQ(Q); QS[name]=e
        print(f"  {name:22s}: {fmt(e)} | rel.drop {(med-v)/med:.3f}",flush=True)
    fig,axs=plt.subplots(1,4,figsize=(20,5))
    for ax,(name,e) in zip(axs,QS.items()):
        M=e['M']; im=ax.imshow(M,vmin=0,vmax=1,cmap="viridis")
        for i in range(d):
            for j in range(d): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="w" if M[i,j]<0.6 else "k",fontsize=8)
        ax.set_xticks(range(d)); ax.set_xticklabels([f"T{j+1}" for j in range(d)]); ax.set_yticks(range(d)); ax.set_yticklabels([f"U{i+1}" for i in range(d)])
        ax.set_title(f"{name}"+chr(10)+f"MCC {e['mcc']:.3f} leak {e['lm']:.3f}/{e['lx']:.3f}",fontsize=9)
    fig.suptitle(f"{model}: global UX concentration metrics |corr(U,T)|",fontsize=12); fig.colorbar(im,ax=axs,shrink=0.8)
    fig.savefig(f"heatU_{model}.png",dpi=100,bbox_inches="tight"); plt.close(fig); print(f"HEAT {model}  [{time.time()-t0:.0f}s]",flush=True)
