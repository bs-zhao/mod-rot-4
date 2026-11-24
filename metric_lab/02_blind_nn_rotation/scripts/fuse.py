# PRE-REGISTERED parameter-free fusions (T only for evaluation). A=rowsp(a=.5,q=.1,eps=1e-4) [a was T-picked earlier: flagged], B=global UX 1-Hoyer, C=UU self-fraction.
#  F1 = A + B (both already in [0,1], plain sum)
#  F2 = zA + zB       z = (m - median_Haar)/SD_Haar  (null = uniformly random rotation; inverse-variance scaling, no free weight)
#  F3 = zA + zB + zC
from lib2 import *
import bench, time
from bench import udshare
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for model in sys.argv[1].split(","):
    t0=time.time(); g=G(model,seed=2); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std); C=g.mom["logit"]
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def ux(Q):
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.einsum("ai,ij,aj->a",bs,C,bs,optimize=True).reshape(d,p)
    def uu(Q): m=transform_moment(C,J2(Q)); return m[:d,:d]
    hoy=lambda U:(lambda v:float(1-(np.sqrt(v.size)-np.abs(v).sum()/np.sqrt((v**2).sum()))/(np.sqrt(v.size)-1)))(U.ravel())
    mA=lambda Q:bench.rowsp(None,ux(Q),a=0.5,q=0.1,eps=1e-4); mB=lambda Q:hoy(ux(Q)); mC=lambda Q:float(np.trace(uu(Q))/uu(Q).sum())
    Hs=special_ortho_group.rvs(d,size=200,random_state=4)
    nul={k:(np.median(v:=[m(Q) for Q in Hs]),np.std(v)+1e-12) for k,m in (("A",mA),("B",mB),("C",mC))}
    z=lambda k,m:(lambda Q:(m(Q)-nul[k][0])/nul[k][1])
    zA,zB,zC=z("A",mA),z("B",mB),z("C",mC)
    mets={"A rowsp":mA,"B global UX 1-Hoyer":mB,"F1 = A+B (plain)":lambda Q:mA(Q)+mB(Q),"F2 = zA+zB":lambda Q:zA(Q)+zB(Q),"F3 = zA+zB+zC":lambda Q:zA(Q)+zB(Q)+zC(Q)}
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
    ec=ev(Rorig); print(f"\n## {model} d={d} p={p} ceiling {f(ec)}",flush=True); R={"ceiling":ec}
    for name,m in mets.items():
        v,Q=strong(m); R[name]=ev(Q); print(f"  {name:22s}: {f(R[name])}",flush=True)
    fig,axs=plt.subplots(2,3,figsize=(15,9.5))
    for ax,(name,e) in zip(axs.ravel(),R.items()):
        M=e['M']; im=ax.imshow(M,vmin=0,vmax=1,cmap="viridis")
        for i in range(d):
            for j in range(d): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="w" if M[i,j]<0.6 else "k",fontsize=9)
        ax.set_xticks(range(d)); ax.set_xticklabels([f"T{j+1}" for j in range(d)]); ax.set_yticks(range(d)); ax.set_yticklabels([f"U{i+1}" for i in range(d)])
        ax.set_title(f"{name}"+chr(10)+f"MCC {e['mcc']:.3f} leak {e['lm']:.3f}/{e['lx']:.3f}",fontsize=10)
    fig.suptitle(f"{model}: parameter-free fusions |corr(U,T)|",fontsize=12); fig.colorbar(im,ax=axs,shrink=0.7)
    fig.savefig(f"heatF_{model}.png",dpi=105,bbox_inches="tight"); plt.close(fig); print(f"HEAT {model} [{time.time()-t0:.0f}s]",flush=True)
