# Energy-definition variants (doc sec.1 table). usage: energydef.py TAG MODELS ; TAG in {cov, abs}
#  cov: centred second moment  Cov(vec H) = E[vecH vecH^T] - mu mu^T  (removes the mean/systematic curvature)  [Var(H_ij) in the doc table]
#  abs: mean ABSOLUTE curvature E|H'_ij| (L1-type, less dominated by large entries), N=1024 per-point Hessians
# metrics: A rowsp, B global UX 1-Hoyer, F1=A+B, F2=zA+zB, F3=zA+zB+zC (C = UU self-fraction). T only for evaluation.
from lib2 import *
import bench, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TAG=sys.argv[1]
for model in sys.argv[2].split(","):
    t0=time.time(); g=G(model,seed=2); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std)
    pts,H,Gr=per_point(g,N=4096,seed=2,target="logit"); H=np.asarray(H); Gr=np.asarray(Gr); N=len(H)
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    if TAG=="cov":
        Hf=H.reshape(N,-1); mu=Hf.mean(0); Cm=Hf.T@Hf/N-np.outer(mu,mu)
        def ux(Q):
            J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
            return np.maximum(np.einsum("ai,ij,aj->a",bs,Cm,bs,optimize=True).reshape(d,p),0)
        def uu(Q): m=transform_moment(Cm,J2(Q)); return np.maximum(m[:d,:d],0)
        info=f"centred moment; mean-curvature share of total energy {1-np.trace(Cm)/np.trace(Hf.T@Hf/N):.3f}"
    elif TAG=="rk1":
        # remove the link-function artefact: per-point rank-one gradient-outer-product term  c*g*g^T,  c = g^T H g / |g|^4  (doc eq.3 structure)
        g2=(Gr**2).sum(1); c=np.einsum("ni,nij,nj->n",Gr,H,Gr)/(g2**2+1e-30); Hres=H-c[:,None,None]*Gr[:,:,None]*Gr[:,None,:]
        Hf=Hres.reshape(N,-1); Cm=Hf.T@Hf/N
        def ux(Q):
            J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
            return np.maximum(np.einsum("ai,ij,aj->a",bs,Cm,bs,optimize=True).reshape(d,p),0)
        def uu(Q): m=transform_moment(Cm,J2(Q)); return np.maximum(m[:d,:d],0)
        info=f"rank-one gradient term removed; share of energy removed {1-np.trace(Cm)/np.trace(H.reshape(N,-1).T@H.reshape(N,-1)/N):.3f}"
    else:
        Hs_=H[:1024]
        def blocks(Q):
            J=J2(Q); return np.einsum("ia,nij,jb->nab",J[:,:d],Hs_,J[:,d:],optimize=True),np.einsum("ia,nij,jb->nab",J[:,:d],Hs_,J[:,:d],optimize=True)
        def ux(Q): return np.abs(blocks(Q)[0]).mean(0)
        def uu(Q): return np.abs(blocks(Q)[1]).mean(0)
        info="mean absolute curvature, 1024 points"
    hoy=lambda U:(lambda v:float(1-(np.sqrt(v.size)-np.abs(v).sum()/np.sqrt((v**2).sum()))/(np.sqrt(v.size)-1)))(U.ravel())
    mA=lambda Q:bench.rowsp(None,ux(Q),a=0.5,q=0.1,eps=1e-4); mB=lambda Q:hoy(ux(Q)); mC=lambda Q:(lambda U:float(np.trace(U)/U.sum()))(uu(Q))
    Hz=special_ortho_group.rvs(d,size=120 if TAG=="abs" else 200,random_state=4); nul={}
    for k,m in (("A",mA),("B",mB),("C",mC)): v=[m(Q) for Q in Hz]; nul[k]=(float(np.median(v)),float(np.std(v))+1e-12)
    z=lambda k,m:(lambda Q:(m(Q)-nul[k][0])/nul[k][1]); zA,zB,zC=z("A",mA),z("B",mB),z("C",mC)
    mets={"A rowsp":mA,"B global UX 1-Hoyer":mB,"F1 = A+B":lambda Q:mA(Q)+mB(Q),"F2 = zA+zB":lambda Q:zA(Q)+zB(Q),"F3 = zA+zB+zC":lambda Q:zA(Q)+zB(Q)+zC(Q)}
    nh,nst,mi=(300,8,900) if TAG=="abs" else (600,20,1500)
    def strong(f,seed=4):
        H6=special_ortho_group.rvs(d,size=nh,random_state=seed); v=np.array([f(Q) for Q in H6]); best=(1e18,None)
        for i in np.argsort(v)[:nst]:
            val,Q=g.opt(f,H6[i],maxiter=mi)
            if val<best[0]: best=(val,Q)
        return best
    def ev(Q):
        U=g.Zs@Q.T; Cc=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-Cc)
        m=Cc[r,c]; mask=np.ones_like(Cc,bool); mask[r,c]=False; off=Cc[mask]
        return dict(mcc=float(m.mean()),mn=float(m.min()),lm=float(off.mean()),lx=float(off.max()),M=Cc[[r[list(c).index(j)] for j in range(d)]])
    f=lambda e:f"MCC {e['mcc']:.3f} (min {e['mn']:.3f}) leak {e['lm']:.3f}/{e['lx']:.3f}"
    ec=ev(Rorig); print(f"\n## [{TAG}] {model} d={d} p={p} ceiling {f(ec)} | {info}",flush=True); R={"ceiling":ec}
    for name,m in mets.items():
        v,Q=strong(m); R[name]=ev(Q); print(f"  {name:22s}: {f(R[name])}",flush=True)
    fig,axs=plt.subplots(2,3,figsize=(15,9.5))
    for ax,(name,e) in zip(axs.ravel(),R.items()):
        M=e['M']; im=ax.imshow(M,vmin=0,vmax=1,cmap="viridis")
        for i in range(d):
            for j in range(d): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="w" if M[i,j]<0.6 else "k",fontsize=9)
        ax.set_xticks(range(d)); ax.set_xticklabels([f"T{j+1}" for j in range(d)]); ax.set_yticks(range(d)); ax.set_yticklabels([f"U{i+1}" for i in range(d)])
        ax.set_title(f"{name}"+chr(10)+f"MCC {e['mcc']:.3f} leak {e['lm']:.3f}/{e['lx']:.3f}",fontsize=10)
    fig.suptitle(f"[{TAG}] {model}: |corr(U,T)|",fontsize=12); fig.colorbar(im,ax=axs,shrink=0.7)
    fig.savefig(f"heatE_{TAG}_{model}.png",dpi=105,bbox_inches="tight"); plt.close(fig); print(f"HEAT {TAG} {model} [{time.time()-t0:.0f}s]",flush=True)
