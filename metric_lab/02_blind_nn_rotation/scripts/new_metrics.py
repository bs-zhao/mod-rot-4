# PRE-DECLARED settings (no tuning against T): F2 uses q=0.5 (L1/L2-type sparsity); composites are equal-weight after Haar standardisation.
from lib2 import *
import bench, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FROZEN=dict(a=0.5,q=0.1,eps=1e-4)
def full(g,Q):
    d=g.d; U=g.Zs@(Q@g.base.R_star).T; C=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-C)
    m=C[r,c]; mask=np.ones_like(C,bool); mask[r,c]=False; off=C[mask]
    return f"MCC {m.mean():.3f} (min {m.min():.3f}) | leak mean {off.mean():.3f} max {off.max():.3f} | err {g.err(Q):.1f}deg"
for model in sys.argv[1].split(","):
    t0=time.time(); g=G(model,seed=2); d,p=g.d,g.p; D=d+p
    pts,H,Gr=per_point(g,N=4096,seed=2,target="logit"); H=np.asarray(H); print(f"\n## {model} d={d} p={p} per-point H {H.shape}",flush=True)
    C0=g.mom["logit"]; ids=np.arange(d)[:,None]*D+d+np.arange(p)[None,:]
    def F1(Q):   # per-axis X-coupling rank-1-ness (sign-aware, uses cross moments of H)
        J=g.J(Q,True); K=np.kron(J.T,J.T); C=K@C0@K.T; val=0.0; tot=0.0
        for i in range(d):
            R=C[np.ix_(ids[i],ids[i])]; R=(R+R.T)/2; tr=np.trace(R)
            if tr>1e-300: val+=tr-np.linalg.eigvalsh(R)[-1]; tot+=tr
        return val/tot
    def F2(Q):   # per-point L1/L2-type sparsity of U x X Hessian block, energy weighted
        J=g.J(Q,True); B=np.einsum("ia,nij,jb->nab",J[:,:d],H,J[:,d:],optimize=True); E=B**2; s=E.sum((1,2))
        Eh=E/(s[:,None,None]+1e-300); return float(np.sum(s*np.mean((Eh+1e-8)**0.5,axis=(1,2)))/s.sum())
    FA=lambda Q:bench.rowsp(*g.blocks("logit",Q),**FROZEN)
    Hb=special_ortho_group.rvs(d,size=120,random_state=7)
    sdv=lambda f:np.std([f(Q) for Q in Hb])+1e-12
    sA,s1,s2=sdv(FA),sdv(F1),sdv(F2)
    metrics={"F1 per-axis rank-1 X-coupling":F1,"F2 per-point L1/L2 sparsity":F2,
             "A+F1 (equal wts)":lambda Q:FA(Q)/sA+F1(Q)/s1,"A+F2 (equal wts)":lambda Q:FA(Q)/sA+F2(Q)/s2}
    QS={"ceiling (supervised R*)":np.eye(d)}; print("  ceiling (supervised R*)        :",full(g,np.eye(d)),flush=True)
    Hh=special_ortho_group.rvs(d,size=600,random_state=4); v=np.array([FA(Q) for Q in Hh]); mins=[]
    for i in np.argsort(v)[:40]:
        val,Q=g.opt(FA,Hh[i],maxiter=1500)
        if not any(g.err(Q@m[1].T)<3 for m in mins): mins.append([val,Q])
    mins.sort(key=lambda r:r[0]); QS["A rowsp (reference)"]=mins[0][1]; print("  A rowsp (reference)            :",full(g,mins[0][1]),flush=True)
    for name,f in metrics.items():
        _,Q=g.search(f,nh=300,nstart=6); QS[name]=Q
        v0=f(np.eye(d)); vt,Qt=g.opt(f,np.eye(d),maxiter=900)
        print(f"  {name:31s}:",full(g,Q),flush=True)
        print(f"      at R*: f={v0:.4f}; local min from R*: f={vt:.4f} -> MCC {g.mcc(Qt)[0]:.3f} err {g.err(Qt):.1f}deg",flush=True)
    names=["ceiling (supervised R*)","A rowsp (reference)"]+list(metrics)
    fig,axs=plt.subplots(2,3,figsize=(15,9.5))
    for ax,name in zip(axs.ravel(),names):
        Q=QS[name]; U=g.Zs@(Q@g.base.R_star).T; Cc=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-Cc)
        order=[r[list(c).index(j)] for j in range(d)]; M=Cc[order]; m=M.diagonal(); off=(M.sum()-m.sum())/(d*d-d)
        im=ax.imshow(M,vmin=0,vmax=1,cmap="viridis")
        for i in range(d):
            for j in range(d): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="w" if M[i,j]<0.6 else "k",fontsize=9)
        ax.set_xticks(range(d)); ax.set_xticklabels([f"T{j+1}" for j in range(d)]); ax.set_yticks(range(d)); ax.set_yticklabels([f"U{order[i]+1}" for i in range(d)])
        ax.set_title(f"{name}"+chr(10)+f"MCC {m.mean():.3f}  leak mean {off:.3f}  max {np.max(M-np.diag(m)):.3f}",fontsize=10)
    fig.suptitle(f"{model}: |corr(U,T)| new metrics (rows = U reordered to match T)",fontsize=12)
    fig.colorbar(im,ax=axs,shrink=0.7,label="|corr|"); fig.savefig(f"heatN_{model}.png",dpi=110,bbox_inches="tight"); plt.close(fig)
    print(f"HEAT {model}  [{time.time()-t0:.0f}s]",flush=True)
