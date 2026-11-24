# V1 = "UX-sparsity valley, then UU tie-break". PRE-REGISTERED, T used only for printed evaluation.
#  stage 1: minimise rowsp (a=.5,q=.1,eps=1e-4 [a is the earlier T-picked value, flagged]) -> v0, Q0
#  tolerance tau = 2 x bootstrap SD of rowsp(Q0) over B=10 bootstrap resamples of the evaluation points (T-free noise level)
#  stage 2: among rotations with rowsp <= v0+tau, minimise UU self-fraction (trace(UU)/sum(UU)) [EXP13's UUd_share] via penalty
from lib2 import *
import bench, time
from bench import udshare
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A_,Q_,EPS=0.5,0.1,1e-4; B=10
for model in sys.argv[1].split(","):
    t0=time.time(); g=G(model,seed=2); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std)
    pts,H,Gr=per_point(g,N=4096,seed=2,target="logit"); N=len(H); Hf=np.asarray(H).reshape(N,D*D); Cfull=g.mom["logit"]
    rng=np.random.default_rng(321); Cb=[]
    for b in range(B):
        h=Hf[rng.integers(0,N,N)]; Cb.append(h.T@h/N)
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def blk(C,Q): m=transform_moment(C,J2(Q)); return m[:d,:d],m[:d,d:]
    def ux_fast(C,Q):
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.einsum("ai,ij,aj->a",bs,C,bs,optimize=True).reshape(d,p)
    def strong(f,nh=600,nst=20,seed=4,maxiter=1500):
        Hs=special_ortho_group.rvs(d,size=nh,random_state=seed); v=np.array([f(Q) for Q in Hs]); best=(1e18,None)
        for i in np.argsort(v)[:nst]:
            val,Q=g.opt(f,Hs[i],maxiter=maxiter)
            if val<best[0]: best=(val,Q)
        return best
    def evalQ(Q):
        U=g.Zs@Q.T; Cc=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-Cc)
        m=Cc[r,c]; mask=np.ones_like(Cc,bool); mask[r,c]=False; off=Cc[mask]
        return dict(mcc=m.mean(),mn=m.min(),lm=off.mean(),lx=off.max(),err=g.err(Q@Rorig.T),M=Cc[[r[list(c).index(j)] for j in range(d)]])
    fmt=lambda e:f"MCC {e['mcc']:.3f} (min {e['mn']:.3f}) | leak mean {e['lm']:.3f} max {e['lx']:.3f} | err {e['err']:.1f}deg"
    ceil=evalQ(Rorig); print(f"\n## {model} d={d} p={p}   ceiling: {fmt(ceil)}",flush=True)
    rs=lambda C,Q:bench.rowsp(None,ux_fast(C,Q),a=A_,q=Q_,eps=EPS); f=lambda Q:rs(Cfull,Q)
    Hs=special_ortho_group.rvs(d,size=600,random_state=4); sH=float(np.std([f(Q) for Q in Hs[:200]]))
    v0,Q0=strong(f); eA=evalQ(Q0); tau=2*float(np.std([rs(C,Q0) for C in Cb]))
    print(f"  stage1 rowsp min: f={v0:.4f}  {fmt(eA)}",flush=True); print(f"  tau={tau:.5f}  (Haar SD {sH:.4f}; tau/SD={tau/sH:.3f})",flush=True)
    ud=lambda Q:udshare(*blk(Cfull,Q)); su=float(np.std([ud(Q) for Q in Hs[:200]]))
    pen=lambda Q:ud(Q)/su+1e3*max(0.0,(f(Q)-v0-tau)/sH)**2
    vp=np.array([pen(Q) for Q in Hs]); best=(1e18,None)
    for i in np.argsort(vp)[:20]:
        val,Q=g.opt(pen,Hs[i],maxiter=1500)
        if val<best[0]: best=(val,Q)
    Qv=best[1]; eV=evalQ(Qv); feas=f(Qv)<=v0+tau+1e-9
    print(f"  V1 (valley + UU tie-break): f_rowsp={f(Qv):.4f} feasible={feas} udshare={ud(Qv):.4f} (at Q0 {ud(Q0):.4f}) -> {fmt(eV)} | moved {g.err(Qv@Q0.T):.1f}deg from rowsp-min",flush=True)
    eT=evalQ(Rorig); print(f"  [diag] at supervised R*: rowsp={f(Rorig):.4f} (v0+tau={v0+tau:.4f}) udshare={ud(Rorig):.4f}",flush=True)
    fig,axs=plt.subplots(1,3,figsize=(15,5)); 
    for ax,(name,e) in zip(axs,[("ceiling (supervised R*)",ceil),("stage 1: rowsp a=.5 min",eA),("V1: valley + UU tie-break",eV)]):
        M=e['M']; im=ax.imshow(M,vmin=0,vmax=1,cmap="viridis")
        for i in range(d):
            for j in range(d): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="w" if M[i,j]<0.6 else "k",fontsize=9)
        ax.set_xticks(range(d)); ax.set_xticklabels([f"T{j+1}" for j in range(d)]); ax.set_yticks(range(d)); ax.set_yticklabels([f"U{i+1}" for i in range(d)])
        ax.set_title(f"{name}"+chr(10)+f"MCC {e['mcc']:.3f}  leak mean {e['lm']:.3f}  max {e['lx']:.3f}",fontsize=10)
    fig.suptitle(f"{model}: V1 valley + UU tie-break |corr(U,T)|",fontsize=12); fig.colorbar(im,ax=axs,shrink=0.8,label="|corr|")
    fig.savefig(f"heatV_{model}.png",dpi=110,bbox_inches="tight"); plt.close(fig); print(f"HEAT {model}  [{time.time()-t0:.0f}s]",flush=True)
