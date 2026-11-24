# T-FREE version of rowsp. PRE-REGISTERED (written before any result was seen):
#  * coordinates: U = Q @ Zs  (NO supervised R*; Zs = standardised Z).  T used ONLY for post-hoc evaluation printouts.
#  * hyperparameters (a,q) chosen per model WITHOUT T: grid a in {0,.25,.5,.75,1} x q in {.1,.5}, eps=1e-4 fixed,
#    selection rule = smallest "instability" = mean pairwise rotation distance (deg, signed-permutation invariant) between the
#    global minimisers found on B=4 bootstrap resamples of the evaluation points; ties (<1 deg) -> larger relative drop
#    (median Haar value - min)/median Haar value.
#  * logit channel, std X, kept as before (a-priori: natural pre-sigmoid output; scale invariance).
from lib2 import *
import bench, time, itertools
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EPS=1e-4; AG=[0,0.25,0.5,0.75,1.0]; QG=[0.1,0.5]; B=4
for model in sys.argv[1].split(","):
    t0=time.time(); g=G(model,seed=2); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std)
    pts,H,Gr=per_point(g,N=4096,seed=2,target="logit"); N=len(H); Hf=np.asarray(H).reshape(N,D*D)
    Cfull=g.mom["logit"]
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def blk(C,Q): m=transform_moment(C,J2(Q)); return m[:d,:d],m[:d,d:]
    def ux_fast(C,Q):   # only the d x p UX block of the transformed energy matrix (cheaper)
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.einsum("ai,ij,aj->a",bs,C,bs,optimize=True).reshape(d,p)
    def strong_search(f,nh=600,nst=20,seed=4,maxiter=1500):
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
    ceil=evalQ(Rorig); print(f"\n## {model} d={d} p={p}   ceiling (supervised): {fmt(ceil)}",flush=True)
    rng=np.random.default_rng(123); Cb=[]
    for b in range(B):
        idx=rng.integers(0,N,N); h=Hf[idx]; Cb.append(h.T@h/N)
    Hh=special_ortho_group.rvs(d,size=120,random_state=7)
    cfg=list(itertools.product(AG,QG)); res={}
    for (a,q) in cfg:
        mk=lambda C:(lambda Q:bench.rowsp(None,ux_fast(C,Q),a=a,q=q,eps=EPS))
        Qb=[strong_search(mk(C))[1] for C in Cb]
        inst=float(np.mean([g.err(Qb[i]@Qb[j].T) for i in range(B) for j in range(i+1,B)]))
        f=mk(Cfull); vh=np.array([f(Q) for Q in Hh]); v,Qf=strong_search(f); drop=float((np.median(vh)-v)/np.median(vh))
        res[(a,q)]=dict(inst=inst,drop=drop,Q=Qf,ev=evalQ(Qf))
        print(f"  a={a:<4} q={q:<3} instability {inst:5.1f}deg  rel.drop {drop:.3f}  | post-hoc: {fmt(res[(a,q)]['ev'])}",flush=True)
    mn=min(r['inst'] for r in res.values()); cand=[k for k,r in res.items() if r['inst']<=mn+1.0]; sel=max(cand,key=lambda k:res[k]['drop'])
    print(f"  SELECTED (T-free rule): a={sel[0]} q={sel[1]}  -> {fmt(res[sel]['ev'])}",flush=True)
    old=res[(0.5,0.1)]['ev']; print(f"  (old T-picked a=0.5,q=0.1 in the new R*-free frame: {fmt(old)})",flush=True)
    panels=[("ceiling (supervised R*)",ceil),(f"SELECTED a={sel[0]} q={sel[1]}",res[sel]['ev']),("old a=.5 q=.1",res[(0.5,0.1)]['ev']),("a=0 q=.1",res[(0,0.1)]['ev']),("a=1 q=.1",res[(1.0,0.1)]['ev']),("a=.5 q=.5",res[(0.5,0.5)]['ev'])]
    fig,axs=plt.subplots(2,3,figsize=(15,9.5))
    for ax,(name,e) in zip(axs.ravel(),panels):
        M=e['M']; im=ax.imshow(M,vmin=0,vmax=1,cmap="viridis")
        for i in range(d):
            for j in range(d): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="w" if M[i,j]<0.6 else "k",fontsize=9)
        ax.set_xticks(range(d)); ax.set_xticklabels([f"T{j+1}" for j in range(d)]); ax.set_yticks(range(d)); ax.set_yticklabels([f"U{i+1}" for i in range(d)])
        ax.set_title(f"{name}"+chr(10)+f"MCC {e['mcc']:.3f}  leak mean {e['lm']:.3f}  max {e['lx']:.3f}",fontsize=10)
    fig.suptitle(f"{model}: T-free rowsp (no R*, hyper-params by bootstrap stability) |corr(U,T)|",fontsize=12)
    fig.colorbar(im,ax=axs,shrink=0.7,label="|corr|"); fig.savefig(f"heatT_{model}.png",dpi=110,bbox_inches="tight"); plt.close(fig)
    print(f"HEAT {model}  [{time.time()-t0:.0f}s]",flush=True)
