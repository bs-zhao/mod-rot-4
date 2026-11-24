# SELECTOR RULE (pre-registered BEFORE this run, after seeing validators on C2,CR4,C3): of the two candidate rotations
#   A = argmin rowsp(a=.5,q=.1,eps=1e-4)   B = argmin global UX 1-Hoyer   (logit, std X, R*-free coordinates)
# pick the one with the LOWER  udshare(Q)/median_Haar(udshare)  (UU self-fraction = trace(UU)/sum(UU)).  T only for printed evaluation.
from lib2 import *
import bench, time
from bench import udshare
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
specs=[s.split(":") for s in sys.argv[1].split(",")]   # model:repeat:seed
for model,rep,sd in specs:
    rep=int(rep); sd=int(sd); t0=time.time(); g=G(model,seed=sd,repeat=rep); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std); C=g.mom["logit"]
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def blk(Q): m=transform_moment(C,J2(Q)); return m[:d,:d],m[:d,d:]
    def ux(Q):
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.einsum("ai,ij,aj->a",bs,C,bs,optimize=True).reshape(d,p)
    hoy=lambda U:(lambda v:float(1-(np.sqrt(v.size)-np.abs(v).sum()/np.sqrt((v**2).sum()))/(np.sqrt(v.size)-1)))(U.ravel())
    mets={"A":lambda U:bench.rowsp(None,U,a=0.5,q=0.1,eps=1e-4),"B":hoy}
    def strong(f,nh=600,nst=20,seed=4,maxiter=1500):
        Hs=special_ortho_group.rvs(d,size=nh,random_state=seed); v=np.array([f(Q) for Q in Hs]); best=(1e18,None)
        for i in np.argsort(v)[:nst]:
            val,Q=g.opt(f,Hs[i],maxiter=maxiter)
            if val<best[0]: best=(val,Q)
        return best
    def ev(Q):
        U=g.Zs@Q.T; Cc=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-Cc)
        m=Cc[r,c]; mask=np.ones_like(Cc,bool); mask[r,c]=False; off=Cc[mask]
        return dict(mcc=float(m.mean()),lm=float(off.mean()),lx=float(off.max()),M=Cc[[r[list(c).index(j)] for j in range(d)]])
    Hs=special_ortho_group.rvs(d,size=200,random_state=4); medud=float(np.median([udshare(*blk(Q)) for Q in Hs]))
    cand={k:strong(lambda Q,m=m:m(ux(Q)))[1] for k,m in mets.items()}
    r={k:udshare(*blk(Q))/medud for k,Q in cand.items()}; pick="A" if r["A"]<=r["B"] else "B"; E={k:ev(Q) for k,Q in cand.items()}; ec=ev(Rorig)
    f=lambda e:f"MCC {e['mcc']:.3f} leak {e['lm']:.3f}/{e['lx']:.3f}"
    print(f"SEL {model} rep{rep} seed{sd}: ceiling {f(ec)} | A rowsp {f(E['A'])} ud/H {r['A']:.3f} | B UX-Hoyer {f(E['B'])} ud/H {r['B']:.3f} | PICK {pick} -> {f(E[pick])} | best-of-two would be {max(E['A']['mcc'],E['B']['mcc']):.3f} [{time.time()-t0:.0f}s]",flush=True)
    if rep==1 and sd==2:
        fig,axs=plt.subplots(1,4,figsize=(20,5))
        for ax,(name,e) in zip(axs,[("ceiling (supervised R*)",ec),("A rowsp a=.5",E["A"]),("B global UX 1-Hoyer",E["B"]),(f"SELECTED = {pick}",E[pick])]):
            M=e['M']; im=ax.imshow(M,vmin=0,vmax=1,cmap="viridis")
            for i in range(d):
                for j in range(d): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="w" if M[i,j]<0.6 else "k",fontsize=8)
            ax.set_xticks(range(d)); ax.set_xticklabels([f"T{j+1}" for j in range(d)]); ax.set_yticks(range(d)); ax.set_yticklabels([f"U{i+1}" for i in range(d)])
            ax.set_title(f"{name}"+chr(10)+f"MCC {e['mcc']:.3f} leak {e['lm']:.3f}/{e['lx']:.3f}",fontsize=9)
        fig.suptitle(f"{model}: two-candidate selector (lower UU self-fraction vs Haar)",fontsize=12); fig.colorbar(im,ax=axs,shrink=0.8)
        fig.savefig(f"heatS2_{model}.png",dpi=100,bbox_inches="tight"); plt.close(fig); print(f"HEAT {model}",flush=True)
