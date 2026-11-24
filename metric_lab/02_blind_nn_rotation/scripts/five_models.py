from lib2 import *
import bench, time
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
QS={}
from bench import neff,udshare
FROZEN=dict(a=0.5,q=0.1,eps=1e-4)
def full(g,Q):
    d=g.d; U=g.Zs@(Q@g.base.R_star).T; C=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-C)
    m=C[r,c]; mask=np.ones_like(C,bool); mask[r,c]=False; off=C[mask]
    return f"MCC {m.mean():.3f} (min {m.min():.3f}) | leak mean {off.mean():.3f} max {off.max():.3f} | err {g.err(Q):.1f}deg"
for model in sys.argv[1].split(","):
    t0=time.time(); g=G(model,seed=2); d=g.d; print(f"\n## {model} d={d} p={g.p}",flush=True)
    QS={"ceiling (supervised R*)":np.eye(d)}
    print("  ceiling (supervised R*)      :",full(g,np.eye(d)),flush=True)
    # A/B: frozen rowsp, logit, stdX
    f=lambda Q:bench.rowsp(*g.blocks("logit",Q),**FROZEN)
    Hh=special_ortho_group.rvs(d,size=600,random_state=4); v=np.array([f(Q) for Q in Hh]); mins=[]
    for i in np.argsort(v)[:40]:
        val,Q=g.opt(f,Hh[i],maxiter=1500)
        if not any(g.err(Q@m[1].T)<3 for m in mins): mins.append([val,Q])
    mins.sort(key=lambda r:r[0]); QA=mins[0][1]; QB=min(mins,key=lambda m:udshare(*g.blocks("logit",m[1])))[1]
    QS["A rowsp(a=.5) pure"]=QA; QS["B rowsp->min UUd pick"]=QB
    print("  A rowsp(a=.5) pure           :",full(g,QA),f"basins={len(mins)}",flush=True)
    print("  B rowsp -> min UUd pick      :",full(g,QB),flush=True)
    # C: EXP13 structure on logit, std X ;  D: composite with rowsp ; E: EXP13 p rawX (project baseline)
    Hb=special_ortho_group.rvs(d,size=120,random_state=7)
    Bl=[g.blocks("logit",Q) for Q in Hb]; Bp=[g.blocks("probability",Q,False) for Q in Hb]
    sd=lambda fn,B:np.std([fn(*b) for b in B])+1e-12
    s_n=sd(lambda UU,UX:neff(UX),Bl); s_u=sd(udshare,Bl); s_r=sd(lambda UU,UX:bench.rowsp(UU,UX,**FROZEN),Bl)
    fC=lambda Q:(lambda b:neff(b[1])/s_n+udshare(*b)/s_u)(g.blocks("logit",Q))
    fD=lambda Q:(lambda b:neff(b[1])/s_n+udshare(*b)/s_u+bench.rowsp(*b,**FROZEN)/s_r)(g.blocks("logit",Q))
    _,QC=g.search(fC,nh=300,nstart=6); _,QD=g.search(fD,nh=300,nstart=6)
    QS["C EXP13-form, logit, std X"]=QC; QS["D C + rowsp"]=QD
    print("  C EXP13-form on logit, std X :",full(g,QC),flush=True)
    print("  D C + rowsp (equal weights)  :",full(g,QD),flush=True)
    s1=sd(lambda UU,UX:neff(UX),Bp); s2=sd(udshare,Bp)
    fE=lambda Q:(lambda b:neff(b[1])/s1+udshare(*b)/s2)(g.blocks("probability",Q,False))
    _,QE=g.search(fE,nh=300,nstart=6)
    print("  E EXP13 baseline (p, raw X)  :",full(g,QE),f"  [{time.time()-t0:.0f}s]",flush=True)

    QS["E EXP13 baseline (p, raw X)"]=QE
    fig,axs=plt.subplots(2,3,figsize=(15,9.5)); 
    for ax,(name,Q) in zip(axs.ravel(),QS.items()):
        U=g.Zs@(Q@g.base.R_star).T; C=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-C)
        order=[r[list(c).index(j)] for j in range(d)]; M=C[order]
        m=M.diagonal(); off=(M.sum()-m.sum())/(d*d-d)
        im=ax.imshow(M,vmin=0,vmax=1,cmap="viridis")
        for i in range(d):
            for j in range(d): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="w" if M[i,j]<0.6 else "k",fontsize=9)
        ax.set_xticks(range(d)); ax.set_xticklabels([f"T{j+1}" for j in range(d)]); ax.set_yticks(range(d)); ax.set_yticklabels([f"U{order[i]+1}" for i in range(d)])
        ax.set_title(f"{name}"+chr(10)+f"MCC {m.mean():.3f}  leak mean {off:.3f}  max {np.max(M-np.diag(m)):.3f}",fontsize=10)
    fig.suptitle(f"{model}: |corr(U,T)| after rotation (rows = U axes reordered to match T; diagonal = matched)",fontsize=12)
    fig.colorbar(im,ax=axs,shrink=0.7,label="|corr|"); fig.savefig(f"heat_{model}.png",dpi=110,bbox_inches="tight"); plt.close(fig)
    print(f"HEAT {model}",flush=True)
