# Inference x metric grid. usage: fuse2.py TAG MODELS   TAG in {prob,gauss,fd,poly}  (I1 = logit autograd was fuse.py)
# metrics identical to fuse.py: A rowsp(a=.5,q=.1,eps=1e-4), B global UX 1-Hoyer, F1=A+B, F2=zA+zB, F3=zA+zB+zC. T only for evaluation.
from lib2 import *
import bench, time, itertools
from bench import udshare
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TAG=sys.argv[1]
def sample_pts(g,N,seed):
    rng=np.random.default_rng(seed); src=g.src
    lat=src.z[rng.choice(len(src.z),N,replace=True)]; x=np.asarray(get_spec(g.model).sample_inputs(N,rng),float); return np.column_stack((lat,x))
def moment_from_H(H): N=len(H); Hf=np.asarray(H).reshape(N,-1); return Hf.T@Hf/N
def get_moment(g,tag):
    d,p=g.d,g.p; D=d+p; f=f"{HERE}/mom2_{tag}_{g.model}.npz"
    if os.path.exists(f): z=np.load(f); return z["C"],z["info"].item() if z["info"].shape==() else str(z["info"])
    dec=Decoder(g.src); info=""
    if tag=="prob": C=g.mom["probability"]
    elif tag=="fd":
        pts,Ha,_=per_point(g,N=4096,seed=2,target="logit"); P=torch.as_tensor(pts,dtype=torch.float64); sd=pts.std(0); h=1e-3*sd
        fn=torch.vmap(lambda v:dec.scalar(v,target="logit")); f0=fn(P); E=torch.eye(D,dtype=torch.float64); hh=torch.as_tensor(h)
        Hfd=torch.zeros(len(pts),D,D,dtype=torch.float64)
        for i in range(D):
            Hfd[:,i,i]=(fn(P+hh[i]*E[i])-2*f0+fn(P-hh[i]*E[i]))/hh[i]**2
            for j in range(i+1,D):
                v=(fn(P+hh[i]*E[i]+hh[j]*E[j])-fn(P+hh[i]*E[i]-hh[j]*E[j])-fn(P-hh[i]*E[i]+hh[j]*E[j])+fn(P-hh[i]*E[i]-hh[j]*E[j]))/(4*hh[i]*hh[j])
                Hfd[:,i,j]=v; Hfd[:,j,i]=v
        Hfd=Hfd.numpy(); C=moment_from_H(Hfd); Cad=moment_from_H(Ha)
        info=f"FD vs AD moment relative Frobenius error {np.linalg.norm(C-Cad)/np.linalg.norm(Cad):.4f}"
    elif tag=="poly":
        Ntr=20000; ptr=sample_pts(g,Ntr,seed=11); mu=ptr.mean(0); sd=ptr.std(0)
        with torch.no_grad(): y=torch.vmap(lambda v:dec.scalar(v,target="logit"))(torch.as_tensor(ptr,dtype=torch.float64)).numpy()
        idx=[()]+[c for k in (1,2,3) for c in itertools.combinations_with_replacement(range(D),k)]
        def phi(ws):  # ws (...,D) torch
            return torch.stack([torch.ones_like(ws[...,0]) if len(c)==0 else torch.stack([ws[...,i] for i in c],-1).prod(-1) for c in idx],-1)
        Wt=torch.as_tensor((ptr-mu)/sd); Phi=phi(Wt).numpy(); lam=1e-6*np.trace(Phi.T@Phi)/Phi.shape[1]
        theta=np.linalg.solve(Phi.T@Phi+lam*np.eye(Phi.shape[1]),Phi.T@y); ntr=int(0.8*Ntr)
        th=np.linalg.solve(Phi[:ntr].T@Phi[:ntr]+lam*np.eye(Phi.shape[1]),Phi[:ntr].T@y[:ntr]); r2=1-((Phi[ntr:]@th-y[ntr:])**2).sum()/((y[ntr:]-y[ntr:].mean())**2).sum()
        T_=torch.as_tensor(theta); mu_=torch.as_tensor(mu); sd_=torch.as_tensor(sd)
        fpoly=lambda w:(phi((w-mu_)/sd_)*T_).sum(-1)
        pts,_,_=per_point(g,N=16,seed=2); pe=sample_pts(g,4096,seed=2)  # same evaluation points as the autograd moment
        Hp=torch.func.vmap(torch.func.hessian(fpoly))(torch.as_tensor(pe,dtype=torch.float64)).numpy(); C=moment_from_H(Hp)
        info=f"poly3 features {len(idx)}, held-out R2 of logit {r2:.4f}"
    np.savez(f,C=C,info=np.array(info)); return C,info
for model in sys.argv[2].split(","):
    t0=time.time()
    if TAG=="gauss": g=G(model,seed=2,lat_mode="gauss"); C=g.mom["logit"]; info="Z ~ N(mu,Sigma) fitted"
    else:
        g=G(model,seed=2); C,info=get_moment(g,TAG)
    d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std)
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def ux(Q):
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.einsum("ai,ij,aj->a",bs,C,bs,optimize=True).reshape(d,p)
    def uu(Q): m=transform_moment(C,J2(Q)); return m[:d,:d]
    hoy=lambda U:(lambda v:float(1-(np.sqrt(v.size)-np.abs(v).sum()/np.sqrt((v**2).sum()))/(np.sqrt(v.size)-1)))(U.ravel())
    mA=lambda Q:bench.rowsp(None,np.maximum(ux(Q),0),a=0.5,q=0.1,eps=1e-4); mB=lambda Q:hoy(np.maximum(ux(Q),0)); mC=lambda Q:float(np.trace(uu(Q))/uu(Q).sum())
    Hs=special_ortho_group.rvs(d,size=200,random_state=4)
    nul={k:(np.median(v:=[m(Q) for Q in Hs]),np.std(v)+1e-12) for k,m in (("A",mA),("B",mB),("C",mC))}
    z=lambda k,m:(lambda Q:(m(Q)-nul[k][0])/nul[k][1]); zA,zB,zC=z("A",mA),z("B",mB),z("C",mC)
    mets={"A rowsp":mA,"B global UX 1-Hoyer":mB,"F1 = A+B":lambda Q:mA(Q)+mB(Q),"F2 = zA+zB":lambda Q:zA(Q)+zB(Q),"F3 = zA+zB+zC":lambda Q:zA(Q)+zB(Q)+zC(Q)}
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
    fig.savefig(f"heatI_{TAG}_{model}.png",dpi=105,bbox_inches="tight"); plt.close(fig); print(f"HEAT {TAG} {model} [{time.time()-t0:.0f}s]",flush=True)
