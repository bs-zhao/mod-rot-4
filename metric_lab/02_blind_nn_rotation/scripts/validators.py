# Which internal (T-free) validator separates the right candidate from the wrong one? T only labels good/bad.
from lib2 import *
import bench, time
from bench import udshare
def kmeans(X,K=3,seed=0,it=30):
    r=np.random.default_rng(seed); c=X[r.choice(len(X),K,replace=False)]
    for _ in range(it):
        l=np.argmin(((X[:,None]-c[None])**2).sum(-1),1); c=np.array([X[l==k].mean(0) if (l==k).any() else c[k] for k in range(K)])
    return l
for model in sys.argv[1].split(","):
    t0=time.time(); g=G(model,seed=2); d,p=g.d,g.p; D=d+p; Rorig=np.asarray(g.base.R_star); zstd=np.asarray(g.base.z_std); Cfull=g.mom["logit"]
    pts,H,Gr=per_point(g,N=4096,seed=2,target="logit"); N=len(H); H=np.asarray(H); Hf=H.reshape(N,D*D)
    lab=kmeans(pts[:,d:]/g.xs,3); Creg=[Hf[lab==k].T@Hf[lab==k]/max(1,(lab==k).sum()) for k in range(3)]
    def J2(Q): J=np.eye(D); J[:d,:d]=np.diag(zstd)@Q.T; J[d:,d:]=np.diag(g.xs); return J
    def blk(C,Q): m=transform_moment(C,J2(Q)); return m[:d,:d],m[:d,d:]
    def ux_fast(C,Q):
        J=J2(Q); bs=np.einsum("ia,jb->abij",J[:,:d],J[:,d:],optimize=True).reshape(d*p,D*D)
        return np.einsum("ai,ij,aj->a",bs,C,bs,optimize=True).reshape(d,p)
    def F2(Q):
        J=J2(Q); B=np.einsum("ia,nij,jb->nab",J[:,:d],H,J[:,d:],optimize=True); E=B**2; s=E.sum((1,2)); Eh=E/(s[:,None,None]+1e-300)
        return float(np.sum(s*np.mean((Eh+1e-8)**0.5,axis=(1,2)))/s.sum())
    def selftail(Q): UU=blk(Cfull,Q)[0]; s=np.sort(np.diag(UU))[::-1]; return float(s[1:].sum()/UU.sum())
    hoy=lambda U:(lambda v:float(1-(np.sqrt(v.size)-np.abs(v).sum()/np.sqrt((v**2).sum()))/(np.sqrt(v.size)-1)))(U.ravel())
    mets={"A":lambda U:bench.rowsp(None,U,a=0.5,q=0.1,eps=1e-4),"B":hoy}
    def strong(f,nh=600,nst=20,seed=4,maxiter=1500):
        Hs=special_ortho_group.rvs(d,size=nh,random_state=seed); v=np.array([f(Q) for Q in Hs]); best=(1e18,None)
        for i in np.argsort(v)[:nst]:
            val,Q=g.opt(f,Hs[i],maxiter=maxiter)
            if val<best[0]: best=(val,Q)
        return best
    def mcc(Q):
        U=g.Zs@Q.T; Cc=np.abs(np.corrcoef(np.c_[U,g.Ts].T)[:d,d:]); r,c=linear_sum_assignment(-Cc); return float(Cc[r,c].mean())
    ceil=mcc(Rorig); Hs=special_ortho_group.rvs(d,size=200,random_state=4)
    base={k:(float(np.median([m(ux_fast(Cfull,Q)) for Q in Hs])),float(np.std([m(ux_fast(Cfull,Q)) for Q in Hs]))) for k,m in mets.items()}
    med={"F2":float(np.median([F2(Q) for Q in Hs[:80]])),"self":float(np.median([selftail(Q) for Q in Hs])),"ud":float(np.median([udshare(*blk(Cfull,Q)) for Q in Hs]))}
    cand={}; mins={}
    for k,m in mets.items():
        v,Q=strong(lambda Q,m=m:m(ux_fast(Cfull,Q))); cand[k]=Q; mins[k]=v
    print(f"\n## {model} d={d} p={p} ceiling MCC {ceil:.3f}",flush=True)
    for k,Q in cand.items():
        other="B" if k=="A" else "A"; me=mets[k]; mo=mets[other]
        regret=(mo(ux_fast(Cfull,Q))-mins[other])/base[other][1]
        drifts=[]
        for C in Creg:
            v,Qr=g.opt(lambda Q,C=C:me(ux_fast(C,Q)),Q,maxiter=600); drifts.append(g.err(Qr@Q.T))
        print(f"  cand {k}({'rowsp' if k=='A' else 'UX 1-Hoyer'}): MCC {mcc(Q):.3f} {'GOOD' if mcc(Q)>=ceil-0.05 else 'bad '} | cross-metric regret {regret:6.2f} SD | region drift {np.mean(drifts):5.1f}deg | self-tail/Haar {selftail(Q)/med['self']:.2f} | udshare/Haar {udshare(*blk(Cfull,Q))/med['ud']:.2f} | F2/Haar {F2(Q)/med['F2']:.2f}",flush=True)
    print(f"VAL {model} [{time.time()-t0:.0f}s]",flush=True)
