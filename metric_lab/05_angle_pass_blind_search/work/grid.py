import os,sys,time,csv
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[k]="1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from scipy.stats import special_ortho_group
from scipy.optimize import minimize, linear_sum_assignment
from scipy.linalg import expm
import metrics as MET
HERE=os.path.dirname(os.path.abspath(__file__))
INF=["i01_logit_AD","i02_prob_AD","i03_logit_gaussZ","i04_logit_FD005","i05_logit_FD02","i06_logit_FD05","i07_poly3","i08_poly4","i09_krr","i10_smooth01","i11_smooth03","i12_a1raw","i13_logp","i14_asin","i15_fisherW","i16_interior"]
_ST={}; _M=None; _HAAR={}
def haar(d,n,seed):
    k=(d,n,seed)
    if k not in _HAAR: _HAAR[k]=special_ortho_group.rvs(d,size=n,random_state=seed)
    return _HAAR[k]
def state(inf,model):
    global _M
    if _M is None: _M=MET.build()
    k=(inf,model)
    if k in _ST: return _ST[k]
    _ST.clear()
    mt=np.load(f"{HERE}/meta_{model}.npz"); d=int(mt["d"]); p=int(mt["p"]); D=d+p
    C=np.load(f"{HERE}/moms/{inf}_{model}.npz")["C"]; C4=C.reshape(D,D,D,D)
    st=dict(d=d,p=p,xs2=mt["xs"]**2,zstd=mt["zstd"],Zs=mt["Zs"],Ts=mt["Ts"],R=mt["R"],Cuu=C4[:d,:d,:d,:d].copy(),Cb=np.stack([C4[:d,d+b,:d,d+b] for b in range(p)]))
    comp=_M["_comp"]; Hz=haar(d,200,4); vals={c:[] for c in comp}
    for Q in Hz:
        UX,UU=uxuu(st,Q)
        for c,f in comp.items(): vals[c].append(f(UX,UU))
    st["null"]={c:(float(np.median(v)),float(np.std(v))+1e-12) for c,v in vals.items()}
    _ST[k]=st; return st
def uxuu(st,Q):
    Jz=st["zstd"][:,None]*Q.T
    UX=np.einsum("ia,bik,ka->ab",Jz,st["Cb"],Jz)*st["xs2"][None,:]
    W=np.einsum("ijkl,ia,ka->ajl",st["Cuu"],Jz,Jz); UU=np.einsum("ajl,jb,lb->ab",W,Jz,Jz)
    return np.maximum(UX,0),np.maximum(UU,0)
def skew(x,d):
    A=np.zeros((d,d)); A[np.triu_indices(d,1)]=x; return A-A.T
def errdeg(Q,R):
    d=Q.shape[0]; Qr=Q@R.T; r,c=linear_sum_assignment(-np.abs(Qr)); P=np.zeros((d,d)); P[r,c]=np.sign(Qr[r,c]); A=Qr@P.T
    if np.linalg.det(A)<0: A=A@np.diag([-1]+[1]*(d-1))
    return float(np.degrees(np.max(np.abs(np.angle(np.linalg.eigvals(A))))))
def evalQ(st,Q):
    d=st["d"]; U=st["Zs"]@Q.T; Cc=np.abs(np.corrcoef(np.c_[U,st["Ts"]].T)[:d,d:]); r,c=linear_sum_assignment(-Cc)
    m=Cc[r,c]; mask=np.ones_like(Cc,bool); mask[r,c]=False; off=Cc[mask]
    return float(m.mean()),float(m.min()),float(off.mean()),float(off.max())
def search(st,f,nh=400,nst=8,maxiter=800):
    d=st["d"]; Hs=haar(d,nh,4); v=np.array([f(Q) for Q in Hs]); best=(1e18,None); npar=d*(d-1)//2; mi=maxiter*(1 if d<=4 else 3)
    for i in np.argsort(v)[:nst]:
        Q0=Hs[i]; r=minimize(lambda x:f(expm(skew(x,d))@Q0),np.zeros(npar),method="Nelder-Mead",options=dict(maxiter=mi,xatol=1e-4,fatol=1e-14))
        if r.fun<best[0]: best=(r.fun,expm(skew(r.x,d))@Q0)
    return best
def run(task):
    inf,model,name=task; st=state(inf,model); fn=_M[name]
    def f(Q):
        UX,UU=uxuu(st,Q)
        try: return fn(UX,UU,st)
        except Exception: return 1e9
    v,Q=search(st,f); mcc,mn,lm,lx=evalQ(st,Q)
    return (inf,model,name,mcc,mn,lm,lx,errdeg(Q,st["R"]),v)
def ceilings(models=("C1","C2","C3","CR4")):
    out={}
    for m in models:
        mt=np.load(f"{HERE}/meta_{m}.npz"); st=dict(d=int(mt["d"]),Zs=mt["Zs"],Ts=mt["Ts"]); out[m]=evalQ(st,mt["R"])
    return out
if __name__=="__main__":
    import multiprocessing as mp
    M=MET.build(); names=[k for k in M if not k.startswith("_")]
    if len(sys.argv)>1 and sys.argv[1]=="test":
        for t in [("i01_logit_AD","CR4","rowsp.a0.5.q0.1"),("i01_logit_AD","CR4","Fz.A+B+C"),("i01_logit_AD","C2","UX.hoy"),("i01_logit_AD","C2","Fz.A+B+C")]:
            t0=time.time(); r=run(t); print(r[:3],"MCC %.3f leak %.3f/%.3f err %.1f  [%.1fs]"%(r[3],r[5],r[6],r[7],time.time()-t0),flush=True)
        sys.exit()
    infs=INF if len(sys.argv)<2 else sys.argv[1].split(","); models=["CR4","C2","C3","C1"] if len(sys.argv)<3 else sys.argv[2].split(","); nproc=int(sys.argv[3]) if len(sys.argv)>3 else 14
    tasks=[(i,m,n) for m in models for i in infs for n in names]; out=os.environ.get("OUT",f"{HERE}/grid_results.csv"); done=set()
    if os.path.exists(out):
        with open(out) as fh: done={tuple(r[:3]) for r in csv.reader(fh)}
    tasks=[t for t in tasks if t not in done]; print(f"{len(tasks)} tasks ({len(names)} metrics x {len(infs)} inferences x {len(models)} models), {nproc} workers",flush=True)
    t0=time.time(); n=0
    with mp.Pool(nproc) as pool, open(out,"a",newline="") as fh:
        w=csv.writer(fh)
        for r in pool.imap_unordered(run,tasks,chunksize=4):
            w.writerow(r); n+=1
            if n%100==0: fh.flush(); print(f"{n}/{len(tasks)} [{time.time()-t0:.0f}s]",flush=True)
    print("ALLDONE",time.time()-t0)
