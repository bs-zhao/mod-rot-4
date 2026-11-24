# 16 inference variants -> second-moment tensor C = E_w[vec(H) vec(H)^T] (D^2 x D^2) on the SAME evaluation points. No retraining; only Z, X, decoder outputs.
# usage: infer.py MODELS [TAGS]
from lib2 import *
import itertools, time, os
SEL=sys.argv[2].split(",") if len(sys.argv)>2 else None
REP=int(os.environ.get('REP','1')); SUF='' if REP==1 else f'r{REP}'
def mk_fn(dec,k,kind):
    def fn(v):
        z,x=v[:k],v[k:]; lg=dec.network(torch.cat(((x-dec.x_mean)/dec.x_std,z))); eta=lg[1]-lg[0]
        if kind=="logit": return eta
        if kind=="prob": return torch.sigmoid(eta)
        if kind=="a1": return lg[1]
        if kind=="logp": return torch.nn.functional.logsigmoid(eta)
        if kind=="asin": return torch.asin(torch.sqrt(torch.sigmoid(eta).clamp(1e-9,1-1e-9)))
        raise ValueError(kind)
    return fn
def hess_at(fn,P,chunk=512):
    out=[]; hv=torch.func.vmap(torch.func.hessian(fn))
    for s in range(0,len(P),chunk): out.append(hv(torch.as_tensor(P[s:s+chunk],dtype=torch.float64)).detach().numpy())
    return np.concatenate(out)
def moment(H,w=None):
    Hf=H.reshape(len(H),-1)
    if w is None: return Hf.T@Hf/len(H)
    w=np.asarray(w); return (Hf.T*w)@Hf/w.sum()
def sample_pts(g,N,seed):
    rng=np.random.default_rng(seed); src=g.src; lat=src.z[rng.choice(len(src.z),N,replace=True)]
    return np.column_stack((lat,np.asarray(get_spec(g.model).sample_inputs(N,rng),float)))
for model in sys.argv[1].split(","):
    t00=time.time(); g=G(model,seed=2,repeat=REP); d,p=g.d,g.p; D=d+p; dec=Decoder(g.src)
    P=sample_pts(g,4096,2)   # same evaluation points as the cached AD moment
    sd=P.std(0); mu=P.mean(0)
    flogit=mk_fn(dec,d,"logit"); 
    def save(tag,C,info=""): np.savez(f"moms/{tag}_{model}{SUF}.npz",C=C,info=np.array(info)); print(f"  [{model}] {tag} saved {info} ({time.time()-t00:.0f}s)",flush=True)
    want=lambda t:(SEL is None or t in SEL)
    if want("i01_logit_AD"): save("i01_logit_AD",g.mom["logit"])
    if want("i02_prob_AD"): save("i02_prob_AD",g.mom["probability"])
    if want("i03_logit_gaussZ"):
        gg=G(model,seed=2,lat_mode="gauss",repeat=REP); save("i03_logit_gaussZ",gg.mom["logit"])
    Hbase=None
    for tag,s in (("i04_logit_FD005",0.05),("i05_logit_FD02",0.2),("i06_logit_FD05",0.5)):
        if not want(tag): continue
        Pt=torch.as_tensor(P,dtype=torch.float64); fn=torch.vmap(lambda v:flogit(v)); f0=fn(Pt); E=torch.eye(D,dtype=torch.float64); h=torch.as_tensor(s*sd)
        Hf=torch.zeros(len(P),D,D,dtype=torch.float64)
        for i in range(D):
            Hf[:,i,i]=(fn(Pt+h[i]*E[i])-2*f0+fn(Pt-h[i]*E[i]))/h[i]**2
            for j in range(i+1,D):
                v=(fn(Pt+h[i]*E[i]+h[j]*E[j])-fn(Pt+h[i]*E[i]-h[j]*E[j])-fn(Pt-h[i]*E[i]+h[j]*E[j])+fn(Pt-h[i]*E[i]-h[j]*E[j]))/(4*h[i]*h[j]); Hf[:,i,j]=v; Hf[:,j,i]=v
        save(tag,moment(Hf.numpy()),f"central differences, step {s} sd")
    # poly surrogates
    for tag,deg in (("i07_poly3",3),("i08_poly4",4)):
        if not want(tag): continue
        Ntr=20000; ptr=sample_pts(g,Ntr,11); m_=ptr.mean(0); s_=ptr.std(0)
        with torch.no_grad(): y=torch.vmap(flogit)(torch.as_tensor(ptr,dtype=torch.float64)).numpy()
        idx=[c for k in range(1,deg+1) for c in itertools.combinations_with_replacement(range(D),k)]; I=np.full((len(idx)+1,deg),D,dtype=np.int64)
        for r,c in enumerate(idx,1): I[r,:len(c)]=c
        It=torch.as_tensor(I)
        def phi(ws): we=torch.cat([ws,torch.ones_like(ws[...,:1])],-1); return we[...,It].prod(-1)
        Phi=phi(torch.as_tensor((ptr-m_)/s_)).numpy(); lam=1e-6*np.trace(Phi.T@Phi)/Phi.shape[1]; ntr=int(0.8*Ntr)
        th=np.linalg.solve(Phi[:ntr].T@Phi[:ntr]+lam*np.eye(Phi.shape[1]),Phi[:ntr].T@y[:ntr]); r2=1-((Phi[ntr:]@th-y[ntr:])**2).sum()/((y[ntr:]-y[ntr:].mean())**2).sum()
        theta=np.linalg.solve(Phi.T@Phi+lam*np.eye(Phi.shape[1]),Phi.T@y); T_=torch.as_tensor(theta); mt=torch.as_tensor(m_); st=torch.as_tensor(s_)
        fp=lambda w:(phi((w-mt)/st)*T_).sum(-1)
        save(tag,moment(hess_at(fp,P,256)),f"poly{deg} {len(idx)+1} features, held-out R2 {r2:.4f}")
    if want("i09_krr"):
        Nc=3000; ptr=sample_pts(g,Nc+1000,13); m_=ptr.mean(0); s_=ptr.std(0); Wt=(ptr-m_)/s_
        with torch.no_grad(): y=torch.vmap(flogit)(torch.as_tensor(ptr,dtype=torch.float64)).numpy()
        Cc=Wt[:Nc]; sub=Wt[:600]; ell=float(np.median(np.sqrt(((sub[:,None]-sub[None])**2).sum(-1))))
        Kmat=np.exp(-((Cc[:,None]-Cc[None])**2).sum(-1)/(2*ell**2)); ym=y[:Nc].mean(); alpha=np.linalg.solve(Kmat+1e-4*np.eye(Nc),y[:Nc]-ym)
        Kte=np.exp(-((Wt[Nc:,None]-Cc[None])**2).sum(-1)/(2*ell**2)); r2=1-((Kte@alpha+ym-y[Nc:])**2).sum()/((y[Nc:]-y[Nc:].mean())**2).sum()
        Ct=torch.as_tensor(Cc); At=torch.as_tensor(alpha); mt=torch.as_tensor(m_); st=torch.as_tensor(s_)
        fk=lambda w:(torch.exp(-((((w-mt)/st)[None]-Ct)**2).sum(-1)/(2*ell**2))*At).sum()+float(ym)
        save("i09_krr",moment(hess_at(fk,P,64)),f"RBF KRR 3000 centres, ell {ell:.2f}, held-out R2 {r2:.4f}")
    # smoothed decoder (average Hessian over Gaussian input perturbations, then square)
    for tag,s in (("i10_smooth01",0.1),("i11_smooth03",0.3)):
        if not want(tag): continue
        rng=np.random.default_rng(5); K=8; Hs=0
        for _ in range(K): Hs=Hs+hess_at(flogit,P+s*sd*rng.standard_normal(P.shape))
        save(tag,moment(Hs/K),f"mean Hessian over {K} Gaussian perturbations, sigma {s} sd")
    for tag,kind in (("i12_a1raw","a1"),("i13_logp","logp"),("i14_asin","asin")):
        if want(tag): save(tag,moment(hess_at(mk_fn(dec,d,kind),P)),f"channel {kind}")
    if want("i15_fisherW") or want("i16_interior"):
        Hl=hess_at(flogit,P)
        with torch.no_grad(): eta=torch.vmap(flogit)(torch.as_tensor(P,dtype=torch.float64)).numpy()
        pr=1/(1+np.exp(-eta))
        if want("i15_fisherW"): save("i15_fisherW",moment(Hl,pr*(1-pr)),"logit Hessian, weights p(1-p)")
        if want("i16_interior"):
            m=(pr>0.1)&(pr<0.9); save("i16_interior",moment(Hl[m]),f"logit Hessian, only 0.1<p<0.9 ({m.mean():.2f} of points)")
    print(f"## {model} done [{time.time()-t00:.0f}s]",flush=True)
