import numpy as np
EPS=1e-12
def _S(v): return float(v.sum())+1e-300
def ent(v): q=v/_S(v); return float(-(q*np.log(q+EPS)).sum())
def hoy(v):
    n=v.size; s2=float(np.sqrt((v*v).sum()))
    if n<2 or s2<=0: return 1.0
    return float(1-(np.sqrt(n)-v.sum()/s2)/(np.sqrt(n)-1))
def hhi(v): q=v/_S(v); return float(-(q*q).sum())
def gini(v):
    s=np.sort(v); n=v.size; return float(-((2*np.arange(1,n+1)-n-1)*s).sum()/(n*_S(v)))
def tail(v,k): s=np.sort(v)[::-1]; return float(s[k:].sum()/_S(v))
def lq(v,q): return float(((v/_S(v))**q).sum())
def logs(v,e): return float(np.log(v/_S(v)+e).sum())
def cover(v,c): s=np.sort(v)[::-1]/_S(v); return float((np.searchsorted(np.cumsum(s),c)+1)/v.size)
def rowsp(UX,a,q,eps=1e-4):
    s=UX.sum(1); sh=s/(s.sum()+1e-300); A=UX/(s[:,None]+1e-30); w=sh**a
    return float(np.sum(w*np.mean((A+eps)**q,axis=1))/(w.sum()+1e-300))
def triu(UU): return UU[np.triu_indices(UU.shape[0])]
def offtri(UU): return UU[np.triu_indices(UU.shape[0],1)]
def build():
    M={}
    def add(name,fn): M[name]=fn
    d_tail=lambda v:tail(v,max(1,int(round(np.sqrt(v.size)))))
    FULL={"ent":ent,"hoy":hoy,"hhi":hhi,"gini":gini,"tail1":lambda v:tail(v,1),"tailsqrt":d_tail,"lq.1":lambda v:lq(v,.1),"lq.25":lambda v:lq(v,.25),"lq.5":lambda v:lq(v,.5),"log1e-3":lambda v:logs(v,1e-3),"log1e-2":lambda v:logs(v,1e-2),"cov.9":lambda v:cover(v,.9)}
    for fn,f in FULL.items():
        add(f"UX.{fn}",lambda UX,UU,st,f=f:f(UX.ravel()))
        add(f"UUtri.{fn}",lambda UX,UU,st,f=f:f(triu(UU)))
        add(f"POOL.{fn}",lambda UX,UU,st,f=f:f(np.concatenate([triu(UU),UX.ravel()])))
    ROW={"ent":ent,"hoy":hoy,"tail1":lambda v:tail(v,1),"lq.25":lambda v:lq(v,.25),"log1e-2":lambda v:logs(v,1e-2)}
    def agg(Mx,f,a,axis):
        Mx=Mx if axis==0 else Mx.T   # rows are the vectors
        s=Mx.sum(1); w=(s/(s.sum()+1e-300))**a
        return float(sum(wi*f(r) for wi,r in zip(w,Mx))/(w.sum()+1e-300))
    for a in (0,0.5,1):
        for fn,f in ROW.items(): add(f"UXrow.a{a}.{fn}",lambda UX,UU,st,f=f,a=a:agg(UX,f,a,0))
    for a in (0,1):
        for fn,f in ROW.items(): add(f"UXcol.a{a}.{fn}",lambda UX,UU,st,f=f,a=a:agg(UX,f,a,1))
    for a in (0.5,1):
        for fn,f in ROW.items():
            add(f"UUrow.a{a}.{fn}",lambda UX,UU,st,f=f,a=a:agg(UU,f,a,0))
            add(f"AXrow.a{a}.{fn}",lambda UX,UU,st,f=f,a=a:agg(np.hstack([UU,UX]),f,a,0))
    for fn,f in {"ent":ent,"hoy":hoy,"hhi":hhi,"gini":gini,"tail1":lambda v:tail(v,1)}.items(): add(f"UXrowtot.{fn}",lambda UX,UU,st,f=f:f(UX.sum(1)))
    for fn,f in {"ent":ent,"hoy":hoy,"hhi":hhi,"gini":gini,"tail1":lambda v:tail(v,1),"lq.25":lambda v:lq(v,.25)}.items(): add(f"UUdiag.{fn}",lambda UX,UU,st,f=f:f(np.diag(UU)))
    for fn,f in {"ent":ent,"hoy":hoy,"lq.25":lambda v:lq(v,.25),"log1e-2":lambda v:logs(v,1e-2)}.items(): add(f"UUoff.{fn}",lambda UX,UU,st,f=f:f(offtri(UU)))
    for fn,f in {"hoy":hoy,"lq.5":lambda v:lq(v,.5),"ent":ent}.items():
        add(f"UXrms.{fn}",lambda UX,UU,st,f=f:f(np.sqrt(UX).ravel()))
        add(f"POOLrms.{fn}",lambda UX,UU,st,f=f:f(np.sqrt(np.concatenate([triu(UU),UX.ravel()]))))
    add("S.udshare",lambda UX,UU,st:float(np.trace(UU)/(UU.sum()+1e-300)))
    add("S.selftail1",lambda UX,UU,st:tail(np.diag(UU),1)*np.trace(UU)/(UU.sum()+1e-300))
    add("S.selftail2",lambda UX,UU,st:tail(np.diag(UU),2)*np.trace(UU)/(UU.sum()+1e-300))
    add("S.UUeig1",lambda UX,UU,st:float(-np.linalg.eigvalsh((UU+UU.T)/2)[-1]/(np.trace(UU)+1e-300)))
    def sv(UX): s=np.linalg.svd(UX,compute_uv=False); return s
    add("S.UXsv1",lambda UX,UU,st:float(-(sv(UX)[0]**2)/((sv(UX)**2).sum()+1e-300)))
    add("S.UXeffrank",lambda UX,UU,st:(lambda s:float(np.exp(-(s*np.log(s+EPS)).sum())))(sv(UX)/(sv(UX).sum()+1e-300)))
    add("S.UXstablerank",lambda UX,UU,st:float((sv(UX)**2).sum()/(sv(UX)[0]**2+1e-300)))
    for a in (0,0.25,0.5,0.75,1.0):
        for q in (0.1,0.5): add(f"rowsp.a{a}.q{q}",lambda UX,UU,st,a=a,q=q:rowsp(UX,a,q))
    # components for z-fusions (null stats in st['null'])
    comp={"A":lambda UX,UU:rowsp(UX,0.5,0.1),"B":lambda UX,UU:hoy(UX.ravel()),"C":lambda UX,UU:float(np.trace(UU)/(UU.sum()+1e-300)),
          "K":lambda UX,UU:agg(UX,hoy,1,1),"G":lambda UX,UU:hoy(UX.sum(1)),"S":lambda UX,UU:tail(np.diag(UU),1)*np.trace(UU)/(UU.sum()+1e-300)}
    M["_comp"]=comp
    def zf(keys): return lambda UX,UU,st:sum((comp[k](UX,UU)-st["null"][k][0])/st["null"][k][1] for k in keys)
    add("F1.A+B (plain)",lambda UX,UU,st:comp["A"](UX,UU)+comp["B"](UX,UU))
    for keys in ("AB","ABC","AC","BC","ABK","ABG","ABCK","ABCG","ABCS","BK","AKC","ABCGK"): add(f"Fz.{'+'.join(keys)}",zf(list(keys)))

    # ---- support ("is the connection there?") metrics: soft-L0 on block-normalised energies  m/(m+eps^2) ----
    def _norm(Mx): return Mx / (Mx.sum() + 1e-300)
    def sl0(m, eps): return float((m / (m + eps ** 2)).sum())
    def sl0m(m, lo=.01, hi=.2): return float((1 - np.log((m + hi ** 2) / (m + lo ** 2)) / (2 * np.log(hi / lo))).sum())
    def uu_up(UU): return (UU / (UU.sum() + 1e-300))[np.triu_indices(UU.shape[0])]
    for eps in (0.01, 0.03, 0.05, 0.1, 0.2):
        add(f"SL0.UX.e{eps}", lambda UX, UU, st, eps=eps: sl0(_norm(UX), eps))
        add(f"SL0.UU.e{eps}", lambda UX, UU, st, eps=eps: sl0(uu_up(UU), eps))
        add(f"SL0.UXUU.e{eps}", lambda UX, UU, st, eps=eps: sl0(_norm(UX), eps) + sl0(uu_up(UU), eps))
    add("SL0m.UX", lambda UX, UU, st: sl0m(_norm(UX))); add("SL0m.UU", lambda UX, UU, st: sl0m(uu_up(UU))); add("SL0m.UXUU", lambda UX, UU, st: sl0m(_norm(UX)) + sl0m(uu_up(UU)))
    add("SL0m.UX.wide", lambda UX, UU, st: sl0m(_norm(UX), .003, .3)); add("SL0m.UXUU.wide", lambda UX, UU, st: sl0m(_norm(UX), .003, .3) + sl0m(uu_up(UU), .003, .3))
    comp["A75"]=lambda UX,UU:rowsp(UX,0.75,0.1)
    comp["K0"]=lambda UX,UU:float(np.mean([hoy(UX[:,j]) for j in range(UX.shape[1])]))
    M["CAND.rowsp.75+UXcol.hoy"]=lambda UX,UU,st:sum((comp[k](UX,UU)-st["null"][k][0])/st["null"][k][1] for k in ("A75","K0"))
    return M
