from lib2 import *
def neff(v): q=np.asarray(v,float).ravel(); q=q/q.sum(); return float(np.exp(-(q*np.log(q+1e-12)).sum()))
def ent(v): q=np.asarray(v,float).ravel(); q=q/q.sum(); return float(-(q*np.log(q+1e-12)).sum())
def rowsp(UU,UX,a=0.25,q=0.1,eps=1e-4):
    s=UX.sum(1); share=s/s.sum(); A=UX/(s[:,None]+1e-30); return float(np.sum((share**a)*np.mean((A+eps)**q,axis=1))/np.sum(share**a))
def udshare(UU,UX): return float(np.trace(UU)/UU.sum())
def tail1(UU,UX): s=np.sort(np.diag(UU))[::-1]; return float(s[1:].sum()/UU.sum())
METRICS={  # name -> (channel, xstd, function of blocks)
 "EXP13_p_rawX":("probability",False,"exp13"),
 "EXP13_p_stdX":("probability",True,"exp13"),
 "EXP13_logit_stdX":("logit",True,"exp13"),
 "UXent_logit_stdX":("logit",True,lambda UU,UX:ent(UX)),
 "rowsp_logit_stdX":("logit",True,rowsp),
 "rowsp_p_stdX":("probability",True,rowsp),
}
def run_metric(g,name,Hs=300):
    ch,xst,f=METRICS[name]; blk=lambda Q:g.blocks(ch,Q,xst)
    if f=="exp13":
        Hh=special_ortho_group.rvs(g.d,size=120,random_state=7); B=[blk(Q) for Q in Hh]
        s1=np.std([neff(b[1]) for b in B])+1e-12; s2=np.std([udshare(*b) for b in B])+1e-12
        fn=lambda Q:(lambda b:neff(b[1])/s1+udshare(*b)/s2)(blk(Q))
    else: fn=lambda Q:f(*blk(Q))
    v,Q=g.search(fn,nh=Hs); return Q
if __name__=="__main__":
    models=sys.argv[1].split(","); names=sys.argv[2].split(",") if len(sys.argv)>2 else list(METRICS)
    for m in models:
        g=G(m); print(f"== {m}: ceiling MCC/leak at R* = {tuple(round(x,3) for x in g.mcc(np.eye(g.d)))}",flush=True)
        for nme in names:
            t0=time.time(); Q=run_metric(g,nme); mc,lk=g.mcc(Q); print(f"  {nme:18s} err={g.err(Q):5.1f}  MCC={mc:.3f} leak={lk:.3f}  ({time.time()-t0:.0f}s)",flush=True)
