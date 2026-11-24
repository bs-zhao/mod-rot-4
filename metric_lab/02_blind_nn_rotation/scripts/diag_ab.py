from lib2 import *
import bench
np.set_printoptions(precision=2,suppress=True,linewidth=140)
F=dict(a=0.5,q=0.1,eps=1e-4)
for model in sys.argv[1].split(","):
    g=G(model,seed=2); d=g.d
    f=lambda Q:bench.rowsp(*g.blocks("logit",Q),**F)
    UU,UX=g.blocks("logit",np.eye(d)); A=UX/UX.sum(1,keepdims=True)
    f0=f(np.eye(d))
    Hh=special_ortho_group.rvs(d,size=600,random_state=4); v=np.array([f(Q) for Q in Hh]); best=(1e9,None)
    for i in np.argsort(v)[:40]:
        val,Q=g.opt(f,Hh[i],maxiter=1500)
        if val<best[0]: best=(val,Q)
    # local optimum started at truth
    vt,Qt=g.opt(f,np.eye(d),maxiter=1500)
    print(f"\n## {model} d={d} p={g.p}")
    print(f"  rowsp at truth R*      = {f0:.4f}   (MCC {g.mcc(np.eye(d))[0]:.3f})")
    print(f"  local min from truth   = {vt:.4f}   (MCC {g.mcc(Qt)[0]:.3f}, err {g.err(Qt):.1f}deg)")
    print(f"  global min found       = {best[0]:.4f}   (MCC {g.mcc(best[1])[0]:.3f}, err {g.err(best[1]):.1f}deg)")
    print("  -> metric prefers wrong orientation" if best[0]<vt-1e-3 else "  -> truth is (near) the global min")
    print("  UX row-normalised at truth (rows=T axes, cols=X columns):"); print(A)
    print("  row energy share:",UX.sum(1)/UX.sum()); print("  eff #X cols per row:",np.exp(-(A*np.log(A+1e-12)).sum(1)),flush=True)
