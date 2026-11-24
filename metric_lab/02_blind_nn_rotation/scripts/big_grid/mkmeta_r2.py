from lib2 import *
for m in ("C1","C2","C3","CR4"):
    g=G(m,seed=2,repeat=2); np.savez(f"meta_{m}r2.npz",d=g.d,p=g.p,xs=np.asarray(g.xs,float),zstd=np.asarray(g.base.z_std,float),Zs=g.Zs,Ts=g.Ts,R=np.asarray(g.base.R_star,float)); print(m,g.d,g.p)
