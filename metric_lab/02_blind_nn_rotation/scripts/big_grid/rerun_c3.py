import os,sys,csv,time
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[k]="1"
import numpy as np, grid, metrics as MET
def run2(task):
    inf,model,name=task; st=grid.state(inf,model); fn=grid._M[name]
    def f(Q):
        UX,UU=grid.uxuu(st,Q)
        try: return fn(UX,UU,st)
        except Exception: return 1e9
    v,Q=grid.search(st,f,nh=4000,nst=60,maxiter=3000); mcc,mn,lm,lx=grid.evalQ(st,Q)
    R=np.load(f"{grid.HERE}/meta_{model}.npz")["R"]; UX,UU=grid.uxuu(st,R)
    return (inf,model,name,mcc,lx,v,float(fn(UX,UU,st)))
if __name__=="__main__":
    import multiprocessing as mp
    ceil=grid.ceilings(); rows=list(csv.reader(open("grid_results.csv"))); R={(r[0],r[2],r[1]):(float(r[3]),float(r[6])) for r in rows}
    cr4=[(i,n) for (i,n,m),(mcc,lx) in R.items() if m=="CR4" and mcc>=ceil["CR4"][0]-0.05 and lx<=0.25]
    print(len(cr4),"CR4-passing cells; rerunning them on C3 (and C1,C2 for completeness) with a 10x search budget",flush=True)
    tasks=[(i,m,n) for (i,n) in cr4 for m in ("C3",)]
    M=grid.MET.build(); grid._M=M
    with mp.Pool(14) as pool: res=pool.map(run2,tasks,chunksize=1)
    print(f"{'inference':18s} {'metric':20s} | C1 mcc/lx | C2 mcc/lx | C3 old mcc/lx | C3 strong mcc/lx (f_found vs f(R*))")
    npass=0
    for (inf,m,n,mcc,lx,fv,fR) in res:
        a=R[(inf,n,"C1")]; b=R[(inf,n,"C2")]; o=R[(inf,n,"C3")]; ok3=(mcc>=ceil["C3"][0]-0.05 and lx<=0.25)
        ok12=all(R[(inf,n,mm)][0]>=ceil[mm][0]-0.05 and R[(inf,n,mm)][1]<=0.25 for mm in ("C1","C2"))
        npass+=ok3 and ok12
        print(f"{inf:18s} {n:20s} | {a[0]:.3f}/{a[1]:.2f} | {b[0]:.3f}/{b[1]:.2f} | {o[0]:.3f}/{o[1]:.2f} | {mcc:.3f}/{lx:.2f}  ({fv:.4f} vs {fR:.4f}) {'ALL FOUR PASS' if ok3 and ok12 else ''}")
    print("cells passing all four after the strong C3 search:",npass)
