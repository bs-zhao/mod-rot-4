import os,sys,csv,time
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[k]="1"
import numpy as np, grid
import rerun_c3
def load(f): return {(r[0],r[2],r[1]):(float(r[3]),float(r[6])) for r in csv.reader(open(f))}
if __name__=="__main__":
    import multiprocessing as mp
    M1=["C1","C2","C3","CR4"]; M2=[m+"r2" for m in M1]; ceil=grid.ceilings(M1+M2)
    R1=load("grid_results.csv"); R2=load("grid_results_r2.csv"); R={**R1,**R2}
    ok=lambda inf,n,m:(R[(inf,n,m)][0]>=ceil[m][0]-0.05 and R[(inf,n,m)][1]<=0.25)
    cells=sorted({(i,n) for (i,n,m) in R1})
    cand=[(i,n) for (i,n) in cells if all(ok(i,n,m) for m in ("C1","C2","CR4","C1r2","C2r2","CR4r2"))]
    print(f"cells passing C1,C2,CR4 on BOTH NNs (default search): {len(cand)}",flush=True)
    for i,n in cand: print("   ",i,n,"C3:",R[(i,n,'C3')],"C3r2:",R[(i,n,'C3r2')])
    near=[(i,n) for (i,n) in cells if all(ok(i,n,m) for m in ("C1","C2","CR4","C1r2","C2r2","CR4r2")) is False and sum(ok(i,n,m) for m in ("C1","C2","CR4","C1r2","C2r2","CR4r2"))==5]
    print(f"cells passing 5 of those 6: {len(near)}")
    grid._M=grid.MET.build()
    tasks=[(i,m,n) for (i,n) in cand for m in ("C3","C3r2")]
    with mp.Pool(14) as pool: res=pool.map(rerun_c3.run2,tasks,chunksize=1)
    S={(r[0],r[2],r[1]):(r[3],r[4]) for r in res}
    print("\ninference          metric               | NN1: C1 C2 C3* CR4 (MCC/leakmax)                          | NN2: C1 C2 C3* CR4")
    full=0
    for i,n in cand:
        c3a=S[(i,n,"C3")]; c3b=S[(i,n,"C3r2")]
        ok3a=c3a[0]>=ceil["C3"][0]-0.05 and c3a[1]<=0.25; ok3b=c3b[0]>=ceil["C3r2"][0]-0.05 and c3b[1]<=0.25; full+=ok3a and ok3b
        f=lambda m:f"{R[(i,n,m)][0]:.3f}/{R[(i,n,m)][1]:.2f}"
        print(f"{i:18s} {n:20s} | {f('C1')} {f('C2')} {c3a[0]:.3f}/{c3a[1]:.2f} {f('CR4')} | {f('C1r2')} {f('C2r2')} {c3b[0]:.3f}/{c3b[1]:.2f} {f('CR4r2')} {'  <== ALL EIGHT PASS' if ok3a and ok3b else ''}")
    print("ALL EIGHT (4 DBs x 2 NNs) PASS:",full)
