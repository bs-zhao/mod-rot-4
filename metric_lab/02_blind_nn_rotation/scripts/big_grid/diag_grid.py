import csv,collections,numpy as np, grid, metrics as MET
M=MET.build(); ceil=grid.ceilings(); rows=list(csv.reader(open("grid_results.csv")))
cnt=collections.Counter(); tot=collections.Counter(); tmp={}
for inf,m,name,mcc,mn,lm,lx,err,f in rows:
    st=grid.state(inf,m) if False else None
import itertools
byk=collections.defaultdict(list)
for r in rows: byk[(r[1],r[0])].append(r)
for (m,inf),rs in byk.items():
    st=grid.state(inf,m); R=np.load(f"meta_{m}.npz")["R"]; UX,UU=grid.uxuu(st,R)
    for inf_,m_,name,mcc,mn,lm,lx,err,f in rs:
        fR=M[name](UX,UU,st); found=float(f); ok=(float(mcc)>=ceil[m][0]-0.05 and float(lx)<=0.25)
        kind="pass" if ok else ("METRIC prefers wrong" if found<fR-1e-6*max(1,abs(fR)) else "search miss")
        cnt[(m,kind)]+=1
for m in ("C1","C2","C3","CR4"): print(m,{k:cnt[(m,k)] for k in ("pass","METRIC prefers wrong","search miss")})
