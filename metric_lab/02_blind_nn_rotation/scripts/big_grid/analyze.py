import csv,sys,collections,numpy as np
import grid
ceil=grid.ceilings(); MOD=["C1","C2","C3","CR4"]
rows=[r for r in csv.reader(open("grid_results.csv"))]
R={}
for inf,m,name,mcc,mn,lm,lx,err,f in rows: R[(inf,name,m)]=(float(mcc),float(mn),float(lm),float(lx),float(err))
def ok(m,v,strict):
    c=ceil[m]; return (v[0]>=c[0]-(0.03 if strict else 0.05)) and (v[3]<=(0.15 if strict else 0.25))
infs=sorted({k[0] for k in R}); names=sorted({k[1] for k in R})
print("ceilings:",{m:tuple(round(x,3) for x in ceil[m]) for m in MOD}); print(f"cells: {len(R)}  inferences {len(infs)}  metrics {len(names)}")
# per model: how many (inference,metric) cells pass
for strict in (False,True):
    print(f"\n== cells passing, {'STRICT' if strict else 'LENIENT'} ==")
    for m in MOD:
        n=sum(ok(m,R[(i,nm,m)],strict) for i in infs for nm in names if (i,nm,m) in R); tot=sum(1 for i in infs for nm in names if (i,nm,m) in R); print(f"  {m}: {n}/{tot}")
    full=[]
    for i in infs:
        for nm in names:
            if all((i,nm,m) in R for m in MOD):
                k=sum(ok(m,R[(i,nm,m)],strict) for m in MOD)
                full.append((k,np.mean([R[(i,nm,m)][0]-ceil[m][0] for m in MOD]),i,nm))
    full.sort(key=lambda t:(-t[0],-t[1]))
    c4=[t for t in full if t[0]==4]; print(f"  cells passing ALL FOUR: {len(c4)}  | passing 3: {sum(1 for t in full if t[0]==3)}")
    for k,gap,i,nm in c4[:25]: print(f"    {i:18s} {nm:22s}  mean MCC gap {gap:+.3f} | "+" | ".join(f"{m} {R[(i,nm,m)][0]:.3f}/{R[(i,nm,m)][3]:.2f}" for m in MOD))
    # per inference: best metric
    print(f"  best metric per inference (most passes, then mean gap):")
    for i in infs:
        best=max([t for t in full if t[2]==i],key=lambda t:(t[0],t[1]),default=None)
        if best: print(f"    {i:18s} {best[3]:22s} passes {best[0]}/4  mean gap {best[1]:+.3f}")
    # metrics robust across inferences
    cnt=collections.Counter(nm for k,gap,i,nm in c4); print("  metrics passing all four under >=1 inference (count of inferences):",cnt.most_common(15))
    cnt2=collections.Counter(i for k,gap,i,nm in c4); print("  inferences with any all-four metric:",cnt2.most_common())
# per-model best cells (lenient)
print("\n== best single cell per model (MCC) ==")
for m in MOD:
    b=sorted(((R[(i,nm,m)][0],R[(i,nm,m)][3],i,nm) for i in infs for nm in names if (i,nm,m) in R),reverse=True)[:6]
    print(f"  {m} (ceiling {ceil[m][0]:.3f}):"); [print(f"     {x[0]:.3f} leakmax {x[1]:.2f}  {x[2]} {x[3]}") for x in b]
