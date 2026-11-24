import csv,collections,numpy as np, grid
ceil=grid.ceilings(); MOD=["C1","C2","C3","CR4"]; R={}
for inf,m,name,mcc,mn,lm,lx,err,f in csv.reader(open("grid_results.csv")): R[(inf,name,m)]=(float(mcc),float(lx))
infs=sorted({k[0] for k in R}); names=sorted({k[1] for k in R})
# minimax: worst MCC gap over the four DBs
rank=[]
for i in infs:
    for nm in names:
        g=[ceil[m][0]-R[(i,nm,m)][0] for m in MOD]; lx=[R[(i,nm,m)][1] for m in MOD]
        rank.append((max(g),max(lx),i,nm,[round(R[(i,nm,m)][0],3) for m in MOD]))
rank.sort()
print("TOP 20 by worst-case MCC gap over the four DBs (C1,C2,C3,CR4):")
for w,lx,i,nm,v in rank[:20]: print(f"  gap {w:.3f}  maxleak {lx:.2f}  {i:18s} {nm:22s} {v}")
# who fails: for cells passing 3 (lenient), which DB fails
fails=collections.Counter()
for i in infs:
    for nm in names:
        ok=[R[(i,nm,m)][0]>=ceil[m][0]-0.05 and R[(i,nm,m)][1]<=0.25 for m in MOD]
        if sum(ok)==3: fails[MOD[ok.index(False)]]+=1
print("\nfor cells passing exactly 3 (lenient), the failing DB:",dict(fails))
# metrics that pass CR4 / C3 anywhere
def passing(m): return {(i,nm) for i in infs for nm in names if R[(i,nm,m)][0]>=ceil[m][0]-0.05 and R[(i,nm,m)][1]<=0.25}
pc3,pcr4=passing("C3"),passing("CR4"); print(f"\nC3-pass cells {len(pc3)}, CR4-pass cells {len(pcr4)}, intersection {len(pc3&pcr4)}")
print("metric names passing CR4 (any inference):",sorted(collections.Counter(nm for i,nm in pcr4).items(),key=lambda t:-t[1])[:20])
print("inferences passing CR4:",collections.Counter(i for i,nm in pcr4).most_common())
print("metric names passing C3 (any inference), top:",sorted(collections.Counter(nm for i,nm in pc3).items(),key=lambda t:-t[1])[:20])
print("inferences passing C3:",collections.Counter(i for i,nm in pc3).most_common())
# near miss relaxed: gap<=0.07 on all four, leak<=0.35
rel=[(max(ceil[m][0]-R[(i,nm,m)][0] for m in MOD),i,nm) for i in infs for nm in names if all(R[(i,nm,m)][1]<=0.35 for m in MOD)]
rel=[t for t in rel if t[0]<=0.07]; print(f"\nrelaxed (all four within 0.07 of ceiling, max leak<=0.35): {len(rel)} cells"); [print("  ",t) for t in sorted(rel)[:12]]
