import csv,collections,numpy as np, grid
M1=["C1","C2","C3","CR4"]; M2=[m+"r2" for m in M1]; ceil=grid.ceilings(M1+M2)
R={}
for f in ("grid_results.csv","grid_results_r2.csv"):
    for r in csv.reader(open(f)): R[(r[0],r[2],r[1])]=(float(r[3]),float(r[6]))
ok=lambda i,n,m:(R[(i,n,m)][0]>=ceil[m][0]-0.05 and R[(i,n,m)][1]<=0.25)
cells=sorted({(i,n) for (i,n,m) in R})
for tag,mods in (("NN1",M1),("NN2 (retrained)",M2)):
    c={m:sum(ok(i,n,m) for i,n in cells) for m in mods}; a4=[(i,n) for i,n in cells if all(ok(i,n,m) for m in mods)]
    print(f"{tag}: cells passing per DB {c}; ALL FOUR (default search): {len(a4)}; 3-of-4: {sum(1 for i,n in cells if sum(ok(i,n,m) for m in mods)==3)}")
    for i,n in a4[:8]: print("    ",i,n,[round(R[(i,n,m)][0],3) for m in mods])
both=[(i,n) for i,n in cells if all(ok(i,n,m) for m in M1+M2)]; print("all eight (default search):",len(both))
# which DB blocks, among cells passing CR4 on both NNs
cr=[(i,n) for i,n in cells if ok(i,n,"CR4") and ok(i,n,"CR4r2")]
print(f"cells passing CR4 on both NNs: {len(cr)}; of those passing C3 (default search) on NN1: {sum(ok(i,n,'C3') for i,n in cr)}, on NN2: {sum(ok(i,n,'C3r2') for i,n in cr)}, both: {sum(ok(i,n,'C3') and ok(i,n,'C3r2') for i,n in cr)}")
print("metric families passing CR4 on both NNs:",collections.Counter(n for i,n in cr).most_common(12))
# C3 across NNs: how stable is the best C3 cell?
c3=[(i,n) for i,n in cells if ok(i,n,"C3") and ok(i,n,"C3r2")]; print(f"cells passing C3 on both NNs: {len(c3)}  e.g.",collections.Counter(n for i,n in c3).most_common(8))
