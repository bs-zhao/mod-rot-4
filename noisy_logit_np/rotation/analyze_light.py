import csv, collections, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grid
MOD = sys.argv[1].split(","); ceil = grid.ceilings(MOD); R = {}
for r in csv.reader(open(Path(grid.HERE) / "grid_results.csv")):
    if r[1] in MOD: R[(r[0], r[2], r[1])] = tuple(float(x) for x in r[3:7])   # mcc, min, leak mean, leak max
infs = sorted({k[0] for k in R}); names = sorted({k[1] for k in R})
base = {}
for m in MOD:
    mt = np.load(Path(grid.HERE) / f"meta_{m}.npz"); base[m] = grid.evalQ(dict(d=int(mt["d"]), Zs=mt["Zs"], Ts=mt["Ts"]), np.eye(int(mt["d"])))
print("baselines  [MCC, min matched, leak mean, leak max]")
for m in MOD: print(f"  {m}: supervised ceiling {np.round(ceil[m],3)} | UNROTATED {np.round(base[m],3)}")
ok = lambda i, n, m, s=False: R[(i, n, m)][0] >= ceil[m][0] - (.03 if s else .05) and R[(i, n, m)][3] <= (.15 if s else (.30 if m.startswith('CR') else .20))   # leak-max tolerance: C-series .20, CR-series .30
print(f"\ncells {len(R)} = {len(names)} metrics x {len(infs)} inferences x {len(MOD)} models; inferences {infs}")
for s in (False, True):
    print(f"\n{'STRICT (gap<=.03, leakmax<=.15)' if s else 'LENIENT (MCC gap<=.05; leak max<.20 for C, <.30 for CR)'}: cells passing per model:", {m: sum(ok(i, n, m, s) for i in infs for n in names) for m in MOD}, "of", len(infs) * len(names))
    allp = [(i, n) for i in infs for n in names if all(ok(i, n, m, s) for m in MOD)]; print(f"  cells passing ALL {len(MOD)}: {len(allp)}")
    cnt = collections.Counter(n for i, n in allp); print("  metrics passing all models (count of inferences):", cnt.most_common(12))
print("\nheadline metrics, MCC / leak mean / leak max per model")
for name in ("rowsp.a0.5.q0.1", "UX.hoy", "F1.A+B (plain)", "Fz.A+B", "Fz.A+B+C"):
    for i in infs: print(f"  {i:13s} {name:16s}: " + " | ".join(f"{m} {R[(i,name,m)][0]:.3f}/{R[(i,name,m)][2]:.3f}/{R[(i,name,m)][3]:.2f}" for m in MOD))
print("\nbest cells per model (by MCC; leak mean/max):")
for m in MOD:
    b = sorted(((R[(i, n, m)][0], R[(i, n, m)][2], R[(i, n, m)][3], i, n) for i in infs for n in names), reverse=True)[:5]
    print(f"  {m}: " + "; ".join(f"{x[0]:.3f} {x[1]:.3f}/{x[2]:.2f} {x[3]} {x[4]}" for x in b))
rk = sorted((max(ceil[m][0] - R[(i, n, m)][0] for m in MOD), max(R[(i, n, m)][3] for m in MOD), i, n) for i in infs for n in names)
print("\nworst-case ranking (smallest worst MCC gap across models):")
for w, lx, i, n in rk[:12]: print(f"  gap {w:.3f} maxleak {lx:.2f} {i:13s} {n:22s} " + str([f'{R[(i,n,m)][0]:.3f}/{R[(i,n,m)][2]:.3f}/{R[(i,n,m)][3]:.2f}' for m in MOD]))
