"""Summarise grid_results.csv (132 metrics x 10 inferences x 5 models) -> results/rotation_summary.txt"""
import csv, collections, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grid
from common import RESULTS
MOD = ["C1", "C2", "C3", "CR3", "CR4"]; ceil = grid.ceilings(MOD); out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))
R = {}
for r in csv.reader(open(Path(grid.HERE) / "grid_results.csv")): R[(r[0], r[2], r[1])] = (float(r[3]), float(r[4]), float(r[5]), float(r[6]))   # mcc, min, leak mean, leak max
infs = sorted({k[0] for k in R}); names = sorted({k[1] for k in R})
ok = lambda i, n, m, strict=False: R[(i, n, m)][0] >= ceil[m][0] - (.03 if strict else .05) and R[(i, n, m)][3] <= (.15 if strict else (.30 if m.startswith('CR') else .20))   # leak-max tolerance: C-series .20, CR-series .30
P("supervised-rotation ceilings (MCC, min, leak mean, leak max):", {m: tuple(round(x, 3) for x in ceil[m]) for m in MOD}); P(f"cells: {len(R)} = {len(names)} metrics x {len(infs)} inferences x {len(MOD)} models")
for strict in (False, True):
    P(f"\n== cells passing per model, {'STRICT (gap<=.03,leakmax<=.15)' if strict else 'LENIENT (gap<=.05,leakmax<=.25)'} ==")
    for m in MOD: P(f"  {m}: {sum(ok(i, n, m, strict) for i in infs for n in names)}/{len(infs)*len(names)}")
    full = [(sum(ok(i, n, m, strict) for m in MOD), np.mean([R[(i, n, m)][0] - ceil[m][0] for m in MOD]), i, n) for i in infs for n in names]; full.sort(key=lambda t: (-t[0], -t[1]))
    for k in (5, 4): P(f"  cells passing {k} of 5 models: {sum(1 for t in full if t[0]==k)}")
    for k, gap, i, n in [t for t in full if t[0] == 5][:20]: P(f"    ALL FIVE: {i:14s} {n:22s} mean gap {gap:+.3f} | " + " ".join(f"{m} {R[(i,n,m)][0]:.3f}/{R[(i,n,m)][3]:.2f}" for m in MOD))
    fails = collections.Counter(); [fails.update([MOD[[ok(i, n, m, strict) for m in MOD].index(False)]]) for i in infs for n in names if sum(ok(i, n, m, strict) for m in MOD) == 4]
    P("  for cells passing exactly 4 of 5, the failing model:", dict(fails))
P("\n== headline metrics (inference i01_AD): MCC / leak mean / leak max ==")
for name in ("rowsp.a0.5.q0.1", "UX.hoy", "F1.A+B (plain)", "Fz.A+B", "Fz.A+B+C"):
    P(f"  {name:20s}: " + " | ".join(f"{m} {R[('i01_AD',name,m)][0]:.3f}/{R[('i01_AD',name,m)][2]:.3f}/{R[('i01_AD',name,m)][3]:.2f}" for m in MOD if ('i01_AD', name, m) in R))
P("\n== best 5 cells per model (MCC) ==")
for m in MOD:
    b = sorted(((R[(i, n, m)][0], R[(i, n, m)][3], i, n) for i in infs for n in names if (i, n, m) in R), reverse=True)[:5]
    P(f"  {m} (ceiling {ceil[m][0]:.3f}): " + "; ".join(f"{x[0]:.3f}/{x[1]:.2f} {x[2]} {x[3]}" for x in b))
P("\n== worst-case ranking over the five models (smallest worst MCC gap) ==")
rk = sorted((max(ceil[m][0] - R[(i, n, m)][0] for m in MOD), max(R[(i, n, m)][3] for m in MOD), i, n) for i in infs for n in names)
for w, lx, i, n in rk[:15]: P(f"  gap {w:.3f} maxleak {lx:.2f}  {i:14s} {n:22s} " + str([round(R[(i, n, m)][0], 3) for m in MOD]))
(RESULTS / "rotation_summary.txt").write_text("\n".join(out)); print("\n".join(out))
