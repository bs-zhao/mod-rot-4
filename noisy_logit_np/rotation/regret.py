"""'Truth regret' of every metric: (f(R*) - f_found) / SD_null   (R* only used here for diagnosis, never inside a metric)."""
import csv, sys, numpy as np, grid, metrics as MET
grid._M = MET.build(); INF = sys.argv[1]; MOD = ["C1", "C2", "C3", "CR3", "CR4"]
found = {}
for r in csv.reader(open("grid_results.csv")):
    if r[0] == INF: found[(r[2], r[1])] = float(r[8])
names = sorted({k[0] for k in found}); out = {}
for m in MOD:
    st = grid.state(INF, m); d = st["d"]; Hz = grid.haar(d, 300, 4)
    UXR, UUR = grid.uxuu(st, st["R"]); vals = {n: [] for n in names}
    for Q in Hz:
        UX, UU = grid.uxuu(st, Q)
        for n in names:
            try: vals[n].append(grid._M[n](UX, UU, st))
            except Exception: vals[n].append(np.nan)
    for n in names:
        sd = np.nanstd(vals[n]) + 1e-12
        try: fR = grid._M[n](UXR, UUR, st)
        except Exception: fR = np.nan
        out[(n, m)] = (fR - found[(n, m)]) / sd
np.save(f"regret_{INF}.npy", out, allow_pickle=True)
rk = sorted(names, key=lambda n: max(out[(n, m)] for m in MOD))
print(f"truth regret in null-SD units (inference {INF}); metrics with the smallest WORST-case regret over the five models")
for n in rk[:25]: print(f"  {n:24s} max {max(out[(n,m)] for m in MOD):5.2f} | " + " ".join(f"{m} {out[(n,m)]:5.2f}" for m in MOD))
print("\nsome reference metrics:")
for n in ("rowsp.a0.5.q0.1", "UX.hoy", "UXcol.a0.ent", "S.udshare", "Fz.A+B+C"): print(f"  {n:24s} " + " ".join(f"{m} {out[(n,m)]:5.2f}" for m in MOD))
