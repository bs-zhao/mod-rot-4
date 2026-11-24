"""For chosen (inference, fusion) cells: truth regret on the given model (diagnosis) + a 10x search budget (T-free search)."""
import os, sys, csv
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, grid, metrics as MET
def run(task):
    inf, model, combo = task; names = combo.split("+"); M = MET.build(); st = grid.state(inf, model); Hz = grid.haar(st["d"], 300, 4); z = {}
    for n in names:
        v = []
        for Q in Hz: UX, UU = grid.uxuu(st, Q); v.append(M[n](UX, UU, st))
        z[n] = (float(np.median(v)), float(np.std(v)) + 1e-12)
    def f(Q):
        UX, UU = grid.uxuu(st, Q); return sum((M[n](UX, UU, st) - z[n][0]) / z[n][1] for n in names)
    fR = f(st["R"]); v0, Q0 = grid.search(st, f, nh=600, nst=12, maxiter=1000); v1, Q1 = grid.search(st, f, nh=4000, nst=60, maxiter=3000)
    return (inf, model, combo, fR, v0, v1, grid.evalQ(st, Q0), grid.evalQ(st, Q1))
if __name__ == "__main__":
    import multiprocessing as mp
    model = sys.argv[1]; cells = [tuple(c.split("|")) for c in sys.argv[2:]]
    ceil = grid.ceilings([model])[model]; print(model, "ceiling", np.round(ceil, 3), flush=True)
    with mp.Pool(4) as pool: res = pool.map(run, [(i, model, c) for i, c in cells], chunksize=1)
    for inf, m, combo, fR, v0, v1, e0, e1 in res:
        print(f"{inf:13s} {combo:36s} f(R*)={fR:7.3f} found(default)={v0:7.3f} found(10x)={v1:7.3f} -> {'SEARCH MISS' if fR < v1 - 1e-6 else 'metric prefers wrong'} | default MCC {e0[0]:.3f} leak {e0[2]:.3f}/{e0[3]:.2f} | 10x MCC {e1[0]:.3f} leak {e1[2]:.3f}/{e1[3]:.2f}")
