"""k-way direct-sum (z-scored, no weights) fusions of NAMED metrics.   usage: python fuse_k.py OUT.csv INFS MODELS WORKERS 'a+b' 'a+b+c' ..."""
import os, sys, time, csv
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import grid, metrics as MET
_ST = {}; _M = None
def run(task):
    global _M
    inf, model, combo = task; names = combo.split("+")
    if _M is None: _M = MET.build(); grid._M = _M
    key = (inf, model)
    if key not in _ST:
        _ST.clear(); st = grid.state(inf, model); Hz = grid.haar(st["d"], 300, 4); allm = {}
        _ST[key] = (st, allm, Hz)
    st, allm, Hz = _ST[key]
    for n in names:
        if n not in allm:
            v = []
            for Q in Hz: UX, UU = grid.uxuu(st, Q); v.append(_M[n](UX, UU, st))
            allm[n] = (float(np.median(v)), float(np.std(v)) + 1e-12)
    def f(Q):
        UX, UU = grid.uxuu(st, Q)
        try: return sum((_M[n](UX, UU, st) - allm[n][0]) / allm[n][1] for n in names)
        except Exception: return 1e9
    v, Q = grid.search(st, f, nh=600, nst=12, maxiter=1000); mcc, mn, lm, lx = grid.evalQ(st, Q); return (inf, model, combo, mcc, mn, lm, lx, grid.errdeg(Q, st["R"]), v)
if __name__ == "__main__":
    import multiprocessing as mp
    out, infs, models, nproc = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), int(sys.argv[4]); combos = sys.argv[5:]
    done = set()
    if os.path.exists(out): done = {tuple(r[:3]) for r in csv.reader(open(out))}
    tasks = [(i, m, c) for m in models for i in infs for c in combos if (i, m, c) not in done]; print(len(tasks), "tasks", flush=True); t0 = time.time(); k = 0
    with mp.Pool(nproc) as pool, open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        for r in pool.imap_unordered(run, tasks, chunksize=2):
            w.writerow(r); k += 1
            if k % 50 == 0: fh.flush(); print(f"{k}/{len(tasks)} [{time.time()-t0:.0f}s]", flush=True)
    print("ALLDONE", time.time() - t0)
