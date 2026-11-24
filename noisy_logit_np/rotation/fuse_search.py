"""Pairwise direct-sum (z-scored, no free weights) fusions among the metrics with the smallest worst-case truth regret.
usage: python fuse_search.py INFERENCE TOPN WORKERS MODELS     -> fuse_pairs_<INF>.csv
z-score = (m - median_null)/SD_null, null = random rotations;  fusion = z_a + z_b.   T is used ONLY to evaluate the found rotation (and, upstream, to rank metrics by regret)."""
import os, sys, time, csv, itertools
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import grid, metrics as MET
_ST = {}; _M = None
def prep(inf, model, names):
    global _M
    if _M is None: _M = MET.build(); grid._M = _M
    k = (inf, model)
    if k in _ST: return _ST[k]
    _ST.clear(); st = grid.state(inf, model); d = st["d"]; Hz = grid.haar(d, 300, 4); v = {n: [] for n in names}
    for Q in Hz:
        UX, UU = grid.uxuu(st, Q)
        for n in names: v[n].append(_M[n](UX, UU, st))
    st["z"] = {n: (float(np.median(v[n])), float(np.std(v[n])) + 1e-12) for n in names}; _ST[k] = st; return st
def run(task):
    inf, model, a, b, names = task; st = prep(inf, model, names); fa, fb = _M[a], _M[b]; za, zb = st["z"][a], st["z"][b]
    def f(Q):
        UX, UU = grid.uxuu(st, Q)
        try: return (fa(UX, UU, st) - za[0]) / za[1] + (fb(UX, UU, st) - zb[0]) / zb[1]
        except Exception: return 1e9
    v, Q = grid.search(st, f, nh=600, nst=12, maxiter=1000); mcc, mn, lm, lx = grid.evalQ(st, Q); return (inf, model, a, b, mcc, mn, lm, lx, grid.errdeg(Q, st["R"]), v)
if __name__ == "__main__":
    import multiprocessing as mp
    inf = sys.argv[1]; topn = int(sys.argv[2]); nproc = int(sys.argv[3]); models = sys.argv[4].split(","); reg = np.load(f"regret_{inf}.npy", allow_pickle=True).item()
    MOD = ["C1", "C2", "C3", "CR3", "CR4"]; allnames = sorted({k[0] for k in reg}); cand = [n for n in allnames if not n.startswith("F")]
    top = sorted(cand, key=lambda n: max(reg[(n, m)] for m in MOD))[:topn]; names = top
    print("base metrics:", top, flush=True)
    tasks = [(inf, m, a, b, names) for m in models for a, b in itertools.combinations(top, 2)]; out = f"fuse_pairs_{inf}.csv"; done = set()
    if os.path.exists(out): done = {tuple(r[:4]) for r in csv.reader(open(out))}
    tasks = [t for t in tasks if (t[0], t[1], t[2], t[3]) not in done]; print(len(tasks), "tasks", flush=True); t0 = time.time(); k = 0
    with mp.Pool(nproc) as pool, open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        for r in pool.imap_unordered(run, tasks, chunksize=2):
            w.writerow(r); k += 1
            if k % 100 == 0: fh.flush(); print(f"{k}/{len(tasks)} [{time.time()-t0:.0f}s]", flush=True)
    print("ALLDONE", time.time() - t0)
