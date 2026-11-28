"""W4-00 gate: unit tests (tests/test_w4.py) -> results/validation_checks.csv, plus the input manifest results/w4_manifest.json
(source commit, file hashes of the reused Lab 07-09 code, payload / checkpoint hashes, Lab 08 / 09 artifacts read, known missing ones)."""
import sys, json, time, traceback, inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4.common import RES, OUT08, OUT09, LAB09, ROOT, PRIMARY, SECONDARY, code_files, code_hash, git_commit, sha256
import pandas as pd

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
    import test_w4 as T
    rows = []
    for name, fn in inspect.getmembers(T, inspect.isfunction):
        if not name.startswith("test_") or name == "test_blind_isolation":
            continue
        t0 = time.time()
        try:
            fn(); ok, msg = True, ""
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {e}"; traceback.print_exc()
        rows.append(dict(test=name, ok=ok, seconds=round(time.time() - t0, 1), message=msg)); print(name, ok, flush=True)
    t0 = time.time()
    try:
        T.test_blind_isolation(); rows.append(dict(test="test_blind_isolation", ok=True, seconds=round(time.time() - t0, 1), message="audit hook (last: process keeps the hook)"))
    except Exception as e:
        rows.append(dict(test="test_blind_isolation", ok=False, seconds=round(time.time() - t0, 1), message=str(e)))
    pd.DataFrame(rows).to_csv(RES / "validation_checks.csv", index=False)
    from wave2.blind_inputs import load_payload
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), git_commit=git_commit(), code_hash=code_hash(),
               code_sha256={str(f.relative_to(ROOT)): sha256(f) for f in code_files()},
               payloads={t: load_payload(t).hashes for t in PRIMARY + SECONDARY},
               reused_artifacts=dict(lab08_moment_cache=str(OUT08 / "moment_cache"), lab09_blind=str(OUT09 / "blind"), lab09_results=str(LAB09 / "results"),
                                     w3_lock_sha256=sha256(LAB09 / "results" / "algorithm_lock.json")),
               missing=[], notes="Lab 07-09 outputs are read-only; Lab 10 caches recompute every measure it searches (Lab 08 caches used for the replay audit only)")
    (RES / "w4_manifest.json").write_text(json.dumps(man, indent=1)); print(sum(r["ok"] for r in rows), "/", len(rows), "tests passed")
