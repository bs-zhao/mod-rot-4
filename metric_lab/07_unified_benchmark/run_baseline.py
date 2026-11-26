"""W1-01 / W1-02: unified metric x inference baseline and the measure / integration batches.

usage: python run_baseline.py --toys all|C1,CR4L,... --batch A|B|C|D|E [--variant both|s9|s9_2] [--seed S] [--resume]
  A  primary panel, Q01, pilot fidelity (256 TEST subjects x 32 trials), seed 31, ALL 415 metrics
  B  primary panel, Q02-Q05 where valid, pilot fidelity (bank: 32 latents x 500 rows), seed 31, ALL metrics
  C  primary panel, Q01-Q05, 512 x 64 (bank 128 x 500), seeds 31/73/101, priority panel (configs/priority_panel.json)
  D  primary panel, Q01-Q05, 1024 x 128 (bank 512 x 500), seeds 31/73/101, escalated metrics (configs/escalation_D.json)
  E  secondary panel, Q01, 512 x 64, seeds 31/73/101, ALL metrics
Sources: oracle = AD of the generating formula (evaluation-only arm), nn = I01 AD of the frozen decoder (blind arm);
Q05 additionally oracle_leg (independent Gaussian T, legacy sensitivity; out-of-support draws excluded and counted).
Channels: class B Y-L and Y-P (one derivative pass), class C Y-C.  Masks: FULL, or MASK-ALL and MASK-COL for fixed banks.
Outputs (never overwritten; --resume skips finished files): outputs/metric_lab/07_unified_benchmark/W1/<toy run>/
  tensors/<Q>_<fid>_s<seed>.npz   base second moments per source|channel|mask (+ coverage, counts, validity)
  losses/<variant>/<Q>_<fid>_s<seed>/<channel>_<mask>.npz   loss per rotation x metric for every source, A/B blocks
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4_2 import common
from wave4_2.common import OUT, CONFIGS, FIDELITY, SEEDS, MEASURES, libraries, library, channels, git_commit, code_hash
import numpy as np
from wave4_2.inputs import load
from wave4_2.surfaces import DecoderSurface
from wave4_2.measures import blind_design, oracle_latent, col_mask, available
from wave4_2.derivatives import ad_chunks, channel_hessian
from wave4_2.moments import Moments, contract, jac, mask_modes
from wave4_2 import metrics as MT
from wave4_2.truth import oracle_eta, prior_draws, in_support

RUN = OUT / "W1"
BATCH = {"A": dict(panel="primary", Q=("Q01",), fid="pilot", seeds=(31,), metrics="all"),
         "B": dict(panel="primary", Q=("Q02", "Q03", "Q04", "Q05"), fid="pilot", seeds=(31,), metrics="all"),
         "C": dict(panel="primary", Q=MEASURES, fid="C", seeds=SEEDS, metrics="priority"),
         "D": dict(panel="primary", Q=MEASURES, fid="D", seeds=SEEDS, metrics="escalated"),
         "E": dict(panel="secondary", Q=("Q01",), fid="C", seeds=SEEDS, metrics="all")}


def lib_dir(lib):
    return RUN / lib["run"]


def metric_ids(kind):
    if kind == "all":
        return list(MT.ALL_IDS)
    f = CONFIGS / ("priority_panel.json" if kind == "priority" else "escalation_D.json")
    if not f.is_file():
        raise FileNotFoundError(f"{f.name} is frozen by analyze.py --freeze-{kind}; run batch A (and C for D) first")
    return list(json.loads(f.read_text())["ids"])


def compute_tensors(b, ev, dec_eta, Q, fid, seed):
    d = blind_design(b, Q, fid, seed)
    xs = (d["x"] - b.x_mean) / b.x_std; cm = col_mask(d["x"], b.x_domain); masks = mask_modes(b); chans = channels(b.cls)
    fo = oracle_eta(ev.family, b.x_mean, b.x_std); prior = lambda n, r: prior_draws(ev.family, n, r)
    arms = [("nn", d["lat"], dec_eta, None)]
    Tl, _ = oracle_latent(d, Q, ev.T, prior, seed, test_ids=b.splits["test"]); arms.append(("oracle", Tl, fo, None))
    if Q == "Q05":
        TL, _ = oracle_latent(d, Q, ev.T, prior, seed, legacy=True, test_ids=b.splits["test"]); ok, _ = in_support(ev.family, TL)
        arms.append(("oracle_leg", TL, fo, ok))
    res, meta = {}, dict(design=d["meta"], n_points=len(xs), coverage_all=float(cm.all(1).mean()), coverage_col=cm.mean(0).tolist())
    for name, lat, f, keep in arms:
        P = np.column_stack((lat, xs)); cmk = cm
        if keep is not None:
            meta["oracle_leg_out_of_support_frac"] = float(1 - keep.mean()); P = P[keep]; cmk = cm[keep]
        acc = {(c, m): Moments(b.k, b.p, m) for c in chans for m in masks}
        for sl, e, g, H in ad_chunks(f, P):
            for c in chans:
                Hc = channel_hessian(e, g, H, c)
                for m in masks:
                    acc[(c, m)].add(Hc, cmk[sl])
        for key, a in acc.items():
            res[(name,) + key] = a.finish()
    return res, meta, d


def save_tensors(path, res, meta):
    arr = {}
    for (s, c, m), r in res.items():
        for kk in ("K", "J", "nJ", "coverage"):
            arr[f"{s}|{c}|{m}|{kk}"] = np.asarray(r[kk])
        arr[f"{s}|{c}|{m}|scalars"] = np.array([r["nK"], r["n_points"], r["nonfinite"], float(r["valid"])])
    arr["meta"] = np.array(json.dumps(meta, default=float))
    path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **arr)


def load_tensors(path):
    f = np.load(path, allow_pickle=False); res = {}
    for key in f.files:
        if key == "meta":
            continue
        s, c, m, kk = key.split("|"); r = res.setdefault((s, c, m), {})
        if kk == "scalars":
            r["nK"], r["n_points"], r["nonfinite"], r["valid"] = f[key][0], f[key][1], f[key][2], bool(f[key][3])
        else:
            r[kk] = f[key]
    return res, json.loads(str(f["meta"]))


def losses(mo, rots, source, ids):
    """loss of every metric at every saved candidate for one source; the nn arm sees only (z_std, R) per candidate"""
    def J(r): return jac(r.t_std, r.G) if source.startswith("oracle") else jac(r.z_std, r.R)
    A0, B0 = contract(mo["K"], mo["J"], J(rots[0])); zero, EA, EB = MT.zero_flags(A0, B0)
    scale0 = rots[0].t_std if source.startswith("oracle") else rots[0].z_std
    st = MT.l03_null(mo["K"], mo["J"], scale0, contract, jac)
    full = len(ids) == len(MT.ALL_IDS)
    L = np.full((len(rots), len(ids)), np.nan); As = np.zeros((len(rots),) + A0.shape, np.float32); Bs = np.zeros((len(rots),) + B0.shape, np.float32)
    for i, r in enumerate(rots):
        A, B = contract(mo["K"], mo["J"], J(r)); As[i], Bs[i] = A, B
        L[i] = MT.evaluate_all(A, B, st, zero) if full else MT.evaluate_ids(A, B, st, zero, ids)
    return L, As, Bs, dict(zero_A=zero["A"], zero_B=zero["B"], EA=EA, EB=EB)


def manifest_rows(rots):
    return [dict(name=r.name, candidate_type=r.candidate_type, axes=list(r.axes), angle=float(r.angle)) for r in rots]


def run_library(lib, batch, variants, resume, seeds=None):
    spec = BATCH[batch]; ids = metric_ids(spec["metrics"]); t0 = time.time()
    b, ev, rots, src = load(lib["toy"]); dec = DecoderSurface(b).eta(); root = lib_dir(lib)
    (root).mkdir(parents=True, exist_ok=True)
    man = dict(toy=lib["toy"], cls=lib["cls"], family=lib["family"], panel=lib["panel"], run=lib["run"], beta=lib["beta"], hashes=b.hashes,
               git_commit=git_commit(), code_hash=code_hash(), dtype="float64", device="cpu")
    (root / "run_manifest.json").write_text(json.dumps(man, indent=1))
    for Q in spec["Q"]:
        if not available(b, Q):
            continue
        for seed in (seeds or spec["seeds"]):
            tag = f"{Q}_{spec['fid']}_s{seed}"; tpath = root / "tensors" / f"{tag}.npz"
            todo = [v for v in variants if v in rots and not all((root / "losses" / v / tag / f"{c}_{m}.npz").is_file()
                                                                   for c in channels(b.cls) for m in mask_modes(b))]
            if not todo and tpath.is_file():
                print(f"  [{lib['toy']}] {tag}: done, skipped"); continue
            if tpath.is_file():
                res, meta = load_tensors(tpath)
            else:
                res, meta, _ = compute_tensors(b, ev, dec, Q, spec["fid"], seed); save_tensors(tpath, res, meta)
            for v in todo:
                for c in channels(b.cls):
                    for m in mask_modes(b):
                        out = root / "losses" / v / tag / f"{c}_{m}.npz"
                        if out.is_file():
                            if not resume:
                                raise FileExistsError(f"{out} exists (no overwrite); use --resume")
                            continue
                        arr = dict(ids=np.array(ids), manifest=np.array(json.dumps(manifest_rows(rots[v]))))
                        for s in [s for s in ("oracle", "nn", "oracle_leg") if (s, c, m) in res]:
                            mo = res[(s, c, m)]
                            L, As, Bs, z = losses(mo, rots[v], s, ids)
                            arr[f"{s}|L"] = L; arr[f"{s}|A"] = As; arr[f"{s}|B"] = Bs
                            arr[f"{s}|info"] = np.array(json.dumps(dict(z, valid=bool(mo["valid"]), nK=int(mo["nK"]), n_points=int(mo["n_points"]),
                                                                        nonfinite=int(mo["nonfinite"]), coverage=np.asarray(mo["coverage"]).tolist()), default=float))
                        arr["meta"] = np.array(json.dumps(meta, default=float))
                        out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, **arr)
            print(f"  [{lib['toy']}] {tag}: {len(ids)} metrics, variants {todo} [{time.time()-t0:.0f}s]", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--toys", default="all"); ap.add_argument("--batch", required=True, choices=list(BATCH))
    ap.add_argument("--variant", default="both"); ap.add_argument("--seed", type=int, default=None); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv); spec = BATCH[a.batch]
    libs = libraries(spec["panel"]) if a.toys == "all" else [library(t) for t in a.toys.split(",")]
    variants = common.VARIANTS if a.variant == "both" else (a.variant,)
    for lib in libs:
        run_library(lib, a.batch, variants, a.resume, seeds=(a.seed,) if a.seed else None)


if __name__ == "__main__":
    main()
