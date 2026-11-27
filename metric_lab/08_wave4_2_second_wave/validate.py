"""W2-00 gate: frozen snapshot, reproduction audit and acceptance checks (02_second_wave section 3.4).

usage: python validate.py      -> results/w2_manifest.json, config_snapshot.json, validation_checks.csv, reproduction_audit.csv
Checks (all must pass before results are interpreted; failures are recorded, never dropped):
  payload    no truth arrays in the blind payload; blind S9_2 frame == saved S8_2 normaliser; splits == S8 splits
  design     Lab 08 design(pool=TEST) == Lab 07 blind_design for Q01-Q05 (pilot, seed 31)
  moments    Lab 08 TEST Q01 C s31 decoder moments == Lab 07 nn tensors (every toy, channel, mask)
  objective  Lab 08 Objective(R) == Lab 07 saved nn losses at every saved S9 / S9_2 rotation (7 searched candidate ids, Q01 C s31)
  decoder    class C output scaling (out * y_std + y_mean) and prediction invariance under V -> R V -> R^T U
  status     Q01 pilot all-library flags of all 415 metrics recomputed from the saved Lab 07 losses == results/pass_summary.csv.gz;
             E01/E02 flags for every saved measure / fidelity / seed / mask; the first-wave claims (24/24, 8/24, 12/24, MASK-COL 4/6)
"""
import sys, json, glob, subprocess, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave2.common import OUT, OUT07, RES, LAB07, CONFIGS, PRIMARY, SECONDARY, VARIANTS, sha256, code_hash, git_commit, toy_entry, cfg
import numpy as np, pandas as pd, torch
from wave2.blind_inputs import load_payload, variant_frame, payload_path
from wave2.designs import design
from wave2.objectives import load_moments, Objective, CACHE, masks_of
from wave2.evaluate_recovery import EvalData
from wave4_2.measures import blind_design
from wave4_2.evaluation import sweeps, curve_stats, library_flags
from wave4_2.surfaces import DecoderSurface

CHECKS = []
CANDS = [c["metric"] for c in cfg("blind_candidates.json")["candidates"] if c["search"]]


def check(name, toy, ok, value=None, detail=""):
    CHECKS.append(dict(check=name, toy=toy, passed=bool(ok), value=value, detail=detail))
    if not ok:
        print(f"  FAIL {name} {toy}: {value} {detail}", flush=True)


def run_dir07(toy):
    return OUT07 / "W1" / toy_entry(toy)["run"]


def chan(pl):
    return "YC" if pl.cls == "C" else "YL"


def payload_checks(toy, pl, ed):
    with np.load(payload_path(toy)) as a:
        keys = set(a.files)
    check("payload_has_no_truth", toy, not (keys & {"parameters", "T", "t", "R_star"}), sorted(keys))
    for v in ed.refs:
        c, s = variant_frame(pl, v)
        check(f"blind_frame_{v}", toy, np.allclose(c, ed.refs[v]["z_mean"], rtol=1e-12, atol=1e-12) and np.allclose(s, ed.refs[v]["z_std"], rtol=1e-12, atol=1e-12),
              float(np.abs(s - ed.refs[v]["z_std"]).max()))
    check("test_split_equal", toy, np.array_equal(np.sort(pl.splits["test"]), np.sort(ed.test)))


def design_checks(toy, pl):
    for Q in ("Q01", "Q02", "Q03", "Q04", "Q05"):
        d8 = design(pl, Q, "pilot", 31, pool="test"); d7 = blind_design(pl, Q, "pilot", 31)
        if d7 is None or d8 is None:
            check(f"design_{Q}_availability", toy, (d7 is None) == (d8 is None)); continue
        check(f"design_{Q}_equals_lab07", toy, np.array_equal(d7["x"], d8["x"]) and np.array_equal(d7["lat"], d8["lat"]))


def moment_checks(toy, pl):
    tf = run_dir07(toy) / "tensors" / "Q01_C_s31.npz"; mf = CACHE / toy.replace("#", "_") / "test_Q01_C_s31.npz"
    if not mf.is_file() or not tf.is_file():
        check("moments_equal_lab07", toy, False, detail=f"missing {'cache' if not mf.is_file() else 'lab07 tensors'}"); return
    mine, _ = load_moments(mf); t = np.load(tf)
    for m in masks_of(pl):
        K7, J7 = t[f"nn|{chan(pl)}|{m}|K"], t[f"nn|{chan(pl)}|{m}|J"]; K8, J8 = mine[(chan(pl), m)]["K"], mine[(chan(pl), m)]["J"]
        err = max(np.abs(K7 - K8).max() / np.abs(K7).max(), np.abs(J7 - J8).max() / np.abs(J7).max())
        check(f"moments_equal_lab07_{m}", toy, err <= 1e-12, float(err))


def objective_checks(toy, pl):
    mf = CACHE / toy.replace("#", "_") / "test_Q01_C_s31.npz"
    if not mf.is_file():
        return
    mine, _ = load_moments(mf)
    for v in VARIANTS:
        for m in masks_of(pl):
            lf = run_dir07(toy) / "losses" / v / "Q01_C_s31" / f"{chan(pl)}_{m}.npz"
            if not lf.is_file():
                continue
            z = np.load(lf); ids = list(z["ids"]); Ls = z["nn|L"]
            rots = _rots(toy, v)
            sc = variant_frame(pl, v)[1]; worst = 0.0
            for mid in CANDS:
                j = ids.index(mid); L = Objective(mine[(chan(pl), m)], sc, mid)
                mine_v = np.array([L(r.R) for r in rots]); ref = Ls[:, j]
                ok = np.isfinite(ref)
                err = float(np.max(np.abs(mine_v[ok] - ref[ok]) / (np.abs(ref[ok]) + 1e-12))) if ok.any() else 0.0
                worst = max(worst, err); same_nan = bool(np.array_equal(np.isnan(mine_v), np.isnan(ref)))
                check(f"objective_equals_lab07_{v}_{m}_{mid}", toy, err <= 1e-9 and same_nan, err)


_ROT = {}


def _rots(toy, v):
    if (toy, v) not in _ROT:
        from s9curves.sources import load_sources, load_rotations
        e = toy_entry(toy); src = load_sources(e.get("model", toy), e["run_dir"]); _ROT[(toy, "s9")] = load_rotations(src, "s9")
        try:
            _ROT[(toy, "s9_2")] = load_rotations(src, "s9_2")
        except FileNotFoundError:
            _ROT[(toy, "s9_2")] = []
    return _ROT[(toy, v)]


def decoder_checks(toy, pl):
    dec = DecoderSurface(pl); rng = np.random.default_rng(3); idx = rng.choice(len(pl.x), 64, replace=False)
    z = pl.z[pl.subject_index[idx]]; xs = (pl.x[idx] - pl.x_mean) / pl.x_std; p0 = dec.predict(z, xs)
    with torch.no_grad():
        o = dec.net(torch.cat((torch.as_tensor(xs), torch.as_tensor(z)), 1)).numpy()
    manual = o[:, 0] * pl.y_std + pl.y_mean if pl.cls == "C" else o[:, 1] - o[:, 0]
    check("decoder_output_scaling", toy, np.allclose(p0, manual, rtol=1e-12, atol=1e-12))
    for v in VARIANTS:
        c, s = variant_frame(pl, v); R = np.linalg.qr(rng.standard_normal((pl.k, pl.k)))[0]
        U = ((z - c) / s) @ R.T; zb = c + s * (U @ R)
        check(f"prediction_invariance_{v}", toy, np.allclose(dec.predict(zb, xs), p0, rtol=1e-10, atol=1e-10))


def status_checks(ps):
    global PSI
    ps = ps[ps.analytic_source == "oracle"]
    PSI = {k: g for k, g in ps.groupby(["toy", "variant", "Q", "fidelity", "seed", "channel", "mask"])}
    rows = []
    for toy in PRIMARY + SECONDARY:
        e = toy_entry(toy); k = e["k"]; n_exp = 2 ** k - k - 1
        for f in sorted(glob.glob(str(run_dir07(toy) / "losses" / "*" / "*" / "*.npz"))):
            p = Path(f); v = p.parent.parent.name; Q, fid, seed = p.parent.name.split("_"); ch, mask = p.stem.split("_", 1)
            if ch not in ("YL", "YC"):
                continue
            z = np.load(f); ids = list(z["ids"]); man = json.loads(str(z["manifest"])); sw = sweeps(man)
            want = [i for i, m in enumerate(ids) if (Q == "Q01" and fid == "pilot") or m in ("L06|B08.M03.c0.01", "L06|B08.M03.c0.03")]
            if not want:
                continue
            fl = {}
            for src in ("oracle", "nn"):
                cs = curve_stats(z[f"{src}|L"][:, want], sw, "harmonised"); fl[src] = library_flags(cs, n_exp)
            g = PSI.get((toy, v, Q, fid, int(seed[1:]), ch, mask), pd.DataFrame(columns=["metric"])).set_index("metric")
            for jj, i in enumerate(want):
                m = ids[i]
                r = dict(toy=toy, variant=v, Q=Q, fidelity=fid, seed=int(seed[1:]), channel=ch, mask=mask, metric=m,
                         O_perfect=bool(fl["oracle"]["perfect"][jj]), O_marginal=bool(fl["oracle"]["marginal"][jj]), I_perfect=bool(fl["nn"]["perfect"][jj]),
                         I_marginal=bool(fl["nn"]["marginal"][jj]), I_strict_marginal=bool(fl["nn"]["strict_marginal"][jj]), O_strict_perfect=bool(fl["oracle"]["strict_perfect"][jj]))
                if m in g.index:
                    gr = g.loc[m]
                    r["match_pass_summary"] = all(bool(gr[c]) == r[c] for c in ("O_perfect", "O_marginal", "I_perfect", "I_marginal", "I_strict_marginal", "O_strict_perfect"))
                else:
                    r["match_pass_summary"] = np.nan
                rows.append(r)
        print(f"  status {toy}", flush=True)
    ra = pd.DataFrame(rows); ra.to_csv(RES / "reproduction_audit.csv", index=False)
    m = ra.match_pass_summary.dropna()
    check("status_reproduces_pass_summary", "all", bool(m.astype(bool).all()), f"{int(m.astype(bool).sum())}/{len(m)}")
    q = ra[(ra.Q == "Q01") & (ra.fidelity == "pilot") & ra.toy.isin(PRIMARY)]
    for mid in ("L06|B08.M03.c0.01", "L06|B08.M03.c0.03"):
        a = q[(q.metric == mid) & q["mask"].isin(["FULL", "MASK-ALL"])]
        check(f"claim_24_24_inferred_marginal_{mid}", "primary", int(a.I_marginal.sum()) == 24 and len(a) == 24, f"{int(a.I_marginal.sum())}/{len(a)}")
        check(f"claim_8_24_analytic_perfect_{mid}", "primary", int(a.O_perfect.sum()) == 8, int(a.O_perfect.sum()))
        check(f"claim_12_24_analytic_marginal_{mid}", "primary", int(a.O_marginal.sum()) == 12, int(a.O_marginal.sum()))
        c = q[(q.metric == mid) & (q["mask"] == "MASK-COL") & q.toy.isin(["CR3", "CR5"])]
        b = q[(q.metric == mid) & q.toy.isin(["CR4"]) & (q["mask"] == "FULL")]
        n = int(c.I_marginal.sum() + b.I_marginal.sum())
        check(f"claim_MASKCOL_B_CR_inferred_marginal_4_6_{mid}", "primary", n == 4, f"{n}/6")
    # seed-status changes at fidelity C (metric_evidence)
    for mid in ("L06|B08.M03.c0.01", "L06|B08.M03.c0.03"):
        a = ra[(ra.metric == mid) & (ra.fidelity == "C") & ra.toy.isin(PRIMARY)]
        ch = a.groupby(["toy", "variant", "Q", "channel", "mask"]).I_marginal.agg(lambda s: s.nunique() > 1).sum()
        CHECKS.append(dict(check=f"info_seed_status_changes_fidelityC_{mid}", toy="primary", passed=True, value=int(ch), detail="cells whose inferred marginal flag differs across seeds 31/73/101"))
    return ra


def main():
    t0 = time.time(); RES.mkdir(exist_ok=True)
    lab07_files = ["results/metric_evidence.csv", "results/next_wave_candidates.csv", "results/lane_rules.json", "results/global_oracle_summary.csv",
                   "results/report_tables.md", "results/pass_summary.csv.gz", "results/global_oracle_search.csv.gz", "configs/priority_panel.json",
                   "configs/toy_registry.json", "configs/escalation_D.json"]
    man = dict(run_id=OUT.name, created=time.strftime("%Y-%m-%d %H:%M:%S"), git_commit=git_commit(), code_hash=code_hash(),
               lab07_inputs={f: sha256(LAB07 / f) for f in lab07_files}, lab07_code={p.name: sha256(p) for p in sorted((LAB07 / "wave4_2").glob("*.py"))},
               configs={p.name: sha256(p) for p in sorted(CONFIGS.glob("*.json"))}, toys={})
    for toy in PRIMARY + SECONDARY:
        pl = load_payload(toy); ed = EvalData(toy)
        man["toys"][toy] = dict(hashes=pl.hashes, splits={k: int(len(v)) for k, v in pl.splits.items()}, fixed_bank=None if pl.fixed_bank is None else list(pl.fixed_bank.shape),
                                masks=list(masks_of(pl)), x_domain=pl.x_domain, variants_with_reference=list(ed.refs))
        payload_checks(toy, pl, ed); design_checks(toy, pl); decoder_checks(toy, pl)
        if toy in PRIMARY:
            moment_checks(toy, pl); objective_checks(toy, pl)
        print(f"  [{toy}] checks done [{time.time()-t0:.0f}s]", flush=True)
    ps = pd.read_csv(LAB07 / "results" / "pass_summary.csv.gz", low_memory=False)
    status_checks(ps)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True, cwd=Path(__file__).resolve().parent)
    check("unit_tests", "all", p.returncode == 0, p.stdout.strip().splitlines()[-1] if p.stdout else p.stderr[-300:])
    (RES / "w2_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    (RES / "config_snapshot.json").write_text(json.dumps({p.name: json.loads(p.read_text()) for p in sorted(CONFIGS.glob("*.json"))}, indent=1))
    df = pd.DataFrame(CHECKS); df.to_csv(RES / "validation_checks.csv", index=False)
    print(f"{len(df)} checks, {int((~df.passed).sum())} failed [{time.time()-t0:.0f}s]")
    sys.exit(1 if (~df.passed).any() else 0)


if __name__ == "__main__":
    main()
