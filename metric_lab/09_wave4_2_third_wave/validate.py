"""W3-00 gate: frozen snapshot, exact replay of the second-wave baselines and acceptance checks (03_third_wave section 4).

usage: python validate.py [--refit]     -> results/w3_manifest.json, config_snapshot.json, validation_checks.csv, replay_audit.csv
Checks (failures are recorded, never dropped):
  1 counts      E01 / E02 BS-B 15/24, 17/24 libraries; E02 S9_2 10/12 toys; X01 / X02 5/12, 8/12 (S9) and 5/12, 7/12 (S9_2) over 3 search seeds
  2 B08 status  inferred marginal 24/24, analytic perfect 8/24, analytic marginal 12/24 (Q01 pilot), MASK-COL B-CR 4/6, recomputed from the
                saved Lab 07 losses with the Lab 07 flag code
  3 loss replay every W2 BS-B E01/E02 and BS-SL X01/X02 selected R re-scored on its frozen cache == archived loss (rel 1e-10);
                RW-POW(a=1, c) == B08.M03.c on every selected R
  4 RBF state   the saved X02 RBF state reproduces the saved TRAIN / VALIDATION D1 moment caches (rel 1e-9); with --refit the deterministic
                fit replay (Lab 08 IF-B code, S-min) reproduces the saved coefficients
  5 search      Lab 09 MX-A replays of E02 / X01 / X02 (seed 31 caches, search seeds 31/73/101) are bit-identical to the W2 rotations
  6-7 tests     tests/test_w3.py (formulas, transport, invariance, zero/undefined, activation continuity, FD vs AD, pair reconstruction, firewall)
"""
import sys, json, glob, subprocess, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import OUT, OUT07, OUT08, RES, LAB07, LAB08, CONFIGS, PRIMARY, VARIANTS, sha256, code_hash, git_commit, toy_entry, chan, prim_mask
import numpy as np, pandas as pd
from wave3 import rowmetrics as RM, jobs as JB
from wave2.blind_inputs import load_payload, variant_frame
from wave2.objectives import Objective, load_moments
from wave4_2.evaluation import sweeps, curve_stats, library_flags

CHECKS, AUDIT = [], []
R8 = LAB08 / "results"


def check(name, toy, ok, value=None, detail=""):
    CHECKS.append(dict(check=name, toy=toy, passed=bool(ok), value=value, detail=detail))
    if not ok:
        print(f"  FAIL {name} {toy}: {value} {detail}", flush=True)


def counts():
    br = pd.read_csv(R8 / "blind_recovery.csv")
    b = br[br.batch == "BS-B"]
    for cid, want in (("E01", 15), ("E02", 17)):
        g = b[b.cid == cid].groupby(["toy", "variant"]).pass_lab05.agg(["all", "size"])
        check(f"replay_count_{cid}_BS-B_libraries", "primary", int(g["all"].sum()) == want and (g["size"] == 9).all(), f"{int(g['all'].sum())}/24")
    g = b[(b.cid == "E02") & (b.variant == "s9_2")].groupby("toy").pass_lab05.all(); check("replay_count_E02_S9_2_toys", "primary", int(g.sum()) == 10, f"{int(g.sum())}/12")
    s = br[br.batch == "BS-SL"]
    for cid, w9, w92 in (("X01", 5, 5), ("X02", 8, 7)):
        g = s[s.cid == cid].groupby(["variant", "toy"]).pass_lab05.agg(["all", "size"])
        for v, want in (("s9", w9), ("s9_2", w92)):
            n = int(g.loc[v]["all"].sum()); check(f"replay_count_{cid}_{v}_toys", "primary", n == want and (g.loc[v]["size"] == 3).all(), f"{n}/12 (3 search seeds, not 9 runs)")
    for (cid, v), g in b[b.cid.isin(["E01", "E02"])].groupby(["cid", "variant"]):
        AUDIT.append(dict(item="BS-B toy pass", cid=cid, variant=v, value=int(g.groupby("toy").pass_lab05.all().sum()), denominator=12))


def b08_status():
    rows = []
    for toy in PRIMARY:
        e = toy_entry(toy); k = e["k"]; n_exp = 2 ** k - k - 1
        for f in sorted(glob.glob(str(OUT07 / "W1" / e["run"] / "losses" / "*" / "Q01_pilot_s31" / "*.npz"))):
            p = Path(f); v = p.parent.parent.name; ch, mask = p.stem.split("_", 1)
            if ch not in ("YL", "YC"):
                continue
            z = np.load(f); ids = list(z["ids"]); sw = sweeps(json.loads(str(z["manifest"])))
            for mid in ("L06|B08.M03.c0.01", "L06|B08.M03.c0.03"):
                j = ids.index(mid); fo = library_flags(curve_stats(z["oracle|L"][:, [j]], sw), n_exp); fi = library_flags(curve_stats(z["nn|L"][:, [j]], sw), n_exp)
                rows.append(dict(toy=toy, variant=v, mask=mask, metric=mid, O_perfect=bool(fo["perfect"][0]), O_marginal=bool(fo["marginal"][0]), I_marginal=bool(fi["marginal"][0])))
    ra = pd.DataFrame(rows)
    for mid in ("L06|B08.M03.c0.01", "L06|B08.M03.c0.03"):
        a = ra[(ra.metric == mid) & ra["mask"].isin(["FULL", "MASK-ALL"])]
        check(f"b08_inferred_marginal_24_{mid}", "primary", int(a.I_marginal.sum()) == 24 and len(a) == 24, f"{int(a.I_marginal.sum())}/{len(a)}")
        check(f"b08_analytic_perfect_8_{mid}", "primary", int(a.O_perfect.sum()) == 8, int(a.O_perfect.sum()))
        check(f"b08_analytic_marginal_12_{mid}", "primary", int(a.O_marginal.sum()) == 12, int(a.O_marginal.sum()))
        c = ra[(ra.metric == mid) & (ra["mask"] == "MASK-COL") & ra.toy.isin(["CR3", "CR5"])]; bb = ra[(ra.metric == mid) & (ra.toy == "CR4")]
        n = int(c.I_marginal.sum() + bb.I_marginal.sum()); check(f"b08_maskcol_BCR_inferred_marginal_4_6_{mid}", "primary", n == 4, f"{n}/6")
    return ra


def loss_replay():
    worst = {}
    for toy in PRIMARY:
        pl = load_payload(toy); c = chan(pl)
        for batch, cid, metric in (("BS-B", "E01", "L06|B08.M03.c0.01"), ("BS-B", "E02", "L06|B08.M03.c0.03"), ("BS-SL", "X01", "L03|SL0.UXUU.e0.01"), ("BS-SL", "X02", "L03|SL0.UXUU.e0.01")):
            f = OUT08 / "blind" / batch / toy / f"{cid}.npz"; z = np.load(f); diag = json.loads(str(z["diag"])); mc = {}
            for key, d in diag.items():
                if d["cache"] not in mc:
                    mc[d["cache"]] = load_moments(OUT08 / "moment_cache" / toy / f"{d['cache']}.npz")[0][(c, d["mask"])]
                sc = variant_frame(pl, d["variant"])[1]; R = z[f"{key}|R"]; L = Objective(mc[d["cache"]], sc, metric); val = L(R)
                err = abs(val - d["f"]) / (abs(d["f"]) + 1e-12); worst[(cid, toy)] = max(worst.get((cid, toy), 0), err)
                if metric.startswith("L06|B08"):
                    cc = float(metric.split(".c")[-1]); RO = JB.build(dict(type="row", kind="POW", a=1.0, c=cc), mc[d["cache"]], sc)
                    e2 = abs(RO(R) - val); worst[("RWPOW", toy)] = max(worst.get(("RWPOW", toy), 0), e2)
        for (cid, t), e in worst.items():
            if t == toy:
                check(f"loss_replay_{cid}", toy, e <= 1e-10, e)
        print(f"  loss replay {toy}", flush=True)


def rbf_state(refit=False):
    import run_inference as RI
    from wave2 import surrogates as S
    from wave2.designs import design
    from wave2.objectives import masks_of
    for toy in PRIMARY:
        pl = load_payload(toy); c = chan(pl); st = OUT08 / "inference" / "estimator_fits" / f"{toy}_RBF-block_D1_Smin.npz"
        z = np.load(st); est = S.RBFEta(z["m"], z["s"], z["C"], float(z["h"]), z["a"], float(z["b"]))
        for pool in ("train", "validation"):
            d = design(pl, "Q01", "C", 31, pool=pool); ref = load_moments(OUT08 / "moment_cache" / toy / f"{pool}_Q01_C_s31_rbfD1.npz")[0]
            mine = RI.moments_from_H(lambda P: est.hessian(P), pl, d, masks_of(pl))
            m = prim_mask(pl); err = max(np.abs(mine[m]["K"] - ref[(c, m)]["K"]).max() / np.abs(ref[(c, m)]["K"]).max(), np.abs(mine[m]["J"] - ref[(c, m)]["J"]).max() / np.abs(ref[(c, m)]["J"]).max())
            check(f"rbf_state_reproduces_{pool}_cache", toy, err <= 1e-9, float(err))
        if refit:
            D = RI.data(pl); fits = RI.fit_rbf_family(pl, D, "block", ("D1",)); f = min(fits, key=lambda g: g["val_loss"]); e2 = f["est"]
            err = float(np.abs(e2.a - est.a).max() / np.abs(est.a).max()); check("rbf_refit_replay_coefficients", toy, err <= 1e-6 and abs(e2.h - est.h) < 1e-12, err,
                                                                                f"val_loss {f['val_loss']:.6g} vs saved {float(z['val_loss']):.6g}")
        print(f"  rbf {toy}", flush=True)


def search_replay():
    pairs = (("MX-B08c03@AD", "BS-B", "E02"), ("MX-SL0UXUU@AD", "BS-SL", "X01"), ("MX-SL0UXUU@RBF", "BS-SL", "X02"))
    for toy in PRIMARY:
        for cid9, b8, cid8 in pairs:
            f9 = JB.job_path("MX-A", toy, cid9); f8 = OUT08 / "blind" / b8 / toy / f"{cid8}.npz"
            if not f9.is_file():
                check(f"search_replay_{cid8}", toy, False, detail="Lab 09 MX-A job missing"); continue
            z9, z8 = np.load(f9), np.load(f8); d9 = json.loads(str(z9["diag"])); same, n = True, 0
            for key in d9:
                if f"{key}|R" in z8.files:
                    n += 1; same &= bool(np.array_equal(z9[f"{key}|R"], z8[f"{key}|R"]))
            check(f"search_replay_{cid8}", toy, same and n == 6, f"{n} keys bit-identical={same}")
            AUDIT.append(dict(item="search replay", cid=cid8, toy=toy, value=bool(same), denominator=n))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--refit", action="store_true"); ap.add_argument("--skip-replay", action="store_true"); a = ap.parse_args()
    t0 = time.time(); RES.mkdir(exist_ok=True)
    lab08_files = ["results/blind_recovery.csv", "results/blind_reference_loss.csv", "results/oracle_metric_practical.csv", "results/inference_estimator_comparison.csv",
                   "results/next_wave_candidates.csv", "configs/blind_candidates.json", "run_shortlist.py"]
    man = dict(run_id=OUT.name, created=time.strftime("%Y-%m-%d %H:%M:%S"), git_commit=git_commit(), code_hash=code_hash(),
               lab08_inputs={f: sha256(LAB08 / f) for f in lab08_files}, configs={p.name: sha256(p) for p in sorted(CONFIGS.glob("*.json"))},
               w2_artifacts=dict(payloads={t: sha256(OUT08 / "blind_payload" / f"{t}.npz") for t in PRIMARY},
                                 rbf_states={t: sha256(OUT08 / "inference" / "estimator_fits" / f"{t}_RBF-block_D1_Smin.npz") for t in PRIMARY}),
               rng="search [seed,77,k]; Haar calibration [20261127,9,k]; grid [seed,101/102/103]", dtype="float64 CPU", missing_artifacts=[])
    counts(); print(f"  counts [{time.time()-t0:.0f}s]", flush=True)
    b08_status(); print(f"  b08 status [{time.time()-t0:.0f}s]", flush=True)
    loss_replay(); rbf_state(a.refit)
    if not a.skip_replay:
        search_replay()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True, cwd=Path(__file__).resolve().parent)
    check("unit_tests", "all", p.returncode == 0, p.stdout.strip().splitlines()[-1] if p.stdout else p.stderr[-300:])
    (RES / "w3_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    (RES / "config_snapshot.json").write_text(json.dumps({p.name: json.loads(p.read_text()) for p in sorted(CONFIGS.glob("*.json"))}, indent=1))
    pd.DataFrame(CHECKS).to_csv(RES / "validation_checks.csv", index=False); pd.DataFrame(AUDIT).to_csv(RES / "replay_audit.csv", index=False)
    df = pd.DataFrame(CHECKS); print(f"{len(df)} checks, {int((~df.passed).sum())} failed [{time.time()-t0:.0f}s]")
    sys.exit(1 if (~df.passed).any() else 0)


if __name__ == "__main__":
    main()
