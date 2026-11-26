"""W1-00: freeze inputs, audit them and run the numerical acceptance tests on the real artifacts.

usage: python validate.py [--toys all|C1,CR4L,...]
writes results/input_audit.csv, results/validation_checks.csv, results/fd_check.csv, results/registry_mapping.csv,
       results/w1_00_manifest.json.   Exit status 1 if any gate check fails.
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4_2 import common
from wave4_2.common import RES, libraries, channels, sha256, git_commit, code_hash, LAB03
import numpy as np, pandas as pd, torch
from wave4_2.inputs import load, blind_rotation
from wave4_2.surfaces import DecoderSurface
from wave4_2.truth import oracle_eta
from wave4_2.measures import blind_design, col_mask
from wave4_2.derivatives import ad_all, channel_hessian, fd_hessian, gfd_hessian
from wave4_2.moments import Moments, contract, jac, mask_modes
from wave4_2 import metrics as MT

EXPECTED = {3: 29, 4: 78, 6: 400}
checks = []


def check(toy, name, value, tol, ok=None, note=""):
    ok = bool(value <= tol) if ok is None else bool(ok)
    checks.append(dict(toy=toy, check=name, value=float(value) if value is not None else np.nan, tolerance=tol, passed=ok, note=note))
    return ok


def registry_consistency():
    sys.path.insert(0, str(LAB03)); from toys import TOYS
    reg = common.registry()["primary"]
    for t, (cls, run) in TOYS.items():
        check(t, "registry_matches_lab03_toys", 0, 0, ok=(t in reg and reg[t]["run"] == run and reg[t]["class"] == cls))
    for t, e in reg.items():
        doc = (common.ROOT / "docs" / "00_project_flow" / "02_test_beds.md").read_text(encoding="utf-8")
        check(t, "registry_run_listed_in_02_test_beds", 0, 0, ok=(e["run"] in doc))


def audit(lib):
    toy = lib["toy"]; row = dict(toy=toy, panel=lib["panel"], cls=lib["cls"], family=lib["family"], run=lib["run"], beta=lib["beta"])
    try:
        b, ev, rots, src = load(toy)
    except Exception as e:                                        # missing artifact: recorded, never silently skipped
        row.update(status=f"MISSING: {e}"); return row, None
    row.update(status="ok", k=b.k, p=b.p, subjects=len(b.counts), trials=len(b.x), fixed_bank=b.fixed_bank is not None, **{f"sha_{k}": v for k, v in b.hashes.items()})
    st = torch.load(src.checkpoint_file, map_location="cpu", weights_only=True)
    ck = st["subject_splits"]; sp = np.load(src.run_dir / "splits.npz")
    for part in ("train", "validation", "test"):
        same = np.array_equal(np.sort(np.asarray(ck[part])), np.sort(b.splits[part])) and np.array_equal(np.sort(sp[part]), np.sort(b.splits[part]))
        check(toy, f"split_{part}_S3=S5=splits.npz", 0, 0, ok=same)
    check(toy, "k_equals_dimT_and_registry", 0, 0, ok=(b.k == ev.T.shape[1] == lib["k"] and b.p == lib["p"]))
    xs = st["extra_state"]["input_standardization"]
    check(toy, "x_scaler_equals_checkpoint", float(np.max(np.abs(np.asarray(xs["x_mean"]) - b.x_mean))), 0.0)
    tr_rows = np.isin(b.subject_index, b.splits["train"])
    check(toy, "x_scaler_refit_on_TRAIN_rows", float(np.max(np.abs(b.x[tr_rows].mean(0) - b.x_mean) / b.x_std)), 1e-6)
    if b.cls == "C":
        ym = b.y[tr_rows].mean(); ysd = b.y[tr_rows].std()
        check(toy, "y_scaler_refit_on_TRAIN_rows", max(abs(ym - b.y_mean), abs(ysd - b.y_std) / b.y_std), 1e-6)
    else:
        check(toy, "binary_y_in_{0,1}", 0, 0, ok=set(np.unique(b.y)) <= {0.0, 1.0})
    # subject pairing sanity (evaluation only): held-out linear Z -> T recovery must be far above chance
    tr, te = b.splits["train"], b.splits["test"]; X1 = np.column_stack((np.ones(len(b.z)), b.z))
    W = np.linalg.lstsq(X1[tr], ev.T[tr], rcond=None)[0]; r2 = 1 - ((X1[te] @ W - ev.T[te]) ** 2).mean(0) / ev.T[te].var(0)
    row["min_test_linear_R2"] = float(r2.min())
    check(toy, "subject_pairing_linear_R2_min>0.3", -float(r2.min()), -0.3)
    for v in common.VARIANTS:
        if v not in rots:
            row[f"{v}_rotations"] = "MISSING"
            check(toy, f"{v}_present", 0, 0, ok=(lib["panel"] == "secondary"), note="secondary run: only S9 was produced (expected, recorded)" if lib["panel"] == "secondary" else "PRIMARY LIBRARY MISSING")
            continue
        rl = rots[v]; row[f"{v}_rotations"] = len(rl)
        subsets = {r.axes for r in rl if r.candidate_type == "sweep"}
        check(toy, f"{v}_rotation_count={EXPECTED[b.k]}", 0, 0, ok=(len(rl) == EXPECTED[b.k] and len(subsets) == 2 ** b.k - b.k - 1))
        check(toy, f"{v}_orthogonality", max(np.abs(r.R @ r.R.T - np.eye(b.k)).max() for r in rl), 1e-7)
        check(toy, f"{v}_R=G_Rstar", max(np.abs(r.G @ r.R_star - r.R).max() for r in rl), 1e-7)
    return row, (b, ev, rots)


def analytic_checks(toy, b, ev):
    fo = oracle_eta(ev.family, b.x_mean, b.x_std); d = blind_design(b, "Q01", "pilot", 31)
    T = ev.T[d["sid"]][:512]; xs = ((d["x"] - b.x_mean) / b.x_std)[:512]; P = np.column_stack((T, xs)); k = b.k
    _, _, H = ad_all(fo, P)
    if ev.family == "C1":                                         # eta = b1 + sqrt3 (b2 x1 + b3 x2 + b4 x3)
        ref = np.zeros_like(H)
        for j in range(3):
            ref[:, 1 + j, k + j] = ref[:, k + j, 1 + j] = np.sqrt(3) * b.x_std[j]
        check(toy, "analytic_C1_closed_form", float(np.abs(H - ref).max()), 1e-10)
    elif ev.family == "C2":                                       # linear in T; d/dX of each basis term, times x_std
        x = d["x"][:512]; q5 = np.sqrt(5) / 2; ref = np.zeros_like(H); s = b.x_std
        ref[:, 1, k + 0] = np.sqrt(3) * s[0]; ref[:, 1, k + 1] = -np.sqrt(3) * s[1]
        ref[:, 2, k + 0] = q5 * 6 * x[:, 0] * s[0]; ref[:, 2, k + 1] = -q5 * 6 * x[:, 1] * s[1]
        ref[:, 3, k + 2] = np.sqrt(3) * s[2]; ref[:, 3, k + 3] = -np.sqrt(3) * s[3]
        ref[:, k + 0, k + 0] = q5 * 6 * T[:, 2] * s[0] ** 2; ref[:, k + 1, k + 1] = -q5 * 6 * T[:, 2] * s[1] ** 2
        ref = ref + np.transpose(ref, (0, 2, 1)) - np.einsum("nii->ni", ref)[:, :, None] * np.eye(ref.shape[1])[None]   # symmetric, diagonal once
        check(toy, "analytic_C2_closed_form", float(np.abs(H - ref).max()), 1e-9)
    if ev.family.startswith("CR"):
        cm = col_mask(d["x"][:512], b.x_domain).all(1)
        Hf = fd_hessian(fo, P[cm], 1e-4, P.std(0))
        check(toy, f"analytic_{ev.family}_AD_vs_FD_interior", float(np.abs(Hf - H[cm]).max() / np.abs(H[cm]).max()), 1e-4)


def decoder_checks(toy, b, ev, rots, src):
    dec = DecoderSurface(b); fe = dec.eta()
    from s9curves.decoder import Decoder as L03Decoder                 # independent historical implementation
    te = np.isin(b.subject_index, b.splits["test"]); idx = np.sort(np.random.default_rng(8).choice(np.where(te)[0], 20000, replace=False))  # random TEST rows (a stride would hit a fixed subset of bank rows)
    xs = (b.x[idx] - b.x_mean) / b.x_std; z = b.z[b.subject_index[idx]]
    ours = dec.predict(z, xs); hist = L03Decoder(src).predict(z, b.x[idx])[0]
    check(toy, "decoder_equals_lab03_decoder", float(np.abs(ours - hist).max()), 1e-10)
    s4 = json.loads((src.run_dir / "s4_evaluate" / "metrics.json").read_text())
    if b.cls == "C":
        r2 = 1 - np.mean((ours - b.y[idx]) ** 2) / np.var(b.y[idx]); ref = s4.get("test", {}).get("r2", None) if isinstance(s4.get("test"), dict) else None
        check(toy, "decoder_R2_vs_observed_y_TEST", abs(r2 - 0.8), 0.03, note=f"R2={r2:.4f} (perfect predictor 0.8); S4 test r2={ref}")
    else:
        acc = np.mean((ours > 0) == (b.y[idx] > .5)); check(toy, "decoder_accuracy_TEST_>0.55", -acc, -0.55, note=f"acc={acc:.4f}")
    # contraction vs direct differentiation of the rotated function, and prediction invariance, at a random R (s9_2 coordinates)
    d = blind_design(b, "Q01", "pilot", 31); P = np.column_stack((d["lat"], (d["x"] - b.x_mean) / b.x_std))[:600]; k = b.k
    for v in rots:
        r = rots[v][0]; Q = np.linalg.qr(np.random.default_rng(5).standard_normal((k, k)))[0]
        for ch in channels(b.cls):
            e, g, H = ad_all(fe, P); Hc = channel_hessian(e, g, H, ch); m = Moments(k, b.p, "FULL"); m.add(Hc, np.ones((len(P), b.p), bool)); mo = m.finish()
            A, B = contract(mo["K"], mo["J"], jac(r.z_std, Q))
            zm, zs = torch.as_tensor(r.z_mean), torch.as_tensor(r.z_std); Qt = torch.as_tensor(Q)
            fch = (lambda w: torch.sigmoid(fe(w))) if ch == "YP" else fe
            gR = lambda w: fch(torch.cat((zm + zs * (Qt.T @ w[:k]), w[k:])))
            U = ((P[:, :k] - r.z_mean) / r.z_std) @ Q.T; PU = np.column_stack((U, P[:, k:]))
            _, _, Hg = ad_all(gR, PU)
            relA = np.abs((Hg[:, :k, :k] ** 2).mean(0) - A).max() / (A.max() + 1e-300); relB = np.abs((Hg[:, :k, k:] ** 2).mean(0) - B).max() / (B.max() + 1e-300)
            check(toy, f"{v}_{ch}_contraction_vs_direct", max(relA if A.max() > 1e-14 else 0, relB), 1e-7)
            inv = torch.max(torch.abs(torch.vmap(gR)(torch.as_tensor(PU)) - torch.vmap(fch)(torch.as_tensor(P)))).item()
            check(toy, f"{v}_{ch}_prediction_invariance", inv, 1e-9)
            # energy invariance over all saved candidates, signed-permutation invariance of the registry, blind-label independence
            EA, EB = [], []; z0, _, _ = MT.zero_flags(*contract(mo["K"], mo["J"], jac(r.z_std, rots[v][0].R)))
            for rr in rots[v]:
                A_, B_ = contract(mo["K"], mo["J"], jac(rr.z_std, rr.R)); EA.append(A_.sum()); EB.append(B_.sum())
            check(toy, f"{v}_{ch}_energy_invariance", max(np.ptp(EA) / (max(EA) + 1e-300) if max(EA) > 1e-14 else 0, np.ptp(EB) / max(EB)), 1e-9)
            st = MT.l03_null(mo["K"], mo["J"], r.z_std, contract, jac); R0 = rots[v][7].R
            perm = np.eye(k)[np.random.default_rng(3).permutation(k)] * np.random.default_rng(4).choice([-1, 1], k)[:, None]
            v1 = MT.evaluate_all(*contract(mo["K"], mo["J"], jac(r.z_std, R0)), st, z0); v2 = MT.evaluate_all(*contract(mo["K"], mo["J"], jac(r.z_std, perm @ R0)), st, z0)
            ok = np.isfinite(v1) & np.isfinite(v2); dev = np.abs(v1 - v2) / (np.abs(v1) + 1e-9)
            bad = [MT.ALL_IDS[i] for i in np.where(ok & (dev > 1e-6))[0]]
            check(toy, f"{v}_{ch}_signed_permutation_invariance", float(dev[ok].max()) if ok.any() else 0, 1e-6, note=";".join(bad[:10]))
            br = blind_rotation(rots[v][7]); v3 = MT.evaluate_all(*contract(mo["K"], mo["J"], jac(br.z_std, br.R)), st, z0)
            check(toy, f"{v}_{ch}_blind_score_label_independent", float(np.nanmax(np.abs(v1 - v3))) if np.isfinite(v1).any() else 0, 0.0)


def fd_check(toy, b, rows):
    dec = DecoderSurface(b); fe = dec.eta(); d = blind_design(b, "Q01", "pilot", 31)
    P = np.column_stack((d["lat"], (d["x"] - b.x_mean) / b.x_std))[:256]; x = d["x"][:256]; sd = P.std(0); k = b.k
    for ch in channels(b.cls):
        e, g, H = ad_all(fe, P); Had = channel_hessian(e, g, H, ch); scale = np.abs(Had).max()
        def ux(Hm): return (Hm[:, :k, k:] ** 2).mean(0)
        res = {}
        for name, fn in (("I02.FD0.01", lambda: fd_hessian(fe, P, .01, sd, ch)), ("I02.FD0.03", lambda: fd_hessian(fe, P, .03, sd, ch)),
                         ("I03.GFD0.03", lambda: gfd_hessian(fe, P, .03, sd, ch)), ("I02.FD0.1(hist)", lambda: fd_hessian(fe, P, .1, sd, ch))):
            Hm = fn(); res[name] = Hm
        if max(np.abs(res[n] - Had).max() / scale for n in ("I02.FD0.01", "I02.FD0.03")) > 1e-3:
            res["I02.FD0.003"] = fd_hessian(fe, P, .003, sd, ch)
        for name, Hm in res.items():
            step = float(name.split("FD")[-1].split("(")[0]); cross = np.zeros(len(x), bool)
            if b.x_domain is not None:                       # perturbation leaves the declared X domain (boundary validity)
                h = step * sd[k:] * b.x_std
                for j, dm in enumerate(b.x_domain):
                    lo = x[:, j] - h[j]; hi = x[:, j] + h[j]
                    cross |= (lo < 0) if dm == "amount" else ((lo < 0) | (hi > 1))
            rows.append(dict(toy=toy, channel=ch, method=name, points=len(P), max_abs_rel_err=float(np.abs(Hm - Had).max() / scale),
                             ux_moment_rel_err=float(np.abs(ux(Hm) - ux(Had)).max() / (ux(Had).max() + 1e-300)), frac_points_crossing_x_boundary=float(cross.mean())))


def registry_mapping():
    rows = [MT.describe(i) for i in MT.ALL_IDS]; df = pd.DataFrame(rows)
    rng = np.random.default_rng(11); vals = []
    for t in range(300):                                   # random energy blocks, both (k,p) shapes of the panel
        k, p = ((4, 4), (3, 6), (6, 4))[t % 3]
        A = rng.random((k, k)) ** 3; A = (A + A.T) / 2; B = rng.random((k, p)) ** 4; B[rng.random((k, p)) < .3] = 0
        st = dict(null={c: (0.0, 1.0) for c in MT.L03["_comp"]}); vals.append(MT.evaluate_all(A, B, st, dict(A=False, B=False)))
    V = np.array(vals); from scipy.stats import rankdata
    ok = np.isfinite(V).all(0) & (np.ptp(np.nan_to_num(V), 0) > 1e-12); Rk = rankdata(V, axis=0)
    eq = [""] * len(MT.ALL_IDS)
    idx = np.where(ok)[0]
    C = np.corrcoef(Rk[:, idx].T)
    for a_, i in enumerate(idx):
        same = [MT.ALL_IDS[idx[j]] for j in np.where(C[a_] > 1 - 1e-12)[0] if idx[j] != i]
        eq[i] = ";".join(same)
    df["numerically_rank_equivalent_to"] = eq
    df["equivalence_test"] = np.where(df.recipe.isin(["Fz", "CAND"]), "not tested (Haar-null dependent)", np.where(ok, "300 random blocks, Spearman = 1", "constant/undefined on random blocks"))
    df["status"] = "kept (no deduplication: names, normalisations and zero-block policies differ)"
    df.to_csv(RES / "registry_mapping.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--toys", default="all"); a = ap.parse_args(); t0 = time.time()
    RES.mkdir(exist_ok=True); registry_consistency()
    libs = libraries("primary") + libraries("secondary")
    if a.toys != "all":
        libs = [l for l in libs if l["toy"] in a.toys.split(",")]
    audit_rows, fd_rows = [], []
    for lib in libs:
        row, got = audit(lib); audit_rows.append(row)
        if got is None:
            continue
        b, ev, rots = got
        if lib["panel"] == "primary":
            _, _, _, src = load(lib["toy"])
            analytic_checks(lib["toy"], b, ev); decoder_checks(lib["toy"], b, ev, rots, src); fd_check(lib["toy"], b, fd_rows)
        print(f"[W1-00] {lib['toy']}: done [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(audit_rows).to_csv(RES / "input_audit.csv", index=False)
    pd.DataFrame(fd_rows).to_csv(RES / "fd_check.csv", index=False)
    ch = pd.DataFrame(checks); ch.to_csv(RES / "validation_checks.csv", index=False)
    mp = registry_mapping()
    man = dict(stage="W1-00", git_commit=git_commit(), code_hash=code_hash(), registry=common.registry(), n_metrics=len(MT.ALL_IDS),
               n_L03=len(MT.L03_NAMES), n_L06=len(MT.L06_NAMES), checks=int(len(ch)), failed=int((~ch.passed).sum()),
               torch=torch.__version__, numpy=np.__version__, dtype="float64", device="cpu")
    (RES / "w1_00_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(ch[~ch.passed].to_string() if (~ch.passed).any() else "ALL CHECKS PASSED", f"({len(ch)} checks)")
    print(pd.DataFrame(fd_rows).groupby("method")[["max_abs_rel_err", "ux_moment_rel_err"]].max().to_string())
    sys.exit(int((~ch.passed).any()))


if __name__ == "__main__":
    main()
