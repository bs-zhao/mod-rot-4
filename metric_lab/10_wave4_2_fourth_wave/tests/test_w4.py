"""W4-00 unit tests (04_fourth_wave section 7).  Run through validate.py (records results/validation_checks.csv) or pytest."""
import sys, json, builtins
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from wave4.common import CAND, spec_of, chan, prim_mask, LAB
from wave4 import designs as DS, moments as MO
from wave3 import rowmetrics as RM
from wave3.objectives import ux_factory, build
from wave2.blind_inputs import load_payload, variant_frame
from wave4_2.derivatives import ad_all, fd_hessian
import run_blind as RB8

TOYS = ("CR4", "CR5", "C1L")


def _gl_from_text(B, lam=.75, c=.03):
    """section 3 formula written independently of rowmetrics.py"""
    B = np.asarray(B, float); k, p = B.shape; r = B.sum(1); E = r.sum()
    if not E > 1e-12:
        return np.nan
    A = r > 1e-15 * E; w = r[A] / r[A].sum(); q = B[A] / r[A, None]; s = (q / (q + c / p)).mean(1)
    v = B / E; G = (v / (v + c / (k * p))).mean()
    return lam * float((w * s).sum()) + (1 - lam) * G


def _haar(k, rng):
    Q, R = np.linalg.qr(rng.standard_normal((k, k))); return Q * np.sign(np.diag(R))


def test_gl_formula_and_invariances():
    rng = np.random.default_rng(1); sp = spec_of(CAND)
    for _ in range(200):
        k, p = rng.integers(3, 7), rng.integers(3, 7); B = rng.gamma(.5, 1, (k, p)) * (rng.random((k, p)) > .3)
        assert np.isclose(RM.value(sp, B), _gl_from_text(B), rtol=1e-12, atol=1e-14)
        P = np.eye(k)[rng.permutation(k)]; assert np.isclose(RM.value(sp, P @ B), RM.value(sp, B), rtol=1e-12)
        t = RM.terms(sp, B); assert np.isclose(t["L"], .75 * t["L_row"] + t["extra"], rtol=1e-13)
        assert np.isclose(t["extra"], .25 * RM.g_support(B, .03), rtol=1e-13)
    assert np.isnan(RM.value(sp, np.zeros((3, 4))))


def test_contraction_vs_direct_and_energy_invariance():
    """B_ij(R) from the base-frame J equals the mean squared mixed derivative of g_R = g(R^T u, x) computed directly by AD"""
    pl = load_payload("CR4"); eta = RB8.eta_factory(pl)(); d = DS.design(pl, "Q05", "C", 31, "train"); n = 256
    dd = dict(d, lat=d["lat"][:n], x=d["x"][:n], rows_per_latent=d["rows_per_latent"]); res, _ = MO.compute(pl, eta, dd, ("FULL",)); J = res[("YL", "FULL")]["J"]
    k = pl.k; rng = np.random.default_rng(2)
    for variant in ("s9", "s9_2"):
        cen, sc = variant_frame(pl, variant); R = _haar(k, rng); Bf = ux_factory(J, sc)
        V = (dd["lat"] - cen) / sc; U = V @ R.T; xs = (dd["x"] - pl.x_mean) / pl.x_std
        gR = lambda w: eta(torch.cat(((torch.as_tensor(R.T) @ w[:k]) * torch.as_tensor(sc) + torch.as_tensor(cen), w[k:])))
        _, _, H = ad_all(gR, np.column_stack((U, xs)), 256); Bd = (H[:, :k, k:] ** 2).mean(0)
        assert np.allclose(Bf(R), Bd, rtol=1e-9, atol=1e-14)
        assert np.isclose(Bf(R).sum(), Bf(np.eye(k)).sum(), rtol=1e-10) if variant == "s9" else True   # trace invariance in the raw frame
        assert np.allclose(Bf(R.T @ R), Bf(np.eye(k)), rtol=1e-12)


def test_ad_vs_fd():
    pl = load_payload("C1L"); eta = RB8.eta_factory(pl)(); d = DS.design(pl, "Q05", "C", 31, "train"); P = np.column_stack((d["lat"][:16], (d["x"][:16] - pl.x_mean) / pl.x_std))
    _, _, H = ad_all(eta, P, 16); Hf = fd_hessian(eta, P, 1e-3, np.ones(P.shape[1]))
    assert np.abs(H - Hf).max() < 1e-4 * max(1.0, np.abs(H).max())


def test_output_scaling_class_c():
    from wave4_2.surfaces import DecoderSurface
    pl = load_payload("C1L"); ds = DecoderSurface(pl); eta = ds.eta(); z = pl.z[:3]; xs = (pl.x[:3] - pl.x_mean) / pl.x_std
    v = torch.as_tensor(np.column_stack((z, xs))); a = np.array([float(eta(r)) for r in v]); b = ds.predict(z, xs)
    o = ds.net(torch.cat((torch.as_tensor(xs), torch.as_tensor(z)), 1))[:, 0].detach().numpy()
    assert np.allclose(a, b) and np.allclose(a, o * pl.y_std + pl.y_mean)


def test_q05_fixed_points_under_rotation():
    """the evaluation law of any R is the push-forward of the SAME base draws: L(R) from the cache == L recomputed from per-point squared
    derivatives at the fixed points (no refit in U coordinates)"""
    pl = load_payload("CR4"); res = MO.get(pl, "Q05", "C", 31, "train", blocks=True); J = res[0][("YL", "FULL")]["J"]; bl = res[2]
    Jb = bl["FULL|SJ"].sum(0) / bl["FULL|NJ"].sum(0)[:, None, None]; assert np.allclose(J, Jb, rtol=1e-12)
    L = build(spec_of(CAND), res[0][("YL", "FULL")], np.ones(3)); rng = np.random.default_rng(3)
    for _ in range(5):
        R = _haar(3, rng); assert np.isclose(L(R), RM.value(spec_of(CAND), ux_factory(Jb, np.ones(3))(R)), rtol=1e-12)


def test_designs():
    for toy in TOYS:
        pl = load_payload(toy); P = np.sort(pl.splits["train"]); Zp = pl.z[P]
        a, b = DS.design(pl, "Q05", "C", 31), DS.design(pl, "M-D", "C", 31)
        assert np.array_equal(a["lat"], b["lat"]) and np.array_equal(a["x"], b["x"])                 # M-D == historical Q05
        assert np.allclose(b["fit"]["mu"], Zp.mean(0)) and np.allclose(b["fit"]["sd"], Zp.std(0, ddof=0))
        b2 = DS.design(pl, "M-D", "C", 31); assert np.array_equal(b["lat"], b2["lat"])                 # deterministic replay
        c = DS.design(pl, "M-D", "C", 73); assert not np.array_equal(b["lat0"], c["lat0"])            # seed streams differ
        J = DS.design(pl, "M-J", "C", 31)
        S = np.cov(Zp, rowvar=False, ddof=0); Sh = DS.sqrtm_psd(S); assert np.allclose(Sh @ Sh, S, rtol=1e-10, atol=1e-12)
        e = (J["lat0"] - J["fit"]["mu"]) @ np.linalg.inv(Sh); assert np.allclose(e, (b["lat0"] - b["fit"]["mu"]) / b["fit"]["sd"], atol=1e-9)   # matched normals
        if pl.fixed_bank is not None:
            nb = len(pl.fixed_bank); assert b["rows_per_latent"] == nb and np.array_equal(b["x"][:nb], pl.fixed_bank) and np.array_equal(b["x"][nb:2 * nb], pl.fixed_bank)
            assert np.all(b["lat"][:nb] == b["lat"][0])
        else:
            assert np.array_equal(b["x"], DS.design(pl, "M-E", "C", 31)["x"])                          # common X panel
        I = DS.design(pl, "M-I", "C", 31); assert all(np.isin(I["lat0"][:, i], Zp[:, i]).all() for i in range(pl.k))
        T = DS.design(pl, "TRN-2", "C", 31); assert (np.abs((T["lat0"] - T["fit"]["mu"]) / T["fit"]["sd"]) <= 2 + 1e-12).all()


def test_mask_denominators():
    pl = load_payload("CR5"); res, _ = MO.get(pl, "Q05", "C", 31, "train")
    from wave4_2.measures import col_mask
    d = DS.design(pl, "Q05", "C", 31, "train"); cm = col_mask(d["x"], pl.x_domain)
    assert np.all(res[("YL", "MASK-ALL")]["nJ"] == cm.all(1).sum()); assert np.array_equal(res[("YL", "MASK-COL")]["nJ"], cm.sum(0))
    assert np.all(res[("YL", "FULL")]["nJ"] == len(d["x"]))


def test_blind_isolation():
    """a blind Q05 search runs with every truth-bearing file unreadable (audit hook)"""
    banned = ("eval_only", "s1_generate__dataset", "s8_supervised_rotation", "s8_2_supervised_rotation", "s9_rotation_sweep", "s9_2_rotation_sweep", "tensors")
    def hook(ev, args):
        if ev == "open" and isinstance(args[0], (str, Path)) and any(b in str(args[0]) for b in banned):
            raise PermissionError(f"blind code opened {args[0]}")
    sys.addaudithook(hook)
    from wave4.search import objective
    pl = load_payload("CR4"); L, _ = objective(pl, CAND, "Q05", "C", 31, "FULL", "s9"); r = RB8.run_cell(L, 3, 31, "S0"); assert np.isfinite(r["f"])
    builtins.__dict__.setdefault("_w4_audit_installed", True)


def test_static_import_split():
    """blind modules never import truth-enabled code"""
    bad = ("eval_only", "evaluate_recovery", "wave4_2.truth", "wave4_2.inputs", "wave4_2.bridge", "global_oracle")
    for f in ("common", "designs", "moments", "search", "reeval"):
        for line in (LAB / "wave4" / f"{f}.py").read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t.startswith(("import ", "from ")):
                assert not any(b in t for b in bad), (f, t)
