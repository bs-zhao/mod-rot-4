"""W2-00 unit tests (02_second_wave section 3.4).  Run: python -m pytest -q tests"""
import sys, os, subprocess, textwrap, ast
from pathlib import Path
import numpy as np
import pytest
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from wave2.common import OUT, LAB07
from wave2.optimizers import haar, d_sp, rel_dist, refine, search, givens
from wave2.objectives import contract_factory, Objective
from wave4_2.moments import contract, jac
from wave4_2 import metrics as MT
from wave1.library import metrics as l06_metrics

BLIND_MODULES = ["wave2/blind_inputs.py", "wave2/designs.py", "wave2/objectives.py", "wave2/optimizers.py", "wave2/conditional_blind.py",
                 "wave2/surrogates.py", "wave2/denoise.py", "run_blind.py"]
TRUTH_NAMES = {"truth", "bridge", "global_oracle", "run_global_oracle", "evaluate_recovery", "oracle_diagnostics", "EvalOnly", "oracle_eta",
               "prior_draws", "eval_only", "extract_eval", "EvalData"}


def _rand_moments(k, p, seed=0):
    rng = np.random.default_rng(seed); H = rng.standard_normal((500, k + p, k + p)); H = (H + H.transpose(0, 2, 1)) / 2
    K = np.einsum("nab,ncd->abcd", H[:, :k, :k], H[:, :k, :k]) / 500
    J = np.stack([H[:, :k, k + j].T @ H[:, :k, k + j] / 500 for j in range(p)])
    return dict(K=K, J=J, valid=True), H


def test_haar_orthogonal_and_both_components():
    Q = haar(4, 200, np.random.default_rng(1))
    assert np.allclose(np.einsum("nij,nkj->nik", Q, Q), np.eye(4)[None], atol=1e-12)
    dets = np.sign(np.linalg.det(Q)); assert (dets > 0).any() and (dets < 0).any()


def test_contract_factory_equals_reference():
    for k, p in ((3, 6), (4, 3), (6, 4)):
        mo, _ = _rand_moments(k, p); s = np.random.default_rng(2).uniform(.5, 2, k)
        for Q in haar(k, 5, np.random.default_rng(3)):
            A1, B1 = contract_factory(mo["K"], mo["J"], s)(Q); A2, B2 = contract(mo["K"], mo["J"], jac(s, Q))
            assert np.allclose(A1, A2, rtol=1e-10, atol=1e-13) and np.allclose(B1, B2, rtol=1e-10, atol=1e-13)


def test_contraction_equals_direct_rotation_of_hessians():
    k, p = 4, 3; mo, H = _rand_moments(k, p, 5); s = np.array([1., 2., .5, 1.5]); R = haar(k, 1, np.random.default_rng(9))[0]
    Jz = jac(s, R); Huu = np.einsum("ia,nij,jb->nab", Jz, H[:, :k, :k], Jz); Hux = np.einsum("ia,nix->nax", Jz, H[:, :k, k:])
    A = np.mean(Huu ** 2, 0); B = np.mean(Hux ** 2, 0); A2, B2 = contract_factory(mo["K"], mo["J"], s)(R)
    assert np.allclose(A, A2, rtol=1e-10) and np.allclose(B, B2, rtol=1e-10)


def test_signed_permutation_invariance_of_objectives():
    mo, _ = _rand_moments(4, 3, 7); s = np.ones(4); R = haar(4, 1, np.random.default_rng(4))[0]
    rng = np.random.default_rng(11)
    for mid in ("L06|B08.M03.c0.01", "L06|B08.M03.c0.03", "L03|UX.lq.1", "L06|B01.M03.c0.01", "L03|Fz.A+B+C", "L06|B09.w75.M03.c0.01"):
        L = Objective(mo, s, mid); l0 = L(R)
        for _ in range(5):
            P = np.eye(4)[rng.permutation(4)] * rng.choice([-1, 1], 4)[:, None]
            assert abs(L(P @ R) - l0) <= 1e-9 * (abs(l0) + 1e-12), mid


def test_fast_single_metric_equals_full_library():
    rng = np.random.default_rng(13)
    for _ in range(20):
        A = rng.exponential(1, (4, 4)); A = (A + A.T) / 2; B = rng.exponential(1, (4, 3)); B[rng.random(B.shape) < .3] = 0
        zero = dict(A=False, B=False); full = l06_metrics(A, B, zero)
        for name in ("B08.M03.c0.01", "B08.M03.c0.03", "B01.M03.c0.01", "B09.w75.M03.c0.01", "CTRL.EB"):
            v1, v2 = MT.single_l06(name)(A, B, zero), full[name]
            assert (np.isnan(v1) and np.isnan(v2)) or abs(v1 - v2) <= 1e-12 * (abs(v2) + 1e-12), name


def test_invariant_control_constant_and_zero_block_undefined():
    mo, _ = _rand_moments(3, 6, 17); L = Objective(mo, np.ones(3), "L06|CTRL.EB")
    vals = [L(Q) for Q in haar(3, 20, np.random.default_rng(2))]; assert np.ptp(vals) <= 1e-9 * abs(vals[0])
    mo0 = dict(mo, J=np.zeros_like(mo["J"])); L0 = Objective(mo0, np.ones(3), "L06|B08.M03.c0.01")
    assert np.isnan(L0(np.eye(3)))


def test_b08_active_row_rule_tiny_rows():
    """rows below 1e-12 * 1e-3 of the block energy are inactive; a tiny row never yields a favourable zero"""
    B = np.array([[1., 0, 0], [0, 1., 0], [1e-20, 1e-20, 1e-20]]); A = np.eye(3); zero = dict(A=False, B=False)
    v = MT.single_l06("B08.M03.c0.01")(A, B, zero); B2 = B.copy(); B2[2] = 0
    assert abs(v - MT.single_l06("B08.M03.c0.01")(A, B2, zero)) < 1e-12


def test_search_seed_reproducible_and_label_free():
    mo, _ = _rand_moments(3, 6, 19); L = Objective(mo, np.ones(3), "L06|B08.M03.c0.01")
    r1 = search(L, 3, 31); r2 = search(L, 3, 31)
    assert np.array_equal(r1["R"], r2["R"]) and r1["f"] == r2["f"]
    assert abs(L(r1["R"]) - r1["f"]) < 1e-12                      # the score depends only on the matrix


def test_refine_finds_planted_axis():
    """objective with a known minimum (distance to a planted rotation modulo signed permutations) is recovered"""
    Q0 = haar(3, 1, np.random.default_rng(8))[0]; L = lambda Q: rel_dist(Q, Q0)
    r = search(L, 3, 5); assert rel_dist(r["R"], Q0) < 1e-3


def test_rel_dist_signed_perm_zero():
    rng = np.random.default_rng(1); Q = haar(4, 1, rng)[0]; P = np.eye(4)[[2, 0, 3, 1]] * np.array([1, -1, 1, -1])[:, None]
    assert rel_dist(P @ Q, Q) < 1e-7 and d_sp(np.eye(4)) == 0


def test_blind_modules_do_not_import_truth():
    for m in BLIND_MODULES:
        f = LAB / m
        if not f.is_file():
            continue
        tree = ast.parse(f.read_text(encoding="utf-8")); names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[-1] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names |= {(node.module or "").split(".")[-1]} | {a.name for a in node.names}
        assert not (names & TRUTH_NAMES), (m, names & TRUTH_NAMES)


def test_truth_file_access_denied_during_blind_objective_and_search():
    """runs payload loading, decoder moments, objective and a short search under an audit hook that forbids the S1 dataset, every S8/S9
    directory, the evaluation-only files and the Lab 07 tensors / losses.  The hook must also demonstrably catch a forbidden open."""
    toy = "CR4"
    pay = OUT / "blind_payload" / f"{toy}.npz"
    if not pay.is_file():
        pytest.skip("payload not prepared")
    code = textwrap.dedent(f"""
        import sys, os, numpy as np
        sys.path.insert(0, r"{LAB}")
        from wave2.common import ROOT, OUT, OUT07, NNOUT
        nc = lambda q: os.path.normcase(os.path.abspath(q))
        FORB = [nc(ROOT / "outputs" / "models" / "generate"), nc(OUT / "eval_only"), nc(OUT07)]
        TOK = [os.sep + "s8_", os.sep + "s9_", os.sep + "s5_extract_z"]
        def hook(ev, args):
            if ev != "open" or not args or isinstance(args[0], int):
                return
            try:
                p = nc(os.fsdecode(args[0]))
            except Exception:
                return
            if any(p.startswith(f) for f in FORB) or any(t in p for t in TOK):
                raise PermissionError("FIREWALL " + p)
        sys.addaudithook(hook)
        from wave2.blind_inputs import load_payload, variant_frame
        from wave2.designs import design
        from wave2.objectives import compute_moments, Objective
        from wave2.optimizers import search
        from wave4_2.surfaces import DecoderSurface
        pl = load_payload("{toy}")
        d = design(pl, "Q01", "pilot", 31, pool="train", sizes=(16, 8, 4))
        mo = compute_moments(pl, DecoderSurface(pl).eta(), d, ("YL",), ("FULL",))[("YL", "FULL")]
        L = Objective(mo, variant_frame(pl, "s9_2")[1], "L06|B08.M03.c0.01")
        r = search(L, pl.k, 31)
        print("BLIND_OK", r["f"])
        try:
            np.load(OUT / "eval_only" / "{toy}.npz")
            print("HOOK_FAILED")
        except PermissionError:
            print("HOOK_CAUGHT")
    """)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=900)
    assert "BLIND_OK" in p.stdout and "HOOK_CAUGHT" in p.stdout, p.stdout + p.stderr


def test_conditional_moments_per_row_per_latent_and_groups():
    """per-row, per-latent and grouped moments of a known function equal direct computation; pooled reconstruction is exact"""
    import torch
    from wave2.conditional import product_moments, group_moments, kmeans_groups
    k, p = 3, 2
    def eta(v):
        u, x = v[:k], v[k:]
        return (u[0] ** 2) * x[0] + u[1] * u[2] * x[1] + torch.sin(u[0] * x[1])
    rng = np.random.default_rng(0); lat = rng.standard_normal((5, k)); X = rng.standard_normal((7, p))
    pm = product_moments(eta, lat, X, np.zeros(p), np.ones(p), None)
    H = np.zeros((5, 7, k + p, k + p))
    for l in range(5):
        for r in range(7):
            H[l, r] = torch.func.hessian(eta)(torch.as_tensor(np.r_[lat[l], X[r]])).numpy()
    Jl = np.einsum("lrax,lrbx->lxab", H[:, :, :k, k:], H[:, :, :k, k:]) / 7
    Jr = np.einsum("lrax,lrbx->rxab", H[:, :, :k, k:], H[:, :, :k, k:]) / 5
    assert np.allclose(pm["per_lat"]["FULL"][1], Jl) and np.allclose(pm["J_r"], Jr)
    lab = kmeans_groups(X, 2, min_group=1); gm = group_moments(pm, lab, "FULL")
    w = np.array([g[2] for g in gm], float); Kp = sum(g[0] * wi for g, wi in zip(gm, w)) / w.sum()
    assert np.allclose(Kp, pm["K_r"].mean(0))
