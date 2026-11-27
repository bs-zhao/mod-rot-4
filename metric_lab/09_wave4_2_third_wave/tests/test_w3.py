"""W3-00 unit tests (03_third_wave section 4, items 5-7).  Run: python -m pytest -q tests"""
import sys, os, subprocess, textwrap, ast
from pathlib import Path
import numpy as np
import pytest
import torch
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from wave3.common import OUT, OUT08
from wave3 import rowmetrics as RM, structure as ST, fusion as FU
from wave3.objectives import RowObjective, build
from wave2.optimizers import haar, search, rel_dist
from wave2.objectives import contract_factory, Objective
from wave4_2 import metrics as MT

BLIND_MODULES = ["wave3/common.py", "wave3/rowmetrics.py", "wave3/objectives.py", "wave3/jobs.py", "wave3/fusion.py", "wave3/structure.py",
                 "wave3/reduced_axis.py", "wave3/context.py", "run_search.py", "prepare.py"]
TRUTH_NAMES = {"truth", "bridge", "global_oracle", "run_global_oracle", "evaluate_recovery", "oracle_diagnostics", "EvalOnly", "oracle_eta",
               "prior_draws", "eval_only", "extract_eval", "EvalData", "load", "inputs"}


def _rand_moments(k, p, seed=0):
    rng = np.random.default_rng(seed); H = rng.standard_normal((400, k + p, k + p)); H = (H + H.transpose(0, 2, 1)) / 2
    K = np.einsum("nab,ncd->abcd", H[:, :k, :k], H[:, :k, :k]) / 400
    J = np.stack([H[:, :k, k + j].T @ H[:, :k, k + j] / 400 for j in range(p)])
    return dict(K=K, J=J, valid=True), H


def _perm(k, rng):
    return np.eye(k)[rng.permutation(k)] * rng.choice([-1, 1], k)[:, None]


# ---------------------------------------------------------------- W3-01 repair family
def test_pow_a1_equals_b08_and_a0_equals_b07_and_gl_endpoint():
    rng = np.random.default_rng(1)
    for _ in range(30):
        B = rng.exponential(1, (4, 6)); B[rng.random(B.shape) < .3] = 0; A = np.eye(4); z = dict(A=False, B=False)
        for c in (0.01, 0.03, 0.1):
            assert abs(RM.value(dict(kind="POW", a=1.0, c=c), B) - MT.single_l06(f"B08.M03.c{c}")(A, B, z)) < 1e-14
            assert abs(RM.value(dict(kind="POW", a=0.0, c=c), B) - MT.single_l06(f"B07.M03.c{c}")(A, B, z)) < 1e-14
        g = RM.value(dict(kind="GL", lam=0.0, a=1.0, c=.03), B); assert abs(g - MT.single_l06("B01.M03.c0.03")(A, B, z)) < 1e-14


def test_smooth_weights_a1_proportional_to_energy_up_to_activation():
    rng = np.random.default_rng(2); B = rng.exponential(1, (5, 4)); r = B.sum(1); E = r.sum(); k = 5
    w, _ = RM.weights(dict(kind="SM", a=1.0, t=0.01), r, E, k)
    e = r / E; d = 0.01 / k; ref = e * e / (e + d) * (e + d) / e; ref = ref / ref.sum()      # (e + d)^1 A_d(e) = e
    assert np.allclose(w, e / e.sum()) and np.allclose(w, ref)


def test_zero_row_has_zero_weight_and_continuity_near_cutoff():
    rng = np.random.default_rng(3); B = rng.exponential(1, (4, 5)); B0 = B.copy(); B0[3] = 0
    for spec in (dict(kind="SM", a=0.0, t=0.01), dict(kind="BL", rho=0.5), dict(kind="SM", a=0.5, t=0.001)):
        t = RM.terms(spec, B0); assert t["w"][3] == 0
        # continuity: a vanishing row changes a smooth-activation loss continuously
        vals = [RM.value(spec, np.vstack([B0[:3], eps * B[3:]])) for eps in (1e-12, 1e-9)]
        assert abs(vals[0] - RM.value(spec, B0)) < 1e-6 and abs(vals[1] - vals[0]) < 1e-5


def test_exact_decomposition_and_signed_perm_invariance():
    mo, _ = _rand_moments(4, 5, 4); s = np.array([1., 2., .5, 1.5]); rng = np.random.default_rng(5)
    rc = RM.registry(__import__("json").loads((LAB / "configs" / "repair_candidates.json").read_text()))
    Qa, Qb = haar(4, 2, rng)
    for cid, spec in rc.items():
        L = RowObjective(mo, s, spec); P = _perm(4, rng)
        assert abs(L(P @ Qa) - L(Qa)) <= 1e-12 * (abs(L(Qa)) + 1e-12), cid
        Ba, Bb = L.B(Qa), L.B(Qb); dec = RM.decompose(spec, Ba, Bb, np.arange(4))
        assert abs(dec["dL"] - (dec["score_term"].sum() + dec["weight_term"].sum() + dec["d_extra"])) < 1e-12, cid
        assert abs(dec["residual"]) < 1e-12, cid


def test_row_objective_undefined_on_zero_block():
    mo, _ = _rand_moments(3, 4, 6); mo0 = dict(mo, J=np.zeros_like(mo["J"])); L = RowObjective(mo0, np.ones(3), dict(kind="POW", a=1.0, c=.03))
    assert np.isnan(L(np.eye(3)))


# ---------------------------------------------------------------- W3-03 gradient / contrast moments
def _grid(f, lat, X):
    g = torch.func.vmap(torch.func.grad(f))
    P = torch.as_tensor(np.column_stack((np.repeat(lat, len(X), 0), np.tile(X, (len(lat), 1)))))
    return g(P).numpy()[:, :lat.shape[1]].reshape(len(lat), len(X), -1)


def test_gradient_identities_and_pair_reconstruction():
    k = 3
    f = lambda v: v[0] * v[3] + torch.sin(v[1]) * v[4] ** 2 + v[2] + 0.3 * v[0] * v[1]
    rng = np.random.default_rng(7); lat = rng.standard_normal((9, k)); X = rng.standard_normal((6, 2))
    G = _grid(f, lat, X); n = len(X)
    allp = np.array([(i, j) for i in range(n) for j in range(n)])                       # every ORDERED pair incl. identical
    M = ST.base_matrices(G, {"U": allp})
    assert np.allclose(M["CG"], M["C0"] + M["CD"], atol=1e-13)
    assert np.allclose(M["PU"].mean(0), 2 * M["CD"], atol=1e-12)                           # E_pair C_pair = 2 C_Delta
    # direct per-row / centred definitions
    assert np.allclose(M["Cx"].mean(0), M["CG"]) and np.allclose(M["Cc"].mean(0), M["CD"])
    # gradient by AD equals finite differences
    h = 1e-6; v0 = np.r_[lat[0], X[0]]; fd = [(f(torch.as_tensor(v0 + h * e)) - f(torch.as_tensor(v0 - h * e))).item() / (2 * h) for e in np.eye(k + 2)[:k]]
    assert np.allclose(fd, G[0, 0], atol=1e-6)


def test_mean_only_axis_is_visible_to_C0_and_invisible_to_contrast():
    """f = a + b x: the intercept axis a has a constant gradient (C0 only); contrasts remove it"""
    f = lambda v: v[0] + v[1] * v[2]
    lat = np.random.default_rng(8).standard_normal((5, 2)); X = np.random.default_rng(9).standard_normal((7, 1))
    G = _grid(f, lat, X); M = ST.base_matrices(G, {"U": np.array([(0, 1), (2, 3), (4, 5)])})
    assert M["C0"][0, 0] > 0.9 and abs(M["CD"][0, 0]) < 1e-14 and np.allclose(M["PU"][:, 0, :], 0)


def test_struct_objectives_signed_perm_invariant_and_rotation_transport():
    k = 3; f = lambda v: v[0] * v[3] + v[1] * v[4] + 0.5 * v[2] * v[3] * v[4]
    rng = np.random.default_rng(10); lat = rng.standard_normal((12, k)); X = rng.standard_normal((10, 2)); G = _grid(f, lat, X)
    pairs, _ = ST.pair_designs(X, 20, 3)
    cache = dict(G=G, mats=ST.base_matrices(G, pairs), valid=True, HE={})
    import json
    specs = [c["spec"] for c in json.loads((LAB / "configs" / "structure_candidates.json").read_text())["candidates"] if c["spec"].get("target") != "HECTRL"]
    s = np.array([1., 2., .5]); Q = haar(k, 1, rng)[0]
    for sp in specs:
        L = ST.StructObjective(cache, s, sp); v = L(Q)
        if sp.get("rows") == "interior":
            assert np.isnan(v); continue
        assert np.isfinite(v) and abs(L(_perm(k, rng) @ Q) - v) <= 1e-10 * (abs(v) + 1e-12), sp
    # transport: rotated gradient energies equal energies of explicitly rotated gradients grad_U = R S a
    Ga = np.einsum("ia,vxa->vxi", Q * s[None, :], G); L = ST.StructObjective(cache, s, dict(target="G", recipe="M03", c=.03))
    v = np.mean(Ga ** 2, (0, 1)); assert abs(L(Q) - ST.sup_vec(v, .03)) < 1e-12


def test_struct_zero_block_is_undefined():
    G = np.zeros((4, 5, 3)); cache = dict(G=G, mats=ST.base_matrices(G, {"U": np.array([(0, 1)])}), valid=True, HE={})
    assert np.isnan(ST.StructObjective(cache, np.ones(3), dict(target="G", recipe="M03", c=.03))(np.eye(3)))


# ---------------------------------------------------------------- W3-02 fusion
def test_fusion_calibration_invariant_component_is_invalid_and_pareto():
    k = 3; const = lambda Q: 1.0; cal = FU.calibrate(const, k); assert cal["invariant"]
    mo, _ = _rand_moments(3, 4, 11); L = Objective(mo, np.ones(3), "L06|B08.M03.c0.03"); calL = FU.calibrate(L, k)
    F = FU.Fused(FU.Z(L, calL), FU.Z(const, cal), "w", w=.5); assert np.isnan(F(np.eye(3)))
    P = np.array([[0, 1], [1, 0], [1, 1], [.5, .5]]); assert FU.pareto(P).tolist() == [True, True, False, True]
    assert FU.select([dict(key="b", zE_val=1., zX_val=0.), dict(key="a", zE_val=0., zX_val=1.)]) == "a"


# ---------------------------------------------------------------- search / firewall
def test_search_reproducible_with_row_objective():
    mo, _ = _rand_moments(3, 6, 12); L = RowObjective(mo, np.ones(3), dict(kind="SM", a=0.0, t=0.01))
    r1 = search(L, 3, 31); r2 = search(L, 3, 31); assert np.array_equal(r1["R"], r2["R"]) and abs(L(r1["R"]) - r1["f"]) < 1e-12


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


def test_truth_file_access_denied_during_w3_blind_scoring():
    toy = "CR3"
    if not (OUT08 / "blind_payload" / f"{toy}.npz").is_file():
        pytest.skip("payload not prepared")
    code = textwrap.dedent(f"""
        import sys, os, numpy as np
        sys.path.insert(0, r"{LAB}")
        from wave3.common import ROOT, OUT08, OUT07
        nc = lambda q: os.path.normcase(os.path.abspath(q))
        FORB = [nc(ROOT / "outputs" / "models" / "generate"), nc(OUT08 / "eval_only"), nc(OUT07), nc(ROOT / "outputs" / "metric_lab" / "09_wave4_2_third_wave" / "W3r1" / "eval_only")]
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
        from wave3 import jobs, structure as ST
        from wave3.objectives import build
        from wave2.blind_inputs import load_payload, variant_frame
        from wave2.optimizers import search
        import run_blind as RB8
        pl = load_payload("{toy}")
        d = ST.grid_design(pl, "train", "pilot", 5); d["lat"] = d["lat"][:4]; d["sid"] = d["sid"][:4]; d["x"] = d["x"][:40]
        G, E, mo = ST.compute_grid(pl, d, RB8.eta_factory(pl)())
        pairs, _ = ST.pair_designs((d["x"] - pl.x_mean) / pl.x_std, 16, 5)
        cache = dict(G=G, mats=ST.base_matrices(G, pairs), valid=True, HE=mo)
        sc = variant_frame(pl, "s9_2")[1]
        for spec in (dict(type="struct", target="PU", recipe="col", c=.03), dict(type="row", kind="SM", a=0.0, t=0.01, c=.03)):
            L = build(spec, cache if spec["type"] == "struct" else mo["MASK-ALL"], sc); r = search(L, pl.k, 31)
        print("BLIND_OK", r["f"])
        try:
            np.load(OUT08 / "eval_only" / "{toy}.npz")
            print("HOOK_FAILED")
        except PermissionError:
            print("HOOK_CAUGHT")
    """)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=900)
    assert "BLIND_OK" in p.stdout and "HOOK_CAUGHT" in p.stdout, p.stdout + p.stderr
