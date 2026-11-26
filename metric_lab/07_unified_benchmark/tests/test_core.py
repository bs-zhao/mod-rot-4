"""W1-00 unit tests that need no data: chain rule, contraction, invariances, ties, registry adapters, blind/truth separation."""
import sys, ast
from pathlib import Path
import numpy as np, torch, pytest
LAB = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(LAB))
from wave4_2 import common  # noqa: F401  (paths)
from wave4_2.derivatives import ad_all, channel_hessian, fd_hessian
from wave4_2.moments import Moments, contract, jac
from wave4_2 import metrics as MT
from wave4_2.evaluation import curve_stats, library_flags, ANGLES

rng = np.random.default_rng(0)


def rand_orth(k, r=rng):
    q, rr = np.linalg.qr(r.standard_normal((k, k))); return q * np.sign(np.diag(rr))


def toy_f(k):
    W = torch.as_tensor(rng.standard_normal((k, 3)))
    return lambda v: torch.tanh(v[:k] @ W @ v[k:]) + (v[:k] ** 2).sum() * v[k] + v[0] * v[1] * v[-1]


def test_channel_chain_rule():
    f = toy_f(3); P = rng.standard_normal((50, 6))
    e, g, H = ad_all(f, P); HP = channel_hessian(e, g, H, "YP")
    _, _, Hd = ad_all(lambda v: torch.sigmoid(f(v)), P)
    assert np.max(np.abs(HP - Hd)) < 1e-10


def test_contraction_equals_direct_rotated_derivative():
    k, p = 3, 3; f = toy_f(k); P = rng.standard_normal((400, k + p)); scale = np.array([.5, 2., 1.3]); centre = rng.standard_normal(k)
    _, _, H = ad_all(f, P); m = Moments(k, p, "FULL"); m.add(H, np.ones((len(P), p), bool)); r = m.finish()
    Q = rand_orth(k); A, B = contract(r["K"], r["J"], jac(scale, Q))
    Qt = torch.as_tensor(Q); st = torch.as_tensor(scale); ct = torch.as_tensor(centre)
    g = lambda w: f(torch.cat((ct + st * (Qt.T @ w[:k]), w[k:])))           # g_R(u, x) = f(raw(u), x), raw = centre + diag(scale) Q^T u
    U = ((P[:, :k] - centre) / scale) @ Q.T
    _, _, Hg = ad_all(g, np.column_stack((U, P[:, k:])))
    A2 = (Hg[:, :k, :k] ** 2).mean(0); B2 = (Hg[:, :k, k:] ** 2).mean(0)
    assert np.allclose(A, A2, rtol=1e-8, atol=1e-12) and np.allclose(B, B2, rtol=1e-8, atol=1e-12)
    # prediction invariance g_R(R v) = f(v)
    vals = torch.vmap(g)(torch.as_tensor(np.column_stack((U, P[:, k:])))); base = torch.vmap(f)(torch.as_tensor(P))
    assert torch.max(torch.abs(vals - base)) < 1e-10


def test_energy_invariance_and_signed_permutation_invariance():
    k, p = 4, 3; f = toy_f(k); P = rng.standard_normal((300, k + p)); _, _, H = ad_all(f, P)
    m = Moments(k, p, "FULL"); m.add(H, np.ones((len(P), p), bool)); r = m.finish(); sc = np.ones(k)
    Q = rand_orth(k); A, B = contract(r["K"], r["J"], jac(sc, Q)); A2, B2 = contract(r["K"], r["J"], jac(sc, rand_orth(k)))
    assert abs(A.sum() - A2.sum()) / A.sum() < 1e-10 and abs(B.sum() - B2.sum()) / B.sum() < 1e-10
    perm = np.eye(k)[rng.permutation(k)] * rng.choice([-1, 1], k)[:, None]
    A3, B3 = contract(r["K"], r["J"], jac(sc, perm @ Q))
    z, _, _ = MT.zero_flags(A, B); st = MT.l03_null(r["K"], r["J"], sc, contract, jac)
    v1 = MT.evaluate_all(A, B, st, z); v3 = MT.evaluate_all(A3, B3, st, z)
    ok = np.isfinite(v1) & np.isfinite(v3)
    assert np.allclose(v1[ok], v3[ok], rtol=1e-7, atol=1e-9)


def test_single_l06_matches_full_library():
    for k, p in ((3, 6), (4, 4), (6, 4)):
        for t in range(5):
            A = rng.random((k, k)) ** 3; A = (A + A.T) / 2; B = rng.random((k, p)) ** 4; B[rng.random((k, p)) < .3] = 0
            z = dict(A=False, B=False); full = MT.l06_metrics(A, B, z)
            for n in MT.L06_NAMES:
                a, b = full[n], MT.single_l06(n)(A, B, z)
                assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-12, atol=1e-14), n


def test_constant_and_tie_rules():
    nm = 3; L = np.zeros((7, nm)); L[:, 0] = 5.0                              # metric 0: constant curve -> invalid
    L[:, 1] = [1, 2, 3, 4, 3, 2, 1]                                          # endpoints tied -> perfect, strict
    L[:, 2] = [1, 2, 1, 4, 3, 2, 3]                                          # tie 0 and 30 -> legacy perfect, NOT strict
    man_sw = [((0, 1), np.arange(7))]
    cs = curve_stats(L, man_sw, "harmonised")
    assert not cs["valid"][0, 0] and not cs["perfect"][0, 0]
    assert cs["perfect"][0, 1] and cs["strict_perfect"][0, 1]
    assert cs["perfect"][0, 2] and not cs["strict_perfect"][0, 2] and cs["interior_tie"][0, 2]
    fl = library_flags(cs, 1); assert list(fl["perfect"]) == [False, True, True]


def test_isotropic_nonidentifiable_case_is_not_a_pass():
    """f = x (a^2 + b^2): every rotation of (a, b) gives the same energies -> constant curves -> no metric can pass"""
    f = lambda v: v[2] * (v[0] ** 2 + v[1] ** 2)
    P = rng.standard_normal((2000, 3)); _, _, H = ad_all(f, P); m = Moments(2, 1, "FULL"); m.add(H, np.ones((len(P), 1), bool)); r = m.finish()
    L = []
    for t in np.deg2rad(ANGLES):
        Q = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]); A, B = contract(r["K"], r["J"], jac(np.ones(2), Q))
        z, _, _ = MT.zero_flags(A, B); L.append(MT.evaluate_all(A, B, MT.l03_null(r["K"], r["J"], np.ones(2), contract, jac), z))
    cs = curve_stats(np.array(L), [((0, 1), np.arange(7))], "harmonised")
    # finite-sample anisotropy can leave tiny non-constant curves; none may show a material range (no false identification)
    rel = cs["range"] / (np.abs(np.array(L)).max(0) + 1e-12)
    assert ((~cs["valid"]) | (rel < 0.05)).mean() > 0.9


def test_fd_matches_ad_on_smooth_function():
    f = toy_f(3); P = rng.standard_normal((40, 6)); _, _, H = ad_all(f, P)
    Hf = fd_hessian(f, P, 1e-3, np.ones(6))
    assert np.max(np.abs(Hf - H)) < 1e-3 * np.abs(H).max()


BLIND = ("surfaces.py", "measures.py", "moments.py", "metrics.py", "derivatives.py", "evaluation.py", "inputs.py")


def test_blind_modules_do_not_import_truth():
    for fn in BLIND:
        tree = ast.parse((LAB / "wave4_2" / fn).read_text(encoding="utf-8"))
        names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(m and ("truth" in m or "oracle" in m or "bridge" in m or "global_oracle" in m) for m in names), (fn, names)
