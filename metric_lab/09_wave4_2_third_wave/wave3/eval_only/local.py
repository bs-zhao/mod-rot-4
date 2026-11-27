"""EVALUATION-ONLY local Gap 1 / Gap 2 (03_third_wave sections 20-21): the same saved S9 / S9_2 sweep rotations, analytic (oracle) and
inferred (frozen decoder) surfaces on IDENTICAL TEST points, Lab 07 flag code (harmonised tolerance) and Eval1 correlations.

HE-based W3 candidates (RW repair recipes + controls) reuse the saved Lab 07 base tensors (oracle and nn arms at the same Q01 points):
Q01 pilot seed 31 (Lab 07 batch A) and Q01 C seeds 31 / 73 / 101 (batch C).
Gradient / contrast candidates use a TEST product grid (pilot 128 x 128 / 32 x 500, seed 31): decoder gradients at S5 Z and analytic gradients
of the generating formula at the paired true T, the same X rows and the same frozen pair designs.
Rotation convention: oracle U = G T_s (d T / d U = diag(t_std) G^T), decoder U = R V (d Z / d U = diag(z_std) R^T), as Lab 07.
"""
import json
import numpy as np, pandas as pd
from ..common import OUT, OUT07, RES, PRIMARY, toy_entry, cfg
from .. import rowmetrics as RM, structure as ST, jobs as JB
from wave4_2.inputs import load
from wave4_2.moments import jac
from wave4_2.evaluation import sweeps, curve_stats, library_flags, eval1
from wave4_2 import metrics as MT
import run_baseline as RB7

LOCAL = OUT / "local"
HE_TENSORS = ("Q01_pilot_s31", "Q01_C_s31", "Q01_C_s73", "Q01_C_s101")


def _B_stack(mo, rots, source):
    out = []
    for r in rots:
        Jz = jac(r.t_std, r.G) if source == "oracle" else jac(r.z_std, r.R)
        out.append(np.maximum(np.einsum("ia,xik,ka->ax", Jz, mo["J"], Jz, optimize=True), 0))
    return np.stack(out)


def he_candidates():
    reg = JB.registry()
    return {c: e["spec"] for c, e in reg.items() if e["family"] in ("RW", "CTRL") and e["surface"] == "AD" and c not in ("X01",)}


def _score_he(spec, Bs):
    if spec["type"] == "row":
        return np.array([RM.value(spec, B) for B in Bs])
    name = spec["metric"].split("|", 1)[1]; f = MT.single_l06(name); A0 = np.zeros((Bs.shape[1], Bs.shape[1]))
    zero = dict(A=False, B=bool(Bs[0].sum() <= 1e-300))
    return np.array([float(f(A0, B, zero)) for B in Bs])


def _flags(Lo, Li, man, sw, n_exp, cids):
    fo, fi = library_flags(curve_stats(Lo, sw), n_exp), library_flags(curve_stats(Li, sw), n_exp); e1 = eval1(Lo, Li, man, sw); rows = []
    for j, cid in enumerate(cids):
        rows.append(dict(cid=cid, O_perfect=bool(fo["perfect"][j]), O_marginal=bool(fo["marginal"][j]), O_strict_marginal=bool(fo["strict_marginal"][j]),
                         O_min_margin=float(fo["min_margin"][j]), O_constant=bool(fo["any_constant"][j]), I_perfect=bool(fi["perfect"][j]), I_marginal=bool(fi["marginal"][j]),
                         I_strict_marginal=bool(fi["strict_marginal"][j]), I_min_margin=float(fi["min_margin"][j]), I_constant=bool(fi["any_constant"][j]),
                         both_marginal=bool(fo["marginal"][j] and fi["marginal"][j]), pearson=float(e1["pearson"][j]), spearman=float(e1["spearman"][j]),
                         curve_mean=float(e1["curve_mean"][j]), curve_min=float(e1["curve_min"][j])))
    return rows


def he_local(toy, cands=None):
    out = LOCAL / "he" / f"{toy.replace('#', '_')}.csv.gz"
    if out.is_file() and cands is None:
        return pd.read_csv(out)
    cands = cands or he_candidates(); cids = list(cands)
    b, ev, rots, _ = load(toy); e = toy_entry(toy); k = b.k; n_exp = 2 ** k - k - 1; root = OUT07 / "W1" / e["run"]; rows = []
    ch = "YC" if b.cls == "C" else "YL"
    for tag in HE_TENSORS:
        tf = root / "tensors" / f"{tag}.npz"
        if not tf.is_file():
            continue
        res, _ = RB7.load_tensors(tf); Q, fid, seed = tag.split("_")
        for (src, c, m), _mo in res.items():
            if src != "nn" or c != ch:
                continue
            for v in rots:
                man = RB7.manifest_rows(rots[v]); sw = sweeps(man)
                Bo, Bi = _B_stack(res[("oracle", c, m)], rots[v], "oracle"), _B_stack(res[("nn", c, m)], rots[v], "nn")
                Lo = np.column_stack([_score_he(cands[cid], Bo) for cid in cids]); Li = np.column_stack([_score_he(cands[cid], Bi) for cid in cids])
                for r in _flags(Lo, Li, man, sw, n_exp, cids):
                    rows.append(dict(toy=toy, variant=v, Q=Q, fidelity=fid, seed=int(seed[1:]), channel=c, mask=m, arm="HE", **r))
    df = pd.DataFrame(rows); out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False); return df


# ---------------------------------------------------------------- gradient / contrast local arm
def struct_caches(toy):
    """TEST pilot grid: decoder and analytic gradients at identical (subject, X row) points; frozen pair designs"""
    from wave4_2.truth import oracle_eta
    from wave2.blind_inputs import load_payload
    import run_blind as RB8
    f = LOCAL / "grad" / f"{toy.replace('#', '_')}.npz"
    pl = load_payload(toy); b, ev, rots, _ = load(toy)
    if f.is_file():
        z = np.load(f); Gn, Go, X, inter = z["Gn"], z["Go"], z["x"], z["interior"]; pairs = {r: z[f"pairs_{r}"] for r in ST.PAIR_RULES}
    else:
        d = ST.grid_design(pl, "test", "pilot", 31)
        Gn, _, _ = ST.compute_grid(pl, d, RB8.eta_factory(pl)(), he=False)
        Go, _, _ = ST.compute_grid(pl, d, oracle_eta(ev.family, pl.x_mean, pl.x_std), lat=ev.T[d["sid"]], he=False)
        X = d["x"]; inter = ST.interior_rows(pl, X); xs = (X - pl.x_mean) / pl.x_std
        pairs, audit = ST.pair_designs(xs, ST.SIZES["pilot"]["pairs"], 31)
        arr = dict(Gn=Gn, Go=Go, x=X, sid=d["sid"], interior=inter, meta=np.array(json.dumps(dict(d["meta"], audit=audit), default=float)))
        if pl.fixed_bank is not None and pl.x_domain is not None:
            pI, _ = ST.pair_designs(xs, ST.SIZES["pilot"]["pairs"], 31, rows_ok=inter)
            for r, p in pI.items():
                arr[f"pairsI_{r}"] = p
        for r, p in pairs.items():
            arr[f"pairs_{r}"] = p
        f.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, **arr)
    z = np.load(f)
    caches = {}
    for name, G in (("nn", Gn), ("oracle", Go)):
        c = dict(G=G, mats=ST.base_matrices(G, pairs), valid=bool(np.isfinite(G).all()), HE={})
        if "pairsI_U" in z.files:
            rows = np.where(inter)[0]; remap = -np.ones(len(inter), int); remap[rows] = np.arange(len(rows))
            c["mats_int"] = ST.base_matrices(G, {r: remap[z[f"pairsI_{r}"]] for r in ST.PAIR_RULES}, rows=rows)
        caches[name] = c
    return caches, b, ev, rots


def struct_local(toy):
    out = LOCAL / "struct" / f"{toy.replace('#', '_')}.csv.gz"
    if out.is_file():
        return pd.read_csv(out)
    caches, b, ev, rots = struct_caches(toy); k = b.k; n_exp = 2 ** k - k - 1
    cands = {c["id"]: c["spec"] for c in cfg("structure_candidates.json")["candidates"] if c["spec"]["target"] != "HECTRL"}; cids = list(cands); rows = []
    for v in rots:
        man = RB7.manifest_rows(rots[v]); sw = sweeps(man); r0 = rots[v][0]
        Lo = np.full((len(rots[v]), len(cids)), np.nan); Li = Lo.copy()
        for j, cid in enumerate(cids):
            So = ST.StructObjective(caches["oracle"], r0.t_std, cands[cid]); Si = ST.StructObjective(caches["nn"], r0.z_std, cands[cid])
            Lo[:, j] = [So(r.G) for r in rots[v]]; Li[:, j] = [Si(r.R) for r in rots[v]]
        for r in _flags(Lo, Li, man, sw, n_exp, cids):
            rows.append(dict(toy=toy, variant=v, Q="GRID", fidelity="pilot", seed=31, channel="L", mask="ALLROWS", arm="GRAD", **r))
    df = pd.DataFrame(rows); out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False); return df
