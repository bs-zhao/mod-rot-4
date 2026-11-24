"""04.2 evaluations on every database / variant / (Q, channel, inference) / metric.   usage: python analyze.py

Evaluation 1: Pearson, Spearman between analytic and inferred metric over the unique rotations (R* + every angle > 0), plus per-curve (7-angle) correlations, NA for constant.
Evaluation 2: per sweep (axis subset) angle of the minimum for analytic and inferred values; perfect = {0,90} in ALL sweeps, marginal = {0,15,75,90} in ALL sweeps,
              both for analytic AND inferred; constant / undefined curves never pass; extra diagnostics (dynamic range, endpoint equivalence, margin, strict-margin fraction, interior ties).
Writes results/pass_summary.csv, results/cfg_diagnostics.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; OUT = ROOT / "outputs" / "metric_lab" / "06_wave1_screen"; RES = HERE / "results"; RES.mkdir(exist_ok=True)
TOL = 1e-9; PERF = {0, 90}; MARG = {0, 15, 75, 90}


def pr(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.ptp(a) <= TOL * (np.abs(a).max() + 1e-300) or np.ptp(b) <= TOL * (np.abs(b).max() + 1e-300): return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def sp(a, b):
    if np.isnan(pr(a, b)): return np.nan
    return float(spearmanr(a, b)[0])


def sweeps(man):
    sw = man[man.candidate_type == "sweep"]; out = []
    for ax, g in sw.groupby("axes_1based"):
        g = g.sort_values("angle_deg"); out.append((ax, g.index.values, g.angle_deg.values.astype(int)))
    return out


def curve_stats(v, idx, ang):
    """-> (valid, min-angle set, range, endpoint-equivalence err, margin, strict, interior_tie)"""
    x = v[idx]
    if not np.isfinite(x).all(): return None
    sc = np.abs(x).max() + 1e-300; rng_ = np.ptp(x)
    if rng_ <= TOL * sc: return None
    tol = TOL * sc; mn = x.min(); best = {int(a) for a, y in zip(ang, x) if y <= mn + tol}
    end = min(x[ang == 0][0], x[ang == 90][0]); inter = x[(ang >= 15) & (ang <= 75)]
    return dict(best=best, rng=rng_ / sc, endeq=abs(x[ang == 0][0] - x[ang == 90][0]) / rng_, margin=(inter.min() - end) / rng_, strict=bool(inter.min() > end + tol), itie=bool(abs(inter.min() - end) <= tol))


def main():
    rows = []; cfgd = []
    for rm in sorted(OUT.glob("*/*/*/run_manifest.json")):
        root = rm.parent; model, cond, rep = root.parents[1].name, root.parents[0].name, root.name; db = f"{model}|{cond[:20]}|{rep}"
        for var in ("s9", "s9_2"):
            f = root / var / "metrics.npz"
            if not f.exists(): continue
            Z = np.load(f); names = list(Z["names"]); man = pd.read_csv(root / var / "rotation_manifest.csv"); uniq = ((man.candidate_type == "baseline_unswept") | (man.angle_deg > 0)).values; sw = sweeps(man)
            cfgs = sorted({k[2:] for k in Z.files if k.startswith("M|")})
            for key in cfgs:
                Q, ch, meth = key.split("|")
                if meth == "oracle": continue
                Mo, Mi = Z[f"M|{Q}|{ch}|oracle"], Z[f"M|{key}"]
                Ao, Bo, Ai, Bi = Z[f"A|{Q}|{ch}|oracle"][uniq].astype(float), Z[f"B|{Q}|{ch}|oracle"][uniq].astype(float), Z[f"A|{key}"][uniq].astype(float), Z[f"B|{key}"][uniq].astype(float)
                relA = float(np.nanmean(np.linalg.norm((Ai - Ao).reshape(len(Ai), -1), axis=1) / (np.linalg.norm(Ao.reshape(len(Ao), -1), axis=1) + 1e-300))) if Ao.sum() > 1e-12 else np.nan
                relB = float(np.mean(np.linalg.norm((Bi - Bo).reshape(len(Bi), -1), axis=1) / (np.linalg.norm(Bo.reshape(len(Bo), -1), axis=1) + 1e-300)))
                zA = Ao < 1e-6 * (Ao.max() + 1e-300); zB = Bo < 1e-6 * (Bo.max() + 1e-300)
                cfgd.append(dict(db=db, variant=var, Q=Q, channel=ch, method=meth, rel_err_A=relA, rel_err_B=relB, fp_energy_A=float(Ai[zA].sum() / (Ai.sum() + 1e-300)) if zA.any() else np.nan, fp_energy_B=float(Bi[zB].sum() / (Bi.sum() + 1e-300)) if zB.any() else np.nan))
                for j, nm in enumerate(names):
                    o, i_ = Mo[:, j], Mi[:, j]; res = dict(db=db, model=model, variant=var, Q=Q, channel=ch, method=meth, metric=nm, pearson=pr(o[uniq], i_[uniq]), spearman=sp(o[uniq], i_[uniq]))
                    cs = [pr(o[idx], i_[idx]) for _, idx, _ in sw]; res["curve_pearson_mean"] = float(np.nanmean(cs)) if np.isfinite(cs).any() else np.nan; res["curve_na_frac"] = float(np.mean(np.isnan(cs)))
                    st_o = [curve_stats(o, idx, ang) for _, idx, ang in sw]; st_i = [curve_stats(i_, idx, ang) for _, idx, ang in sw]; n = len(sw)
                    for tag, st in (("or", st_o), ("inf", st_i)):
                        ok = [s for s in st if s is not None]
                        res[f"{tag}_valid_frac"] = len(ok) / n; res[f"{tag}_perfect"] = len(ok) == n and all(s["best"] & PERF for s in ok); res[f"{tag}_marginal"] = len(ok) == n and all(s["best"] & MARG for s in ok)
                        res[f"{tag}_n_perfect"] = sum(bool(s["best"] & PERF) for s in ok); res[f"{tag}_n_marginal"] = sum(bool(s["best"] & MARG) for s in ok)
                        if ok:
                            res[f"{tag}_range"] = float(np.median([s["rng"] for s in ok])); res[f"{tag}_endeq"] = float(np.median([s["endeq"] for s in ok])); res[f"{tag}_margin"] = float(np.median([s["margin"] for s in ok]))
                            res[f"{tag}_strict_frac"] = float(np.mean([s["strict"] for s in ok])); res[f"{tag}_interior_tie_frac"] = float(np.mean([s["itie"] for s in ok]))
                    res["n_sweeps"] = n; res["perfect_both"] = bool(res["or_perfect"] and res["inf_perfect"]); res["marginal_both"] = bool(res["or_marginal"] and res["inf_marginal"]); rows.append(res)
        print(db, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(RES / "pass_summary.csv.gz", index=False); pd.DataFrame(cfgd).to_csv(RES / "cfg_diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
