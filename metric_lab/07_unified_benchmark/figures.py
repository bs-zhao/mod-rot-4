"""Figures of the first wave, drawn only from saved tables / loss files.   usage: python figures.py -> results/figures/*.png"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4_2.common import RES, CONFIGS, OUT
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = RES / "figures"; FIG.mkdir(exist_ok=True)
ORDER = ["C1", "C2", "C3", "CR3", "CR4", "CR5"]


def pass_heatmaps(ps, pan):
    a = ps[(ps.panel == "primary") & (ps.Q == "Q01") & (ps.fidelity == "pilot") & (ps.analytic_source == "oracle") & ps["mask"].isin(["FULL", "MASK-ALL"]) & ps.metric.isin(pan["ids"])]
    for cls, chans in (("B", ("YL", "YP")), ("C", ("YC",))):
        for ch in chans:
            d = a[(a.cls == cls) & (a.channel == ch)]
            code = d.O_marginal.astype(int) * 2 + d.I_marginal.astype(int)        # 3 both, 2 analytic only, 1 inferred only, 0 neither
            d = d.assign(code=code, lib=d.toy + "|" + d.variant)
            libs = [t + ("L" if cls == "C" else "") + "|" + v for t in ORDER for v in ("s9", "s9_2")]
            M = d.pivot_table(index="metric", columns="lib", values="code").reindex(columns=libs)
            M = M.loc[M.sum(1).sort_values(ascending=False).index]
            fig, ax = plt.subplots(figsize=(1.2 + .45 * len(libs), 1 + .16 * len(M)))
            cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#f4a582", "#92c5de", "#1a9850"])
            ax.imshow(M.values, cmap=cmap, vmin=0, vmax=3, aspect="auto")
            ax.set_xticks(range(len(libs))); ax.set_xticklabels(libs, rotation=90, fontsize=7); ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=5)
            ax.set_title(f"class {cls} {ch}: Q01 pilot, priority panel, marginal all-sweep pass\n(green both, blue analytic only, orange inferred only, grey neither)", fontsize=8)
            fig.tight_layout(); fig.savefig(FIG / f"pass_heatmap_{cls}_{ch}.png", dpi=130); plt.close(fig)


def minimum_angle_confusion():
    f = OUT / "W1" / "analysis" / "sweep_minima.csv.gz"
    if not f.is_file():
        return
    sw = pd.read_csv(f, usecols=["toy", "Q", "fidelity", "mask", "side", "metric", "valid", "minset", "channel", "variant", "sweep"])
    sw = sw[(sw.Q == "Q01") & (sw.fidelity == "pilot") & sw["mask"].isin(["FULL", "MASK-ALL"]) & (sw.channel != "YP")]
    def first(ms): return int(np.argmax([(int(ms) >> i) & 1 for i in range(7)])) if ms else -1
    sw["amin"] = sw.minset.map(first)
    wide = sw.pivot_table(index=["toy", "variant", "channel", "metric", "sweep"], columns="side", values="amin", aggfunc="first").dropna()
    wide = wide[(wide.O >= 0) & (wide.I >= 0)]
    for fg in ("C", "CR"):
        w = wide[[t.startswith("CR") == (fg == "CR") for t in wide.index.get_level_values(0)]]
        C = pd.crosstab(w.O, w.I).reindex(index=range(7), columns=range(7), fill_value=0)
        fig, ax = plt.subplots(figsize=(4.6, 4)); ax.imshow(np.log1p(C.values), cmap="viridis")
        lab = [str(a) for a in (0, 15, 30, 45, 60, 75, 90)]
        ax.set_xticks(range(7)); ax.set_xticklabels(lab); ax.set_yticks(range(7)); ax.set_yticklabels(lab)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, int(C.values[i, j]), ha="center", va="center", fontsize=6, color="w")
        ax.set_xlabel("decoder (inferred) minimum angle"); ax.set_ylabel("analytic minimum angle"); ax.set_title(f"{fg} family, both classes, all 415 metrics, Q01 pilot", fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / f"minimum_angle_confusion_{fg}.png", dpi=130); plt.close(fig)


def global_scatter():
    s = pd.read_csv(RES / "global_oracle_search.csv.gz")
    for tgt in sorted(s.target.unique()):
        g = s[s.target == tgt]
        fig, ax = plt.subplots(figsize=(5, 3.6))
        rel = g.dL / (g.L_identity.abs() + 1e-12)
        ax.scatter(g.d_sp_norm, np.sign(rel) * np.log10(1 + np.abs(rel) * 1e4), s=4, alpha=.4)
        ax.axhline(0, color="k", lw=.6); ax.set_xlabel("d_SP / sqrt(2k) of the refined start"); ax.set_ylabel("signed log10(1 + 1e4 |L - L(I)| / |L(I)|)")
        ax.set_title(f"{tgt}: every refined start of every priority metric", fontsize=8); fig.tight_layout(); fig.savefig(FIG / f"global_{tgt.replace(':', '_')}.png", dpi=120); plt.close(fig)


def interpolation():
    bc = pd.read_csv(RES / "bridge_curves.csv.gz"); bc = bc[bc["mask"].isin(["FULL", "MASK-ALL"])]
    O = bc[bc.arm == "O"][["toy", "variant", "channel", "metric", "arm_marginal"]].rename(columns={"arm_marginal": "O"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, br in zip(axes, ("BO", "BA")):
        for (cls, fg), mk in ((("B", "C"), "o-"), (("B", "CR"), "s-"), (("C", "C"), "o--"), (("C", "CR"), "s--")):
            ys = []
            for arm in (br, f"{br}:0.25", f"{br}:0.5", f"{br}:0.75", "D"):
                g = bc[bc.arm == arm].merge(O, on=["toy", "variant", "channel", "metric"]); g = g[g.O & (g.toy.str.endswith("L") == (cls == "C")) & (g.toy.str.startswith("CR") == (fg == "CR"))]
                ys.append(g.arm_marginal.mean())
            ax.plot([0, .25, .5, .75, 1], ys, mk, label=f"class {cls}, {fg} family")
        ax.set_title(f"(1-a) {br} + a decoder", fontsize=9); ax.set_xlabel("a"); ax.grid(alpha=.3)
    axes[0].set_ylabel("fraction of analytic passes kept"); axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "bridge_interpolation.png", dpi=130); plt.close(fig)


def convergence(ps, pan):
    d = ps[(ps.panel == "primary") & (ps.analytic_source == "oracle") & ps.metric.isin(pan["ids"]) & (ps.channel != "YP") & ps["mask"].isin(["FULL", "MASK-ALL"])]
    rows = d.groupby(["cls", "fidelity", "Q", "seed"]).agg(O=("O_marginal", "mean"), I=("I_marginal", "mean"), both=("both_marginal", "mean")).reset_index()
    fid = {"pilot": 0, "C": 1, "D": 2}; rows["x"] = rows.fidelity.map(fid)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, cls in zip(axes, ("B", "C")):
        for Q, g in rows[rows.cls == cls].groupby("Q"):
            for side, ls in (("O", "-"), ("I", "--")):
                gg = g.groupby("x")[side].agg(["mean", "min", "max"])
                ax.errorbar(gg.index + {"Q01": -.12, "Q02": -.06, "Q03": 0, "Q04": .06, "Q05": .12}[Q], gg["mean"], yerr=[gg["mean"] - gg["min"], gg["max"] - gg["mean"]], fmt=ls, label=f"{Q} {'analytic' if side == 'O' else 'decoder'}", lw=1, capsize=2)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["pilot 256x32", "C 512x64", "D 1024x128"]); ax.set_title(f"class {cls}: share of (library, metric) cells passing", fontsize=9); ax.grid(alpha=.3)
    axes[1].legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(FIG / "integration_convergence.png", dpi=130); plt.close(fig)


def paired_curves(pan):
    keys = ["L03|UX.lq.1", "L06|B01.M03.c0.01", "L03|rowsp.a0.75.q0.1", "L06|B08.M03.c0.01"]
    for toy, run in (("CR4", "CR4/beta0p0005_lambda0_ctx0p25to0p8_L3_09fe72d2fe/repeat01"), ("CR4L", "CR4L/beta0p001_lambda0_ctx0p25to0p8_L3_d0bd42a54e/repeat01"),
                     ("C3L", "C3L/beta0p006_lambda0_ctx0p25to0p8_L6_4ade5ca369/repeat01")):
        ch = "YC" if toy.endswith("L") else "YL"; f = OUT / "W1" / run / "bridge" / f"s9_{ch}_FULL.npz"
        if not f.is_file():
            continue
        z = np.load(f, allow_pickle=False); ids = [str(i) for i in z["ids"]]; man = json.loads(str(z["manifest"]))
        sw = {}
        for i, r in enumerate(man):
            if r["candidate_type"] == "sweep": sw.setdefault(tuple(r["axes"]), []).append((r["angle"], i))
        subs = sorted(sw)[:4]
        fig, axes = plt.subplots(len(keys), len(subs), figsize=(3 * len(subs), 2.2 * len(keys)), squeeze=False)
        for a_, m in enumerate(keys):
            j = ids.index(m)
            for b_, s in enumerate(subs):
                ax = axes[a_][b_]; lst = sorted(sw[s]); idx = [i for _, i in lst]; ang = [t for t, _ in lst]
                for arm, st in (("O", "k-"), ("BO", "b--"), ("BA", "c:"), ("D", "r-"), ("S", "g-.")):
                    v = z[f"{arm}|L"][idx, j]; v = (v - v.mean()) / (v.std() + 1e-12); ax.plot(ang, v, st, lw=1.2, label=arm)
                ax.set_title(f"{m}  axes {'-'.join(str(x + 1) for x in s)}", fontsize=6); ax.set_xticks([0, 45, 90])
        axes[0][0].legend(fontsize=6); fig.suptitle(f"{toy} S9 {ch}: z-scored loss vs angle (O analytic, BO/BA bridges, D decoder, S surrogate)", fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / f"paired_curves_{toy}.png", dpi=120); plt.close(fig)


def main():
    pan = json.loads((CONFIGS / "priority_panel.json").read_text())
    ps = pd.read_csv(RES / "pass_summary.csv.gz", low_memory=False)
    pass_heatmaps(ps, pan); convergence(ps, pan); interpolation(); global_scatter(); paired_curves(pan); minimum_angle_confusion()
    print(sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
