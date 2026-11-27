"""Figures of 02_second_wave section 23, all reproduced from saved results (results/ and outputs/.../W2r1).
usage: python figures.py  -> results/figures/*.png"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave2.common import RES, OUT, PRIMARY, VARIANTS
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)


def heat(ax, C, title):
    ax.imshow(np.abs(C), vmin=0, vmax=1, cmap="viridis"); ax.set_title(title, fontsize=6); ax.set_xticks([]); ax.set_yticks([])
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{abs(C[i, j]):.2f}", ha="center", va="center", fontsize=4, color="w" if abs(C[i, j]) < .6 else "k")


def fig1_heatmaps():
    """T-U |corr| for every toy x variant: unrotated (N01), supervised (R*), blind BS-B E01 / E02 (integration 31, search 31) - failures included"""
    from wave2.blind_inputs import load_payload
    from wave2.evaluate_recovery import EvalData, supervised, matched, v_test
    d = pd.read_csv(RES / "blind_recovery.csv"); C = np.load(RES / "recovery_correlations.npz")
    for cid in ("E01", "E02"):
        fig, axes = plt.subplots(len(PRIMARY) * 2, 3, figsize=(4.2, 30))
        for r_i, (toy, v) in enumerate([(t, v) for t in PRIMARY for v in VARIANTS]):
            pl = load_payload(toy); ed = EvalData(toy); sup = supervised(pl, ed, v)
            base = d[(d.batch == "baseline") & (d.toy == toy) & (d.variant == v) & (d.cid == "N01")]
            bl = d[(d.batch == "BS-B") & (d.toy == toy) & (d.variant == v) & (d.cid == cid) & (d.iseed == 31) & (d.sseed == 31)]
            heat(axes[r_i, 0], C[base.row_id.iloc[0]], f"{toy} {v} unrotated")
            heat(axes[r_i, 1], sup["signed"], f"supervised MCC {sup['MCC']:.3f}")
            if len(bl):
                rr = bl.iloc[0]; heat(axes[r_i, 2], C[rr.row_id], f"blind {cid} MCC {rr.MCC:.3f} leak {rr.leak_max:.2f} {'PASS' if rr.pass_lab05 else 'fail'}")
        fig.tight_layout(); fig.savefig(FIG / f"fig1_TU_heatmaps_{cid}.png", dpi=160); plt.close(fig)


def fig2_loss_vs_mcc():
    t = pd.read_csv(RES / "blind_search_trace.csv.gz"); t = t[(t.batch == "BS-B") & t.cid.isin(["E01", "E02"])]
    fig, axes = plt.subplots(3, 4, figsize=(12, 8)); axes = axes.ravel()
    for i, toy in enumerate(PRIMARY):
        g = t[t.toy == toy]; ax = axes[i]
        for cid, mk in (("E01", "o"), ("E02", "^")):
            gg = g[g.cid == cid]; ax.scatter(gg.L_final, gg.MCC, s=6, marker=mk, c=np.where(gg.selected, "C3", "C0"), alpha=.5, label=cid)
        ax.set_title(toy, fontsize=8); ax.set_xlabel("blind loss (finalists)", fontsize=6); ax.set_ylabel("MCC (eval only)", fontsize=6); ax.tick_params(labelsize=6)
    axes[0].legend(fontsize=6); fig.suptitle("BS-B finalists: blind objective vs MCC (red = selected; selection never used MCC)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig2_blind_loss_vs_mcc.png", dpi=140); plt.close(fig)


def fig3_oracle_distance():
    f = RES / "oracle_finalist_recovery.csv.gz"
    if not f.is_file():
        return
    r = pd.read_csv(f); a = pd.read_csv(RES / "oracle_recovery.csv"); a = a[a.solution == "best_any"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for fg, col in (("C", "C0"), ("CR", "C3")):
        g = r[(r.fam_group == fg) & (r.dL < 0)]; axes[0].scatter(g.d_sp, g.ideal_MCC, s=2, c=col, alpha=.3, label=f"{fg} lower-loss finalists")
        axes[1].scatter(g.d_sp, g.ideal_leak_max, s=2, c=col, alpha=.3, label=fg)
        aa = a[a.fam_group == fg]; axes[2].scatter(aa.d_sp, aa.transfer_s9_MCC - aa.transfer_s9_supMCC, s=4, c=col, alpha=.5, label=fg)
    axes[0].axhline(.95, ls="--", c="k", lw=.8); axes[1].axhline(.2, ls="--", c="k", lw=.8); axes[2].axhline(-.05, ls="--", c="k", lw=.8)
    axes[0].set(xlabel="d_SP / sqrt(2k)", ylabel="ideal MCC"); axes[1].set(xlabel="d_SP / sqrt(2k)", ylabel="ideal max leak")
    axes[2].set(xlabel="d_SP of best analytic solution", ylabel="transferred MCC - supervised (S9)")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig3_oracle_distance_vs_recovery.png", dpi=140); plt.close(fig)


def fig4_fine_angle():
    f = RES / "oracle_fine_sweeps.csv.gz"
    if not f.is_file():
        return
    fs = pd.read_csv(f); mets = ["L03|UX.lq.1", "L06|B08.M03.c0.01", "L03|SL0.UX.e0.03", "L06|B09.w50.M03.c0.01"]
    tg = sorted(fs.target.unique()); fig, axes = plt.subplots(len(mets), len(tg), figsize=(2.2 * len(tg), 2.0 * len(mets)), squeeze=False)
    for i, m in enumerate(mets):
        for j, t in enumerate(tg):
            g = fs[(fs.metric == m) & (fs.target == t) & fs.path.str.startswith(("plane", "dir"))]; ax = axes[i, j]
            for pth, gg in g.groupby("path"):
                gg = gg.sort_values("theta"); ax.plot(gg.theta, gg.dL, lw=.6, c="C0" if pth.startswith("plane") else "C1", alpha=.7)
            ax.axhline(0, c="k", lw=.5); ax.set_title(f"{m.split('|')[1]}\n{t}", fontsize=5); ax.tick_params(labelsize=5)
    fig.tight_layout(); fig.savefig(FIG / "fig4_fine_angle_curves.png", dpi=150); plt.close(fig)


def fig5_conditional():
    f = RES / "conditional_coverage.csv"
    if not f.is_file():
        return
    c = pd.read_csv(f); c = c[(c.fid == "pilot") & (c.seed == 31)]
    cfgs = ["P0", "P1", "P2", "P4K4", "P4K8", "P4K16", "P4K8eq"]; mets = sorted(c.metric.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, col, ttl in zip(axes, ["CR_O_perfect", "CR_I_marginal", "C_I_marginal"], ["CR analytic perfect (of 12)", "CR inferred marginal (of 12)", "C inferred marginal (of 12)"]):
        M = np.array([[c[(c.metric == m) & (c.config == k)][col].sum() if len(c[(c.metric == m) & (c.config == k)]) else np.nan for k in cfgs] for m in mets])
        ax.imshow(M, vmin=0, vmax=12, cmap="magma"); ax.set_xticks(range(len(cfgs))); ax.set_xticklabels(cfgs, rotation=45, fontsize=7); ax.set_yticks(range(len(mets)))
        ax.set_yticklabels(mets, fontsize=6); ax.set_title(ttl, fontsize=9)
        for i in range(len(mets)):
            for j in range(len(cfgs)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=6, color="w" if M[i, j] < 7 else "k")
    fig.tight_layout(); fig.savefig(FIG / "fig5_pooled_vs_conditional.png", dpi=140); plt.close(fig)


def fig6_surrogates():
    rows = [pd.read_csv(RES / f"inference_{b}_summary.csv.gz") for b in ("IF-A", "IF-B") if (RES / f"inference_{b}_summary.csv.gz").is_file()]
    if not rows:
        return
    d = pd.concat(rows, ignore_index=True); d = d[d["mask"].isin(["FULL", "MASK-ALL"]) & d.toy.str.startswith("CR")]
    d["selector"] = d.get("selector", pd.Series(index=d.index, dtype=object)).fillna("-")
    g = d.groupby(["estimator", "source", "selector", "setting", "toy"]).agg(UX=("UX_rel_err", "median"), UU=("UU_rel_err", "median"), Imarg=("I_marginal", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (est, src), gg in g.groupby(["estimator", "source"]):
        axes[0].scatter(gg.UX, gg.Imarg, s=10, label=f"{est} {src}"); axes[1].scatter(gg.UU, gg.Imarg, s=10)
    axes[0].set(xlabel="UX relative error (CR)", ylabel="inferred marginal fraction (95 priority metrics)"); axes[1].set(xlabel="UU relative error (CR)")
    axes[0].legend(fontsize=5); fig.tight_layout(); fig.savefig(FIG / "fig6_surrogate_derivative_vs_rotation.png", dpi=140); plt.close(fig)


def fig7_pass_matrix():
    g = pd.read_csv(RES / "blind_toy_variant.csv"); g = g[(g.batch == "BS-B")]
    cids = ["E01", "E02", "C01", "C02", "C03", "C04", "N03"]; rows = [(t, v) for t in PRIMARY for v in VARIANTS]
    M = np.array([[g[(g.toy == t) & (g.variant == v) & (g.cid == c)].pipe(lambda s: s.n_pass.iloc[0] / s.n.iloc[0] if len(s) else np.nan) for c in cids] for t, v in rows])
    fig, ax = plt.subplots(figsize=(5, 9)); ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn"); ax.set_xticks(range(len(cids))); ax.set_xticklabels(cids)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{t} {v}" for t, v in rows], fontsize=6)
    for i in range(len(rows)):
        for j in range(len(cids)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5)
    ax.set_title("BS-B: fraction of seeds passing the Lab 05 criterion", fontsize=8); fig.tight_layout(); fig.savefig(FIG / "fig7_blind_pass_matrix.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    for f in (fig1_heatmaps, fig2_loss_vs_mcc, fig3_oracle_distance, fig4_fine_angle, fig5_conditional, fig6_surrogates, fig7_pass_matrix):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e))
