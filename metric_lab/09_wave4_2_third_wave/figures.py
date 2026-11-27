"""Third-wave figures, every one generated from saved numeric outputs in results/ (03_third_wave section 27).  usage: python figures.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import RES, PRIMARY
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)


def read(n):
    f = RES / n
    return pd.read_csv(f) if f.is_file() else None


def fig1_decomposition():
    d = read("repair_decomposition.csv.gz")
    if d is None:
        return
    t = d[d.axis == "total"].groupby(["toy", "variant"])[["score_term", "weight_term", "dL"]].mean().reset_index()
    t["lab"] = t.toy + " " + t.variant.str.replace("s9_2", "S9_2").str.replace("s9", "S9")
    fig, ax = plt.subplots(figsize=(11, 4)); x = np.arange(len(t))
    ax.bar(x - .2, t.score_term, .4, label="row-score term  sum wbar ds"); ax.bar(x + .2, t.weight_term, .4, label="weight term  sum sbar dw")
    ax.plot(x, t.dL, "k.", label="L(R_hat) - L(R*)"); ax.axhline(0, c="grey", lw=.5); ax.set_xticks(x); ax.set_xticklabels(t.lab, rotation=70, fontsize=7)
    ax.set_ylabel("E02 loss change (TRAIN cache)"); ax.set_title("W3-01 RW-A: exact decomposition of the E02 blind solution vs R* (mean of 9 cells)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig1_row_decomposition.png", dpi=150); plt.close(fig)


def fig2_repair_heatmap():
    c = read("coverage.csv"); loc = read("local_flags.csv.gz")
    if c is None:
        return
    r = c[(c.batch == "RW-C")].copy()
    if not len(r):
        return
    r = r.sort_values("cid"); fig, ax = plt.subplots(figsize=(6, 11)); M = r[["S9 toys", "S9_2 toys"]].fillna(0).values
    ax.imshow(M, cmap="viridis", vmin=0, vmax=12, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(2):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", color="w", fontsize=6)
    ax.set_yticks(range(len(r))); ax.set_yticklabels(r.cid, fontsize=6); ax.set_xticks([0, 1]); ax.set_xticklabels(["S9", "S9_2"])
    ax.set_title("W3-01 RW-C blind pilot: toys recovered /12\n(all 3 search seeds, seed-31 TRAIN cache)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig2_repair_coverage.png", dpi=150); plt.close(fig)


def fig3_pareto():
    p = read("pareto_front.csv")
    if p is None:
        return
    toys = ["CR3", "CR5", "CR4", "C3"]; fig, axs = plt.subplots(2, 4, figsize=(14, 6.5))
    for j, t in enumerate(toys):
        for i, v in enumerate(("s9", "s9_2")):
            ax = axs[i, j]; d = p[(p.toy == t) & (p.variant == v)]
            if not len(d):
                continue
            ax.scatter(d.z_E, d.z_X, c=np.where(d.pass_lab05, "tab:green", "tab:red"), s=8, alpha=.5)
            f = d[d.on_front].sort_values("z_E"); ax.plot(f.z_E, f.z_X, "k-", lw=.8)
            ax.set_title(f"{t} {v}  (green = recovered, post hoc)", fontsize=8); ax.set_xlabel("z_E (B08 AD)", fontsize=7); ax.set_ylabel("z_X (SL0 RBF)", fontsize=7)
    fig.suptitle("W3-02 Pareto diagnostics of E / X / fusion finalists (truth colouring evaluation-only)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig3_pareto.png", dpi=150); plt.close(fig)


def fig4_contrast():
    b = read("pilot_recovery.csv")
    if b is None:
        return
    ids = ["ST-DU-glob", "ST-DU-glob-INT", "ST-DU-col", "ST-DU-col-INT", "ST-DA-glob", "ST-DA-glob-INT", "ST-DA-col", "ST-DA-col-INT", "ST-HE-B08c03", "ST-HE-B01c03"]
    d = b[b.cid.isin(ids) & b.toy.isin(["CR3", "CR5", "CR3L", "CR5L"])]
    if not len(d):
        return
    g = d.groupby(["cid", "variant"]).MCC_gap.mean().unstack().reindex(ids)
    fig, ax = plt.subplots(figsize=(9, 4)); g.plot.bar(ax=ax); ax.set_ylabel("mean MCC gap to supervised"); ax.set_title("W3-03 ST-C bank designs: full rows vs interior rows vs pooled HE")
    fig.tight_layout(); fig.savefig(FIG / "fig4_contrast_full_vs_interior.png", dpi=150); plt.close(fig)


def fig5_local():
    l = read("local_flags.csv.gz")
    if l is None:
        return
    p = l[(l.fidelity == "pilot") & l["mask"].isin(["FULL", "MASK-ALL", "ALLROWS"])]
    g = p.groupby("cid")[["O_perfect", "O_marginal", "I_marginal", "both_marginal"]].sum().sort_values("both_marginal", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 18)); ax.imshow(g.values, cmap="viridis", vmin=0, vmax=24, aspect="auto")
    for i in range(g.shape[0]):
        for j in range(4):
            ax.text(j, i, int(g.values[i, j]), ha="center", va="center", color="w", fontsize=5)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index, fontsize=5); ax.set_xticks(range(4)); ax.set_xticklabels(g.columns, fontsize=7, rotation=20)
    ax.set_title("Local Gap 1 / Gap 2 flags (/24 libraries)", fontsize=8); fig.tight_layout(); fig.savefig(FIG / "fig5_local_flags.png", dpi=150); plt.close(fig)


def fig6_context():
    d = read("context_diagnostics.csv"); r = read("context_recovery.csv")
    if d is None:
        return
    order = ["N|31|25", "N|31|50", "N|31|100", "N|31|200", "N|31|all", "full"]; d = d[d.rep.isin(order)]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for t, g in d.groupby("toy"):
        g = g.set_index("rep").reindex(order); axs[0].plot(range(len(order)), g.R2_mean, marker="o", label=t); axs[1].plot(range(len(order)), g.sup_MCC_s9_2, marker="o")
    if r is not None:
        rr = r[r.variant == "s9_2"].groupby(["cid", "rep"]).MCC_gap.mean().unstack().reindex(columns=order)
        for cid, row in rr.iterrows():
            axs[2].plot(range(len(order)), row.values, marker="o", label=cid)
        axs[2].legend(fontsize=6)
    for ax, ttl in zip(axs, ("linear T~Z R2 (TEST)", "same-context supervised MCC (S9_2)", "blind MCC gap (S9_2, mean over toys)")):
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, fontsize=7); ax.set_title(ttl, fontsize=8)
    axs[0].legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(FIG / "fig6_context.png", dpi=150); plt.close(fig)


def fig7_heatmaps():
    """Locked candidate (results/algorithm_lock.json) and the E02 baseline (S9_2): |corr(T, U)| after matching, every toy, first seed cell."""
    import json
    f = RES / "finalist_found_rotations.npz"; fr = read("finalist_recovery.csv")
    if fr is None or not f.is_file():
        return
    from wave2.blind_inputs import load_payload
    from wave2.evaluate_recovery import EvalData, matched, v_test
    lk = RES / "algorithm_lock.json"
    L = json.loads(lk.read_text()) if lk.is_file() else dict(candidate=fr[fr.batch == "FIN"].groupby("cid").pass_lab05.mean().idxmax(), variant="s9_2")
    panels = [(L["candidate"], L["variant"], "locked"), ("MX-B08c03@AD", "s9_2", "E02 baseline")]
    R = np.load(f); fig, axs = plt.subplots(4, 6, figsize=(16, 11))
    for p_i, (cid, var, lab) in enumerate(panels):
        for ax, toy in zip(axs[2 * p_i:2 * p_i + 2].ravel(), PRIMARY):
            rows = fr[(fr.cid == cid) & (fr.toy == toy) & (fr.variant == var) & (fr.batch == "FIN")]
            ax.set_xticks([]); ax.set_yticks([])
            if not len(rows):
                continue
            rid = rows.iloc[0].row_id; pl = load_payload(toy); ed = EvalData(toy); V = v_test(pl, ed, var); m = matched(ed.T[ed.test], V @ R[rid].T)
            C = np.abs(m["signed"])[:, m["perm"]]; ax.imshow(C, vmin=0, vmax=1, cmap="magma")
            ax.set_title(f"{lab}: {toy} {var} | MCC {m['MCC']:.3f} leak {m['leak_max']:.2f} {'pass' if rows.iloc[0].pass_lab05 else 'FAIL'}", fontsize=7)
    fig.suptitle(f"|corr(T, U)| after Hungarian matching (first seed cell): rows 1-2 locked {L['candidate']} {L['variant']}, rows 3-4 E02 S9_2", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig7_blind_heatmaps.png", dpi=150); plt.close(fig)


def fig8_oracle():
    o = read("oracle_summary.csv")
    if o is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cid, g in o.groupby("cid"):
        ax.scatter(g.best_dsp, -g.best_dL, label=cid, s=18)
    ax.axvline(.1, c="grey", ls=":"); ax.axvline(.2, c="grey", ls="--"); ax.set_xlabel("d_SP of the analytic global minimum"); ax.set_ylabel("L(I) - L(best)")
    ax.legend(fontsize=6); ax.set_title("W3-05 exact vs practical Gap 1 (8 analytic targets)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig8_oracle_global.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    for f in (fig1_decomposition, fig2_repair_heatmap, fig3_pareto, fig4_contrast, fig5_local, fig6_context, fig7_heatmaps, fig8_oracle):
        try:
            f(); print("ok", f.__name__)
        except Exception as ex:
            print("FAILED", f.__name__, ex)
