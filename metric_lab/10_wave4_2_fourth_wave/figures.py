"""Figures from saved numeric results (EVALUATION-ONLY; run after the stages are evaluated).  -> results/figures/fig1..fig6*.png"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4.common import RES, OUT09, PRIMARY, SECONDARY, CAND, E02, prim_mask, leak_threshold
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = RES / "figures"


def read(n):
    f = RES / n
    return pd.read_csv(f) if f.is_file() else None


def fig1_heatmaps():
    from wave2.blind_inputs import load_payload
    from wave2.evaluate_recovery import EvalData, recovery, supervised, v_test, matched
    from wave4.search import job_path
    for variant in ("s9", "s9_2"):
        fig, ax = plt.subplots(12, 3, figsize=(9.5, 36))
        for i, toy in enumerate(PRIMARY):
            pl = load_payload(toy); ed = EvalData(toy); mask = prim_mask(pl); sup = supervised(pl, ed, variant)
            R5 = np.load(job_path("REP-1", toy, CAND, "Q05"))[f"{variant}|Q05@C|{mask}|i31|s31|S0|R"]
            R1 = np.load(OUT09 / "blind" / "FIN" / toy / f"{CAND}.npz")[f"{variant}|Q01|{mask}|i31|s31|S0|R"]
            V = v_test(pl, ed, variant); ref = ed.refs[variant]
            for j, (lab, U) in enumerate((("Q05 (W4-Q05)", V @ R5.T), ("Q01 (W3 lock)", V @ R1.T), ("supervised R*", (V - ref["z_center"]) @ ref["R_star"].T))):
                m = matched(ed.T[ed.test], U); C = m["signed"][:, m["perm"]]
                ok = m["MCC"] >= sup["MCC"] - .05 and m["leak_max"] < leak_threshold(toy)
                ax[i, j].imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
                for a in range(C.shape[0]):
                    for b in range(C.shape[1]):
                        ax[i, j].text(b, a, f"{C[a, b]:+.2f}", ha="center", va="center", fontsize=6)
                ax[i, j].set_title(f"{toy} {lab}\nMCC {m['MCC']:.3f} leak {m['leak_max']:.3f}" + ("" if j == 2 else (" pass" if ok else " FAIL")), fontsize=7)
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        fig.suptitle(f"TEST T x U (Hungarian-matched columns), {variant}, iseed 31 / sseed 31", fontsize=9); fig.tight_layout(); fig.savefig(FIG / f"fig1_heatmaps_{variant}.png", dpi=110); plt.close(fig)


def fig2_replication():
    rep = read("rep_recovery.csv")
    if rep is None:
        return
    m = rep[(rep.cid == CAND) & (rep.Q == "Q05") & (rep.budget == "S0") & (rep.panel == "primary")].copy()
    for variant in ("s9", "s9_2"):
        x = m[m.variant == variant].copy(); x["col"] = x.batch + " " + x.fidelity + " i" + x.iseed.astype(str) + " s" + x.sseed.astype(str)
        P = x.pivot_table(index="toy", columns="col", values="pass_lab05", aggfunc="first").reindex(PRIMARY)
        G = x.pivot_table(index="toy", columns="col", values="MCC_gap", aggfunc="first").reindex(PRIMARY)
        Lk = x.pivot_table(index="toy", columns="col", values="leak_max", aggfunc="first").reindex(PRIMARY)
        fig, ax = plt.subplots(1, 3, figsize=(22, 5.5))
        ax[0].imshow(P.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"); ax[0].set_title(f"pass (Lab 05), {variant}: {int(P.values.sum())}/{P.size} runs")
        for a, (M, t) in zip(ax[1:], ((G, "MCC gap (supervised - blind); line .05"), (Lk, "max leakage; CR .3 / C .2"))):
            for i, toy in enumerate(PRIMARY):
                a.scatter(M.loc[toy].values, np.full(M.shape[1], i) + np.random.default_rng(i).uniform(-.2, .2, M.shape[1]), s=8)
            a.set_yticks(range(12)); a.set_yticklabels(PRIMARY); a.set_title(t)
        ax[1].axvline(.05, color="k", lw=.8); ax[2].axvline(.2, color="k", lw=.8, ls=":"); ax[2].axvline(.3, color="k", lw=.8)
        ax[0].set_yticks(range(12)); ax[0].set_yticklabels(PRIMARY); ax[0].set_xticks(range(P.shape[1])); ax[0].set_xticklabels(P.columns, rotation=90, fontsize=5)
        fig.tight_layout(); fig.savefig(FIG / f"fig2_replication_{variant}.png", dpi=110); plt.close(fig)


def fig3_landscape():
    g = read("cr4_geodesics.csv.gz"); mr = read("measure_recovery.csv")
    if g is None:
        return
    toys = [t for t in ("CR4", "CR4L", "CR3", "C1") if t in set(g.toy)]
    fig, ax = plt.subplots(len(toys), 3, figsize=(16, 3.6 * len(toys)), squeeze=False)
    cols = [c for c in g.columns if c.startswith("L_")]
    for i, toy in enumerate(toys):
        for j, path in enumerate(("R_Q05->R_Q01", "R_star->R_Q01", "R_star->R_Q05")):
            x = g[(g.toy == toy) & (g.variant == "s9") & (g.path == path)]
            for c in cols:
                y = x[c].values; ax[i, j].plot(x.t, y - y[np.argmin(np.abs(x.t.values))], label=c[2:], lw=1.4 if c in ("L_Q01", "L_M-D", "L_M-E", "L_M-J") else .7)
            ax[i, j].axvline(0, color="k", lw=.5); ax[i, j].axvline(1, color="k", lw=.5); ax[i, j].set_title(f"{toy} s9 geodesic {path} (L - L(start))", fontsize=8)
        ax[i, 0].legend(fontsize=5, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "fig3_cr4_landscape.png", dpi=110); plt.close(fig)
    if mr is not None:
        ms = mr[mr.batch.isin(["MS", "MIX"])]; c = ms.groupby(["Q", "toy"]).pass_lab05.mean().unstack()
        fig, ax = plt.subplots(figsize=(10, 7)); ax.imshow(c.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"); ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index, fontsize=7)
        ax.set_xticks(range(c.shape[1])); ax.set_xticklabels(c.columns, rotation=45); ax.set_title("sensitivity laws: run pass rate (s9; MS seed 31 x 3, MIX 3 x 3)")
        fig.tight_layout(); fig.savefig(FIG / "fig3b_sensitivity.png", dpi=110); plt.close(fig)


def fig4_support():
    te = read("tail_energy.csv"); sd = read("support_diagnostics.csv")
    if te is None or sd is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    p = te[te["mask"].isin(["FULL", "MASK-ALL"])].pivot_table(index="law", columns="toy", values="energy_tail2")
    ax[0].imshow(p.values, cmap="viridis", aspect="auto"); ax[0].set_yticks(range(len(p))); ax[0].set_yticklabels(p.index, fontsize=7); ax[0].set_xticks(range(p.shape[1]))
    ax[0].set_xticklabels(p.columns, rotation=45); ax[0].set_title("share of UX energy from draws with max|z_std| > 2")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax[0].text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", fontsize=5, color="w")
    q = sd.pivot_table(index="law", columns="toy", values="frac_maha_gt_val99")
    ax[1].imshow(q.values, cmap="magma", aspect="auto"); ax[1].set_yticks(range(len(q))); ax[1].set_yticklabels(q.index, fontsize=7); ax[1].set_xticks(range(q.shape[1]))
    ax[1].set_xticklabels(q.columns, rotation=45); ax[1].set_title("fraction of latent draws beyond the 99% VALIDATION Mahalanobis radius")
    fig.tight_layout(); fig.savefig(FIG / "fig4_support_energy.png", dpi=110); plt.close(fig)


def fig5_oracle():
    og = read("oracle_global.csv"); g2 = read("gap2_curves.csv.gz")
    if og is None and g2 is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    if og is not None:
        for c, x in og.groupby("cid"):
            ax[0].scatter(x.best_dsp, x.best_dL, label=c)
            for _, r in x.iterrows():
                ax[0].annotate(r.target.replace(":L:", " "), (r.best_dsp, r.best_dL), fontsize=5)
        ax[0].axhline(0, color="k", lw=.5); ax[0].set_xlabel("d_SP of best finalist"); ax[0].set_ylabel("L(best) - L(I)"); ax[0].legend(fontsize=7); ax[0].set_title("Gap 1 (O-true)")
    if g2 is not None:
        p = g2[g2.mask_role == "primary"].groupby(["cid", "inferred", "seed"]).agg(both=("both_marginal", "sum"), n=("toy", "size")).reset_index()
        lab = p.cid.str[:10] + " " + p.inferred + " s" + p.seed.astype(str); ax[1].barh(range(len(p)), p.both / p.n); ax[1].set_yticks(range(len(p))); ax[1].set_yticklabels(lab, fontsize=6)
        ax[1].set_title("Gap 2: fraction of libraries (toy x variant) with both-marginal minima")
    fig.tight_layout(); fig.savefig(FIG / "fig5_oracle_gap.png", dpi=110); plt.close(fig)


def fig6_secondary_confirmation():
    rb = read("robustness_recovery.csv"); c = read("confirmation_recovery.csv")
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    if rb is not None:
        s = rb[rb.panel == "secondary"].dropna(subset=["MCC"]).copy(); s["pass_lab05"] = s.pass_lab05.astype(float); p = s.groupby(["toy", "cid", "Q", "variant"]).pass_lab05.mean().unstack(["cid", "Q", "variant"])
        ax[0].imshow(p.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"); ax[0].set_yticks(range(len(p))); ax[0].set_yticklabels(p.index)
        ax[0].set_xticks(range(p.shape[1])); ax[0].set_xticklabels([" ".join(map(str, c_)) for c_ in p.columns], rotation=60, fontsize=6); ax[0].set_title("secondary saved NPs: run pass rate")
    if c is not None:
        c["pass_lab05"] = c.pass_lab05.astype(float); c["arm"] = c.cid.str[:10] + " " + c.Q + " " + c.variant; p = c.groupby(["toy_id", "arm"]).pass_lab05.mean().unstack()
        ax[1].imshow(p.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"); ax[1].set_yticks(range(len(p))); ax[1].set_yticklabels(p.index, fontsize=5)
        ax[1].set_xticks(range(p.shape[1])); ax[1].set_xticklabels(p.columns, rotation=60, fontsize=6); ax[1].set_title("CONF-A: run pass rate per NP (all 9 seeds)")
    fig.tight_layout(); fig.savefig(FIG / "fig6_secondary_confirmation.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig2_replication, fig3_landscape, fig4_support, fig5_oracle, fig6_secondary_confirmation, fig1_heatmaps):
        try:
            f(); print("ok", f.__name__, flush=True)
        except Exception as e:
            print("FAILED", f.__name__, e, flush=True)
