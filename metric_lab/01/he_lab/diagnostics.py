"""Correlation/ranking of HE metrics across unique R, plus angle curves."""
from collections import defaultdict
import csv
import math
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr


def _rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Run metrics stage first: {path}")
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _csv(path,rows,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def _unique_rotation_ids(rotations):
    # Different subsets' angle zero is the same physical rotation.
    seen=set()
    ids=[]
    for rot in rotations:
        key=np.round(rot.G,decimals=9).tobytes()
        if key not in seen:
            seen.add(key);ids.append(rot.name)
    return set(ids)


def diagnose(root,variant,rotations,selected_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=_rows(root/"metrics"/variant/"all_metrics.csv")
    metric_lookup=defaultdict(dict)
    for r in rows:
        key=(r["section"],r["method"],r["y_name"],r["metric"],r["target"])
        metric_lookup[key][r["rotation_id"]]=float(r["value"])
    oracle={
        (r["metric"],r["target"]):metric_lookup[
            ("oracle","analytic_oracle","true_"+r["target"],r["metric"],r["target"])]
        for r in rows if r["section"]=="oracle"
    }
    unique=_unique_rotation_ids(rotations)
    correlations=[]
    for (section,method,y_name,metric,target),vals in metric_lookup.items():
        if section!="inference":
            continue
        truth=oracle.get((metric,target),{})
        overlap=sorted(unique & vals.keys() & truth.keys())
        a=np.array([truth[x] for x in overlap])
        b=np.array([vals[x] for x in overlap])
        good=np.isfinite(a)&np.isfinite(b)
        a,b=a[good],b[good]
        if len(a)>=3 and a.std()>1e-12 and b.std()>1e-12:
            pr=float(pearsonr(a,b).statistic)
            sr=float(spearmanr(a,b).statistic)
        else:
            pr=sr=float("nan")
        correlations.append(dict(method=method,y_name=y_name,metric=metric,
            target=target,n_unique_rotations=len(a),pearson_r=pr,spearman_r=sr,
            rmse=float(np.sqrt(np.mean((a-b)**2))) if len(a) else float("nan"),
            mae=float(np.mean(abs(a-b))) if len(a) else float("nan"),
            oracle_best_id=(overlap[int(np.nanargmin(np.array([truth[x] for x in overlap])))]
                            if overlap and all(np.isfinite(truth[x]) for x in overlap) else ""),
            inference_best_id=(overlap[int(np.nanargmin(np.array([vals[x] for x in overlap])))]
                               if overlap and all(np.isfinite(vals[x]) for x in overlap) else "")))
    dest=root/"diagnostics"/variant
    cols=("method","y_name","metric","target","n_unique_rotations","pearson_r",
          "spearman_r","rmse","mae","oracle_best_id","inference_best_id")
    _csv(dest/"correlations.csv",correlations,cols)
    # Each axes subset is a distinct ordered Givens path, never mix at the
    # same angle. Baseline is a first-class candidate in CSV but not double-counted.
    by_axes=defaultdict(list)
    for r in rotations:
        if r.candidate_type=="sweep":
            by_axes[r.axes].append(r)
    for key in selected_metrics:
        for axes,subset in by_axes.items():
            subset=sorted(subset,key=lambda r:r.angle)
            x=[r.angle for r in subset]
            for target in ("probability","logit"):
                pair=oracle.get((key,target),{})
                ys=[pair.get(r.name,np.nan) for r in subset]
                if not np.isfinite(ys).any():
                    continue
                fig,ax=plt.subplots(figsize=(7.3,4.6))
                ax.plot(x,ys,marker="o",color="black",linewidth=2,label="Analytic oracle")
                nlines=0
                for (sec,method,y_name,m,t),v in metric_lookup.items():
                    if sec!="inference" or m!=key or t!=target:
                        continue
                    line=[v.get(r.name,np.nan) for r in subset]
                    if not np.isfinite(line).any():
                        continue
                    ax.plot(x,line,marker=".",alpha=.72,label=f"{method}: {y_name}")
                    nlines+=1
                ax.set(xlabel="Givens sweep angle (degrees)",ylabel=key,
                       title=f"{variant}: axes {','.join(str(z+1) for z in axes)} / {target}")
                ax.set_xticks([0,15,30,45,60,75,90])
                ax.grid(alpha=.2)
                if nlines:
                    ax.legend(fontsize=6.5,ncol=2,loc="best")
                fig.tight_layout()
                filename=f"{key.replace('.','__')}_axes_{'_'.join(str(z+1) for z in axes)}_{target}.png"
                target_path=dest/"curves"/filename
                target_path.parent.mkdir(parents=True,exist_ok=True)
                fig.savefig(target_path,dpi=145)
                plt.close(fig)
    print(f"[DIAGNOSE] {variant}: {len(correlations)} comparisons; {dest}")
