"""S9 subject-level variance equality diagnostics and RZ axis correlations.

Pitman-Morgan tests exploit *paired* axis values from the same subjects;
the test assumes bivariate normality for exact Student-t inference.
Never treat the latent axes as independent groups for Levene/Bartlett.
"""
from __future__ import annotations
import csv
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def correlation_of_axes(x):
    """Pearson correlation across subjects, with zero-variance axes set to 0."""
    x=np.asarray(x,dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite subject-by-axis array with >=2 rows")
    centered=x-x.mean(axis=0,keepdims=True)
    gram=centered.T@centered
    den=np.sqrt(np.maximum(np.diag(gram),0.0))
    out=np.divide(gram,den[:,None]*den[None,:],
                  out=np.zeros_like(gram),where=(den[:,None]*den[None,:])>1e-24)
    out=np.clip(out,-1.0,1.0)
    np.fill_diagonal(out,(den>1e-12).astype(float))
    return out


def paired_variance_comparison(z, axis_names=None, alpha=.05):
    """Axis-wise variance and pairwise Pitman-Morgan with Holm p correction.

    This is not a proof of exact homoscedasticity, merely a statistical
    diagnostic; equality of variances is weaker than Gaussian isotropy.
    """
    z=np.asarray(z,dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 4 or z.shape[1] < 2:
        raise ValueError("Need >=4 subjects and >=2 axes")
    if not np.isfinite(z).all():
        raise ValueError("Z has nonfinite values")
    d=z.shape[1]
    axis_names = list(axis_names) if axis_names is not None else [
        f"z{i+1}" for i in range(d)]
    if len(axis_names)!=d:
        raise ValueError("Axis names must match Z dimensions")
    var=np.var(z,axis=0,ddof=1)
    std=np.sqrt(var)
    positive=var[var>1e-20]
    ratio=float(np.max(var)/np.min(var)) if np.min(var)>1e-20 else float("inf")
    rows=[dict(axis=axis_names[j],mean=float(z[:,j].mean()),
               variance=float(var[j]),std=float(std[j]),
               variance_ratio_to_mean=float(var[j]/np.mean(var))
               if np.mean(var)>0 else None)
          for j in range(d)]
    pairs=[]
    n=len(z)
    for i,j in combinations(range(d),2):
        # Pitman-Morgan: equal marginal variances iff Cov(X+Y,X-Y)=0.
        s=z[:,i]+z[:,j]
        diff=z[:,i]-z[:,j]
        if np.std(s,ddof=1)<1e-12 or np.std(diff,ddof=1)<1e-12:
            p=1.0 if abs(var[i]-var[j])<=1e-12 else 0.0
            statistic=None
        else:
            r=float(np.corrcoef(s,diff)[0,1])
            statistic=float(r*np.sqrt((n-2)/max(1e-15,1-r*r)))
            p=float(2*stats.t.sf(abs(statistic),df=n-2))
        pairs.append(dict(axis_i=axis_names[i],axis_j=axis_names[j],
                          variance_i=float(var[i]),variance_j=float(var[j]),
                          variance_difference=float(var[i]-var[j]),
                          pitman_morgan_t=statistic,p_raw=p))
    # Holm's step-down familywise adjustment for the m paired tests.
    sorted_idx=sorted(range(len(pairs)),key=lambda idx:pairs[idx]["p_raw"])
    running=0.0
    m=len(pairs)
    for rank,idx in enumerate(sorted_idx):
        running=max(running,min(1.,pairs[idx]["p_raw"]*(m-rank)))
        pairs[idx]["p_holm"]=float(running)
        pairs[idx]["reject_equal_variance"]=bool(running<alpha)
    summary=dict(n_subjects=int(n),n_axes=int(d),alpha=float(alpha),
                 variances=var.tolist(),mean_variance=float(var.mean()),
                 min_variance=float(var.min()),max_variance=float(var.max()),
                 max_min_variance_ratio=ratio if np.isfinite(ratio) else None,
                 any_zero_variance_axis=bool(np.min(var)<=1e-20),
                 coefficient_of_variation_of_axis_variances=float(
                     var.std(ddof=0)/var.mean()) if var.mean()>0 else None,
                 pairwise_comparisons=m,
                 significant_pair_count=int(sum(p["reject_equal_variance"] for p in pairs)),
                 all_pairwise_equal_variances_not_rejected=bool(
                     not any(p["reject_equal_variance"] for p in pairs)),
                 method="Paired Pitman-Morgan t tests of equality of marginal "
                        "variances; Holm step-down correction",
                 interpretation="Non-rejection does not prove equal variances; "
                        "p-values use a paired bivariate-Gaussian assumption.")
    return rows,pairs,summary


def _save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_z_variance_diagnostics(z,splits,folder):
    from .files import save_json
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    z=np.asarray(z,dtype=np.float64)
    groups=dict(all=np.arange(len(z),dtype=int),
                train=np.asarray(splits["train"],dtype=int),
                validation=np.asarray(splits["validation"],dtype=int),
                test=np.asarray(splits["test"],dtype=int))
    reports={}
    fig, ax=plt.subplots(figsize=(max(7,z.shape[1]*1.1),4.5))
    for key,ids in groups.items():
        rows,pairs,report=paired_variance_comparison(z[ids])
        _save_csv(folder/f"z_variances_{key}.csv",rows)
        _save_csv(folder/f"paired_variance_tests_{key}.csv",pairs)
        reports[key]=report
        if key=="all":
            ax.bar(range(z.shape[1]),report["variances"])
    ax.set(xlabel="Original Z coordinate (posterior mean)",
           ylabel="Between-subject sample variance (ddof=1)",
           title="Original Z between-subject variance by axis")
    ax.set_xticks(range(z.shape[1]),[f"z{i+1}" for i in range(z.shape[1])])
    fig.tight_layout()
    fig.savefig(folder/"z_axis_variances.png",dpi=170)
    plt.close(fig)
    save_json(dict(
        definition="S5 all-trial posterior means; each row is one subject",
        primary_inference_split="train",
        note="Pairwise tests use correlated/paired axis observations; "
             "equal variance does not establish rotational invariance "
             "without a joint isotropic Gaussian distribution",
        splits=reports),folder/"z_variance_equal_tests.json")
    return reports


def save_rz_axis_correlation(z_rotated,folder):
    """All L x L rotated axes, TEST subjects, never use T here."""
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    correlation=correlation_of_axes(z_rotated)
    names=[f"RZ{i+1}" for i in range(correlation.shape[1])]
    with (folder/"rz_axis_correlations.csv").open(
            "w",newline="",encoding="utf-8") as file:
        w=csv.writer(file)
        w.writerow(["axis",*names])
        for name,row in zip(names,correlation,strict=True):
            w.writerow([name,*[float(x) for x in row]])
    fig,ax=plt.subplots(figsize=(
        max(5,1.0*len(names)+2), max(4,0.9*len(names)+2)))
    im=ax.imshow(correlation,cmap="coolwarm",vmin=-1,vmax=1)
    ax.set(xticks=range(len(names)),yticks=range(len(names)),
           xticklabels=names,yticklabels=names,
           xlabel="Rotated latent axis",ylabel="Rotated latent axis",
           title="TEST correlation among all rotated Z axes")
    for i in range(len(names)):
        for j in range(len(names)):
            value=correlation[i,j]
            ax.text(j,i,f"{value:+.2f}",ha="center",va="center",
                    fontsize=8,color="white" if abs(value)>.60 else "black")
    fig.colorbar(im,ax=ax,shrink=.82,label="Pearson r")
    fig.tight_layout()
    fig.savefig(folder/"rz_axis_correlations.png",dpi=175)
    plt.close(fig)
    return correlation
