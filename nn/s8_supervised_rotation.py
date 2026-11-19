"""S8: supervised orthogonal rotation R* toward T, then TEST-only analysis."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modrot4_nn.cli import parse_args
from modrot4_nn.pipeline import (
    stage_folder, sha256, dataset_path, train_path, latent_path, condition)
from modrot4_nn.analysis import load_frozen, corr, ridge_probe, param_names
from modrot4_nn.supervised_rotation import fit_supervised_rotation
from modrot4_nn.files import save_json


def _matrix_csv(path, matrix, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.writer(handle)
        writer.writerow(["T_true",*columns])
        for name, row in zip(rows, matrix, strict=True):
            writer.writerow([name,*[float(x) for x in row]])


def _plot_matrix(matrix, row_names, col_names, title, path):
    nrow, ncol=matrix.shape
    fig, ax=plt.subplots(figsize=(max(6,1.25*ncol+2),max(4,1.1*nrow+1.5)))
    im=ax.imshow(matrix,vmin=-1,vmax=1,cmap="coolwarm")
    ax.set(xticks=np.arange(ncol),yticks=np.arange(nrow),
           xticklabels=col_names,yticklabels=row_names,
           xlabel="Rotated axis / predicted parameter",
           ylabel="True parameter",title=title)
    for i in range(nrow):
        for j in range(ncol):
            value=matrix[i,j]
            ax.text(j,i,f"{value:+.3f}",ha="center",va="center",
                    color="white" if abs(value)>.63 else "black",fontsize=10)
    fig.colorbar(im,ax=ax,label="TEST Pearson r",shrink=.82)
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def _plot_scatter_grid(theta, values, row_names, col_names, correlations,
                       title, path, max_points=2000):
    """All T_i x U_j or all T_i x Tpred_j; share TEST subjects and labels."""
    nrow,ncol=len(row_names),len(col_names)
    fig,axs=plt.subplots(nrow,ncol,figsize=(max(8,ncol*3.25),max(6,nrow*3)),
                         squeeze=False)
    if len(theta)>max_points:
        show=np.linspace(0,len(theta)-1,max_points,dtype=int)
    else:
        show=np.arange(len(theta))
    for i in range(nrow):
        for j in range(ncol):
            axis=axs[i,j]
            axis.scatter(theta[show,i],values[show,j],s=7,alpha=.30,
                         linewidths=0,rasterized=True)
            axis.set_title(f"{row_names[i]} vs {col_names[j]}   r={correlations[i,j]:+.3f}",
                           fontsize=9)
            if i==nrow-1:axis.set_xlabel(f"True {row_names[i]}")
            if j==0:axis.set_ylabel(col_names[j])
    fig.suptitle(title,fontsize=14)
    fig.tight_layout(rect=(0,0,1,.965))
    fig.savefig(path,dpi=155)
    plt.close(fig)


def main(args):
    z, theta, splits, z_source=load_frozen(args)
    tr, va, te = (np.asarray(splits[k],dtype=np.int64)
                   for k in ("train","validation","test"))
    n, latent_dim=z.shape
    t_dim=theta.shape[1]
    if latent_dim < t_dim:
        raise ValueError(
            f"S8 needs latent_dim >= T_dim; got {latent_dim} < {t_dim}. "
            "Train a larger NP latent dimension before S8.")

    destination=stage_folder(args,"s8_supervised_rotation")
    # Prevent partial overwrites or accidental mixing of previous S8 settings.
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"S8 output already exists: {destination}. "
                              "Archive it before regenerating.")
    rotated=fit_supervised_rotation(z,theta,tr)
    rz=rotated["rz"]
    u=rotated["u"]                      # use exactly first T_dim rotated Z axes
    labels=param_names(args.model)
    if len(labels)!=t_dim:
        raise ValueError("Parameter-name count does not match true T dimension")
    axis_labels=[f"RZ{j+1}" for j in range(t_dim)]

    direct_test_corr=corr(theta[te],u[te])
    # Multi-target Ridge, standardized only from TRAIN; select alpha using
    # VALIDATION, refit TRAIN+VAL; report correlations using untouched TEST.
    probe=ridge_probe(u,theta,tr,va,te)
    prediction=np.asarray(probe["theta_linear_pred"],dtype=np.float64)
    predicted_test_corr=corr(theta[te],prediction)
    destination.mkdir(parents=True,exist_ok=False)

    np.savez_compressed(
        destination/"rotation.npz",
        R_star=rotated["R"],R=rotated["R"],
        z_train_mean=rotated["z_train_mean"],
        t_train_mean=rotated["t_train_mean"],
        t_train_std=rotated["t_train_std"],
        singular_values=rotated["singular_values"],
        n_latent=np.array(latent_dim),n_theta=np.array(t_dim),
        checkpoint_sha256=np.array(sha256(train_path(args))),
        latent_sha256=np.array(sha256(z_source)),
        dataset_sha256=np.array(sha256(dataset_path(args.model))),
        train_ids=tr,validation_ids=va,test_ids=te)
    np.savez_compressed(destination/"rotated_z.npz",
                        rz=rz,u=u,
                        train_ids=tr,validation_ids=va,test_ids=te,
                        rotation_policy=np.array("full LxL orthogonal R; first T_dim axes used"))
    np.savez_compressed(
        destination/"test_predictions.npz",
        subject_ids=te,theta_true=theta[te],theta_pred=prediction,
        u_test=u[te],rz_test=rz[te],
        ridge_alpha=np.asarray(probe["selected_ridge_alpha"],dtype=float))

    _matrix_csv(destination/"rz_vs_t_correlation.csv",direct_test_corr,
                labels,axis_labels)
    _matrix_csv(destination/"pred_vs_t_correlation.csv",predicted_test_corr,
                labels,[f"pred_{s}" for s in labels])
    _plot_matrix(direct_test_corr,labels,axis_labels,
                 f"{args.model}: TEST T vs R*Z (first {t_dim} axes)",
                 destination/"rz_vs_t_correlation.png")
    _plot_matrix(predicted_test_corr,labels,
                 [f"pred_{s}" for s in labels],
                 f"{args.model}: TEST T vs Ridge(R*Z)",
                 destination/"pred_vs_t_correlation.png")
    _plot_scatter_grid(theta[te],u[te],labels,axis_labels,direct_test_corr,
                       "TEST true T versus rotated latent axes",
                       destination/"rz_vs_t_scatter.png")
    _plot_scatter_grid(theta[te],prediction,labels,
                       [f"pred_{s}" for s in labels],predicted_test_corr,
                       "TEST true T versus predicted T from rotated Z",
                       destination/"pred_vs_t_scatter.png")
    report={
        "model":args.model,"condition":condition(args),
        "subjects":n,"split_counts":{k:int(len(v)) for k,v in splits.items()},
        "latent_dim":latent_dim,"theta_dim":t_dim,
        "rotation_shape":[latent_dim,latent_dim],
        "features_used_for_regression":axis_labels,
        "rotation_objective":rotated["objective"],
        "coordinate_convention":rotated["coordinate_convention"],
        "fitting_policy":"R* TRAIN only; Ridge alpha VALIDATION only; Ridge TRAIN+VAL refit; correlations TEST only",
        "procrustes_train_mse":rotated["procrustes_train_mse"],
        "orthogonality_max_error":rotated["orthogonality_max_error"],
        "direct_t_vs_rz_test_corr":direct_test_corr.tolist(),
        "t_vs_pred_test_corr":predicted_test_corr.tolist(),
        "ridge_test_r2":probe["test_r2"],
        "ridge_test_pearson_r":probe["test_pearson_r"],
        "ridge_test_rmse":probe["test_rmse"],
        "ridge_selected_alpha":probe["selected_ridge_alpha"],
        "source_latent_sha256":sha256(z_source),
        "source_checkpoint_sha256":sha256(train_path(args)),
        "source_dataset_sha256":sha256(dataset_path(args.model)),
    }
    save_json(report,destination/"summary.json")
    print(f"[S8] {args.model} R*={latent_dim}x{latent_dim}, "
          f"regression uses first {t_dim} rotated axes",flush=True)
    print("[S8] direct TEST T vs RZ:",np.round(direct_test_corr,3),sep="\n",flush=True)
    print("[S8] TEST T vs predicted T:",np.round(predicted_test_corr,3),sep="\n",flush=True)
    print(f"[S8] Ridge TEST R2={np.round(probe['test_r2'],4)}",flush=True)
    print(f"[S8] saved {destination}",flush=True)


if __name__=="__main__":
    main(parse_args("S8: supervised orthogonal rotation R* toward true T"))
