"""S9: enumerate reproducible Givens sweeps around S8's frozen R*.

Default: rotate all subsets of the first d=T_dim aligned RZ axes, for
k=2..d and angles 0,15,...,90 degrees. Save every resulting full R
separately. At angle 45, save T x *all* RZ axis correlation and scatter
grids in a separate figures_45deg folder, evaluated on untouched TEST.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from modrot4_nn.cli import parse_args
from modrot4_nn.analysis import load_frozen, corr, param_names
from modrot4_nn.pipeline import (
    condition, dataset_path, latent_path, sha256, stage_folder, train_path)
from modrot4_nn.files import save_json
from modrot4_nn.s9_rotation import (
    ANGLES, axis_subsets, count_subsets, expected_rotation_count,
    rotate_from_oracle, rotation_id)
from s8_supervised_rotation import _matrix_csv, _plot_matrix, _plot_scatter_grid


def configure_parser(parser):
    parser.add_argument(
        "--axes", choices=("theta", "all"), default="theta",
        help="theta: rotate first d=T_dim S8-aligned axes (default); "
             "all: rotate all L latent axes")
    parser.add_argument(
        "--max-test-scatter-points", type=int, default=2000,
        help="Maximum shown TEST subjects per 45-degree scatter cell; "
             "all TEST subjects are used to calculate Pearson correlations")


def _read_oracle(args, z, splits, z_file):
    path = stage_folder(args, "s8_supervised_rotation") / "rotation.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Run S8 first: {path}")
    with np.load(path, allow_pickle=False) as stored:
        required = ("R_star", "n_latent", "n_theta", "latent_sha256",
                    "dataset_sha256", "checkpoint_sha256",
                    "train_ids", "validation_ids", "test_ids")
        missing = sorted(set(required).difference(stored.files))
        if missing:
            raise ValueError(f"S8 rotation.npz is missing {missing}")
        r = np.asarray(stored["R_star"], np.float64)
        if (int(stored["n_latent"]) != z.shape[1] or
                int(stored["n_theta"]) < 1 or
                str(stored["latent_sha256"]) != sha256(z_file) or
                str(stored["dataset_sha256"]) != sha256(dataset_path(args.model)) or
                str(stored["checkpoint_sha256"]) != sha256(train_path(args))):
            raise ValueError("S8 rotation is incompatible with current dataset/checkpoint/Z")
        for name in ("train", "validation", "test"):
            if not np.array_equal(np.asarray(stored[f"{name}_ids"], int),
                                  np.asarray(splits[name], int)):
                raise ValueError(f"S8 and S5 differ in {name} subjects")
        tdim = int(stored["n_theta"])
    if r.shape != (z.shape[1], z.shape[1]):
        raise ValueError("S8 R* does not match latent dimension")
    if not np.allclose(r @ r.T, np.eye(len(r)), atol=1e-7):
        raise ValueError("S8 R* must be orthogonal")
    return r, tdim, path


def main(args):
    if args.max_test_scatter_points < 1:
        raise ValueError("--max-test-scatter-points must be positive")
    z, t, splits, z_file = load_frozen(args)
    r_star, tdim, oracle_file = _read_oracle(args, z, splits, z_file)
    if t.shape[1] != tdim:
        raise ValueError("True T dimension does not match S8")
    n_latent = z.shape[1]
    n_axes = tdim if args.axes == "theta" else n_latent
    if n_axes < 2:
        raise ValueError("At least two selected axes are needed for S9")
    count = count_subsets(n_axes)
    total = expected_rotation_count(n_axes)
    dest = stage_folder(args, "s9_rotation_sweep")
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(
            f"S9 output already exists: {dest}. Archive before rerunning.")
    if args.axes == "all" and n_axes > 12:
        print(f"[S9 WARNING] exhaustive --axes all with L={n_axes} "
              f"will create {total:,} R files and {count:,} two-figure sets.",
              flush=True)
    print(f"[S9] model={args.model} L={n_latent} T_dim={tdim} "
          f"axis_scope={args.axes} subsets={count} "
          f"angles={list(ANGLES)} rotations={total} figures_45deg={count*2}",
          flush=True)

    folder = dest / "rotations"
    figure_folder = dest / "figures_45deg"
    folder.mkdir(parents=True, exist_ok=False)
    figure_folder.mkdir(parents=True, exist_ok=False)
    names = param_names(args.model)
    if len(names) != tdim:
        raise ValueError("T parameter labels do not match S1 data")
    rotated_axis_names = [f"RZ{i+1}" for i in range(n_latent)]
    test = np.asarray(splits["test"], dtype=np.int64)
    truth = t[test]
    records = []
    source_hashes = dict(
        s8_rotation_sha256=sha256(oracle_file),
        latent_sha256=sha256(z_file),
        checkpoint_sha256=sha256(train_path(args)),
        dataset_sha256=sha256(dataset_path(args.model)))
    for subset_index, axes in enumerate(axis_subsets(n_axes), 1):
        k = len(axes)
        subset_dir = folder / f"k{k:02d}"
        label = "_".join(f"{a+1:02d}" for a in axes)
        picture_dir = figure_folder / f"k{k:02d}" / f"axes_{label}"
        for angle in ANGLES:
            r, delta = rotate_from_oracle(r_star, axes, angle)
            error = float(np.max(np.abs(r @ r.T - np.eye(n_latent))))
            if error > 1e-8:
                raise ArithmeticError(f"Non-orthogonal rotation {axes}, {angle}")
            name = rotation_id(axes, angle)
            subset_dir.mkdir(parents=True, exist_ok=True)
            outfile = subset_dir / f"{name}.npz"
            np.savez_compressed(
                outfile, R=r, R_star=r_star, delta_R=delta,
                axes=np.array(axes, dtype=np.int64),
                axes_1based=np.array(axes, dtype=np.int64)+1,
                subset_size=np.array(k, dtype=np.int64),
                angle_degrees=np.array(angle, dtype=np.int64),
                rotation_policy=np.array("ordered_adjacent_Givens_sweep_v1"),
                coordinate_convention=np.array("R = G @ R_star; rows RZ = Z @ R.T"),
                source_s8_rotation_sha256=np.array(source_hashes["s8_rotation_sha256"]),
                source_latent_sha256=np.array(source_hashes["latent_sha256"]))
            records.append(dict(
                subset_size=k,
                axes_1based=",".join(str(a+1) for a in axes),
                angle_degrees=angle,
                rotation_file=str(outfile.relative_to(dest)).replace("\\", "/"),
                orthogonality_max_error=error))
            if angle == 45:
                picture_dir.mkdir(parents=True, exist_ok=True)
                rz_test = z[test] @ r.T
                corr_test = corr(truth, rz_test)
                _matrix_csv(picture_dir / "rz_vs_t_correlation.csv",
                            corr_test, names, rotated_axis_names)
                _plot_matrix(
                    corr_test, names, rotated_axis_names,
                    f"{args.model} TEST: T vs RZ | axes {label} | 45 deg",
                    picture_dir / "rz_vs_t_heatmap.png")
                _plot_scatter_grid(
                    truth, rz_test, names, rotated_axis_names, corr_test,
                    f"{args.model} TEST: T vs RZ | axes {label} | 45 deg",
                    picture_dir / "rz_vs_t_scatter.png",
                    max_points=args.max_test_scatter_points)
        if subset_index % 10 == 0 or subset_index == count:
            print(f"[S9] completed subsets={subset_index}/{count} "
                  f"matrices={len(records)}/{total}", flush=True)
    with (dest / "rotation_index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    save_json(dict(
        model=args.model,
        condition=condition(args),
        latent_dim=n_latent,
        theta_dim=tdim,
        axis_scope=args.axes,
        rotated_axes_1based=list(range(1,n_axes+1)),
        subset_sizes=list(range(2,n_axes+1)),
        angle_degrees=list(ANGLES),
        subset_count=count,
        matrix_count=len(records),
        figure_count=count * 2,
        figure_policy="At 45 deg, TEST-only Pearson T x all L RZ axes; "
                      "scatter plot may subsample points but not statistics",
        rotation_policy="For ascending selected axes a1..ak, left-multiply "
                        "G(a1,a2,theta), then G(a2,a3,theta), ..., "
                        "G(a[k-1],ak,theta). For k>2 this is an ordered "
                        "composition of k-1 plane rotations, not a unique "
                        "single k-dimensional rotation angle.",
        source_hashes=source_hashes,
        matrix_filenames="rotations/kXX/axes_...__deg_XXX.npz",
        test_subject_count=len(test),
    ), dest/"summary.json")
    print(f"[S9] saved {len(records)} orthogonal R matrices and "
          f"{count} complete 45-degree T-vs-RZ figure sets in {dest}",flush=True)


if __name__ == "__main__":
    main(parse_args("S9: exhaustive rotation sweep around supervised R*",
                    configure_parser=configure_parser))
