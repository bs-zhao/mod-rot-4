"""Independent, read-only S1-S9 -> Oracle / Inference / Metrics pipeline."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .sources import load_sources, load_rotations, output_base, sha256
from .measures import evaluation_points, training_indices, PointSet
from .energy import HEResult, result_from_moment
from .oracle import oracle_base_moment, validity
from .decoder import Decoder
from .coordinates import full_jacobian
from .inference_wy import METHODS as WY_METHODS, infer_he as infer_wy
from .inference_wy_mlp import METHODS as MLP_METHODS, infer_he as infer_mlp, FrozenWMLP
from .inference_wy_mlp.reference_moments import estimate_reference_moment
from .he_metrics import compute_metrics

Y_NAMES = ("true_choice", "nn_choice", "nn_probability", "nn_logit")
METHODS = (*MLP_METHODS.keys(), *WY_METHODS.keys())


def _json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True,
                               default=lambda o: (o.tolist() if isinstance(o, np.ndarray)
                                                  else o.item() if isinstance(o, np.generic)
                                                  else str(o))), encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _id(args, src):
    config = {k: getattr(args,k) for k in ("n_eval","latent_draws","fit_rows",
              "seed","boundary_policy","degree","max_centers","neighbors","measure",
              "finite_step","chunk","max_rotations")}
    config.update(dataset_sha=sha256(src.dataset_file),
                  latent_sha=sha256(src.latent_file),
                  checkpoint_sha=sha256(src.checkpoint_file))
    tag=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:10]
    return f"eval{args.n_eval}_fit{args.fit_rows}_{tag}",config


def _context(args):
    src = load_sources(args.model,args.run_dir)
    tag, cfg = _id(args,src)
    root = output_base(src)/tag
    _json(root/"run_config.json",dict(config=cfg,run_dir=str(src.run_dir),
          model=args.model,version="metric_lab_01_modular_v2",
          note="S1-S9 artifacts are read-only; ideal T-space oracle vs actual learned U"))
    return src,root


def _variants(args):
    return ("s9","s9_2") if args.variant == "both" else (args.variant,)


def _rotations(src,args,variant):
    rotations=load_rotations(src,variant)
    if args.max_rotations:
        rotations=rotations[:args.max_rotations]
    if not rotations:
        raise ValueError("No S9 candidates found")
    return rotations


def _points(src,args,source):
    pts = evaluation_points(src,source=source,measure=args.measure,
            n_eval=args.n_eval,seed=args.seed+111,latent_draws=args.latent_draws)
    if args.boundary_policy == "interior" and src.model.startswith("CR"):
        # The same domain restriction MUST apply to Oracle and inference.
        # Filtering only the oracle would compare different target measures.
        keep = validity(src.model, pts).all(axis=(1, 2))
        if not np.any(keep):
            raise ValueError("No fully interior CPT evaluation points")
        pts = PointSet(pts.latent[keep],pts.x[keep],pts.weights[keep],
                       pts.name,pts.source,
                       {**pts.meta,"boundary_policy":"interior",
                        "kept":int(keep.sum()),"removed":int((~keep).sum())})
    return pts


def _base_cache(root,source,method,target,args):
    return root/"cache"/source/f"{args.measure}_{method}_{target}.npz"


def _get_oracle_moment(root,src,args,target):
    cache=_base_cache(root,"oracle",args.boundary_policy,target,args)
    if cache.exists():
        with np.load(cache,allow_pickle=False) as f:
            return f["moment"],f["coverage"].astype(bool),{"cached":True}
    pts=_points(src,args,"oracle")
    moment,coverage,meta=oracle_base_moment(src,pts,target,
            chunk=args.chunk,boundary_policy=args.boundary_policy)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,moment=moment,coverage=coverage)
    _json(cache.with_suffix(".json"),dict(meta,measure=args.measure,source="oracle",
                                            pointset=pts.meta))
    return moment,coverage,meta


def _get_nn_moment(root,src,args,decoder,method,target):
    """Cache original Z/X Hessian moments, never a pre-squared rotated HE."""
    cache=_base_cache(root,"nn",method,target,args)
    if cache.exists():
        with np.load(cache,allow_pickle=False) as f:
            return f["moment"],{"cached":True}
    pts=_points(src,args,"nn")
    moment=estimate_reference_moment(decoder,pts,method=method,target=target,
        chunk=args.chunk,step=args.finite_step,
        z_scale=np.maximum(src.z.std(axis=0),1e-5),x_scale=src.x_std)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,moment=moment)
    meta={"source":"frozen_decoder","measure":args.measure,
          "target":target,"pointset":pts.meta,"n":len(pts.latent),"method":method}
    _json(cache.with_suffix(".json"),meta)
    return moment,meta


def _save_he(root,variant,section,method,y_name,rotation,result):
    path=(root/section/variant/result.measure/method/y_name/
          f"{rotation.name}.npz")
    result.save(path)
    _json(path.with_suffix(".json"),dict(
        method=result.method,target=result.target,rotation_id=rotation.name,
        candidate_type=rotation.candidate_type,
        axes=list(rotation.axes),angle_degrees=rotation.angle,
        measure=result.measure,valid_cells=int(result.valid_mask.sum()),
        matrix_dim=result.M.shape[0],source=section,metadata=result.meta))
    return path


def run_oracle(args):
    src,root=_context(args)
    for target in ("logit","probability"):
        moment,coverage,meta=_get_oracle_moment(root,src,args,target)
        for variant in _variants(args):
            rotations=_rotations(src,args,variant)
            for i,rot in enumerate(rotations):
                result=result_from_moment(moment,rot,src.x_dim,"oracle",
                    target,args.measure,"analytic_oracle",coverage,
                    meta={**meta,"interpretation":"ideal T-space G-rotated analytical HE",
                          "boundary_policy":args.boundary_policy})
                _save_he(root,variant,"oracle","analytic_oracle",
                         f"true_{target}",rot,result)
            print(f"[ORACLE] {src.model} {variant} {target}: {len(rotations)} rotations",flush=True)
    return root


def run_prepare(args):
    src,root=_context(args)
    outfile=root/"prepared"/"responses.npz"
    if outfile.exists() and not args.overwrite:
        print(f"[PREPARE] preserved {outfile}")
        return root
    idx=training_indices(src,args.fit_rows,args.seed+202)
    decoder=Decoder(src)
    eta,p=decoder.predict(src.z[src.subject_ids[idx]],src.x[idx])
    rng=np.random.default_rng(args.seed+203)
    sampled=(rng.random(len(p))<p).astype(np.int8)
    outfile.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(outfile,row_index=idx,
                        true_choice=src.true_choice[idx],
                        nn_choice=sampled,nn_probability=p,nn_logit=eta)
    _json(outfile.with_suffix(".json"),dict(row_count=len(idx),
          seed=args.seed, subject_draw_policy="subject equal / trial within subject",
          response_policy="NN choice sampled once and identical across rotations",
          checkpoint_sha256=sha256(src.checkpoint_file)))
    print(f"[PREPARE] {len(idx)} common fit rows, four Y types: {outfile}")
    return root


def _prepared(root):
    path=root/"prepared"/"responses.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run stage prepare first: {path}")
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def run_infer(args):
    """Pass only W,Y (plus frozen MLP for that family) to independent registries."""
    src,root=_context(args)
    prepared=_prepared(root)
    all_methods=(*MLP_METHODS.keys(),*WY_METHODS.keys())
    methods=all_methods if args.methods=="all" else tuple(
        m.strip() for m in args.methods.split(",") if m.strip())
    unknown=set(methods)-set(all_methods)
    if unknown:
        raise ValueError(f"Unknown inference methods: {unknown}")
    if not methods:
        raise ValueError("No inference methods specified")
    decoder=Decoder(src) if any(m in MLP_METHODS for m in methods) else None
    eval_nn=_points(src,args,"nn")
    idx=prepared["row_index"]
    errors=[]
    for variant in _variants(args):
        rotations=_rotations(src,args,variant)
        nn_cache={}
        for method in methods:
            if method in MLP_METHODS:
                for target in ("logit","probability"):
                    nn_cache[(method,target)]=_get_nn_moment(
                        root,src,args,decoder,method,target)
        for n,rotation in enumerate(rotations,1):
            W=np.column_stack((rotation.z_to_u(src.z[src.subject_ids[idx]]),src.x[idx]))
            eval_W=np.column_stack((rotation.z_to_u(eval_nn.latent),eval_nn.x))
            trained=FrozenWMLP(decoder,rotation) if decoder is not None else None
            for method in methods:
                for y_name in Y_NAMES:
                    Y=prepared[y_name]
                    target="logit" if y_name=="nn_logit" else "probability"
                    binary=y_name.endswith("choice")
                    try:
                        if method in MLP_METHODS:
                            reference_moment,meta=nn_cache[(method,target)]
                            result=infer_mlp(method,W,Y,trained,
                                eval_W=eval_W,weights=eval_nn.weights,
                                dim_u=src.latent_dim,target=target,
                                rotation_id=rotation.name,measure=args.measure,
                                chunk=args.chunk,relative_step=args.finite_step,
                                reference_moment=reference_moment,
                                reference_jacobian=full_jacobian(
                                    rotation.jacobian("nn"),src.x_dim))
                            result.meta.update(meta)
                            result.meta["y_name"]=y_name
                            result.meta["warning"]="Frozen MLP not refit to Y"
                        else:
                            result=infer_wy(method,W,Y,
                                eval_W=eval_W,weights=eval_nn.weights,
                                dim_u=src.latent_dim,target=target,
                                binary=binary,degree=args.degree,
                                max_centers=args.max_centers,
                                neighbors=args.neighbors,seed=args.seed,
                                rotation_id=rotation.name,measure=args.measure)
                            result.meta["y_name"]=y_name
                            result.meta["evaluation_meta"]=eval_nn.meta
                        _save_he(root,variant,"inference",method,y_name,rotation,result)
                    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                        errors.append(dict(variant=variant,method=method,
                            y_name=y_name,rotation=rotation.name,error=str(exc)))
            print(f"[INFER] {variant}: {n}/{len(rotations)} rotations",flush=True)
    _json(root/"inference_failures.json",errors)
    if errors:
        print(f"[INFER] {len(errors)} errors recorded in inference_failures.json",flush=True)
    return root


def _iter_he(root,variant,section):
    base=root/section/variant
    if not base.exists():
        return
    for path in base.glob("*/*/*/*.npz"):
        # <measure>/<method>/<y_name>/<rotation>.npz
        relative=path.relative_to(base)
        measure,method,y_name,_=relative.parts
        with np.load(path,allow_pickle=False) as f:
            yield (measure,method,y_name,path.stem,
                   np.asarray(f["M"]),int(f["dim_u"]))


def _write_csv(path,rows,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_metrics(args):
    src,root=_context(args)
    for variant in _variants(args):
        rotations=_rotations(src,args,variant)
        lookup={r.name:r for r in rotations}
        records=[]
        for section in ("oracle","inference"):
            for measure,method,y_name,rot_id,m,dimu in _iter_he(root,variant,section):
                if rot_id not in lookup:
                    continue
                rotation=lookup[rot_id]
                target=("logit" if y_name.endswith("logit") else "probability")
                for metric,value in compute_metrics(m,dimu,top_k=args.top_k).items():
                    records.append(dict(section=section,variant=variant,
                        measure=measure,method=method,y_name=y_name,
                        target=target,rotation_id=rot_id,
                        candidate_type=rotation.candidate_type,
                        axes="_".join(str(k+1) for k in rotation.axes),
                        angle_degrees=rotation.angle,metric=metric,value=value))
        columns=("section","variant","measure","method","y_name","target",
                 "rotation_id","candidate_type","axes","angle_degrees","metric","value")
        _write_csv(root/"metrics"/variant/"all_metrics.csv",records,columns)
        print(f"[METRICS] {variant}: {len(records)} metric rows")
    return root


def run_diagnose(args):
    from .diagnostics import diagnose
    src,root=_context(args)
    for variant in _variants(args):
        diagnose(root,variant,_rotations(src,args,variant),
                 selected_metrics=tuple(s.strip() for s in args.plot_metrics.split(",")))
    return root


def run(args):
    src,root=_context(args)
    if args.stage=="validate":
        from .validation import smoke
        smoke(src,_rotations(src,args,_variants(args)[0]),root)
        return root
    if args.stage in ("all","oracle"):
        run_oracle(args)
    if args.stage in ("all","prepare"):
        run_prepare(args)
    if args.stage in ("all","infer"):
        run_infer(args)
    if args.stage in ("all","metrics"):
        run_metrics(args)
    if args.stage in ("all","diagnose"):
        run_diagnose(args)
    print(f"[METRIC LAB 01] {root}",flush=True)
    return root
