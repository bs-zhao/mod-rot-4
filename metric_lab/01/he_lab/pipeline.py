"""Independent, read-only S1-S9 -> Oracle / Inference / Metrics pipeline."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .sources import load_sources, load_rotations, output_base, sha256
from .measures import evaluation_points, training_indices
from .energy import HEResult, result_from_moment, moment_from_hessians
from .oracle import oracle_base_moment
from .decoder import Decoder, finite_difference_hessians
from .derivatives import hessians
from .estimators import REGISTRY, estimate_wy
from .metrics import compute_metrics

Y_NAMES = ("true_choice", "nn_choice", "nn_probability", "nn_logit")
METHODS = ("nn_autograd", "nn_finite_difference", *REGISTRY.keys())


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
              "seed","boundary_policy","degree","max_centers","neighbors","measure")}
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
          model=args.model,version="metric_lab_01_v1",
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
    return evaluation_points(src,source=source,measure=args.measure,
            n_eval=args.n_eval,seed=args.seed+111,latent_draws=args.latent_draws)


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
    cache=_base_cache(root,"nn",method,target,args)
    if cache.exists():
        with np.load(cache,allow_pickle=False) as f:
            return f["moment"],{"cached":True}
    pts=_points(src,args,"nn")
    base=np.column_stack((pts.latent,pts.x))
    fn=lambda v: decoder.scalar(v,target)
    if method=="nn_autograd":
        hh=hessians(fn,base,chunk=args.chunk)
    elif method=="nn_finite_difference":
        typical=np.r_[np.maximum(src.z.std(0),1e-5),src.x_std]
        hh=finite_difference_hessians(fn,base,
                        steps=args.finite_step*np.maximum(typical,1e-5))
    else:
        raise ValueError(method)
    moment=moment_from_hessians(hh,pts.weights)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,moment=moment)
    meta={"source":"frozen_decoder","measure":args.measure,
          "target":target,"pointset":pts.meta,"n":len(base),"method":method}
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
    src,root=_context(args)
    prepared=_prepared(root)
    methods=METHODS if args.methods=="all" else tuple(s.strip() for s in args.methods.split(","))
    unknown=set(methods)-set(METHODS)
    if unknown:
        raise ValueError(f"Unknown inference methods: {unknown}")
    if not methods:
        raise ValueError("No inference methods specified")
    needs_decoder=any(m.startswith("nn_") for m in methods)
    decoder=Decoder(src) if needs_decoder else None
    eval_nn=_points(src,args,"nn")
    errors=[]
    for variant in _variants(args):
        rotations=_rotations(src,args,variant)
        # Shared NN base Hessian moment: independently calculated per target,
        # rotated algebraically for every R. No true generating T is read.
        nn_cache={}
        for method in methods:
            if method.startswith("nn_"):
                for target in ("logit","probability"):
                    nn_cache[(method,target)]=_get_nn_moment(
                        root,src,args,decoder,method,target)
        total=len(rotations)*len(Y_NAMES)*len(methods)
        done=0
        for rotation in rotations:
            for method in methods:
                for y_name in Y_NAMES:
                    target="logit" if y_name=="nn_logit" else "probability"
                    try:
                        if method.startswith("nn_"):
                            moment,meta=nn_cache[(method,target)]
                            result=result_from_moment(moment,rotation,src.x_dim,
                                "nn",target,args.measure,method,
                                meta={**meta, "y_name":y_name,
                                      "warning":"Frozen pretrained decoder does not refit to the selected Y variant"})
                        else:
                            result=estimate_wy(method,rotation,src,prepared,
                                eval_nn,y_name,args.measure,
                                degree=args.degree,max_centers=args.max_centers,
                                neighbors=args.neighbors,seed=args.seed)
                        _save_he(root,variant,"inference",method,y_name,
                                 rotation,result)
                    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                        errors.append(dict(variant=variant,method=method,y_name=y_name,
                                      rotation=rotation.name,error=str(exc)))
                    done+=1
            print(f"[INFER] {variant} rotation {rotation.name}: {done}/{total}",flush=True)
        print(f"[INFER] {variant}: {done} attempted, {len(errors)} cumulative failures")
    _json(root/"inference_failures.json",errors)
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
