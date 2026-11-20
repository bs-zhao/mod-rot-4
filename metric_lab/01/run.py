"""CLI for independent metric_lab/01 experiment; run from repo root."""
import argparse
from he_lab.pipeline import run, METHODS

def parser():
    p=argparse.ArgumentParser(allow_abbrev=False,
        description="Metric Lab 01: analytical oracle vs blind/NN HE rotation sweeps")
    p.add_argument("--model",required=True,choices=["C1","C2","C3","CR3","CR4","CR5"])
    p.add_argument("--run-dir",required=True,
        help="Exact outputs/nn/MODEL/condition/repeatXX directory; no condition guessing")
    p.add_argument("--stage",default="all",
        choices=["all","validate","oracle","prepare","infer","metrics","diagnose"])
    p.add_argument("--variant",default="both",choices=["s9","s9_2","both"])
    p.add_argument("--measure",default="theory_x",
        choices=["theory_x","observed_joint"],
        help="theory_x: oracle T prior or Z Gaussian, X model design; observed_joint: rows sampled by subject")
    p.add_argument("--n-eval",type=int,default=256,
        help="Number of reference W points (non-fixed X); 0 uses all observed rows")
    p.add_argument("--latent-draws",type=int,default=16,
        help="Theory-X for fixed CPT bank: latent draws x all 500 X questions")
    p.add_argument("--fit-rows",type=int,default=3000,
        help="WY training rows (sampled subject-equally); 0 uses all, potentially huge")
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--chunk",type=int,default=32,
        help="Batch size for torch.func.vmap(torch.func.hessian)")
    p.add_argument("--boundary-policy",default="mask",choices=["mask","interior"],
        help="CPT X boundary Hessians: mark undefined or restrict evaluation support")
    p.add_argument("--methods",default="nn_autograd,wy_polynomial,wy_kernel_ridge",
        help=f"Comma-separated inference methods; all means {METHODS}")
    p.add_argument("--degree",type=int,default=3)
    p.add_argument("--max-centers",type=int,default=500)
    p.add_argument("--neighbors",type=int,default=180)
    p.add_argument("--finite-step",type=float,default=0.003)
    p.add_argument("--top-k",type=int,default=1)
    p.add_argument("--plot-metrics",default="UX.hoyer,UX.tail_l1,UU.diagonal.hoyer,UU.self_tail_q1")
    p.add_argument("--max-rotations",type=int,default=0,
        help="Only for smoke runs: 0=every S9/S9_2 saved R including unswept baseline")
    p.add_argument("--overwrite",action="store_true",
        help="Recreate prepared Y sample; otherwise preserve the existing Y cache")
    return p

if __name__=="__main__":
    args=parser().parse_args()
    if args.n_eval<0 or args.fit_rows<0 or args.latent_draws<=0 or args.chunk<=0:
        raise SystemExit("Invalid sample counts")
    if args.n_eval==0 and args.measure=="theory_x":
        raise SystemExit("--n-eval 0 not applicable to theory_x; specify a positive MC count")
    run(args)
