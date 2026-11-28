"""W4-05 CONF-A orchestration adapter: run the UNCHANGED S1 / S2 / S3 / S5 / S8 / S8_2 scripts in an isolated output root.

The generator (models/src/modrot4) and NP pipeline (nn/src/modrot4_nn) derive every path from a module-level ROOT evaluated at call time
(pipeline.out_dir / dataset_path / precheck_path / run_dir).  `use_root(root)` points both ROOTs at
outputs/metric_lab/10_wave4_2_fourth_wave/<RUN_ID>/confirmation/<data id>/ so nothing under outputs/models or outputs/nn can be written.
The fixed CR3 / CR5 banks are COPIED (byte-identical, hash recorded) into the isolated root before generation: the bank is reused, the
subjects / responses are fresh (04_fourth_wave section 19).  No script is edited; arguments go through the scripts' own parse_args.
This module is an ORCHESTRATOR: it sees T (it generates it).  Blind code never imports it; it only writes files.
"""
import sys, json, shutil, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "models", ROOT / "models" / "src", ROOT / "nn", ROOT / "nn" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DESIGN_SEED = 20261023          # historical design seed (fixed banks are reused)
PRECHECK_SEED = 20261006


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def use_root(root):
    # import every script first: modules that bind ROOT at import time (fixed_bank.REF, model_specs.ref = models/reference) must see the
    # repository root; only the call-time output paths are redirected
    import s1_generate, s2_precheck, s3_train, s5_extract_z, s8_supervised_rotation, s8_2_supervised_rotation   # noqa: F401
    import modrot4.pipeline as MP, modrot4_nn.pipeline as NPp
    root = Path(root).resolve(); MP.ROOT = root; NPp.ROOT = root
    assert MP.dataset_path("C1").is_relative_to(root) and NPp.dataset_path("C1").is_relative_to(root)
    return root


def copy_banks(root):
    out = {}
    for m in ("CR3", "CR5"):
        src = ROOT / "outputs" / "models" / "generate" / m / "s0_design"; dst = Path(root) / "outputs" / "models" / "generate" / m / "s0_design"
        if not dst.is_dir():
            shutil.copytree(src, dst)
        for f in sorted(src.glob("*.npz")):
            assert sha(f) == sha(dst / f.name), f"bank copy differs: {f.name}"; out[f"{m}/{f.name}"] = sha(f)
    return out


def generate(root, model, data_seed):
    use_root(root); copy_banks(root)
    import s1_generate as S1, s2_precheck as S2
    from modrot4.cli import parse_args
    from modrot4.pipeline import dataset_path, precheck_path
    a = parse_args("S1", ["--model", model, "--data-seed", str(data_seed), "--design-seed", str(DESIGN_SEED), "--precheck-seed", str(PRECHECK_SEED)])
    if not dataset_path(model).is_file():
        S1.main(a)
    if not precheck_path(model).is_file():
        S2.main(a)
    return dataset_path(model)


def np_args(model, k, beta, train_seed, extra=()):
    from modrot4_nn.cli import parse_args
    return parse_args("CONF", ["--model", model, "--latent-dim", str(k), "--beta", repr(float(beta)), "--repeat", "1", "--train-seed", str(train_seed),
                               "--split-seed", "31"] + list(extra))


def train_and_extract(root, model, k, beta, train_seed, max_epochs=None):
    """S3 (frozen rule: per-toy beta / latent dim of the registry, default stopping rule, split seed 31, new training seed), S5, S8, S8_2"""
    use_root(root)
    import s3_train as S3, s5_extract_z as S5, s8_supervised_rotation as S8, s8_2_supervised_rotation as S82
    from modrot4_nn.pipeline import train_path, latent_path, run_dir, stage_folder
    extra = [] if max_epochs is None else ["--max-epochs", str(max_epochs)]
    a = np_args(model, k, beta, train_seed, extra)
    if not train_path(a).is_file():
        S3.train(a)
    if not latent_path(a).is_file():
        S5.extract(a)
    if not (stage_folder(a, "s8_supervised_rotation") / "rotation.npz").is_file():
        S8.main(a)
    if not (stage_folder(a, "s8_2_supervised_rotation") / "rotation.npz").is_file():
        S82.main(a)
    return run_dir(a)


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--model", required=True); ap.add_argument("--data-seed", type=int, required=True)
    ap.add_argument("--k", type=int); ap.add_argument("--beta", type=float); ap.add_argument("--train-seed", type=int); ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--generate-only", action="store_true"); a = ap.parse_args(); t0 = time.time()
    generate(a.root, a.model, a.data_seed); print(f"[CONF] generated {a.model} seed {a.data_seed} [{time.time()-t0:.0f}s]", flush=True)
    if not a.generate_only:
        rd = train_and_extract(a.root, a.model, a.k, a.beta, a.train_seed, a.max_epochs); print(f"[CONF] trained {rd} [{time.time()-t0:.0f}s]", flush=True)
