"""Shared paths/constants for the noisy-logit regression experiment.

Everything this experiment writes lives inside this folder.  The only thing
imported from the rest of the project is the *definition* of the five
generative models (parameter priors, stimulus samplers, logit functions) from
``models/src``; nothing there is modified and nothing is written there.
"""
from pathlib import Path
import sys

EXP = Path(__file__).resolve().parent
PROJECT = EXP.parent
sys.path.insert(0, str(PROJECT / "models" / "src"))

MODELS = ("C1", "C2", "C3", "CR3", "CR4")
LATENT_DIM = {"C1": 4, "C2": 4, "C3": 6, "CR3": 4, "CR4": 3}   # = number of true parameters
R2_TARGET = 0.8            # share of observed-target variance that a perfect predictor explains
SUBJECTS = 8000
DATA_SEED = {"C1": 20261006, "C2": 20261006, "C3": 20261006, "CR3": 20261028, "CR4": 20261028}
DESIGN_SEED = 20261023
NOISE_SEED = 777

DATA = EXP / "data"
CKPT = EXP / "checkpoints"
RESULTS = EXP / "results"
for p in (DATA, CKPT, RESULTS):
    p.mkdir(parents=True, exist_ok=True)


def base(m): return m[:-2] if m.endswith("r2") else m     # "C1r2" = independent replicate (new subjects, new noise, new NP) of C1
def data_seed(m): return DATA_SEED[base(m)] + (1000 if m.endswith("r2") else 0)
def noise_seed(m): return NOISE_SEED + MODELS.index(base(m)) + (100 if m.endswith("r2") else 0)
MODELS_R2 = tuple(m + "r2" for m in MODELS)


def data_path(model): return DATA / f"{model}_noisy_logit.npz"
def ckpt_path(model): return CKPT / f"{model}_np_mse.pt"
def latent_path(model): return RESULTS / f"{model}_latent.npz"
