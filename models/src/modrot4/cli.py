"""CLI options for the original model simulator and S2 precheck only.

The true model formulas and sampler RNG semantics live in the frozen sources.
"""
from __future__ import annotations
import argparse

MODELS = ("C1", "C2", "C3", "CR3", "CR4", "CR5")


def parse_args(description: str, argv=None):
    p = argparse.ArgumentParser(description=description, allow_abbrev=False)
    p.add_argument("--model", choices=MODELS, default="CR3")
    p.add_argument("--subjects", type=int, default=8000)
    p.add_argument("--data-seed", type=int)
    p.add_argument("--design-seed", type=int, default=20261023)
    p.add_argument("--precheck-seed", type=int, default=20261006)
    p.add_argument("--oracle-precheck", action="store_true")
    p.add_argument("--oracle-subjects", type=int, default=100)
    p.add_argument("--oracle-starts", type=int, default=4)
    p.add_argument("--overwrite", action="store_true",
                   help="Compatibility only: existing datasets cannot be overwritten in place.")
    args = p.parse_args(argv)
    if args.data_seed is None:
        args.data_seed = 20261028 if args.model in ("CR3", "CR4", "CR5") else 20261006
    if args.subjects < 1 or args.oracle_subjects < 1 or args.oracle_starts < 1:
        p.error("subjects, oracle-subjects, and oracle-starts must be positive")
    return args
