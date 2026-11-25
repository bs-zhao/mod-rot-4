"""One set of CLI arguments for six model names."""
import argparse

from modrot4.pipeline import ALL_MODELS

MODELS = ALL_MODELS


def parse_args(description, argv=None):
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--subjects", type=int, default=8000)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--design-seed", type=int, default=20261023)
    parser.add_argument("--precheck-seed", type=int, default=20261006)
    args = parser.parse_args(argv)
    if args.subjects < 1:
        parser.error("--subjects must be positive")
    if args.data_seed is None:
        args.data_seed = (20261028 if args.model.startswith("CR")
                          else 20261006)
    return args
