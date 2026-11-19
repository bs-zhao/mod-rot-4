"""Common run selectors and hyperparameters for five independent NN stages."""
from __future__ import annotations
import argparse

MODELS = ("C1","C2","C3","CR3","CR4","CR5")


def parse_args(description, argv=None, *, configure_parser=None):
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--run-id", default=None,
                        help="Optional stable run name; by default a fingerprint of full training configuration")
    parser.add_argument("--beta", type=float, default=0.)
    parser.add_argument("--lambda-sparse", type=float, default=0.)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--context-min", type=float, default=.25)
    parser.add_argument("--context-max", type=float, default=.80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=.0005)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=0.)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-seed", type=int, default=31)
    parser.add_argument("--split-seed", type=int, default=31)
    parser.add_argument("--inference-batch", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--pc-var-threshold", type=float, default=.05)
    parser.add_argument("--z-var-threshold", type=float, default=.01)
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args(argv)
    if (args.latent_dim<1 or args.batch_size<1 or args.max_epochs<1 or
        args.patience<1 or args.repeat<1 or args.inference_batch<1 or
        args.beta<0 or args.lambda_sparse<0 or
        not 0<args.context_min<=args.context_max<=1):
        parser.error("Invalid NN hyperparameters")
    return args
