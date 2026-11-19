"""S9_2: the original S9 Givens sweep but on globally standardized Zz.

Reads S8_2's exact all-subject means/std and R*, with no new fitting.
Produces independent s9_2_rotation_sweep outputs; does not modify S9.
"""
from __future__ import annotations

from s9_rotation_sweep import configure_parser, main
from modrot4_nn.cli import parse_args

if __name__=="__main__":
    main(parse_args("S9_2: Givens rotation sweep in globally standardized Z space",
                    configure_parser=configure_parser),
         variant="global_zscore")
