# Source provenance

Mathematical reference implementations were imported from
[mod-rot-3](https://github.com/bs-zhao/mod-rot-3/tree/98979c9ef522caa7d24de7ccc683148b762d92a5/np).

The active mod-rot-4 pipeline is deliberately smaller:

- `src/modrot4/core.py`: one T/X/logit/p/choice Generate loop and NPZ schema.
- `src/modrot4/model_specs.py`: six mathematical model specifications.
- `src/modrot4/fixed_bank.py`: original CPT bank-design recipe run independently
  in CR3 or CR5's own output directory.
- `src/modrot4/unified_precheck.py`: one T/X/choice-based diagnostic evaluator.
- `s1_generate.py` and `s2_precheck.py`: short common CLI entrypoints.

Archived source in `models/reference/` still contains the original
mathematical functions required by this implementation, including the
original CR2 CPT/logit, CR3 design and CR4 three-parameter CPT MLE.
Old standalone reference command-line experiments are not run by S1 or S2.

**Data schema change:** The old `y` / `predictive_true` files omitted
`logit`. New model datasets have `choice`, `p` and `logit` as explicit
nonduplicated columns. Old datasets cannot be mistaken for new ones.
