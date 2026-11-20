# HE matrix → scalar metrics (shared by Oracle + both inference families)

Input **only**: HE matrix M (symmetric elementwise expected squared Hessian,
not necessarily PSD), dim_u, top_k. No access to Y, T, model/decoder, or R.

```python
from he_lab.he_metrics import compute_metrics,register_metric
scores = compute_metrics(M,dim_u,top_k=1)
```

The `scopes.py` defines FULL, UU, UU.diagonal, UU.offdiag_pairs (upper
triangular, no duplicate), UX, UX.row_totals, XX. Every ordinary registered
metric is run against each scope. `structure.py` functions handle metrics
requiring the original matrix (e.g. UU.self_tail_q1 and UX row entropy).

Individual metric functions:
- `energy.py`: total_energy.
- `sparsity.py`: hhi, entropy, effective_n, hoyer, tail_l1,
  tail_l2, top_k_mass.
- `structure.py`: uu_offdiag_energy_double_count, uu_self_tail_q1,
  ux_active_axis_count, ux_mean_active_row_entropy.

To register an additional vector-index function:

```python
def my_index(v,*,top_k=1):
    ...
register_metric("my_index", my_index, direction="minimize")
```

For a matrix-structural index use `register_structure_metric(name, fn)`;
its callable gets `(M,dim_u,*,top_k=1)`. Set direction to
`minimize`, `maximize`, or `descriptive` for honest R ranking.
Missing/undefined HE entries must not be silently zeroed.
