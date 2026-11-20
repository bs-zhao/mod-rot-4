# W,Y-only inference → HE matrix

**Hard boundary:** algorithms see only W=[U,Xraw], Y and evaluation W/weights.
They must NOT read generating model, true T, fitted T-hat, pretrained MLP or
the Oracle HE matrix.

Public contract in `registry.py`:

```python
result = infer_he(method_name, W, Y,
    eval_W=eval_W, weights=eval_weights, dim_u=latent_dim,
    target="logit" or "probability", binary=is_binary, **options)
```

- `polynomial.py:estimate_he`: polynomial Ridge or logistic polynomial.
- `kernel_ridge.py:estimate_he`: RBF-KRR proxy and analytic kernel derivatives.
- `local_quadratic.py:estimate_he`: local weighted quadratic regression.
- Additional direct quadratic-functional estimators may register here once the
  relevant statistical assumptions on Q and observations are established.
  Such estimators need not fit a surrogate function, but **must still return an
  HEResult containing the same full [dimW,dimW] M**.

A new algorithm: add one file exporting
`estimate_he(W,Y,*,eval_W,weights,dim_u,target,**options)->HEResult`,
register it in `METHODS` or call `register_method("wy_<name>", function)`
at initialization, then select `--methods wy_<name>`. No pipeline edits.

For binary Y, infer the conditional probability (0/1 does not itself
possess a classical Hessian). All HE is computed on raw physical X coordinates.
