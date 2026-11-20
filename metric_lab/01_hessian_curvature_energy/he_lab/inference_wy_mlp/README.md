# W,Y + frozen pretrained MLP inference → HE matrix

**Hard boundary:** input is W,Y plus an already-trained frozen MLP callable
with `mlp(W, target)`. There is NO access to generating formula or true T.

Public API:

```python
adapter = FrozenWMLP(trained_decoder, rotation)
result = infer_he(method_name, W, Y, adapter,
                  eval_W=W_eval, weights=q_weights, dim_u=dim_u,
                  target="probability")
```

- `adapter.py`: map W=(U,Xraw) through inverse S9/S9_2 rotation and TRAIN
  X scaling to the actual frozen NN decoder.
- `autograd.py:estimate_he`: exact NN function Hessian; accepts Y in its
  standardized interface but does not retrain on it.
- `finite_difference.py:estimate_he`: numerically differentiate frozen NN
  predictions; likewise does not retrain on Y.
- `residual_polynomial.py:estimate_he`: **jointly uses W,Y and frozen NN**.
  Fit a polynomial residual correction for smooth outcomes, or logistic
  calibration on NN logit plus W-polynomial for Bernoulli responses. Different
  Y genuinely produces a different estimated HE. Frozen weights never change.
- `reference_moments.py`: optional fast exact base-(Z,Xraw) second moments
  for the first two methods; each rotation contracts the unsquared Hessian
  moment with its Jacobian. The residual fitting method is rotation-specific
  and intentionally does **not** use this shortcut.

To add new methods, write a separate file with
`estimate_he(W,Y,mlp,*,eval_W,weights,dim_u,target,...)->HEResult`,
then register the callable. No changes to the main pipeline.
