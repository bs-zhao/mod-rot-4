# 02 — Test beds

This note lists the simulated test beds used at the current stage. Each test bed is a generating function `f(X, T)` together with a way to draw `T` and `X`. We use six models, `C1`, `C2`, `C3`, `CR3`, `CR4` and `CR5`. Each one has a **binary-Y version** and a **continuous-Y version**. The two versions share the same `T`, `X` and `f`. Only the observation step differs.

Notation. Each test bed outputs a scalar `η(X, T)` for every trial, which we call the **true logit**. For C1–C3 `η` is linear in `T`. For CR3–CR5 `η` is a cumulative-prospect-theory (CPT) value difference scaled by `τ`.

## 1. Observation step: binary vs continuous Y

| Version | Observed Y | Noise |
|---|---|---|
| **Binary** | `Y ~ Bernoulli(p)`, with `p = sigmoid(η(X, T))` | Bernoulli sampling |
| **Continuous** | `Y = η(X, T) + ε`, `ε ~ N(0, s²)` independent across trials | `s² = Var(η) · (1 − R²) / R²` with `R² = 0.8` |

For the continuous version, `Var(η)` is the variance of the true logit over all trials of all subjects in the dataset. With `R² = 0.8` this gives `s² = Var(η) / 4`. A predictor that knows `η` exactly explains 80% of the variance of the observed `Y`, and the other 20% is noise that no model can remove. We call `0.8` the **signal ratio** of the continuous test beds.

Both versions use the same random streams for `T`, trial counts and `X`. A subject in the continuous version has the same `T` and `X` as in the binary version. Only the observation is different.

Continuous datasets that have been generated so far (8000 subjects each):

| Model | Trials | `dim(T)` | `dim(X)` | `Var(η)` | noise SD `s` | realised R² (perfect predictor) |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 3,206,036 | 4 | 3 | 0.993 | 0.498 | 0.7998 |
| C2 | 3,206,036 | 4 | 4 | 1.739 | 0.659 | 0.8002 |
| C3 | 3,206,036 | 6 | 4 | 2.721 | 0.825 | 0.8001 |
| CR3 | 4,000,000 | 4 | 6 | 1.949 | 0.698 | 0.8002 |
| CR4 | 4,000,000 | 3 | 6 | 9.105 | 1.509 | 0.7999 |

CR5 is defined in section 5 and has a binary version. A continuous CR5 dataset has not been generated yet. It would use the same procedure as the other models.

## 2. Common design

| | C1 / C2 / C3 | CR3 / CR4 / CR5 |
|---|---|---|
| Subjects | 8000 | 8000 |
| Trials per subject | integer uniform on 300–500 | 500 |
| `T` | independent per subject, see each model | independent per subject, see section 6 |
| `X` | i.i.d. per trial, per subject | CR4: i.i.d. per trial, per subject. CR3 and CR5: one fixed 500-row bank shared by all subjects |
| Link (binary) | sigmoid | sigmoid |

`X` is drawn independently of `T`. Within a subject, trials are independent.

## 3. C1 — linear logit (4 parameters, 3 stimuli)

- **Parameters** `T = (b1, b2, b3, b4)`, each `b_k ~ N(0, 0.5²)` independently.
- **Stimuli** `X = (x1, x2, x3)`, each `x_j ~ Uniform(−1, 1)` independently.
- **True logit**

  `η = b1 + √3 · b2 · x1 + √3 · b3 · x2 + √3 · b4 · x3`

- `b1` is an overall bias. `b2, b3, b4` are the slopes on `x1, x2, x3`. The factor `√3` gives each design feature unit variance under `Uniform(−1, 1)`.

## 4. C2 and C3 — option comparison with quadratic terms

Both take four raw stimuli `X = (X1_A, X1_B, X2_A, X2_B)`: attributes 1 and 2 of option A and of option B. Each is `Uniform(−1, 1)`, independent. In the binary version `Y = 1` means option A is chosen.

Parameters: each `b_k ~ N(0, 0.5²)`, independent.

### C2 (4 parameters)

`T = (b1, b2, b3, b4)`

```
η = b1
  + √3      · b2 · (X1_A − X1_B)
  + (√5/2)  · b3 · [ (3·X1_A² − 1) − (3·X1_B² − 1) ]
  + √3      · b4 · (X2_A − X2_B)
```

- `b1` is a bias, `b2` is the linear effect of attribute 1, `b3` is its quadratic effect, and `b4` is the linear effect of attribute 2.
- `η` is nonlinear in `X` but linear in `T`.

### C3 (6 parameters, nests C2)

`T = (b1, …, b6)`

```
η = b1
  + √3      · b2 · (X1_A − X1_B)
  + (√5/2)  · b3 · [ (3·X1_A² − 1) − (3·X1_B² − 1) ]
  + √3      · b4 · (X2_A − X2_B)
  + (√5/2)  · b5 · [ (3·X2_A² − 1) − (3·X2_B² − 1) ]
  + 3       · b6 · (X1_A·X2_A − X1_B·X2_B)
```

- It adds the quadratic effect of attribute 2 (`b5`) and the mixed `X1 × X2` effect (`b6`). With `b5 = b6 = 0` it reduces to C2.
- `η` is linear in `T`, so for C1–C3 the second derivative of `η` with respect to `T` is zero. All curvature is in `X`.

## 5. CR3, CR4, CR5 — cumulative-prospect-theory choice

Each trial is a choice between two lotteries A and B. A lottery has three numbers: a gain amount `g`, a loss amount `l ≥ 0` and the probability `p` of the gain. With probability `1 − p` the subject loses `l`.

`X = (gain_A, loss_A, p_A, gain_B, loss_B, p_B)`, six dimensions.

### Generating function

For a lottery `(g, l, p)`:

```
w(p)  = sigmoid( γ · logit(p) )                      # probability weighting; p clipped to [1e-12, 1 − 1e-12]
V(g, l, p) = w(p) · (g/100)^α  −  λ · w(1 − p) · (l/100)^α
η(X, T) = τ · [ V(lottery A) − V(lottery B) ]
```

- `α` is the curvature of the value function, `λ` (loss aversion) scales losses, `γ` controls the distortion of probabilities, and `τ` is the choice sensitivity (the inverse of response randomness).
- For `g = 0` or `l = 0` the corresponding power term is exactly zero.
- Binary version: `P(Y = 1) = sigmoid(η)`. Continuous version: `Y = η + ε`.

### The three models

| Model | `T` | `α` | `X` sampling |
|---|---|---|---|
| **CR3** | `(α, λ, γ, τ)`, 4 parameters | free | fixed 500-row bank (below) |
| **CR4** | `(λ, γ, τ)`, 3 parameters | fixed at 1 | i.i.d. per trial and subject (below) |
| **CR5** | `(λ, γ, τ)`, 3 parameters | fixed at 1 | the same fixed 500-row bank as CR3 |

CR4 and CR5 use the CR3 formula with `α = 1`. They keep the CR3 prior for `λ, γ, τ`.

### CR4 stimuli: i.i.d. uniform

For every trial of every subject, the six components are drawn independently:

- `gain_A, loss_A, gain_B, loss_B ~ Uniform(5, 200)`
- `p_A, p_B ~ Uniform(0.05, 0.95)`

There is no rejection step and no design optimisation. Some trials may therefore have one lottery that dominates the other.

### CR3 and CR5 stimuli: a fixed 500-row bank

All subjects see the **same 500 rows in the same order**. The bank is built once (design seed `20261023`) and then frozen. It has four trial types:

| Type | Rows | Content |
|---|---:|---|
| `gain_only` | 60 | both lotteries have no loss, different probabilities |
| `gamma_neutral` | 60 | both probabilities equal 0.5 |
| `loss_only` | 40 | both lotteries have no gain, different probabilities |
| `general` | 340 | sampled without replacement from an earlier optimised CR2 4-parameter bank (amounts and probabilities on a grid) |

The amount grid is `{5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200}` and the probability grid is `{0.05, 0.10, …, 0.95}`. Rows with strict first-order stochastic dominance are excluded, and there are no duplicate or mirrored rows. The CR5 bank has exactly the same rows, order and type labels as the CR3 bank (the code checks this).

Because the bank is fixed, the stimuli are identical across subjects, and the observed difference between subjects comes only from `T`.

## 6. Sampling of T

For C1–C3 see sections 3 and 4: `b_k ~ N(0, 0.5²)` i.i.d. For the CPT models each subject's `T` is drawn independently from the following marginals:

| Parameter | Distribution | Range |
|---|---|---|
| `α` (CR3 only) | `Beta(12, 3)` | (0, 1), bounds `[0.02, 0.999]` |
| `λ` (loss aversion) | `Gamma(shape 9, scale 0.22)`, redrawn when outside the bounds | `[0.3, 5]` |
| `γ` (probability weighting) | `Beta(9, 4)` | bounds `[0.02, 0.999]` |
| `τ` (sensitivity) | normal, mean 2.2, SD 0.55, truncated | `[0.5, 4]` |

The four parameters are drawn independently. CR4 and CR5 use the same draw as CR3 with the `α` column dropped. Out-of-range `λ` values are redrawn, not clipped, so there is no atom at the bounds.

## 7. Summary of the test beds

| Model | `dim(T)` | `dim(X)` | `X` | trials / subject | `η` is linear in `T`? |
|---|---:|---:|---|---:|---|
| C1 | 4 | 3 | i.i.d. U(−1, 1) | 300–500 | yes |
| C2 | 4 | 4 | i.i.d. U(−1, 1) | 300–500 | yes |
| C3 | 6 | 4 | i.i.d. U(−1, 1) | 300–500 | yes |
| CR3 | 4 | 6 | fixed 500-row CPT bank | 500 | no |
| CR4 | 3 | 6 | i.i.d. U(5, 200) amounts, U(0.05, 0.95) probabilities | 500 | no |
| CR5 | 3 | 6 | the CR3 bank | 500 | no |

C1–C3 are the "classic" controls: `η` is linear in `T`, with the nonlinearity only in `X`. CR3–CR5 are the cognitive-model cases, where `T` enters nonlinearly through `α`, `γ`, `λ` and `τ`.

## 8. Code locations

- Model definitions (T prior, X sampler, true logit): `models/src/modrot4/model_specs.py`, `models/src/modlearn_rot/generators/choice/` (C1–C3), `models/reference/` (CR models).
- Binary data generation: `models/s1_generate.py`.
- Continuous data generation: `noisy_logit_np/gen_data.py`.
