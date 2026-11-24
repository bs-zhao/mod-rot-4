# 03 — The simplicity-based search principle

This note states the principle on which the current search for the rotation `R` is built, and the three gaps that must be closed for the principle to work. It uses the terms of note 01 (`Z`, `T`, `U = R Z`, the frozen decoder `d`, the **Search Success Criterion**) and the test beds of note 02.

## 1. What we have and what we do not have

What the search has:

- the data `X`, `Y`;
- the subject representation `Z`;
- the frozen decoder `d`, which approximates the function `(X, Z) → Y`.

There is a function that really maps `(X, Z)` to `Y`. `d` is an approximation of it. We do not know that underlying function itself. We only have its approximation `d`.

For any rotation `R`, let `U = R Z`. Then the same input–output relation becomes a function of `(U, X)` instead of `(Z, X)`:

```
f'(U, X) → Y          (the underlying function in the coordinates U)
d'(U, X) → Y          (the decoder in the coordinates U; d'(U, X) = d(R⁻¹ U, X))
```

For every `R` we observe `U`, `X` and `Y`, and we have `d'` as an approximation of `f'`. We never see `f'`.

## 2. The output does not depend on the rotation

Earlier work found that `Z` can be linearly regressed onto `T`. The true function `f(T, X) → Y` gives the same output whatever coordinates we use for the subject parameters. So `Z` may mix the dimensions of `T` and still be mathematically equivalent and give the same output. A rotated version `U = R Z` also gives the same output.

This is the problem. The output alone cannot tell us which rotation is the right one. Every rotation reproduces `Y` equally well. Something other than the fit to `Y` has to choose among them.

## 3. Why is the `U` that corresponds one-to-one to `T` the best one?

The answer is the **simplicity of the function `f'`**. The following two examples show what we mean.

### Example 1

True parameters `T = [a, b]`, stimuli `X = [x1, x2]`, generating function

```
f:   Ŷ = a·x1 + b·x2
```

Two rotations:

- **Rotation A:** `u1 = a`, `u2 = b`. Then
  ```
  f'_A:   Ŷ = u1·x1 + u2·x2
  ```
- **Rotation B:** `u1 = a + b`, `u2 = a − b`. Then `a = (u1 + u2)/2` and `b = (u1 − u2)/2`, so
  ```
  f'_B:   Ŷ = 0.5·(u1 + u2)·x1 + 0.5·(u1 − u2)·x2
  ```

In practice we do not know `f'_A` or `f'_B`. What we know are the input–output pairs `U, X → Y` and the approximation `d'(U, X)`. But if we could see the two functions, we would find that `f'_B` is more complex. Each `u` interacts with more of the `x`: in `f'_A` each `u` interacts with one `x`, in `f'_B` each `u` interacts with both.

**Our aim is to infer the complexity of the unknown `f'` from `U`, `X`, `Y` and `d'` alone.**

### Example 2

True parameters `T = [a, b]`, stimuli `X = [x1, x2]`, generating function

```
f:   Ŷ = a·x1 + a·b·x2
```

Two rotations:

- **Rotation A:** `u1 = a`, `u2 = b`. Then
  ```
  f'_A:   Ŷ = u1·x1 + u1·u2·x2
  ```
- **Rotation B:** `u1 = a + b`, `u2 = a − b`. Then `a = (u1 + u2)/2` and `a·b = (u1² − u2²)/4`, so
  ```
  f'_B:   Ŷ = 0.5·(u1 + u2)·x1 + 0.25·(u1² − u2²)·x2
  ```

If we look only at the interaction between `u1` and `u2`, `f'_B` seems simpler, because it has no `u1·u2` term. But the researcher's judgement is that `f'_A` is clearly the simplest. This can be quantified in three ways:

1. **Fewer interactions between `u` and `x`.** In `f'_A`, `u1` interacts with `x1` and `x2` and `u2` interacts only with `x2` (3 couplings). In `f'_B`, both `u1` and `u2` interact with both `x1` and `x2` (4 couplings).
2. **A simpler mathematical form in `u`.** Squares and other nonlinear forms make the function more complex. `f'_B` contains the squares `u1²` and `u2²`.
3. **Fewer terms when the function is expanded.** `f'_A` has 2 terms (`u1·x1`, `u1·u2·x2`), `f'_B` has 4 (`u1·x1`, `u2·x1`, `u1²·x2`, `u2²·x2`).

So the measure of simplicity cannot be only the interaction among the `u`. It has to reflect how many couplings and how much functional structure the function needs.

### The requirement on the test beds

The test beds in note 02, and any that we add later, all satisfy this requirement: **the way the true `f` is written has been assessed to be the simplest, and every rotation that is not a one-to-one correspondence between `U` and `T` makes the functional form more complex.**

### The supervised reference

With the true `T` as supervision, we have shown that a search for the rotation `R` (`U = R Z`) can always find an `R*` that makes `U` and `T` one-to-one. This supervised `R*` is the reference that a blind search is compared with.

## 4. The principle

In this project we only have `X`, `Z`, `Y` and `d`. To find `U` that corresponds one-to-one to `T`, we have to **find the `R` that makes `f'(U, X)` the simplest.**

The measure of simplicity is called the **simplicity metric**. We write it as `l = L(R)`. Because it assigns a number to every rotation, it is the loss function of the search over `R`.

Three terms are fixed here and used in later notes:

- **Simplicity metric** `l = L(R)`: the number that says how simple `f'(U, X)` is, for the rotation `R`. It is the loss for the search.
- **Inference**: the process of estimating `l` for a given `R` from `Z`, `X`, `Y`, `d` alone, i.e. from `U = R Z`, `X`, `Y` and `d'`, without access to `f'`.
- **Search**: the algorithm that looks for `R` by minimising `l`.

## 5. The three gaps

For the principle to deliver the **Search Success Criterion** (note 01), three things have to work. Each is a gap that has to be closed.

### Gap 1 — the simplicity metric prefers the one-to-one rotation (metric success)

The first gap is the simplicity metric `l` itself: it has to prefer the rotation for which `R Z` and `T` are one-to-one. This is called the **success of the metric**.

**Operational definition.** Use the analytic form of `f`. Write `U = Q T` for a rotation `Q` of the (standardised) true parameters, and let `f'_Q(U, X) = f(Qᵀ U, X)` be the true function in these coordinates. Compute the simplicity metric of `f'_Q` from the analytic formula, `l_analytic(Q)`. The metric succeeds when the `Q` that gives the optimal (simplest) `l_analytic(Q)` is exactly the one for which `U` and `T` are one-to-one.

Points that make the definition precise:

- *One-to-one means up to order and sign.* The simplest `Q` is not a single matrix. Every signed permutation of the axes gives the same `U` up to the labels and signs of its dimensions. The metric should give the same value to all of them, and these should be the optimal rotations. An exact one-to-one correspondence means matched correlation 1 and zero leakage.
- *Global optimum.* The metric must prefer the one-to-one rotation over **all** rotations, not only over the ones that were tried. A search for the optimum of `l_analytic` has to be run from many starts. If the search finds a rotation with a worse `l` than the true coordinates have, the failure is of the search and not of the metric. If it finds a rotation with a better `l` that is not one-to-one, the metric has failed.
- *Coordinates.* The rotation acts on the standardised `T`, so that the scale of the parameters does not influence the result.
- *Ties.* If the test bed itself has several equally simple coordinate systems, no metric can pass Gap 1. This is the reason the test beds must satisfy the requirement in section 3.
- *What this does not show.* A pass in Gap 1 shows only that the metric ranks rotations of the analytic function correctly. It does not show that the metric can be computed without the analytic function. That is Gap 2.

### Gap 2 — the metric can be inferred without the analytic function (inference fidelity)

If the metric succeeds with the analytic `f`, it does not follow that it succeeds without it. Without the analytic form we have to compute `l` from `Z`, `X`, `Y` and `d`: given `R`, we know `U = R Z`, `X` and `Y`, and we **infer** the simplicity metric of the unknown function `f'(U, X) → Y` behind them.

The decoder `d` approximates the output of `f'` very well. That is, for given `U` and `X`, the predicted `Y` is accurate. But a metric computed from `d` does not have to be aligned with the metric of the true function.

So we have to check whether the `l` inferred from `Z`, `X`, `Y`, `d` is a faithful reconstruction of the `l` computed from the analytic formula. Both are functions of `R`, `l = L(R)`, and Gap 2 is the question of how well the first reproduces the second. The word to remember is **inference**.

### Gap 3 — the search finds the rotation (search)

If Gap 1 and Gap 2 are both passed, Gap 3 should follow naturally. With only `X`, `Z`, `Y` and `d`, minimising the inferred metric over `R` should give the `R` with the smallest loss, and that `R Z = U` should be in one-to-one correspondence with `T`, close to what the supervised (oracle) rotation achieves. In other words, the search meets the **Search Success Criterion**.

If this step fails while the first two gaps are passed, the likely cause is the optimisation and search algorithm.

## 6. Summary

| Gap | Question | What is used | Passes when |
|---|---|---|---|
| **1. Metric success** | Does the simplicity metric `l` prefer the rotation in which `U` and `T` are one-to-one? | analytic `f` (with `T`) | the rotation with the optimal analytic `l` is exactly the one-to-one one |
| **2. Inference** | Can `l` be computed from `Z`, `X`, `Y`, `d` without the analytic form, and does it reproduce the analytic `l = L(R)`? | `Z`, `X`, `Y`, `d` (analytic only for the comparison) | the inferred `l` is a faithful reconstruction of the analytic `l` |
| **3. Search** | Does a search that minimises the inferred `l` find an `R` with the one-to-one correspondence, similar to the oracle? | `Z`, `X`, `Y`, `d` | the Search Success Criterion is met |

The three research tasks are therefore:

1. choose a suitable **simplicity metric**;
2. choose a suitable **inference** method for this metric;
3. choose a suitable **search** algorithm for `R`.
