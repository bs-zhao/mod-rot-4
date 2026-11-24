# 04 — Simplicity from the Hessian squared-curvature energy

This note builds the simplicity metric `l = L(R)` of note 03 from the Hessian squared-curvature energy. It follows the three gaps of note 03:

| Part | Content | Gap in note 03 |
|---|---|---|
| 1 | Definition of the energy matrix (`H matrix`) when the analytic `f` is known | — |
| 2 | Simplicity metrics `l` computed from the `H matrix`, and how the expectation is taken | Gap 1 (metric) |
| 3 | **Inference**: obtaining `l` for a given `R` from `Z`, `R`, `X`, `Y`, `d'` | Gap 2 (inference) |
| 4 | Search for `R` by minimising the inferred `l` | Gap 3 (search) |

A longer background note on the same energy (Chinese, with literature) is [`docs/01_metrics/01_hessian_mean_squared_curvature_energy.md`](../01_metrics/01_hessian_mean_squared_curvature_energy.md).

---

# Part 1. Definition

## 1.1 The input W and its structure

When the analytic function is known, the response is generated from

```
Y ~ f(W),        W = [T, X]
```

`W` is the full input of `f`: the subject parameters `T` and the stimuli `X`. The dimension of `W` is `dim W = dim T + dim X`.

The data have a subject level, so the input `W` has a structure:

- `T` is one vector per subject. It is the same for all trials of that subject.
- `X` varies from trial to trial.

This structure matters whenever an expectation over `W` is taken (section 2.3).

When the rotation `R` is applied, `T` is replaced by `U` and the input becomes `W' = [U, X]`, with the same structure.

For binary `Y`, `f` stands for the output that is differentiated. It can be the logit or the probability. These give different matrices, so one of them has to be chosen and used throughout.

## 1.2 The Hessian

For every input `w`, the Hessian of `f` is the `dim W × dim W` matrix of second derivatives:

```
H(w)_ij = ∂² f(w) / ∂W_i ∂W_j
```

Its entries are functions of `w`, so `H` is a **matrix of functions**.

## 1.3 The squared-curvature energy and the H matrix

The expectation of the square of each entry gives the squared-curvature energy:

```
M_ij = E_W [ ( ∂² f(W) / ∂W_i ∂W_j )² ]
```

`M` is a numeric `dim W × dim W` matrix. In the rest of the project it is called the **H matrix**.

Because `W = [U, X]` (after rotation), the H matrix has blocks:

```
            U         X
   U  [  UU block | UX block ]
   X  [  XU block | XX block ]
```

- **UU:** the curvature between pairs of `U` dimensions (including each dimension with itself).
- **UX:** the curvature between each `U` dimension and each `X` dimension, i.e. the interaction between `u` and `x`.
- **XX:** the curvature among `X` dimensions. It does not depend on `R`.

The H matrix depends on the rotation. We write `H(R)` when this has to be explicit.

---

# Part 2. From the H matrix to the simplicity metric `l`

## 2.1 The metric is a function of the H matrix

From the H matrix we can compute many numbers that measure how "simple" the function is. They are the candidates for the simplicity metric `l`. The structure behind `W` can also be used, i.e. the split into `T` (or `U`) and `X`. For example, one can quantify the following for the **UX** block and the **UU** block:

- the **sparsity** of the interactions between `U` and `X`, and between `U` and `U`;
- their **complexity**;
- their **nonlinearity**;
- and so on.

**Definition used in this note:** the loss function `l = L(R)` is necessarily a function of the H matrix of the rotation `R`. Every metric of this note is built only from `H(R)`.

## 2.2 Why this fits the principle of note 03

In note 03 the simplest `f'` is the one in which each `u` interacts with the fewest `x`, has the simplest functional form and the fewest terms. A second derivative `∂²f / ∂u_i ∂x_j` is non-zero exactly where `u_i` and `x_j` interact. A second derivative with respect to a single `u` is non-zero where `f'` is nonlinear in `u`. So the H matrix summarises the interaction and nonlinearity structure that note 03 calls the complexity of `f'`.

## 2.3 A separate issue: expectation by integral or by average

The definition of `M_ij` contains an **expectation**. An expectation over `W` can be computed in two ways: as an **integral** over a distribution, or as an **average** over concrete points. Which one is used must be decided for each part of `W`. The decision has to be thought through at the time, both in the **analytic computation** and in the **inference** of Part 3.

**For `T` (or `Z`, `U`).** We know the distribution of the parameters. We can

- integrate over this distribution, or
- average over the concrete points of the individual subjects.

**For `X`.** The test beds differ (note 02):

- In some test beds every subject has a different `X`, drawn at random from a distribution (C1, C2, C3, CR4). Here we can integrate over the distribution, or average over the drawn trials.
- In other test beds all subjects share one trial set, a fixed bank of stimuli (CR3, CR5). Here there is no random draw per subject. The choice is between treating the set as a sample of a distribution and averaging over exactly the stimuli of the set.

So for every test bed, the integral and the average can differ, and they do not have to give the same `l`. The point is relevant for the analytic `l`, and equally for the `l` that is inferred without the analytic function.

---

# Part 3. Inference

**Inference** was fixed in note 03 as the process of estimating `l` for a given `R` from `Z`, `X`, `Y`, `d` alone. Here the situation is the following.

For a given `R` we have only `U = R Z`, `X`, `Y` and `d'(U, X)`. We know that `Y` is the output of a function `f'(U, X)` and that `d'` is an approximation of `f'`. We do not have any information about `f'` itself. We want to infer an attribute of this unknown `f'`: the simplicity metric `l`.

Once `l` has been chosen in Part 2, the task is to infer this `L` from the only things we have: `Z`, `R`, `X`, `Y` and `d'`. There are many branches. The list below names the branches that are currently considered. It is not closed.

## Branch 1 — infer the H matrix first, or infer `l` directly?

- Infer the **H matrix** first, then compute `l` from it.
- Infer **`l` directly**, without an explicit H matrix.

## Branch 2 — how much of `d'` is used?

- **Do not use `d'`.** Work directly with `R Z`, `X`, `Y`.
- **Use `d'` as the function.** Differentiate or evaluate `d'` itself.
- **An intermediate state.** Feed `R Z` and `X` into `d'` and obtain `Y`-like outputs from it, then use these outputs. For a binary `d'` the outputs that can be taken are the **logit**, the probability **p** and the **choice**.

## Branch 3 — an intermediate analytic form?

- Is a **closed-form approximation of the `(U, X) → Y` function** fitted first, with `l` computed from this approximate formula afterwards?

## Branch 4 — integrate over `Z` or average over subjects?

- The `Z` learned by the NP can be approximated as normally distributed. Then `Z` can be treated as a distribution and **integrated over it**, instead of **averaging over all subjects**.
- This is the choice of section 2.3 applied to inference. The same question has to be asked for `X`.

## And further

The branches are not exclusive and can be combined. Every combination is an inference method. The aim of Part 3 is to choose the inference method with which the inferred `L` reproduces the analytic `L` (Gap 2 in note 03).

---

# Part 4. Optimisation and search

Last comes the search algorithm. A rotation `R` is proposed, the inference of Part 3 turns `Z`, `R`, `X`, `Y` and `d'` into the number `l`, and the search looks for the `R` for which `l` is smallest. The goal is that, at the smallest `l`, the resulting `U = R Z` is in one-to-one correspondence with `T`, i.e. the **Search Success Criterion** of note 01 is met.

If Gap 1 and Gap 2 hold, a failure of this step would most likely be a failure of the optimisation and search algorithm (note 03).

---

**Note.** Earlier work already tried a number of simplicity metrics and inference methods in a rather unsystematic way. Those attempts are for reference only; this note does not rely on them. The metric library is in [`docs/01_metrics/01_hessian_mean_squared_curvature_energy.md`](../01_metrics/01_hessian_mean_squared_curvature_energy.md).
