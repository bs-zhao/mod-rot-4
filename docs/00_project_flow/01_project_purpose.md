# 01 — Project purpose

This is the first note in a series that walks through the whole project pipeline. It states what the project is for. It defines the data, the generative assumption and the end goal. It does not cover methods or results.

## 1. Data structure

The data come from many subjects, and each subject contributes many trials:

```
subject  ->  trials (i.i.d. within subject)  ->  (X, Y)
```

- **X** is the stimulus of a trial. It can be discrete or continuous. In a lottery task, for example, X holds the amounts and probabilities of the lotteries. X can be long, i.e. high-dimensional.
- **Y** is the response of a trial. It is a single variable, and it can be discrete or continuous. In a lottery task it can be a choice or a price offer.
- Within a subject, the trials are independent and identically distributed.

## 2. Generative assumption

Assume a generating function `f(X, T)` that is known to the simulator but not to the analyst. `T` is the vector of **individual-difference parameters** of a subject, for example risk preference or response randomness. Each subject has one fixed `T`. For simulation we draw `T` and `X` in some way and generate `Y`:

- **Binary Y.** `P(Y = 1) = f(X, T)`, with `f` taking values in (0, 1). The link function inside `f` is not fixed in advance. It can be a sigmoid, for example.
- **Continuous Y.** `Y = f(X, T) + ε`, where `ε` is independent Gaussian noise.

Simulation gives us data with a known ground truth (`f` and `T`). This lets us check whether a method works. In the real setting, only `(X, Y)` is observed.

## 3. The end goal

The setting we care about: **we only have the `(X, Y)` data of a group of subjects. We do not know `f` and we do not know `T`.** The goal is to recover a representation of each subject in which every axis corresponds to one individual-difference parameter. This is done in two stages.

### Stage A — learn a subject representation Z (premise, already established)

Learn a representation `Z` for each subject from their `(X, Y)` data with a Neural Process (NP). The NP encoder maps a subject's trials to `Z`. The NP decoder is an MLP `d(Z, X) -> Y`.

- We only consider the case where `dim(Z) = dim(T)`. We assume the number of dimensions was found in earlier steps and take it as given.
- **Premise.** Take all `Z` as independent variables and regress each component of `T` on them linearly. Each `T` component must be predictable. This ability has already been shown. We treat it as the premise for the next stage. If an NP does not meet it, we do not go on to Stage B with that NP.

Linear predictability does **not** mean that each axis of `Z` is one parameter. `Z` may be a mixture of the `T` components (an arbitrary linear mix of them, or something close to it). The axes of `Z` are not yet interpretable.

### Stage B — find a rotation R without using T or f

Find an invertible rotation matrix `R` and define

```
U = R Z
```

such that **each dimension of `U` is matched to one dimension of `T`**:

- the correlation between this `U` dimension and its matched `T` dimension is high, and
- the correlations between this `U` dimension and the other `T` dimensions are low.

We call this a **one-to-one correspondence between `U` and `T`**.

The constraint on the algorithm is the core difficulty of the project:

- It **must not** use any information about `T` or `f`.
- It **may only** use the observed `(X, Y)` data, and optionally the frozen NP decoder `d(Z, X) -> Y`.

`T` and `f` are used only after the fact, to evaluate whether the rotation found by the algorithm gives a one-to-one correspondence.

## 4. Search Success Criterion

> **Search Success Criterion.** A search is *successful* when, using only `Z`, `X`, `Y` and optionally the frozen decoder `d` (and never `T` or `f`), it finds a rotation matrix `R` such that `U = R Z` is in **one-to-one correspondence** with `T`.

This is a fixed term. Later notes refer to it as the **Search Success Criterion**.

- **What the search gets.** The inputs are `Z`, `X`, `Y` and, optionally, `d`. The search may not look at `T` or `f`.
- **What is checked.** The outcome is that every dimension of `U` is matched to one dimension of `T`, with a high correlation to its matched dimension and low correlations to the other dimensions (the one-to-one correspondence defined in Stage B).
- **Where `T` is used.** Only after the search, to judge whether the `R` that was found meets the criterion. `T` never enters the search itself.
- **Quantitative form.** This note fixes only the meaning of the term. The numbers used to decide "high" and "low" will be set in a later note.

## 5. Summary

| Item | Known to the algorithm? |
|---|---|
| Subject trials `(X, Y)` | yes |
| NP encoder / representation `Z` (`dim Z = dim T`) | yes (Stage A, learned) |
| Frozen NP decoder `d(Z, X) -> Y` | yes, optional |
| Generating function `f` | no (evaluation only) |
| Individual-difference parameters `T` | no (evaluation only) |

**Project goal.** From `(X, Y)` alone, and optionally the frozen decoder `d`, find a rotation `R` so that every axis of `U = R Z` corresponds one-to-one to a parameter in `T`. A search that does this meets the **Search Success Criterion** (section 4).
