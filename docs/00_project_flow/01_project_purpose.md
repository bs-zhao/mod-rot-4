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

## 4. The current NP: architecture and training

The same NP design is used for both observation types. Only the output layer and the reconstruction loss differ.

| | Binary Y | Continuous Y |
|---|---|---|
| Decoder output | 2 class logits; the decoder logit is `class1 − class0` | 1 real number (the predicted logit) |
| Reconstruction loss | cross-entropy | mean squared error |

### Encoder (subject trials → Z)

1. **Input.** Each trial is the vector `[X, Y]`. The inputs are standardised per dimension with the mean and SD of the training subjects (for `Y` as well).
2. **Trial embedding.** A two-layer MLP maps each trial to 32 dimensions.
3. **Set Transformer.** Two self-attention blocks over the set of trials (4 heads, residual connection, layer norm, a feed-forward layer). They are followed by one attention-pooling layer (PMA) with a single learned seed vector. The result is one vector per subject. A linear layer maps it to a 64-dimensional summary. The trials are an unordered set, so the output does not depend on trial order, and trial numbers can differ between subjects.
4. **Gaussian latent.** Two linear heads map the summary to a mean `μ` and a log-variance for a diagonal Gaussian `q(Z | trials)`. The latent dimension equals `dim(T)` for each test bed (4, 4, 6, 4, 3 for C1, C2, C3, CR3, CR4).

### Decoder `d` (Z, X → Y)

The decoder is an MLP on `[X standardised, Z]`: `dim(X)+dim(Z) → 128 → 256 → 128 → output`, with GELU between the layers. It is applied to each trial separately. It is this network that is frozen after training and offered to the search as the optional `d`. For the continuous version its output is in standardised units and is converted back to logit units with the training mean and SD of `Y`.

### Training

- **Data split.** Subjects are split 70% / 15% / 15% into train / validation / test (seed 31). The split is by subject, never by trial.
- **Context.** In every training step, the encoder sees only a random subset of each subject's trials. The size is a fraction drawn uniformly from 25% to 80% of that subject's trials. The decoder predicts the responses of all trials of the subject. Validation and test use context subsets that are drawn once and kept fixed.
- **Sampling.** In training, `Z = μ + σ·ε` with the reparameterisation trick.
- **Loss.** `reconstruction + β · KL`, where the KL term is between `q(Z | trials)` and a standard normal. The runs use `β = 5e-4`. The reconstruction term is cross-entropy for binary Y and MSE for continuous Y.
- **Optimiser.** Adam, learning rate `5e-4`, no weight decay, batch of 128 subjects, gradient norm clipped at 1.0.
- **Stopping.** At most 500 epochs, with early stopping on the validation loss (patience 50). The checkpoint with the best validation loss is kept.

### After training

The network is frozen. Each subject's `Z` is the posterior mean `μ`, computed with **all** of the subject's observed trials as context. All later steps work with this `Z` and, optionally, the frozen decoder.

### Premise check for Stage A

Before an NP is used in Stage B, `Z` is regressed linearly on each `T` component (fit on the training subjects, `R²` on the test subjects). The NP goes on to Stage B only if every parameter reaches the threshold:

- C1, C2, C3: `R² ≥ 0.70` for every parameter.
- CR3, CR4: `R² ≥ 0.65` for every parameter.

`T` is used here only to decide whether an NP is valid. It is not used to choose methods or metrics. The KL weight `β` is a training setting that affects this check. For the continuous CR4 replicate, `β = 5e-4` did not pass the check, and the replicate was retrained with `β = 5e-5`.

## 5. Search Success Criterion

> **Search Success Criterion.** A search is *successful* when, using only `Z`, `X`, `Y` and optionally the frozen decoder `d` (and never `T` or `f`), it finds a rotation matrix `R` such that `U = R Z` is in **one-to-one correspondence** with `T`.

This is a fixed term. Later notes refer to it as the **Search Success Criterion**.

- **What the search gets.** The inputs are `Z`, `X`, `Y` and, optionally, `d`. The search may not look at `T` or `f`.
- **What is checked.** The outcome is that every dimension of `U` is matched to one dimension of `T`, with a high correlation to its matched dimension and low correlations to the other dimensions (the one-to-one correspondence defined in Stage B).
- **Where `T` is used.** Only after the search, to judge whether the `R` that was found meets the criterion. `T` never enters the search itself.
- **Quantitative form.** This note fixes only the meaning of the term. The numbers used to decide "high" and "low" will be set in a later note.

## 6. Summary

| Item | Known to the algorithm? |
|---|---|
| Subject trials `(X, Y)` | yes |
| NP encoder / representation `Z` (`dim Z = dim T`) | yes (Stage A, learned) |
| Frozen NP decoder `d(Z, X) -> Y` | yes, optional |
| Generating function `f` | no (evaluation only) |
| Individual-difference parameters `T` | no (evaluation only) |

**Project goal.** From `(X, Y)` alone, and optionally the frozen decoder `d`, find a rotation `R` so that every axis of `U = R Z` corresponds one-to-one to a parameter in `T`. A search that does this meets the **Search Success Criterion** (section 5).
