"""BLIND response surfaces of v = [latent_raw (S5 Z), X standardised by the saved S3 scaler].

DecoderSurface : the frozen S3 decoder.  Class B: eta = class1 - class0 logit.  Class C: eta = out * y_std + y_mean (Y units).
PolySurface    : the minimal T-free observed-data comparator of W1-03 (polynomial ridge / logistic ridge).
Channels are derived from eta, its gradient and Hessian: YL / YC = eta ;  YP = sigmoid(eta).
"""
import numpy as np, torch
from modrot4_nn.net.discrete import DiscreteNPConfig, DiscreteNeuralProcess


class DecoderSurface:
    def __init__(self, blind):
        st = torch.load(blind.checkpoint_file, map_location="cpu", weights_only=True)
        cfg = dict(st["model_config"]); cfg["decoder_hidden_dims"] = tuple(cfg["decoder_hidden_dims"])
        net = DiscreteNeuralProcess(DiscreteNPConfig(**cfg)); net.load_state_dict(st["model_state_dict"], strict=True); net.eval().double()
        for q in net.parameters():
            q.requires_grad_(False)
        self.net = net.decoder; self.k = blind.k; self.continuous = int(cfg["num_choices"]) == 1
        if self.continuous != (blind.cls == "C"):
            raise ValueError("decoder output head does not match the observation class")
        self.ym, self.ys = float(blind.y_mean), float(blind.y_std)
        self.full_net = net

    def eta(self):
        k, net = self.k, self.net
        if self.continuous:
            ym, ys = self.ym, self.ys
            return lambda v: net(torch.cat((v[k:], v[:k])))[0] * ys + ym
        return lambda v: (lambda o: o[1] - o[0])(net(torch.cat((v[k:], v[:k]))))

    def predict(self, z, xs):
        with torch.no_grad():
            o = self.net(torch.cat((torch.as_tensor(xs, dtype=torch.float64), torch.as_tensor(z, dtype=torch.float64)), 1))
        return (o[:, 0] * self.ys + self.ym).numpy() if self.continuous else (o[:, 1] - o[:, 0]).numpy()


# ---------------------------------------------------------------- polynomial surrogate (W1-03, arm S)
def monomials(D, degree):
    """all monomials of total degree 1..degree as index lists padded with D (an appended constant 1): shape (R, degree)"""
    out = []
    def rec(prefix, start, left):
        if prefix:
            out.append(prefix + [D] * (degree - len(prefix)))
        if left == 0:
            return
        for i in range(start, D):
            rec(prefix + [i], i, left - 1)
    rec([], 0, degree)
    return np.array(out, int)


def features_np(U, I):
    Ue = np.column_stack((U, np.ones(len(U))))
    return np.prod(Ue[:, I], axis=2)


class PolySurface:
    """eta(v) = b + w . phi((v - m) / s);  fitted once in the base (raw-Z) frame, evaluated identically for every R"""
    def __init__(self, m, s, E, w, b):
        self.m = torch.as_tensor(m, dtype=torch.float64); self.s = torch.as_tensor(s, dtype=torch.float64)
        self.I = torch.as_tensor(E, dtype=torch.long); self.w = torch.as_tensor(w, dtype=torch.float64); self.b = float(b)

    def eta(self):
        m, s, I, w, b = self.m, self.s, self.I, self.w, self.b
        one = torch.ones(1, dtype=torch.float64)
        def f(v):
            u = torch.cat(((v - m) / s, one))
            return b + (w * torch.prod(u[I], dim=1)).sum()
        return f


def fit_poly(blind, rng_seed=7, n_train=1024, n_val=256, trials=64, degrees=(2, 3), lambdas=(1e-4, 1e-2, 1.0)):
    """T-free fit on TRAIN subjects, selection of (degree, lambda) by VALIDATION-subject predictive loss only.
    Objective (both classes): mean loss + lambda * ||w||^2 (intercept unpenalised).  C: squared error on y (original units);
    B: Bernoulli log-likelihood with logit link (Y-L and Y-P are the logit and its sigmoid)."""
    from sklearn.linear_model import Ridge, LogisticRegression
    rng = np.random.default_rng(rng_seed)
    def rows(pool, n):
        sub = rng.choice(pool, min(n, len(pool)), replace=False); idx = []
        for s in sub:
            c = int(blind.counts[s]); idx.append(blind.starts[s] + rng.choice(c, min(trials, c), replace=False))
        idx = np.concatenate(idx); return idx
    tr = rows(blind.splits["train"], n_train); va = rows(blind.splits["validation"], n_val)
    def inputs(idx):
        return np.column_stack((blind.z[blind.subject_index[idx]], (blind.x[idx] - blind.x_mean) / blind.x_std))
    Vtr, Vva = inputs(tr), inputs(va); ytr, yva = blind.y[tr], blind.y[va]
    m = Vtr.mean(0); s = Vtr.std(0); s[s <= 0] = 1
    Utr, Uva = (Vtr - m) / s, (Vva - m) / s
    grid = []
    for deg in degrees:
        E = monomials(Vtr.shape[1], deg); Ftr, Fva = features_np(Utr, E), features_np(Uva, E); n = len(Ftr)
        fs = Ftr.std(0); fs[fs <= 0] = 1                      # feature scaling for conditioning only; folded back into w
        for lam in lambdas:
            if blind.cls == "C":
                mdl = Ridge(alpha=lam * n).fit(Ftr / fs, ytr)
                pred = mdl.predict(Fva / fs); loss = float(np.mean((pred - yva) ** 2)); tr_loss = float(np.mean((mdl.predict(Ftr / fs) - ytr) ** 2))
            else:
                mdl = LogisticRegression(C=1.0 / (2 * lam * n), max_iter=5000, tol=1e-8).fit(Ftr / fs, ytr.astype(int))
                pv = np.clip(mdl.predict_proba(Fva / fs)[:, 1], 1e-12, 1 - 1e-12); loss = float(-np.mean(yva * np.log(pv) + (1 - yva) * np.log(1 - pv)))
                pt = np.clip(mdl.predict_proba(Ftr / fs)[:, 1], 1e-12, 1 - 1e-12); tr_loss = float(-np.mean(ytr * np.log(pt) + (1 - ytr) * np.log(1 - pt)))
            w = np.ravel(mdl.coef_) / fs; b = float(np.ravel(mdl.intercept_)[0]) if np.ndim(mdl.intercept_) else float(mdl.intercept_)
            grid.append(dict(degree=deg, lam=lam, val_loss=loss, train_loss=tr_loss, n_features=len(E), E=E, w=w, b=b))
    best = min(grid, key=lambda g: g["val_loss"])
    diag = dict(n_train_rows=len(tr), n_val_rows=len(va), n_train_subjects=int(len(np.unique(blind.subject_index[tr]))),
                n_val_subjects=int(len(np.unique(blind.subject_index[va]))), selected_degree=best["degree"], selected_lambda=best["lam"],
                grid=[{k: v for k, v in g.items() if k not in ("E", "w")} for g in grid])
    return PolySurface(m, s, best["E"], best["w"], best["b"]), diag
