"""W3-04 disjoint-context representations with the FROZEN encoder (03_third_wave section 18).  Truth-free (blind store).

Encoder path = exactly the S5 extraction (nn/s5_extract_z.py): x standardised with the checkpoint's TRAIN scaler (float32), class C responses
standardised with its y scaler, observation channels normalised by the checkpoint encoder mean / SD, set encoder with a keep mask,
posterior mean.  The set transformer accepts any context subset through its padding mask; context sizes outside the training range
(context fraction 0.25-0.80 of a subject's trials) are flagged out-of-distribution, not removed.
Splits per subject (frozen before any encoding; no response or truth used):
  query   fixed held-out query set: 100 trials (20% of n if n < 500), RNG [9, subject] - identical for every size and context seed
  DC-N    context seed cs in {31, 73, 101}: one permutation of the non-query trials (RNG [cs, 11, subject]); nested contexts = its first m trials,
          m in {25, 50, 100, 200, all non-query}
  DC-R    two DISJOINT blocks of equal size m in {50, 100, 200 where feasible} from the same permutation: [0, m) and [m, 2m)
Assertions: context and query disjoint; nested contexts are prefixes; DC-R blocks disjoint.
"""
import json
import numpy as np, torch
from .common import OUT

CTX = OUT / "context"
SIZES = (25, 50, 100, 200)
CSEEDS = (31, 73, 101)
REP_SIZES = (50, 100, 200)


def query_ids(n, subject):
    nq = 100 if n >= 500 else int(round(0.2 * n))
    return np.sort(np.random.default_rng([9, int(subject)]).permutation(n)[:nq])


def split_manifest(pl):
    """per subject: query trial ids (local indices) and the context permutation of each seed"""
    man = dict(query={}, perm={cs: {} for cs in CSEEDS})
    for s in range(len(pl.counts)):
        n = int(pl.counts[s]); q = query_ids(n, s); rest = np.setdiff1d(np.arange(n), q)
        man["query"][s] = q
        for cs in CSEEDS:
            man["perm"][cs][s] = rest[np.random.default_rng([cs, 11, s]).permutation(len(rest))]
    return man


class Encoder:
    def __init__(self, pl, device="cpu"):
        from modrot4_nn.net.discrete import DiscreteNeuralProcess, DiscreteNPConfig
        st = torch.load(pl.checkpoint_file, map_location="cpu", weights_only=True)
        net = DiscreteNeuralProcess(DiscreteNPConfig(**st["model_config"])); net.load_state_dict(st["model_state_dict"]); net.eval().to(device)
        self.net, self.device = net, device; inp = st["extra_state"]["input_standardization"]
        self.xm = torch.as_tensor(inp["x_mean"], dtype=torch.float32); self.xs = torch.as_tensor(inp["x_std"], dtype=torch.float32)
        self.continuous = "y_mean" in inp; self.ym = float(inp.get("y_mean", 0.0)); self.ysd = float(inp.get("y_std", 1.0))
        self.em = st["preprocessing"]["encoder_mean"].reshape(-1).float().to(device); self.es = st["preprocessing"]["encoder_std"].reshape(-1).float().to(device)
        self.ctx_frac = tuple(st["extra_state"]["full_training_config"].get("context_fraction", (0.25, 0.8))) if isinstance(st["extra_state"].get("full_training_config"), dict) else (0.25, 0.8)
        self.pl = pl
        n = len(pl.counts); T = int(pl.counts.max()); p = pl.p
        x = torch.zeros((n, T, p)); y = torch.zeros((n, T)); self.valid = torch.zeros((n, T), dtype=torch.bool)
        xx = torch.as_tensor(np.asarray(pl.x, np.float32)); yy = torch.as_tensor(np.asarray(pl.y, np.float32))
        for s in range(n):
            a, c = int(pl.starts[s]), int(pl.counts[s]); x[s, :c] = xx[a:a + c]; y[s, :c] = yy[a:a + c]; self.valid[s, :c] = True
        x = (x - self.xm) / self.xs; x[~self.valid] = 0
        if self.continuous:
            y = (y - self.ym) / self.ysd; y[~self.valid] = 0
        self.x, self.y = x, y

    @torch.no_grad()
    def encode(self, keep, batch=512):
        """keep (n_subjects, T) bool context mask -> posterior mean Z (float64), log-variance"""
        n = keep.shape[0]; Z = np.zeros((n, self.net.config.latent_dim)); LV = np.zeros_like(Z); keep = torch.as_tensor(keep)
        for i in range(0, n, batch):
            sl = slice(i, i + batch); x = self.x[sl].to(self.device); y = self.y[sl].float().to(self.device).unsqueeze(-1)
            obs = (torch.cat([x, y], -1) - self.em) / self.es; p = x.shape[-1]
            mu, lv = self.net.encode_context(obs[..., :p], obs[..., p:], keep[sl].to(self.device))
            Z[sl] = mu.double().cpu().numpy(); LV[sl] = lv.double().cpu().numpy()
        return Z, LV


def keep_mask(pl, man, kind, cs=None, m=None, block=0):
    n = len(pl.counts); T = int(pl.counts.max()); K = np.zeros((n, T), bool)
    for s in range(n):
        if kind == "full":
            K[s, :int(pl.counts[s])] = True; continue
        perm = man["perm"][cs][s]
        if kind == "nested":
            idx = perm if m == "all" else perm[:m]
        elif kind == "block":
            idx = perm[block * m:(block + 1) * m]
        assert not np.intersect1d(idx, man["query"][s]).size
        K[s, idx] = True
    return K


def representations(pl, device="cpu"):
    """encode every DC-N / DC-R / full representation of one toy -> OUT/context/<toy>/reps.npz (blind store: Z, log-var, ids; no truth)"""
    f = CTX / pl.toy.replace("#", "_") / "reps.npz"
    if f.is_file():
        return f
    man = split_manifest(pl); enc = Encoder(pl, device); arr = {}; info = {}
    Zf, LVf = enc.encode(keep_mask(pl, man, "full")); arr["full|Z"] = Zf; arr["full|LV"] = LVf
    info["full_reproduces_S5_maxabs"] = float(np.abs(Zf - pl.z).max())
    lo, hi = enc.ctx_frac
    for cs in CSEEDS:
        for m in SIZES + ("all",):
            K = keep_mask(pl, man, "nested", cs, m); Z, LV = enc.encode(K); key = f"N|{cs}|{m}"
            arr[f"{key}|Z"] = Z; arr[f"{key}|LV"] = LV; nctx = K.sum(1); frac = nctx / pl.counts
            info[key] = dict(n_context_median=float(np.median(nctx)), frac_median=float(np.median(frac)), ood_frac=float(np.mean((frac < lo) | (frac > hi))))
        for m in REP_SIZES:
            if np.all(pl.counts - [len(man["query"][s]) for s in range(len(pl.counts))] >= 2 * m):
                for b in (0, 1):
                    Z, LV = enc.encode(keep_mask(pl, man, "block", cs, m, b)); arr[f"R|{cs}|{m}|{b}|Z"] = Z; arr[f"R|{cs}|{m}|{b}|LV"] = LV
    arr["query_ids"] = np.concatenate([man["query"][s] for s in range(len(pl.counts))]); arr["query_counts"] = np.array([len(man["query"][s]) for s in range(len(pl.counts))])
    for cs in CSEEDS:
        arr[f"perm_{cs}"] = np.concatenate([man["perm"][cs][s] for s in range(len(pl.counts))])
    arr["meta"] = np.array(json.dumps(dict(info=info, sizes=list(SIZES) + ["all"], cseeds=list(CSEEDS), rep_sizes=list(REP_SIZES), train_ctx_fraction=[lo, hi],
                                            device=device), default=float))
    f.parent.mkdir(parents=True, exist_ok=True); tmp = f.with_suffix(".tmp.npz"); np.savez_compressed(tmp, **arr); tmp.replace(f)
    return f
