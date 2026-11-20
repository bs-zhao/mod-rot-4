"""Frozen S3 decoder: [X_standardized,Z_original] -> (class1-class0) logit."""
from functools import lru_cache
import numpy as np
import torch
from .sources import ROOT

for p in (ROOT / "nn" / "src",):
    import sys
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from modrot4_nn.net.discrete import DiscreteNPConfig, DiscreteNeuralProcess  # noqa: E402


class Decoder:
    def __init__(self, src):
        state = torch.load(src.checkpoint_file, map_location="cpu", weights_only=True)
        cfg = dict(state["model_config"])
        cfg["decoder_hidden_dims"] = tuple(cfg["decoder_hidden_dims"])
        model = DiscreteNeuralProcess(DiscreteNPConfig(**cfg))
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval().double()
        for p in model.parameters():
            p.requires_grad_(False)
        self.network = model.decoder
        self.x_mean = torch.tensor(src.x_mean, dtype=torch.float64)
        self.x_std = torch.tensor(src.x_std, dtype=torch.float64)
        self.k = src.latent_dim
        self.p = src.x_dim

    def scalar(self, v, target="logit"):
        z, x = v[:self.k], v[self.k:]
        logits = self.network(torch.cat(((x-self.x_mean)/self.x_std, z)))
        eta = logits[1]-logits[0]
        if target == "logit":
            return eta
        if target == "probability":
            return torch.sigmoid(eta)
        raise ValueError(target)

    def predict(self, z, x, batch_size=4096):
        z = np.asarray(z, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        values = []
        with torch.no_grad():
            for start in range(0, len(z), batch_size):
                zz = torch.as_tensor(z[start:start+batch_size], dtype=torch.float64)
                xx = torch.as_tensor(x[start:start+batch_size], dtype=torch.float64)
                yy = self.network(torch.cat(((xx-self.x_mean)/self.x_std, zz), dim=1))
                values.append((yy[:,1]-yy[:,0]).numpy())
        eta = np.concatenate(values)
        return eta, 1/(1+np.exp(-np.clip(eta,-700,700)))


def finite_difference_hessians(fn, points, steps):
    """Central differences, scalar decoder API, for AD independent validation."""
    x = np.asarray(points, dtype=float)
    steps = np.asarray(steps, dtype=float)
    n, d = x.shape
    h = np.empty((n,d,d), dtype=np.float64)
    for row, v in enumerate(x):
        center = float(fn(torch.from_numpy(v)).item())
        for i in range(d):
            hi = np.zeros(d); hi[i] = steps[i]
            fp = float(fn(torch.from_numpy(v+hi)).item())
            fm = float(fn(torch.from_numpy(v-hi)).item())
            h[row,i,i] = (fp-2*center+fm)/(steps[i]**2)
            for j in range(i+1,d):
                hj = np.zeros(d);hj[j]=steps[j]
                a = float(fn(torch.from_numpy(v+hi+hj)).item())
                b = float(fn(torch.from_numpy(v+hi-hj)).item())
                c = float(fn(torch.from_numpy(v-hi+hj)).item())
                dd = float(fn(torch.from_numpy(v-hi-hj)).item())
                h[row,i,j]=h[row,j,i]=(a-b-c+dd)/(4*steps[i]*steps[j])
    return h
