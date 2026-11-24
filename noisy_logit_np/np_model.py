"""Regression Neural Process: Set-Transformer subject encoder -> Gaussian latent z; MLP decoder [x, z] -> scalar.

Same architecture as the project's categorical NP (encoder: embed 32, 2 SAB + PMA, 4 heads, summary 64;
decoder: [x,z] -> 128-256-128 GELU MLP) except that the output is ONE real number (the predicted logit)
instead of two class logits, and the reconstruction loss is MSE.
"""
import torch
import torch.nn as nn


class SetAttentionBlock(nn.Module):
    def __init__(s, e, h):
        super().__init__(); s.mha = nn.MultiheadAttention(e, h, batch_first=True); s.ln1 = nn.LayerNorm(e)
        s.fc = nn.Sequential(nn.Linear(e, e), nn.ReLU(), nn.Linear(e, e)); s.ln2 = nn.LayerNorm(e)
    def forward(s, x, pad=None):
        a, _ = s.mha(x, x, x, key_padding_mask=pad); x = s.ln1(x + a); return s.ln2(x + s.fc(x))


class PMA(nn.Module):
    def __init__(s, e, h):
        super().__init__(); s.seed = nn.Parameter(torch.randn(1, 1, e)); s.mha = nn.MultiheadAttention(e, h, batch_first=True)
        s.ln1 = nn.LayerNorm(e); s.fc = nn.Sequential(nn.Linear(e, e), nn.ReLU(), nn.Linear(e, e)); s.ln2 = nn.LayerNorm(e)
    def forward(s, x, pad=None):
        seeds = s.seed.repeat(x.shape[0], 1, 1); a, _ = s.mha(seeds, x, x, key_padding_mask=pad)
        x = s.ln1(seeds + a); return s.ln2(x + s.fc(x)).squeeze(1)


class SetTransformer(nn.Module):
    def __init__(s, input_dim, summary_dim=64, heads=4, embed=32):
        super().__init__()
        s.embed = nn.Sequential(nn.Linear(input_dim, embed), nn.ReLU(), nn.Linear(embed, embed))
        s.sab1 = SetAttentionBlock(embed, heads); s.sab2 = SetAttentionBlock(embed, heads)
        s.pma = PMA(embed, heads); s.out = nn.Linear(embed, summary_dim)
    def forward(s, x, pad=None):
        o = s.sab2(s.sab1(s.embed(x), pad), pad); return s.out(s.pma(o, pad))


def pack_context(obs, keep):
    """Move kept trials to the front (vectorised). obs (B,T,D), keep (B,T) bool -> packed, pad_mask (True = padding)."""
    B, T, D = obs.shape
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    packed = torch.gather(obs, 1, order.unsqueeze(-1).expand(-1, -1, D))
    n = keep.sum(1); ar = torch.arange(T, device=obs.device)[None, :]
    pad = ar >= n[:, None]
    pad[n == 0, 0] = False                       # keep one dummy slot so attention is defined
    tmax = max(1, int(n.max()))
    packed = packed[:, :tmax].masked_fill(pad[:, :tmax].unsqueeze(-1), 0.0)
    return packed, pad[:, :tmax]


class RegNP(nn.Module):
    def __init__(s, x_dim, latent_dim, summary_dim=64, decoder_hidden=(128, 256, 128)):
        super().__init__()
        s.x_dim, s.latent_dim = x_dim, latent_dim
        s.encoder_backbone = SetTransformer(x_dim + 1, summary_dim)
        s.fc_mu = nn.Linear(summary_dim, latent_dim); s.fc_logvar = nn.Linear(summary_dim, latent_dim)
        dims = [x_dim + latent_dim, *decoder_hidden, 1]; layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2: layers.append(nn.GELU())
        s.decoder = nn.Sequential(*layers)

    def encode(s, x, y, keep):
        packed, pad = pack_context(torch.cat([x, y.unsqueeze(-1)], -1), keep)
        h = s.encoder_backbone(packed, pad); return s.fc_mu(h), s.fc_logvar(h)

    def decode(s, x, z):                          # x (B,T,p), z (B,L) -> (B,T)
        return s.decoder(torch.cat([x, z.unsqueeze(1).expand(-1, x.shape[1], -1)], -1)).squeeze(-1)

    def forward(s, x, y, keep, sample=True):
        mu, lv = s.encode(x, y, keep)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if sample else mu
        return s.decode(x, z), mu, lv, z


def kl_loss(mu, lv):
    return (-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1)).mean()


def sample_keep(valid, pmin, pmax, gen):
    """Random context subset per subject, size ~ U[floor(pmin n), round(pmax n)] (vectorised)."""
    B, T = valid.shape; n = valid.sum(1)
    lo = torch.clamp(torch.floor(pmin * n.float()).long(), min=1); hi = torch.maximum(lo, torch.round(pmax * n.float()).long())
    npick = lo + (torch.rand(B, generator=gen, device=valid.device) * (hi - lo + 1).float()).long().clamp(max=(hi - lo))
    score = torch.rand(B, T, generator=gen, device=valid.device).masked_fill(~valid, 2.0)
    rank = torch.argsort(torch.argsort(score, dim=1), dim=1)
    return (rank < npick[:, None]) & valid
