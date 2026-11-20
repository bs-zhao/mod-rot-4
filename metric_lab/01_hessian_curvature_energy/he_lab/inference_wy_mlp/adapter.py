"""Present the frozen S3 decoder as a differentiable function of W=[U,X_raw].

The supplied W and Y are observational inputs; Y never modifies frozen weights.
"""
import torch
import numpy as np

class FrozenWMLP:
    def __init__(self, decoder, rotation):
        self.decoder=decoder
        self.rotation=rotation
        self.dim_u=rotation.dim
        self.dim_x=decoder.p
        if decoder.k != rotation.dim:
            raise ValueError("Decoder latent dim and R dimension differ")
        # For row W: z=(u @ R) * sd_z + mu_z.
        self.R=torch.as_tensor(rotation.R,dtype=torch.float64)
        self.mu=torch.as_tensor(rotation.z_mean,dtype=torch.float64)
        self.sd=torch.as_tensor(rotation.z_std,dtype=torch.float64)

    def scalar(self, w, target="logit"):
        if w.shape[-1]!=self.dim_u+self.dim_x:
            raise ValueError("W dimension incompatible with trained MLP")
        z=(w[:self.dim_u]@self.R)*self.sd+self.mu
        return self.decoder.scalar(torch.cat((z,w[self.dim_u:])),target=target)

    def __call__(self, w, target="logit"):
        return self.scalar(w,target)
