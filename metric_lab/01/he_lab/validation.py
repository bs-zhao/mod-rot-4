"""Mathematical and upstream-contract smoke checks before expensive runs."""
import numpy as np
from .energy import moment_from_hessians, transform_moment
from .coordinates import full_jacobian
from .oracle import oracle_scalar
from .sources import get_spec


def smoke(src, rotations, root):
    import torch
    from .pipeline import _json
    rng=np.random.default_rng(2026)
    h=rng.normal(size=(10,src.latent_dim+src.x_dim,src.latent_dim+src.x_dim))
    h=(h+h.transpose(0,2,1))/2
    m=moment_from_hessians(h)
    for source in ("oracle","nn"):
        j=full_jacobian(rotations[0].jacobian(source),src.x_dim)
        new=np.einsum("ia,nij,jb->nab",j,h,j)
        expected=np.mean(new**2,axis=0)
        actual=transform_moment(m,j)
        if not np.allclose(actual,expected,rtol=1e-9,atol=1e-9):
            raise AssertionError(f"Moment transform failed: {source}")
    rot=rotations[0]
    z=src.z[:min(len(src.z),10)]
    if not np.allclose(rot.u_to_z(rot.z_to_u(z)),z,atol=1e-9):
        raise AssertionError("S9 z inversion failed")
    t=src.parameters[:min(len(src.parameters),10)]
    if not np.allclose(rot.oracle_u_to_t(rot.t_to_oracle_u(t)),t,atol=1e-9):
        raise AssertionError("Oracle T inversion failed")
    # Compare differentiable oracle against the canonical source formula.
    spec=get_spec(src.model)
    x=src.x[0]
    tt=t[0]
    exact=float(spec.logit(x[None,:],tt)[0])
    from_torch=float(oracle_scalar(src.model,
        torch.as_tensor(np.r_[tt,x],dtype=torch.float64),"logit").item())
    if not np.isclose(exact,from_torch,rtol=1e-9,atol=1e-9):
        raise AssertionError(f"Oracle formula differs from source: {exact} != {from_torch}")
    _json(root/"validation"/"smoke.json",dict(ok=True,model=src.model,
          moment_transform=True,inverse_rotations=True,
          oracle_formula_abs_error=abs(exact-from_torch)))
    print(f"[VALIDATE] {src.model}: core mathematical and source checks passed")
