"""S8: full-dimensional genuine orthogonal R, with rectangular L > dim(T)."""
import unittest

import numpy as np

from modrot4_nn.supervised_rotation import fit_supervised_rotation
from modrot4_nn.analysis import corr


class TestSupervisedRotation(unittest.TestCase):
    def test_recovers_corresponding_axes_with_extra_latent_dimensions(self):
        rng=np.random.default_rng(40)
        n,latent_dim,tdim=700,6,3
        theta=rng.normal(size=(n,tdim))
        nuisance=rng.normal(size=(n,latent_dim-tdim))
        true_r,_=np.linalg.qr(rng.normal(size=(latent_dim,latent_dim)))
        latent=np.concatenate((theta,nuisance),axis=1) @ true_r.T
        train=np.arange(400)
        result=fit_supervised_rotation(latent,theta,train)
        self.assertEqual(result["R"].shape,(latent_dim,latent_dim))
        self.assertEqual(result["u"].shape,(n,tdim))
        np.testing.assert_allclose(result["R"]@result["R"].T,
                                   np.eye(latent_dim),atol=1e-10)
        test_corr=corr(theta[500:],result["u"][500:])
        self.assertGreater(float(np.min(np.diag(test_corr))),.92)
        self.assertGreater(float(np.min(np.diag(test_corr))),
                           float(np.max(np.abs(test_corr-np.diag(np.diag(test_corr))))))
        # No centering or per-coordinate scaling hidden in RZ: exact raw rotation.
        np.testing.assert_allclose(result["rz"],latent@result["R"].T,
                                   rtol=1e-11,atol=1e-11)

    def test_requires_enough_latent_dimensions(self):
        z=np.ones((20,2))
        t=np.ones((20,3))
        with self.assertRaisesRegex(ValueError,"latent_dim"):
            fit_supervised_rotation(z,t,np.arange(12))


if __name__=="__main__":
    unittest.main()
