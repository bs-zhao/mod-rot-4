"""No upstream data needed: pure-math regression tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import torch

from he_lab.coordinates import Rotation,full_jacobian
from he_lab.energy import moment_from_hessians,transform_moment,result_from_moment
from he_lab.oracle import oracle_scalar
from he_lab.derivatives import hessians
from he_lab.he_metrics import compute_metrics
from he_lab.inference_wy.polynomial import _poly_terms,polynomial_hessians


def rotation(k=2):
    a=np.pi/5
    g=np.eye(k)
    g[:2,:2]=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    return Rotation("test","s9","sweep",(0,1),36.,g,np.eye(k),g,
                    np.zeros(k),np.ones(k),np.zeros(k),
                    np.r_[np.array([2.,3.]),np.ones(k-2)])


class Tests(unittest.TestCase):
    def test_affine_inverse_and_G(self):
        r=rotation()
        r.validate()
        rng=np.random.default_rng(4)
        z=rng.normal(size=(50,2))
        t=rng.normal(size=(50,2))
        np.testing.assert_allclose(r.u_to_z(r.z_to_u(z)),z,atol=1e-12)
        np.testing.assert_allclose(r.oracle_u_to_t(r.t_to_oracle_u(t)),t,atol=1e-12)

    def test_he_moment_transformation(self):
        rng=np.random.default_rng(3)
        h=rng.normal(size=(20,4,4))
        h=(h+np.swapaxes(h,1,2))/2
        weights=rng.uniform(.1,2,size=20)
        cross=moment_from_hessians(h,weights)
        r=rotation()
        j=full_jacobian(r.jacobian("oracle"),2)
        transformed=np.einsum("ia,nij,jb->nab",j,h,j)
        expect=np.average(transformed**2,axis=0,weights=weights)
        got=transform_moment(cross,j)
        np.testing.assert_allclose(got,expect,rtol=1e-11,atol=1e-11)
        self.assertTrue(np.allclose(got,got.T))

    def test_invariant_frobenius_energy_when_only_rotating(self):
        rng=np.random.default_rng(2)
        h=rng.normal(size=(7,4,4));h=(h+h.transpose(0,2,1))/2
        cross=moment_from_hessians(h)
        g=rotation().G
        j=full_jacobian(g.T,2)
        got=transform_moment(cross,j)
        self.assertAlmostEqual(float(np.sum(got)),float(np.mean(np.sum(h**2,axis=(1,2)))),places=10)

    def test_c1_oracle_hessian(self):
        arr=np.array([.1,.2,.3,.4,.5,-.3,.8])
        h=hessians(lambda v: oracle_scalar("C1",v,"logit"),arr[None,:])[0]
        self.assertAlmostEqual(h[1,4],np.sqrt(3),places=10)
        self.assertAlmostEqual(h[2,5],np.sqrt(3),places=10)
        self.assertAlmostEqual(h[3,6],np.sqrt(3),places=10)
        self.assertAlmostEqual(h[0,0],0.,places=10)
        self.assertAlmostEqual(h[1,2],0.,places=10)

    def test_c2_oracle_x_quadratic(self):
        arr=np.array([.1,.2,.3,.4,.7,.2,-.2,.6])
        h=hessians(lambda v: oracle_scalar("C2",v,"logit"),arr[None,:])[0]
        self.assertAlmostEqual(h[4,4],3*np.sqrt(5)*arr[2],places=10)
        self.assertAlmostEqual(h[2,4],3*np.sqrt(5)*arr[4],places=10)

    def test_c3_and_cpt_oracle_finite(self):
        c3=np.r_[np.ones(6)*.3,np.array([.6,.4,.1,-.5])]
        cr3=np.r_[np.array([.8,2.,.7,2.]),np.array([80.,40.,.4,100.,70.,.7])]
        cr4=np.r_[np.array([2.,.7,2.]),cr3[4:]]
        for model,data in [("C3",c3),("CR3",cr3),("CR4",cr4),("CR5",cr4)]:
            for target in ("logit","probability"):
                hh=hessians(lambda v:oracle_scalar(model,v,target),data[None,:])
                self.assertTrue(np.isfinite(hh).all(),msg=(model,target))

    def test_metrics_zero_and_sparse(self):
        m=np.zeros((4,4))
        m[0,2]=m[2,0]=3.
        x=compute_metrics(m,2)
        self.assertAlmostEqual(x["UX.energy"],3.)
        self.assertAlmostEqual(x["UX.hhi"],1.)
        self.assertAlmostEqual(x["FULL.energy"],6.)
        self.assertTrue(np.isnan(x["UU.diagonal.hoyer"]))

    def test_polynomial_terms_exact(self):
        q=np.array([[1.2,2.3],[-.4,.5]])
        coef=np.array([.5,2.])
        powers=np.array([[2,0],[1,1]])
        value,g,h=_poly_terms(q,coef,powers)
        np.testing.assert_allclose(h,np.array([[[1.,2.],[2.,0.]]]*len(q)))

    def test_polynomial_estimator(self):
        rng=np.random.default_rng(0)
        w=rng.normal(size=(400,2))
        y=w[:,0]*w[:,1]+.5*w[:,0]**2+.3
        hh=polynomial_hessians(w,y,w[:10],binary=False,degree=2,alpha=1e-8)
        np.testing.assert_allclose(hh[0],[[1.,1.],[1.,0.]],atol=1e-5)


if __name__=="__main__":
    unittest.main()
