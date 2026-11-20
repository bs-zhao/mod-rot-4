"""Public function contracts for three independent and extensible HE algorithm families."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from he_lab.inference_wy import infer_he as infer_wy, METHODS as WY
from he_lab.inference_wy_mlp import infer_he as infer_mlp, METHODS as MLP
from he_lab.he_metrics import compute_metrics,register_metric,METRICS
from he_lab.he_metrics.registry import metric_direction


class QuadraticFrozenMLP:
    """Synthetic fixed trained surrogate in physical W coordinates."""
    def __call__(self, w, target="logit"):
        if target!="logit":
            raise ValueError("Only logit in this mathematical fixture")
        return 0.5*w[0]*w[0]+w[0]*w[1]+0.2


class Contracts(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(45)
        self.W=rng.normal(size=(250,2))
        self.Y=0.5*self.W[:,0]**2+self.W[:,0]*self.W[:,1]+0.2
        self.eval=self.W[:12].copy()
        self.weights=np.linspace(1,3,12)

    def test_every_family_algorithm_is_a_function(self):
        self.assertEqual(set(WY),{"wy_polynomial","wy_local_quadratic","wy_kernel_ridge"})
        self.assertEqual(set(MLP),{"nn_autograd","nn_finite_difference"})
        for fn in (*WY.values(),*MLP.values()):
            self.assertTrue(callable(fn))

    def test_blind_polynomial_exact_he(self):
        result=infer_wy("wy_polynomial",self.W,self.Y,eval_W=self.eval,
                        weights=self.weights,dim_u=1,target="logit",
                        degree=2,alpha=1e-8)
        np.testing.assert_allclose(result.M,[[1,1],[1,0]],atol=2e-4)
        self.assertEqual(result.M_UX.shape,(1,1))

    def test_blind_kernel_and_local(self):
        for method in ("wy_kernel_ridge","wy_local_quadratic"):
            x=infer_wy(method,self.W,self.Y,eval_W=self.eval[:3],
                       dim_u=1,target="logit",max_centers=100,neighbors=50)
            self.assertEqual(x.M.shape,(2,2))
            self.assertTrue(np.isfinite(x.M).all())
            self.assertTrue((x.M>=-1e-9).all())

    def test_frozen_mlp_autograd_and_finite_diff_direct(self):
        model=QuadraticFrozenMLP()
        for method in ("nn_autograd","nn_finite_difference"):
            x=infer_mlp(method,self.W,self.Y,model,eval_W=self.eval,
                        weights=self.weights,dim_u=1,target="logit",
                        relative_step=1e-3)
            np.testing.assert_allclose(x.M,[[1,1],[1,0]],atol=1e-5)
            self.assertFalse(x.meta["uses_Y_for_fit"])

    def test_metric_is_an_independent_function_with_registry(self):
        scores=compute_metrics(np.array([[1.,2.],[2.,0.]]),1,top_k=1)
        self.assertAlmostEqual(scores["UX.energy"],2.)
        self.assertAlmostEqual(scores["FULL.energy"],5.)
        self.assertEqual(metric_direction("UX.hoyer"),"maximize")
        self.assertEqual(metric_direction("FULL.energy"),"descriptive")
        # New metrics require adding one callable and registering once;
        # the pipeline never needs a new per-metric if/else branch.
        name="test_sum_of_squares"
        try:
            register_metric(name,lambda v,*,top_k=1: float(np.dot(v,v)),direction="minimize")
            self.assertAlmostEqual(compute_metrics(np.array([[1.,2.],[2.,0.]]),1)
                                   ["FULL."+name],9.)
        finally:
            METRICS.pop(name,None)
            from he_lab.he_metrics.registry import DIRECTIONS
            DIRECTIONS.pop(name,None)

    def test_binary_requires_probability_target(self):
        with self.assertRaises(ValueError):
            infer_wy("wy_polynomial",self.W,np.zeros(len(self.W)),
                     eval_W=self.eval,dim_u=1,target="logit",binary=True)


if __name__=="__main__":
    unittest.main()
