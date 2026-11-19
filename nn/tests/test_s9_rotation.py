"""S9 exhaustive subset enumeration and ordered Givens sweep tests."""
from __future__ import annotations
import math
import unittest
import numpy as np

from modrot4_nn.s9_rotation import (
    ANGLES, axis_subsets, count_subsets, expected_rotation_count,
    givens_sweep, rotate_from_oracle, rotation_id)


class TestS9Sweep(unittest.TestCase):
    def test_combinatorial_coverage(self):
        for d in (2,3,4,6,8):
            sets=list(axis_subsets(d))
            self.assertEqual(len(sets),count_subsets(d))
            self.assertEqual(len(sets),2**d-d-1)
            self.assertEqual(len(set(sets)),len(sets))
            self.assertEqual(sets[0],(0,1))
            self.assertEqual(sets[-1],tuple(range(d)))
            for subset in sets:
                self.assertGreaterEqual(len(subset),2)
                self.assertEqual(tuple(sorted(subset)),subset)
            self.assertEqual(expected_rotation_count(d),7*len(sets))
        self.assertEqual(expected_rotation_count(3),28)
        self.assertEqual(expected_rotation_count(4),77)

    def test_givens_2d_exact_angles(self):
        for angle in ANGLES:
            g=givens_sweep(2,(0,1),angle)
            rad=math.radians(angle)
            expected=np.array([[math.cos(rad),-math.sin(rad)],
                               [math.sin(rad),math.cos(rad)]])
            np.testing.assert_allclose(g,expected,atol=1e-14)
        np.testing.assert_allclose(givens_sweep(4,(1,3),90)[:,[0,2]],
                                   np.eye(4)[:,[0,2]],atol=1e-14)

    def test_all_subsets_4d_angles_preserve_orthogonality_and_zero(self):
        rng=np.random.default_rng(3)
        oracle,_=np.linalg.qr(rng.normal(size=(6,6)))
        z=rng.normal(size=(20,6))
        for axes in axis_subsets(4):
            for angle in ANGLES:
                r,g=rotate_from_oracle(oracle,axes,angle)
                np.testing.assert_allclose(r,r if angle else oracle,atol=1e-12)
                np.testing.assert_allclose(r@r.T,np.eye(6),atol=1e-12)
                np.testing.assert_allclose(r,g@oracle,atol=1e-12)
                inactive=np.array(sorted(set(range(6))-set(axes)))
                np.testing.assert_allclose(g[inactive],np.eye(6)[inactive],
                                           atol=1e-12)
                np.testing.assert_allclose(z@r.T,(z@oracle.T)@g.T,atol=1e-12)
        self.assertEqual(rotation_id((0,2,3),45),"axes_01_03_04__deg_045")


if __name__ == "__main__":
    unittest.main()
