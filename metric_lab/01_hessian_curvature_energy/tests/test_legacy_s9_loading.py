"""Regression tests against historical S9 producer schemas (not synthetic modern-only)."""
import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from he_lab.sources import load_rotations


class HistoricalSweepSchemas(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dim = 4
        rng = np.random.default_rng(55)
        self.z = rng.normal(size=(14, self.dim))
        self.src = SimpleNamespace(
            run_dir=self.root, latent_dim=self.dim, z=self.z,
            parameters=np.zeros((len(self.z), self.dim))
        )
        self.Rstar = np.eye(self.dim)
        theta = np.deg2rad(15.)
        self.G = np.eye(self.dim)
        self.G[:2, :2] = [[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]]
        for name in ("s8_supervised_rotation", "s8_2_supervised_rotation"):
            path = self.root / name
            path.mkdir()
            keys = dict(R_star=self.Rstar, t_train_mean=np.zeros(self.dim),
                        t_train_std=np.ones(self.dim))
            if name == "s8_2_supervised_rotation":
                keys["global_z_mean"] = self.z.mean(0)
                keys["global_z_std"] = self.z.std(0, ddof=0)
            np.savez_compressed(path / "rotation.npz", **keys)

    def write_s9(self, variant, *, modern=False, with_z_stats=False,
                 wrong_stats=False, include_baseline=False):
        stage = ("s9_rotation_sweep" if variant == "s9"
                 else "s9_2_rotation_sweep")
        base = self.root / stage
        (base / "rotations" / "k02").mkdir(parents=True)
        records = []
        path = "rotations/k02/axes_01_02__deg_015.npz"
        keys = dict(R=self.G, R_star=self.Rstar, delta_R=self.G)
        if with_z_stats:
            keys["global_z_mean"] = (self.z.mean(0) if variant=="s9_2"
                                    else np.zeros(self.dim))
            keys["global_z_std"] = (self.z.std(0,ddof=0) if variant=="s9_2"
                                   else np.ones(self.dim))
            if wrong_stats:
                keys["global_z_std"] = np.ones(self.dim)*7
        np.savez_compressed(base / path, **keys)
        records.append(dict(axes_1based="1,2",angle_degrees="15",
                            rotation_file=path))
        if modern:
            records[0]["candidate_type"] = "sweep"
        if include_baseline:
            (base / "rotations" / "baseline").mkdir()
            path2 = "rotations/baseline/unswept.npz"
            base_keys = dict(R=self.Rstar, delta_R=np.eye(self.dim))
            if with_z_stats:
                base_keys["global_z_mean"] = keys["global_z_mean"]
                base_keys["global_z_std"] = keys["global_z_std"]
            np.savez_compressed(base / path2, **base_keys)
            row = dict(axes_1based="",angle_degrees="",rotation_file=path2)
            if modern:
                row["candidate_type"] = "baseline_unswept"
            records.insert(0,row)
        with (base / "rotation_index.csv").open("w",newline="") as f:
            cols = list(records[0])
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(records)

    def test_legacy_s9_without_kind_or_z_scaler_recovers_baseline(self):
        self.write_s9("s9")
        got = load_rotations(self.src, "s9")
        self.assertEqual([a.name for a in got],
                         ["baseline_unswept", "axes_01_02__deg_015"])
        np.testing.assert_array_equal(got[0].G, np.eye(self.dim))
        np.testing.assert_array_equal(got[1].z_std, np.ones(self.dim))
        np.testing.assert_array_equal(got[1].z_mean, np.zeros(self.dim))
        np.testing.assert_allclose(got[1].R, self.G)

    def test_new_s9_keeps_existing_baseline_once(self):
        self.write_s9("s9", modern=True, with_z_stats=True,include_baseline=True)
        got = load_rotations(self.src, "s9")
        self.assertEqual(len(got),2)
        self.assertEqual(got[0].candidate_type,"baseline_unswept")

    def test_legacy_s9_2_missing_npz_scaler_uses_saved_s8_2(self):
        self.write_s9("s9_2",modern=False,with_z_stats=False)
        got = load_rotations(self.src, "s9_2")
        np.testing.assert_allclose(got[1].z_mean,self.z.mean(0))
        np.testing.assert_allclose(got[1].z_std,self.z.std(0,ddof=0))
        np.testing.assert_allclose(got[1].u_to_z(got[1].z_to_u(self.z)),self.z)

    def test_s9_2_invalid_scaler_cannot_be_silently_replaced(self):
        self.write_s9("s9_2",modern=True,with_z_stats=True,wrong_stats=True)
        with self.assertRaisesRegex(ValueError,"inconsistent"):
            load_rotations(self.src,"s9_2")

    def test_s9_2_without_upstream_scaler_fails_explicitly(self):
        self.write_s9("s9_2")
        np.savez_compressed(
            self.root / "s8_2_supervised_rotation" / "rotation.npz",
            R_star=self.Rstar,t_train_mean=np.zeros(self.dim),
            t_train_std=np.ones(self.dim))
        with self.assertRaisesRegex(ValueError,"requires S8_2"):
            load_rotations(self.src,"s9_2")

if __name__=="__main__":
    unittest.main()
