"""Small tests for the actual shared Generate/Precheck interface."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models" / "src"))

from modrot4.core import FIELDS, generate, load, save
from modrot4.model_specs import get_spec, MODELS, CR4_LOW, CR4_HIGH
from modrot4.unified_precheck import run_precheck
from modlearn_rot.utils.seeds import make_rng_streams


class TestModels(unittest.TestCase):
    def _run_small(self, model_id, people=5):
        spec = get_spec(model_id)
        if model_id in ("CR3", "CR5"):
            # Use a small deterministic test bank; production independently
            # builds the complete optimized 500-X model bank.
            rng = np.random.default_rng(3)
            bank = np.column_stack([
                rng.uniform(5, 200, 500), rng.uniform(5, 200, 500),
                rng.uniform(.05, .95, 500), rng.uniform(5, 200, 500),
                rng.uniform(5, 200, 500), rng.uniform(.05, .95, 500)])
            spec.bank = bank
            with patch.object(spec, "prepare"):
                data = generate(spec, people, 20261028, 20261023)
        else:
            data = generate(
                spec, people, 20261006 if model_id.startswith("C") and
                not model_id.startswith("CR") else 20261028, 20261023)
        return spec, data

    def test_identical_generation_schema_and_logit_p_choice_for_six(self):
        for name in MODELS:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                spec, data = self._run_small(name)
                file = Path(tmp) / "dataset.npz"
                save(data, file)
                with np.load(file) as stored:
                    self.assertEqual(set(stored.files), set(FIELDS))
                loaded = load(file)
                for field in FIELDS:
                    np.testing.assert_array_equal(
                        getattr(data, field), getattr(loaded, field))
                self.assertEqual(data.parameters.shape, (5, spec.n_parameters))
                self.assertEqual(data.x.shape[0], int(data.trial_counts.sum()))
                self.assertEqual(data.logit.shape, data.choice.shape)
                self.assertEqual(data.p.shape, data.choice.shape)
                np.testing.assert_array_equal(data.p, expit(data.logit))
                self.assertTrue(np.isin(data.choice, (0, 1)).all())
                offsets = np.r_[0, np.cumsum(data.trial_counts)]
                for subject in range(5):
                    a, b = offsets[subject:subject+2]
                    np.testing.assert_allclose(
                        data.logit[a:b],
                        spec.logit(data.x[a:b], data.parameters[subject]),
                        rtol=0, atol=1e-12)

    def test_precheck_reads_T_X_choice_with_same_report_structure(self):
        keys = None
        for name in MODELS:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                spec, data = self._run_small(name)
                file = Path(tmp) / "dataset.npz"
                save(data, file)
                output = Path(tmp) / "precheck" / "s2_precheck__report.json"
                with patch("modrot4.unified_precheck.precheck_path",
                           return_value=output):
                    report = run_precheck(spec, load(file), seed=20261006,
                                          data_path=file)
                present = tuple(sorted(report["checks"].keys()))
                if keys is None:
                    keys = present
                self.assertEqual(present, keys)
                self.assertTrue(report["checks"]["structure"]["passed"])
                self.assertFalse(
                    report["checks"]["parameter_recovery"]["conclusive"])
                self.assertEqual(
                    report["checks"]["parameter_recovery"]["sample_size"], 5)
                self.assertTrue(output.is_file())
                self.assertTrue(
                    (output.parent / "s2_precheck__recovery.csv").is_file())

    def test_iid_x_algorithm_is_shared_between_c1_c2_c3_cr4(self):
        names = ("parameters", "trial_counts", "inputs", "observations")
        for name in ("C1", "C2", "C3", "CR4"):
            with self.subTest(model=name):
                spec, data = self._run_small(name, 3)
                seed = 20261006 if name.startswith("C") and not name.startswith("CR") else 20261028
                rng = make_rng_streams(seed, names)["inputs"]
                low = CR4_LOW if name == "CR4" else -1.
                high = CR4_HIGH if name == "CR4" else 1.
                expected = np.concatenate([
                    rng.uniform(low, high, size=(int(count), spec.x_dim))
                    for count in data.trial_counts
                ])
                np.testing.assert_array_equal(data.x, expected)


if __name__ == "__main__":
    unittest.main()
