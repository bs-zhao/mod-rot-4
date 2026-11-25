"""<MODEL>L: same T / X / true logit as <MODEL>, observation y = logit + noise (perfect-predictor R^2 = 0.8)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models" / "src"))

from modrot4.core import FIELDS, LOGIT_FIELDS, generate, load, save
from modrot4.model_specs import get_spec, MODELS
from modrot4.noisy_logit_precheck import run_noisy_logit_precheck
from modrot4.pipeline import LOGIT_MODELS, dataset_path


def _bank():
    rng = np.random.default_rng(3)
    return np.column_stack([
        rng.uniform(5, 200, 500), rng.uniform(5, 200, 500), rng.uniform(.05, .95, 500),
        rng.uniform(5, 200, 500), rng.uniform(5, 200, 500), rng.uniform(.05, .95, 500)])


def _small(model_id, people):
    spec = get_spec(model_id)
    seed = 20261028 if model_id.startswith("CR") else 20261006
    if spec.base_id in ("CR3", "CR5"):
        spec.bank = _bank()
        with patch.object(spec, "prepare"):
            return spec, generate(spec, people, seed, 20261023)
    return spec, generate(spec, people, seed, 20261023)


class TestLogitModels(unittest.TestCase):
    def test_same_T_X_logit_as_choice_model_and_noise_level(self):
        for base in MODELS:
            with self.subTest(model=base):
                _, choice = _small(base, 40)
                spec, noisy = _small(base + "L", 40)
                self.assertEqual(spec.base_id, base)
                for field in ("parameters", "trial_counts", "subject_index", "trial_index", "x", "logit"):
                    np.testing.assert_array_equal(getattr(choice, field), getattr(noisy, field))
                self.assertFalse(hasattr(noisy, "choice"))
                residual = noisy.y - noisy.logit
                self.assertAlmostEqual(residual.std() / np.sqrt(np.var(noisy.logit) * 0.25), 1.0, delta=0.05)
                _, again = _small(base + "L", 40)
                np.testing.assert_array_equal(noisy.y, again.y)

    def test_schema_roundtrip_and_precheck_report(self):
        for base in MODELS:
            with self.subTest(model=base), tempfile.TemporaryDirectory() as tmp:
                spec, data = _small(base + "L", 5)
                file = Path(tmp) / "dataset.npz"
                save(data, file)
                with np.load(file) as stored:
                    self.assertEqual(set(stored.files), set(LOGIT_FIELDS))
                    self.assertNotEqual(set(stored.files), set(FIELDS))
                output = Path(tmp) / "precheck" / "s2_precheck__report.json"
                with patch("modrot4.noisy_logit_precheck.precheck_path", return_value=output):
                    report = run_noisy_logit_precheck(spec, load(file), seed=20261006, data_path=file)
                self.assertEqual(sorted(report["checks"]),
                                 ["fisher_information", "parameter_recovery", "response_balance", "structure"])
                self.assertTrue(report["checks"]["response_balance"]["passed"])
                self.assertTrue(output.is_file())

    def test_logit_models_have_their_own_folders(self):
        for base, name in zip(MODELS, LOGIT_MODELS):
            self.assertEqual(name, base + "L")
            self.assertNotEqual(dataset_path(name), dataset_path(base))


if __name__ == "__main__":
    unittest.main()
