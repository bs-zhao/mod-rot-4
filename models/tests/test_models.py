"""Fast, deterministic S1/S2 smoke tests; no optimized CR2 bank regeneration."""
from __future__ import annotations
import json
import shutil
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models" / "src"))
sys.path.insert(0, str(ROOT / "models" / "reference"))
sys.path.insert(0, str(ROOT / "models" / "reference" / "cr4"))

from modlearn_rot.generators.registry import get_model
from modlearn_rot.generators.dataset import generate_dataset, load_dataset
from s0_s1_cr4 import (sample_subject_x, sample_bank, sample_theta, theta_as_cr3,
                      logits, verify_x)
from s0_cr2_design_recovery import logits as cr2_logits


class TestModels(unittest.TestCase):
    def test_all_defaults_and_shapes(self):
        for name, dim, xdim in [("C1",4,3), ("C2",4,4), ("C3",6,4)]:
            with self.subTest(name=name):
                model = get_model(name)
                self.assertEqual(model.n_subjects, 8000)
                self.assertEqual(model.n_factors, dim)
                model.n_subjects = 3
                dataset = generate_dataset(model, 20261006)
                self.assertEqual(dataset.parameters.shape, (3, dim))
                self.assertEqual(dataset.x.shape[1], xdim)
                self.assertEqual(dataset.trial_counts.shape, (3,))
                self.assertTrue(np.all((dataset.trial_counts >= 300) &
                                       (dataset.trial_counts <= 500)))
                self.assertTrue(np.isfinite(dataset.predictive_true).all())
                self.assertTrue(np.isin(dataset.y, [0, 1]).all())

    def test_c1_c2_c3_seeded_identical_generation(self):
        for name in ("C1","C2","C3"):
            with self.subTest(name=name):
                model = get_model(name)
                model.n_subjects = 2
                a = generate_dataset(model, 20261006)
                b = generate_dataset(model, 20261006)
                for key in ("parameters","trial_counts","subject_index",
                            "trial_index","x","y","predictive_true"):
                    np.testing.assert_array_equal(getattr(a,key), getattr(b,key))

    def test_cr4_iid_and_alpha_one_equivalence(self):
        x = sample_subject_x(3, seed=20263032)
        self.assertEqual(x.shape,(3,500,6))
        self.assertTrue(verify_x(x))
        self.assertFalse(np.array_equal(x[0],x[1]))
        self.assertFalse(np.array_equal(x[0,0],x[0,1]))
        np.testing.assert_array_equal(x[:2],sample_subject_x(2, seed=20263032))
        # Regression: the per-subject implementation is byte-equivalent to the
        # upstream vectorized IID sampler for the same seed/trial count.
        np.testing.assert_array_equal(x, sample_bank(seed=20263032, n=1500).reshape(3,500,6))
        t = sample_theta(3,np.random.default_rng(19))
        self.assertEqual(t.shape,(3,3))
        np.testing.assert_allclose(logits(x,t[:,None,:]),
                                   cr2_logits(x,theta_as_cr3(t)[:,None,:]),
                                   rtol=0,atol=0)

    def test_output_path_contract(self):
        from modrot4.pipeline import (dataset_path, precheck_path, _cr2_bank,
                                      _cr3_bank, _cr5_bank)
        self.assertEqual(
            dataset_path("CR4"),
            ROOT/"outputs"/"models"/"generate"/"CR4"/"s1_generate__dataset.npz")
        self.assertEqual(
            precheck_path("CR4"),
            ROOT/"outputs"/"models"/"precheck"/"CR4"/"s2_precheck__report.json")
        self.assertIn("models/generate/CR2/s0_design_recovery",
                      _cr2_bank().as_posix())
        self.assertIn("models/generate/CR3/s0_design",
                      _cr3_bank().as_posix())
        self.assertIn("models/generate/CR5/s0_design",
                      _cr5_bank().as_posix())

    def test_cr4_pipeline_and_precheck(self):
        # Exercise the CLI scripts without contaminating a real 8000-subject run.
        with tempfile.TemporaryDirectory() as temp:
            sandbox = Path(temp)
            shutil.copytree(ROOT/"models", sandbox/"models",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            path = (sandbox/"outputs"/"models"/"generate"/"CR4"/
                    "s1_generate__dataset.npz")
            for stage, tail in (("s1_generate.py",["--subjects","5"]),
                                ("s2_precheck.py",[])):
                complete = subprocess.run(
                    [sys.executable, str(sandbox/"models"/stage),
                     "--model", "CR4", *tail],
                    check=(stage == "s1_generate.py"), cwd=sandbox)
                if stage == "s2_precheck.py":
                    # Five subjects cannot establish an r>=0.8 recovery gate;
                    # the complete S2 report should still be written.
                    self.assertNotEqual(complete.returncode, 0)
            data = load_dataset(path)
            self.assertEqual(data.parameters.shape, (5, 3))
            self.assertEqual(data.x.shape, (5*500, 6))
            report_file = (sandbox/"outputs"/"models"/"precheck"/"CR4"/
                           "s2_precheck__report.json")
            report = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertFalse(report["qualified"])
            self.assertEqual(report["subjects"], 5)
            for key in ("structure", "generator_fidelity", "stimulus_design",
                        "response_balance", "fisher_information"):
                self.assertIn(key, report["checks"])
                self.assertTrue(report["checks"][key]["passed"], key)
            self.assertFalse(report["checks"]["parameter_recovery"]["conclusive"])
            self.assertEqual(report["checks"]["parameter_recovery"]["sample_size"], 5)
            self.assertTrue((sandbox/"outputs"/"models"/"precheck"/"CR4"/
                             "s2_precheck__oracle_subjects.npz").is_file())
            self.assertFalse((sandbox/"outputs"/"choice").exists())

    def test_uniform_precheck_c1_c2_c3(self):
        # All three classic generators must run the SAME S2 entrypoint.
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp)
            shutil.copytree(ROOT/"models", sandbox/"models",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for model in ("C1", "C2", "C3"):
                subprocess.run(
                    [sys.executable, str(sandbox/"models"/"s1_generate.py"),
                     "--model", model, "--subjects", "5"],
                    check=True, cwd=sandbox, capture_output=True, text=True)
                result = subprocess.run(
                    [sys.executable, str(sandbox/"models"/"s2_precheck.py"),
                     "--model", model],
                    check=False, cwd=sandbox, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                report_file = (sandbox/"outputs"/"models"/"precheck"/model/
                               "s2_precheck__report.json")
                self.assertTrue(report_file.is_file(), result.stderr)
                report = json.loads(report_file.read_text(encoding="utf-8"))
                self.assertTrue(report["checks"]["structure"]["passed"])
                self.assertTrue(report["checks"]["generator_fidelity"]["passed"])
                self.assertFalse(report["checks"]["parameter_recovery"]["conclusive"])
                self.assertEqual(report["checks"]["parameter_recovery"]["sample_size"],5)
                self.assertTrue((report_file.parent/
                                 "s2_precheck__oracle_subjects.npz").is_file())

    def test_unified_cpt_estimators_on_real_saved_choice_rows(self):
        # Unit-test common CR3/CR5 MLE logic with synthetic saved X/Y data.
        # Does not replace bank-provenance end-to-end tests.
        from unittest.mock import patch
        from scipy.special import expit
        from modrot4.unified_precheck import run_unified_precheck
        from modlearn_rot.generators.dataset import GeneratedDataset, save_dataset
        from s0_cr2_design_recovery import sample_prior, logits as cpt_logits
        for model_id in ("CR3", "CR5"):
            with self.subTest(model=model_id), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                n, trials = 4, 500
                x = sample_subject_x(n, seed=72)
                theta4 = sample_prior(n, 4, np.random.default_rng(33))
                theta = theta4 if model_id == "CR3" else theta4[:, 1:]
                extended = theta4 if model_id == "CR3" else theta_as_cr3(theta)
                prob = expit(cpt_logits(x, extended[:, None, :]))
                choices = (np.random.default_rng(2026).random(prob.shape)
                           < prob).astype(np.int64)
                data = GeneratedDataset(
                    parameters=theta,
                    trial_counts=np.full(n, trials, dtype=np.int64),
                    subject_index=np.repeat(np.arange(n), trials),
                    trial_index=np.tile(np.arange(trials), n),
                    x=x.reshape(-1, 6),
                    y=choices.reshape(-1, 1),
                    predictive_true=prob.reshape(-1, 1))
                input_path = root/"saved_dataset.npz"
                save_dataset(data, input_path)
                output_path = root/"s2_precheck__report.json"
                with patch("modrot4.unified_precheck.precheck_path", return_value=output_path):
                    report = run_unified_precheck(
                        model_id, data, seed=20261006,
                        provenance={"unit_test": True}, data_path=input_path)
                self.assertTrue(report["checks"]["generator_fidelity"]["passed"])
                self.assertEqual(report["checks"]["parameter_recovery"]["sample_size"],n)
                self.assertTrue(output_path.is_file())
                recovered = np.load(root/"s2_precheck__oracle_subjects.npz")
                self.assertEqual(recovered["theta_mle"].shape, theta.shape)
                self.assertTrue(np.isfinite(recovered["theta_mle"]).all())

    def test_cli_default_8000(self):
        from modrot4.cli import parse_args
        for model in ("C1","C2","C3","CR3","CR4","CR5"):
            args = parse_args("test",["--model",model])
            self.assertEqual(args.subjects,8000)
            self.assertEqual(args.data_seed,
                             20261028 if model.startswith("CR") else 20261006)


if __name__ == "__main__":
    unittest.main()
