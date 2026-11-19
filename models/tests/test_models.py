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
        from modrot4.model_specs import get_spec
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
                        get_spec(model_id), data, seed=20261006,
                        provenance={"unit_test": True}, data_path=input_path)
                self.assertTrue(report["checks"]["generator_fidelity"]["passed"])
                self.assertEqual(report["checks"]["parameter_recovery"]["sample_size"],n)
                self.assertTrue(output_path.is_file())
                recovered = np.load(root/"s2_precheck__oracle_subjects.npz")
                self.assertEqual(recovered["theta_mle"].shape, theta.shape)
                self.assertTrue(np.isfinite(recovered["theta_mle"]).all())

    def test_shared_x_sampler_keeps_original_classic_random_sequences(self):
        # C1/C2/C3 must draw exactly the same numbers as their original
        # single-rng uniform(-1,+1) implementation.
        from modlearn_rot.utils.seeds import make_rng_streams
        for model_id in ("C1","C2","C3"):
            with self.subTest(model=model_id):
                model=get_model(model_id)
                seed=20261006
                a=np.random.default_rng(seed)
                b=np.random.default_rng(seed)
                np.testing.assert_array_equal(
                    model.sample_inputs(57,a),
                    b.uniform(-1.0,1.0,size=(57,model.n_input_features)))
                model.n_subjects=4
                ds=generate_dataset(model,seed)
                inp=make_rng_streams(seed,(
                    "parameters","trial_counts","inputs","observations"))["inputs"]
                expected_x=np.concatenate([
                    inp.uniform(-1.0,1.0,size=(int(t),model.n_input_features))
                    for t in ds.trial_counts
                ])
                np.testing.assert_array_equal(ds.x,expected_x)

    def test_cr4_x_is_exactly_classic_trialwise_single_rng_scenario(self):
        from types import SimpleNamespace
        from scipy.special import expit
        from modrot4.model_specs import (get_spec,CR4_X_LOW,CR4_X_HIGH,
                                         CR4_X_POLICY)
        from modlearn_rot.utils.seeds import make_rng_streams
        from s0_s1_cr4 import logits as three_logits
        args=SimpleNamespace(subjects=4,data_seed=20261028,design_seed=20261023)
        generated=get_spec("CR4").generate(args)
        input_rng=make_rng_streams(
            args.data_seed,("parameters","trial_counts","inputs","observations"))["inputs"]
        # Direct independent NumPy oracle: one *single* RNG, one
        # rng.uniform(low,high,size=(500,6)) call PER SUBJECT.
        expected_x=np.concatenate([
            input_rng.uniform(CR4_X_LOW,CR4_X_HIGH,size=(500,6))
            for _ in range(args.subjects)
        ])
        np.testing.assert_array_equal(generated.x,expected_x)
        np.testing.assert_array_equal(generated.trial_counts,
                                      np.full(4,500,dtype=np.int64))
        self.assertEqual(generated.y.ndim,1)
        self.assertEqual(generated.predictive_true.ndim,1)
        self.assertFalse(np.array_equal(generated.x[:500],generated.x[500:1000]))
        self.assertFalse(np.array_equal(generated.x[0],generated.x[1]))
        np.testing.assert_array_equal(
            generated.parameters,sample_theta(4,np.random.default_rng(20261028+4)))
        theta=generated.parameters
        probability=expit(three_logits(expected_x.reshape(4,500,6),theta[:,None,:]))
        expected_choices=(np.random.default_rng(20261028+1004).random(probability.shape)
                          < probability).astype(np.int64)
        np.testing.assert_array_equal(generated.predictive_true,probability.reshape(-1))
        np.testing.assert_array_equal(generated.y,expected_choices.reshape(-1))
        self.assertEqual(CR4_X_POLICY,"iid_uniform_single_inputs_seedsequence_v2")

    def test_cr4_legacy_dataset_cannot_be_silently_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            sandbox=Path(tmp)
            shutil.copytree(ROOT/"models",sandbox/"models",
                            ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            s1=sandbox/"models"/"s1_generate.py"
            s2=sandbox/"models"/"s2_precheck.py"
            args=[sys.executable,str(s1),"--model","CR4","--subjects","5"]
            created=subprocess.run(args,cwd=sandbox,capture_output=True,text=True)
            self.assertEqual(created.returncode,0,created.stderr)
            location=sandbox/"outputs"/"models"/"generate"/"CR4"
            dataset_file=location/"s1_generate__dataset.npz"
            manifest_file=location/"s1_generate__manifest.json"
            with np.load(dataset_file) as d:
                original_x=d["x"].copy()
            manifest=json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["x_sampling_policy"],
                             "iid_uniform_single_inputs_seedsequence_v2")
            manifest["x_sampling_policy"]="iid_subject_trial_uniform_v1"
            manifest_file.write_text(json.dumps(manifest),encoding="utf-8")
            refused=subprocess.run(args,cwd=sandbox,capture_output=True,text=True)
            self.assertNotEqual(refused.returncode,0)
            self.assertIn("OLD two-stream X generation",refused.stderr)
            precheck=subprocess.run([sys.executable,str(s2),"--model","CR4"],
                                    cwd=sandbox,capture_output=True,text=True)
            self.assertNotEqual(precheck.returncode,0)
            self.assertIn("Old CR4 X sampling stream",precheck.stderr)
            with np.load(dataset_file) as d:
                np.testing.assert_array_equal(d["x"],original_x)

    def test_model_spec_generation_retains_cr3_cr5_sampling(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from scipy.special import expit
        from modrot4.model_specs import get_spec, ModelSpec
        from s0_cr2_design_recovery import sample_prior, logits as cr3_logits
        args=SimpleNamespace(subjects=3,data_seed=20261028,design_seed=20261023)
        rng=np.random.default_rng(40)
        bank=np.column_stack((
            rng.uniform(5,200,500),rng.uniform(5,200,500),
            rng.uniform(.05,.95,500),rng.uniform(5,200,500),
            rng.uniform(5,200,500),rng.uniform(.05,.95,500)))
        with tempfile.TemporaryDirectory() as tmp:
            bank_file=Path(tmp)/"synthetic_reference.npz"
            np.savez_compressed(bank_file,X=bank)
            for model_name in ("CR3","CR5"):
                with self.subTest(model=model_name):
                    patch_target=("_cr3_bank" if model_name=="CR3" else "_cr5_bank")
                    with (patch.object(ModelSpec,"_prepare_stimuli",return_value=None),
                          patch("modrot4.model_specs."+patch_target,return_value=bank_file)):
                        dataset=get_spec(model_name).generate(args)
                    original_theta=sample_prior(3,4,np.random.default_rng(20261028+4))
                    expected_t=(original_theta if model_name=="CR3"
                                else original_theta[:,1:])
                    np.testing.assert_array_equal(dataset.parameters,expected_t)
                    np.testing.assert_array_equal(dataset.x,
                                                  np.tile(bank,(3,1)))
                    full_t=original_theta if model_name=="CR3" else theta_as_cr3(expected_t)
                    expected_p=expit(cr3_logits(bank[None],full_t[:,None,:]))
                    np.testing.assert_array_equal(dataset.predictive_true,
                                                  expected_p.reshape(-1,1))
                    expected_y=(np.random.default_rng(20261028+1004).random(expected_p.shape)<expected_p)
                    np.testing.assert_array_equal(dataset.y,
                                                  expected_y.astype(np.int64).reshape(-1,1))

    def test_cr5_generates_and_prechecks_without_cr3_or_cr2(self):
        """End-to-end CR5 from its OWN local bank, with zero other-model outputs.

        An inexpensive input source is substituted ONLY for CI; the real
        full-parameter CR2 optimizer remains the production design algorithm.
        """
        from unittest.mock import patch
        from modrot4 import fixed_bank
        from s0_cr2_design_recovery import make_candidates
        with tempfile.TemporaryDirectory() as tmp:
            sandbox=Path(tmp)
            shutil.copytree(ROOT/"models",sandbox/"models",
                            ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            out=sandbox/"outputs"/"models"/"generate"
            cr5=out/"CR5"/"s0_design"
            bank5=cr5/"CR5_3P_cr3bank500.npz"
            info5=cr5/"CR5_3P_design.json"
            bank3=out/"CR3"/"s0_design"/"CR3_4P_fixed500.npz"
            info3=out/"CR3"/"s0_design"/"CR3_4P_design.json"
            source,_=make_candidates(720,np.random.default_rng(2026))
            meta=dict(objective=0.1,swaps_accepted=1,family_counts={})
            with (patch("modrot4.fixed_bank.bank_path",
                        side_effect=lambda m: bank5 if m=="CR5" else bank3),
                  patch("modrot4.fixed_bank.design_path",
                        side_effect=lambda m: info5 if m=="CR5" else info3),
                  patch("modrot4.fixed_bank._optimise_4p_source",
                        return_value=(source,meta))):
                fixed_bank.ensure_local_bank("CR5",20261023)
                self.assertTrue(bank5.is_file())
                self.assertFalse(bank3.exists())
                self.assertFalse((out/"CR2").exists())
                fixed_bank.inspect_local_bank("CR5")
                fixed_bank.ensure_local_bank("CR3",20261023)
                fixed_bank.inspect_local_bank("CR3")
            with np.load(bank5) as five, np.load(bank3) as three:
                np.testing.assert_array_equal(five["X"],three["X"])
                np.testing.assert_array_equal(five["type_codes"],three["type_codes"])
            # CR3/CR2 output folders may not exist when CR5 S1/S2 run.
            shutil.rmtree(out/"CR3")
            self.assertFalse((out/"CR3").exists())
            self.assertFalse((out/"CR2").exists())
            generate=subprocess.run(
                [sys.executable,str(sandbox/"models"/"s1_generate.py"),
                 "--model","CR5","--subjects","5"],
                cwd=sandbox,capture_output=True,text=True)
            self.assertEqual(generate.returncode,0,generate.stderr)
            self.assertTrue((out/"CR5"/"s1_generate__dataset.npz").is_file())
            self.assertFalse((out/"CR3").exists())
            self.assertFalse((out/"CR2").exists())
            check=subprocess.run(
                [sys.executable,str(sandbox/"models"/"s2_precheck.py"),
                 "--model","CR5"],
                cwd=sandbox,capture_output=True,text=True)
            # n=5 fails the universal minimum n=30 recovery gate,
            # not because any CR2/CR3 output is missing.
            self.assertNotEqual(check.returncode,0)
            self.assertIn("Precheck failed",check.stderr)
            self.assertNotIn("CR3",check.stderr)
            report=sandbox/"outputs"/"models"/"precheck"/"CR5"/"s2_precheck__report.json"
            self.assertTrue(report.is_file(),check.stderr)
            diagnostics=json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(diagnostics["checks"]["stimulus_design"]["passed"])
            self.assertTrue(diagnostics["checks"]["generator_fidelity"]["passed"])
            self.assertFalse(diagnostics["checks"]["parameter_recovery"]["conclusive"])

    def test_generate_existing_wrong_subject_count_rejected(self):
        # Preserving old files is mandatory, but must not silently reuse N=5 for N=8.
        with tempfile.TemporaryDirectory() as tmp:
            sandbox=Path(tmp)
            shutil.copytree(ROOT/"models",sandbox/"models",
                            ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            script=sandbox/"models"/"s1_generate.py"
            first=subprocess.run([sys.executable,str(script),"--model","C1","--subjects","5"],
                                 cwd=sandbox,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            second=subprocess.run([sys.executable,str(script),"--model","C1","--subjects","8"],
                                  cwd=sandbox,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)
            self.assertIn("existing_subjects=5",second.stderr)
            self.assertIn("requested_subjects=8",second.stderr)
            with np.load(sandbox/"outputs"/"models"/"generate"/"C1"/"s1_generate__dataset.npz") as data:
                self.assertEqual(data["parameters"].shape[0],5)

    def test_cli_default_8000(self):
        from modrot4.cli import parse_args
        for model in ("C1","C2","C3","CR3","CR4","CR5"):
            args = parse_args("test",["--model",model])
            self.assertEqual(args.subjects,8000)
            self.assertEqual(args.data_seed,
                             20261028 if model.startswith("CR") else 20261006)


if __name__ == "__main__":
    unittest.main()
