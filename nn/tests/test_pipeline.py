"""End-to-end six-model NN schema and S3-S7 miniature smoke test."""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"nn"/"src"))
from modrot4_nn.pipeline import fit_x_scaler,_discrete_data
from modrot4_nn.cli import parse_args
from modrot4_nn.dataset import ChoiceDataset


class TestNNPipeline(unittest.TestCase):
    def test_pre_nn_normalizer_uses_train_only_and_all_x_channels(self):
        rng=np.random.default_rng(14)
        n=7; trials=6; xdim=6
        x=rng.normal(size=(n*trials,xdim))
        x[4*trials:]*=1000
        values=ChoiceDataset(
            parameters=np.zeros((n,3)),
            trial_counts=np.full(n,trials),
            subject_index=np.repeat(np.arange(n),trials),
            trial_index=np.tile(np.arange(trials),n),
            x=x,logit=np.zeros(n*trials),p=np.full(n*trials,.5),
            choice=rng.integers(0,2,n*trials))
        args=parse_args("test",["--model","CR4"])
        raw=_discrete_data(args,values)
        normalizer=fit_x_scaler(raw,np.arange(4))
        expected=x[:4*trials]
        np.testing.assert_allclose(normalizer["x_mean"],expected.mean(0),atol=1e-6)
        np.testing.assert_allclose(normalizer["x_std"],expected.std(0),atol=1e-6)
        normalized=_discrete_data(args,values,preprocessing=normalizer)
        np.testing.assert_allclose(normalized.x[:4].numpy().mean((0,1)),
                                   np.zeros(xdim),atol=1e-5)
        np.testing.assert_allclose(normalized.x[:4].numpy().std((0,1)),
                                   np.ones(xdim),atol=1e-5)
        self.assertFalse(np.array_equal(normalized.x[4].numpy(),
                                        raw.x[4].numpy()))

    def test_s3_through_s7_all_outputs_from_s1(self):
        # Create a sandbox repo with *only NN code and S1 files*; no other
        # model-specific reference scripts or legacy NP data are used.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            shutil.copytree(ROOT/"nn",root/"nn",
                ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            n=36; trials=14; d=6
            rng=np.random.default_rng(113)
            x=rng.uniform(0,1,size=(n,trials,d)).astype(np.float32)
            theta=rng.normal(size=(n,3))
            eta=(theta[:,None,0]+theta[:,None,1]*(x[:,:,0]-.5)+
                 theta[:,None,2]*(x[:,:,2]-.5))
            p=expit(eta)
            choice=(rng.random(p.shape)<p).astype(np.int64)
            ds=root/"outputs"/"models"/"generate"/"CR4"/"s1_generate__dataset.npz"
            ds.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(ds,parameters=theta,
                trial_counts=np.full(n,trials),
                subject_index=np.repeat(np.arange(n),trials),
                trial_index=np.tile(np.arange(trials),n),
                x=x.reshape(-1,d),logit=eta.reshape(-1),
                p=p.reshape(-1),choice=choice.reshape(-1))
            h=hashlib.sha256(ds.read_bytes()).hexdigest()
            precheck=root/"outputs"/"models"/"precheck"/"CR4"/"s2_precheck__report.json"
            precheck.parent.mkdir(parents=True,exist_ok=True)
            precheck.write_text(json.dumps({"qualified":True,"dataset_sha256":h}))
            base=["--model","CR4","--latent-dim","3","--beta","0.0005",
                  "--batch-size","12","--max-epochs","2","--patience","2",
                  "--log-every","1","--inference-batch","12"]
            stages=[
                ("s3_train.py",[], "s3_train/checkpoint.pt"),
                ("s4_evaluate.py",[], "s4_evaluate/metrics.json"),
                ("s5_extract_z.py",[], "s5_extract_z/latent.npz"),
                ("s6_z_diagnostics.py",["--fa-factors","3","--pc-var-threshold","0"],
                 "s6_z_diagnostics/s12__pca.npz"),
                ("s7_regression.py",["--pca-threshold","0","--z-var-threshold","0"],
                 "s7_regression/s12__diagnostics.json"),
                ("s8_supervised_rotation.py",[],
                 "s8_supervised_rotation/rotation.npz"),
                ("s9_rotation_sweep.py",[],
                 "s9_rotation_sweep/rotation_index.csv")]
            for script,additional,expected in stages:
                proc=subprocess.run(
                    [sys.executable,str(root/"nn"/script),*base,*additional],
                    cwd=root,capture_output=True,text=True,timeout=180,
                    env=__import__("os").environ | {"OMP_NUM_THREADS":"1",
                                                       "MKL_NUM_THREADS":"1"})
                self.assertEqual(proc.returncode,0,
                    f"{script}\n{proc.stdout[-2500:]}\n{proc.stderr[-5000:]}")
                matches=list((root/"outputs"/"nn"/"CR4").glob("*/repeat01/"+expected))
                self.assertEqual(len(matches),1,f"Missing {expected}")
            zfile=list((root/"outputs"/"nn"/"CR4").glob("*/repeat01/s5_extract_z/latent.npz"))[0]
            with np.load(zfile) as saved:
                self.assertEqual(saved["z"].shape,(n,3))
                self.assertEqual(saved["z_logvar"].shape,(n,3))
                self.assertNotIn("theta_true",saved.files)
            s6=zfile.parent.parent/"s6_z_diagnostics"
            self.assertTrue((s6/"z_histograms.png").exists())
            self.assertTrue((s6/"z_variance_kl.png").exists())
            self.assertTrue((s6/"s12__ica.npz").exists())
            self.assertTrue((s6/"s12__factor_analysis.npz").exists())
            s8=zfile.parent.parent/"s8_supervised_rotation"
            self.assertTrue((s8/"rz_vs_t_correlation.png").exists())
            self.assertTrue((s8/"pred_vs_t_correlation.png").exists())
            self.assertTrue((s8/"rz_vs_t_scatter.png").exists())
            self.assertTrue((s8/"pred_vs_t_scatter.png").exists())
            with np.load(s8/"rotation.npz") as model:
                r=model["R_star"]
                np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-9)
            with np.load(s8/"test_predictions.npz") as predictions:
                self.assertEqual(predictions["theta_pred"].shape[1],3)
                self.assertEqual(predictions["u_test"].shape[1],3)
            s9=zfile.parent.parent/"s9_rotation_sweep"
            self.assertTrue((s9/"summary.json").exists())
            self.assertEqual(len(list((s9/"rotations").rglob("*.npz"))),28)
            self.assertEqual(len(list((s9/"figures_45deg").rglob("*heatmap.png"))),4)
            self.assertEqual(len(list((s9/"figures_45deg").rglob("*scatter.png"))),4)
            with np.load(next((s9/"rotations").rglob("*deg_000.npz"))) as example:
                np.testing.assert_allclose(example["R"],r,atol=1e-9)
            with np.load(next((s9/"rotations").rglob("*deg_045.npz"))) as example:
                np.testing.assert_allclose(example["R"] @ example["R"].T,
                                           np.eye(3),atol=1e-9)


if __name__=="__main__":
    unittest.main()
