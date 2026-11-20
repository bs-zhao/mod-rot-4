"""Synthetic end-to-end S1/S3/S5/S8/S9 test; never writes real outputs."""
import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from he_lab import sources
from he_lab.pipeline import run
from he_lab.sources import sha256
from modrot4_nn.net.discrete import DiscreteNPConfig, DiscreteNeuralProcess


class PipelineFixture(unittest.TestCase):
    def test_all_stages_both_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            run_dir=root/"outputs"/"nn"/"C1"/"test_condition"/"repeat01"
            dataset=root/"outputs"/"models"/"generate"/"C1"/"s1_generate__dataset.npz"
            dataset.parent.mkdir(parents=True)
            rng=np.random.default_rng(12)
            n=8
            counts=np.full(n,5,dtype=np.int32)
            theta=rng.normal(0,.5,size=(n,4))
            x=rng.uniform(-1,1,size=(n*5,3))
            sid=np.repeat(np.arange(n),5)
            tid=np.tile(np.arange(5),n)
            eta=theta[sid,0]+np.sqrt(3)*np.sum(theta[sid,1:]*x,axis=1)
            prob=1/(1+np.exp(-eta))
            choice=(rng.random(n*5)<prob).astype(np.int8)
            np.savez_compressed(dataset,parameters=theta,trial_counts=counts,
                subject_index=sid,trial_index=tid,x=x,logit=eta,p=prob,choice=choice)
            ckpt=run_dir/"s3_train"/"checkpoint.pt"
            ckpt.parent.mkdir(parents=True)
            cfg=DiscreteNPConfig(x_dim=3,latent_dim=4)
            net=DiscreteNeuralProcess(cfg)
            torch.save(dict(model_config=cfg.to_dict(),model_state_dict=net.state_dict(),
                extra_state=dict(input_standardization=dict(
                    x_mean=[0.,0.,0.],x_std=[1.,1.,1.]))),ckpt)
            z=rng.normal(0,.4,size=(n,4))
            latent=run_dir/"s5_extract_z"/"latent.npz"
            latent.parent.mkdir(parents=True)
            np.savez_compressed(latent,z=z,dataset_sha256=np.array(sha256(dataset)),
                checkpoint_sha256=np.array(sha256(ckpt)),
                context_mode=np.array("all_observed_trials"),
                train_ids=np.arange(6),validation_ids=np.array([6]),
                test_ids=np.array([7]))
            givens=np.eye(4)
            angle=np.pi/12
            givens[:2,:2]=[[np.cos(angle),-np.sin(angle)],
                           [np.sin(angle),np.cos(angle)]]
            for variant,stage,oracle in (
                ("s9","s9_rotation_sweep","s8_supervised_rotation"),
                ("s9_2","s9_2_rotation_sweep","s8_2_supervised_rotation")):
                s8=run_dir/oracle
                s8.mkdir(parents=True)
                np.savez_compressed(s8/"rotation.npz",R_star=np.eye(4),
                    t_train_mean=np.zeros(4),t_train_std=np.ones(4))
                stage_dir=run_dir/stage
                (stage_dir/"rotations"/"baseline").mkdir(parents=True)
                (stage_dir/"rotations"/"k02").mkdir(parents=True)
                mean=z.mean(0) if variant=="s9_2" else np.zeros(4)
                std=z.std(0) if variant=="s9_2" else np.ones(4)
                rows=[]
                cases=[
                    ("baseline_unswept","", "",np.eye(4),"rotations/baseline/unswept.npz"),
                    ("sweep","1,2",15,givens,"rotations/k02/axes_01_02__deg_015.npz"),
                ]
                for candidate_type,axes,angle,r,path in cases:
                    np.savez_compressed(stage_dir/path,R=r,delta_R=r,
                        global_z_mean=mean,global_z_std=std)
                    rows.append(dict(candidate_type=candidate_type,axes_1based=axes,
                        angle_degrees=angle,rotation_file=path))
                with (stage_dir/"rotation_index.csv").open("w",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            args=SimpleNamespace(model="C1",run_dir=str(run_dir),stage="all",
                variant="both",measure="theory_x",n_eval=8,latent_draws=2,
                fit_rows=60,seed=10,chunk=4,boundary_policy="mask",
                methods="nn_autograd,wy_polynomial",degree=2,
                max_centers=50,neighbors=20,finite_step=.005,top_k=1,
                plot_metrics="UU.diagonal.hoyer",max_rotations=0,overwrite=False)
            with patch.object(sources,"ROOT",root):
                result=run(args)
            for variant in ("s9","s9_2"):
                self.assertTrue((result/"metrics"/variant/"all_metrics.csv").exists())
                self.assertTrue((result/"diagnostics"/variant/"correlations.csv").exists())
                self.assertTrue((result/"oracle"/variant/"theory_x"/"analytic_oracle"/
                                "true_logit"/"baseline_unswept.npz").exists())
                self.assertTrue((result/"inference"/variant/"theory_x"/"nn_autograd"/
                                "nn_probability"/"baseline_unswept.npz").exists())
                self.assertTrue((result/"inference"/variant/"theory_x"/"wy_polynomial"/
                                "true_choice"/"axes_01_02__deg_015.npz").exists())
            with np.load(result/"oracle"/"s9"/"theory_x"/"analytic_oracle"/
                         "true_logit"/"baseline_unswept.npz") as data:
                m=data["M"]
                self.assertEqual(m.shape,(7,7))
                self.assertTrue(np.all(np.isfinite(m)))
                self.assertTrue(np.all(m>=-1e-10))


if __name__=="__main__":
    unittest.main()
