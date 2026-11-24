# Metric Lab 05 — 通过 S9 角度检验的指标，在四个 choice NP 上盲搜

**选指标：** 在 S9（repeat01）神经网络一侧，若某库**所有**轴子集扫描的最优角度都在 {0,15,75,90}（离 oracle 不超过 15°），该指标在该库"成功过"；取在 C1/C2/C3/CR4 **至少一个库**成功过的指标，共 94 个（`selected_metrics.csv`；来自 `03_s9_metric_curves/results/angle_pref_long.csv`）。
**盲搜：** 每个指标在 C1、C2、C3、CR4 的 choice NP（repeat01）上，用自动微分 logit 能量（`i01_logit_AD`）、R\*-free 的 Haar 起点 + Nelder–Mead 搜索 R；C1/C2/CR4 用 400 起点取 8 个精修（同原网格），C3（d=6）用 2000/30。T 只用于评价。成功 = MCC 距监督上限 ≤0.05 且最大泄露 <0.2（C）/ <0.3（CR）。
**文件：** `results/blind_all.csv`（376 个单元：MCC、最差匹配、泄露均值/最大、与 R\* 的角度误差、是否成功）、`results/summary_by_metric.csv`、`results/figures/tu_<库>.png`（每个指标一个 |corr(T,U)| 热力图，绿框 = 成功）、`results/found_rotations.npz`。脚本 `run.py`（搜索）、`plot.py`（作图）。

| 库 | 成功 / 94 | MCC 中位 / 最大（上限） |
|---|---|---|
| C1 | 77 | 0.972 / 0.972（0.972） |
| C2 | 67 | 0.976 / 0.977（0.977） |
| C3 | 3（`Fz.A+B+K`、`UX.log1e-2`、`UX.log1e-3`） | 0.912 / 0.960（0.970） |
| CR4 | 3（`rowsp.a0.5.q0.1`、`Fz.A+B+C`、`Fz.A+C`） | 0.646 / 0.841（0.845） |

没有任何指标在四个库同时成功，C3 与 CR4 也没有交集。
