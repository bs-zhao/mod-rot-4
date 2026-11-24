# Metric Lab 04 — 盲搜成功指标的 |corr(T,U)| 热力图

对二元选择版 NP（`outputs/nn/`，repeat01 与 repeat02，各 C1/C2/C3/CR4）上、自动微分 logit 推断（`i01_logit_AD`）下**所有盲搜成功的指标**，画盲搜解 U 与真实参数 T 的 |相关系数| 热力图（行 = 真参数 T，列 = 匈牙利匹配后的 U 轴，对角线 = 匹配对）。T 仅用于这一步评价。

**成功判据**（与盲搜日志相同）：MCC 距监督上限 ≤ 0.05，且最大泄露 < 0.2（C 系列）/ < 0.3（CR 系列）。成功单元取自 `metric_lab/02_blind_nn_rotation/results/big_grid/grid_results.csv`（repeat01）与 `grid_results_r2.csv`（repeat02）。当时没有保存找到的旋转，所以对每个成功单元用同一确定性流程（Haar 起点种子 4、400 起点取 8 个精修）**重新搜索**，重算的 MCC、最大泄露与已存值逐单元核对，**242/242 个单元完全一致**。

| 数据库 | 成功指标数 | MCC 范围（上限） | 最大泄露的最大值 |
|---|---|---|---|
| C1 repeat01 | 64 | 0.956–0.972（0.972） | 0.184 |
| C1 repeat02 | 67 | 0.956–0.972（0.972） | 0.179 |
| C2 repeat01 | 54 | 0.967–0.977（0.977） | 0.124 |
| C2 repeat02 | 49 | 0.965–0.976（0.976） | 0.192 |
| C3 repeat01 | 1（`UX.lq.25`） | 0.946（0.970） | 0.196 |
| C3 repeat02 | 0 | — | — |
| CR4 repeat01 | 3 | 0.815–0.841（0.845） | 0.264 |
| CR4 repeat02 | 4 | 0.840–0.845（0.848） | 0.206 |

图：`results/figures/tu_heatmaps_<库>.png`（C3 repeat02 无成功指标，没有图）。每个小图标题是指标名、MCC、泄露均值/最大。`results/successful_cells.csv`（逐单元数值，含与已存值核对）、`results/found_rotations.npz`（找到的旋转）。`run_heatmaps.py` 重搜并作图，`replot.py` 只重画；`work/` 是拷贝的盲搜代码（`lib2.py` 仍读取 `metric_lab/01` 与 `outputs/nn`，只读）。

只含 `i01_logit_AD` 一种推断；其余 15 种推断的成功单元未画。
