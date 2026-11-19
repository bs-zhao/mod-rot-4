# 平均平方 Hessian 曲率能量：数学定义与仅从输入–输出数据估计的算法综述

**研究对象：** 逐元素平均平方 Hessian（elementwise expected squared Hessian / mean-squared second-derivative energy）

**核心指标：**

$$
\boxed{M_{ij}(f;P_W)=\mathbb E_{W\sim P_W}\!\left[
\left(\frac{\partial^2 f(W)}{\partial W_i\partial W_j}\right)^2
\right]}
$$

**文档定位：** 研究备忘录 / 文献综述 / 后续盲矩阵实验的设计依据；**不是已实现的 S1–S9 模块**。这里的 *infer* 指只能使用给定输入矩阵与输出向量拟合或直接估计此泛函，而不是读取真实解析函数或 oracle Hessian。

**项目关联：** [项目总览](../README.md)；[模型定义与 S1–S2](../models/README.md)；[NP、S5 的 Z 和 S8/S9 旋转](../nn/README.md)。

---

# Part I. 指标的数学定义、含义与性质

## 1. 统计对象（estimand）必须先说清楚

给定未知但足够光滑的函数：

$$
f:\Omega\subseteq\mathbb R^d\to\mathbb R,\qquad
W=(W_1,\ldots,W_d)^\top\sim P_W,
$$

Hessian 是 **随输入变化的矩阵函数**：

$$
H_f(w)=\nabla_w^2 f(w),\qquad
[H_f(w)]_{ij}=\partial_{ij}f(w)
=\frac{\partial^2 f(w)}{\partial w_i\partial w_j}.
$$

我们要估计的不是某个点的 Hessian，而是该函数相对于指定输入分布的逐元素二阶能量：

$$
\boxed{
M_{ij}(f;P_W)=\int_\Omega [\partial_{ij}f(w)]^2\,dP_W(w).
}
\tag{1}
$$

相应的 RMS 尺度为：

$$
\boxed{D_{ij}(f;P_W)=\sqrt{M_{ij}(f;P_W)}.}
\tag{2}
$$

对称性：$H_{ij}(w)=H_{ji}(w)$（二阶连续可微时），故 $M_{ij}=M_{ji}$；元素非负。但 **$M$ 不一定是半正定矩阵**：它是逐元素平方期望 $\mathbb E[H\odot H]$，不是 $\mathbb E[H^\top H]$。

还必须区分以下对象：

| 对象 | 公式 | 含义与局限 |
|---|---|---|
| 平均带符号二阶导数 | $\mathbb E[H_{ij}]$ | 正负抵消；不等于曲率能量 |
| 平均绝对二阶导数 | $\mathbb E[|H_{ij}|]$ | 不抵消，较稳健 |
| **平均平方曲率能量** | **$\mathbb E[H_{ij}^2]$** | 本文研究主体 |
| RMS 曲率 | $\sqrt{\mathbb E[H_{ij}^2]}$ | 与原二阶导数量纲一致 |
| 二阶导数方差 | $\mathrm{Var}(H_{ij})$ | $\mathbb E[H_{ij}^2]-\mathbb E[H_{ij}]^2$，单独衡量位置间变化 |
| 全矩阵二阶能量 | $\mathbb E[\|H\|_F^2]=\sum_{ij}M_{ij}$ | 衡量总曲率；对于完整的正交坐标旋转不变 |

**强度不等于简单性。** $M_{ij}$ 大说明对应二阶作用在所选 $P_W$ 下能量大，不自动表示函数“不简单”。简单性是对 *这些能量的分布模式、稀疏性、分组结构、零函数和坐标系* 的另一个研究问题。

## 2. 与现有数学统计领域的准确关系

1. **非参数二次泛函估计（nonparametric quadratic-functional estimation）：** 直接研究未知回归函数的导数平方积分。例如 Efromovich & Samarov (2000) [R1]。**注意** 该文的具体光滑性、积分测度、维数及噪声条件不能未经证明地推广到任意 9–10 维、相关输入与二元选择。
2. **Sobolev 半范数 / thin-plate spline bending energy：** $\int \|H_f(w)\|_F^2\,dw$ 是二阶导数粗糙度的一种形式；本文的 $M_{ij}$ 则在特定 $P_W$ 下保留每个元素的贡献 [R3]。
3. **交叉导数全局敏感性指标（crossed-DGSM）：** 对 $i\ne j$，$\mathbb E[(\partial_{ij}f)^2]$ 可以衡量变量对的混合效应 [R2]。在合适的光滑性与全维支撑假设下，$M_{ij}=0$ 等价于 $\partial_{ij}f=0$ 在所选分布下几乎处处；这并不意味着它们在所有未观测区域均为零。
4. **非参数导数估计（nonparametric derivative estimation）：** 先由观测点估计 $\partial_{ij}f$，然后计算其平方积分。局部多项式、样条、KRR、GP、正交多项式和其他代理模型属于这类实现 [R4–R11]。

## 3. 输入–输出矩阵的统计定义

**一般形式：**

$$
\mathcal D=\{(w_n,y_n)\}_{n=1}^N,\quad
m(w)=\mathbb E[Y\mid W=w].
$$

必须明确输出是哪一种：

- **连续回归 / 已知概率 / logit：** 若 $y_n=f(w_n)+\varepsilon_n$ 且 $\mathbb E[\varepsilon_n\mid W_n]=0$，目标函数是 $f=m$。
- **二元 choice：** $Y\mid W=w\sim\mathrm{Bernoulli}(q(w))$。从 choices 能识别的是条件选择概率 $q(w)$；如果研究 **logit Hessian**，应估计 $f(w)=\operatorname{logit}q(w)$，而不是把每条 0/1 choice 当作可微连续函数。
- **概率输出：** 若 $f=\operatorname{logit}q$，令 $g=\nabla f$，有

$$
H_q=q(1-q)\left[H_f+(1-2q)gg^\top\right].
\tag{3}
$$

所以 $M^{\mathrm{logit}}$ 与 $M^{\mathrm{prob}}$ 是**不同的目标**，不能混用。直接对概率输出拟合的 Hessian 也不等于对 logit 拟合的 Hessian。若 $q$ 接近 0 或 1，取 logit 还会放大误差。

**本项目场景：** 将冻结的个体潜变量与试次特征组成 $W_{s,t}=[Z_s,X_{s,t}]$，只对指定的 $Y_{s,t}$ 版本做推断。$Z_s$ 对同一被试的所有试次固定，不能把 $N_{\mathrm{subj}}\times N_{\mathrm{trials}}$ 当作 $Z$ 方向上同样多的独立设计点。盲矩阵方法不能使用真实 $T$、解析 logits、oracle Hessian 来训练或选择方法；这些只能留在单独的事后 benchmark。

## 4. 必须明确期望分布

$$
M_{ij}(f;P_W)=\mathbb E_{W\sim P_W}[(\partial_{ij}f(W))^2].
$$

$P_W$ 可以选为：

1. **观测经验分布：** $P_W^{\mathrm{emp}}=\frac1N\sum_n\delta_{w_n}$；估计样本覆盖到的函数曲率。此时仍需估计连续空间里的导数。
2. **已知生成分布：** 对真实参数与模拟刺激的理论分布积分；只在 oracle benchmark 中使用。
3. **参考测度：** 预定义标准化坐标中的高斯分布或均匀网格；需要确认其支撑覆盖受观测数据约束的区域。
4. **分组 / 条件分布：** $M_{ij}^{(s)}=\mathbb E[H_{ij}^2\mid S=s]$；适于不同刺激类别对参数作用的互补约束。

**CR4 vs CR5：** 它们的三参数 CPT 函数相同，但 CR4 的 X 每人每试次重新抽样、CR5 的 X 使用全被试共享的固定题库。即使解析 Hessian 函数相同，选用两种 $P_X$ 得到的 $M$ 仍可能不同。

不同被试试次数不等时，须指定“按观察行平均”还是“先被试等权、再被试内试次等权”；前者隐含以试次数为权重。

## 5. Hessian 各区块与简单性

若 $W=(U,X)$，$U\in\mathbb R^k$ 是待旋转的潜在个体坐标，$X\in\mathbb R^p$ 是固定物理刺激：

$$
H_f(W)=
\begin{pmatrix}
H_{UU}(W)&H_{UX}(W)\\
H_{XU}(W)&H_{XX}(W)
\end{pmatrix}.
$$

可分别研究：

$$
\begin{aligned}
M^{UU}_{ij}&=\mathbb E[(\partial_{U_iU_j}f)^2],\\
M^{UX}_{ia}&=\mathbb E[(\partial_{U_iX_a}f)^2],\\
M^{XX}_{ab}&=\mathbb E[(\partial_{X_aX_b}f)^2].
\end{aligned}
\tag{4}
$$

其中 **UU 对角项**描述参数轴自身曲率；**UU 非对角项**描述参数间的混合二阶作用；**UX** 描述个体参数与刺激间的混合作用；**XX** 描述刺激方向上的曲率。

这些是二阶导数能量，不是 Sobol 方差贡献，也不自动等于具有唯一因果解释的“交互强度”。同一 $H_{ii}$ 可以依赖其它变量；仅测量 $H_{ii}$ 并不能保证函数可分。

## 6. 旋转变化和不变性：哪些量能当 loss？

采用列向量记号：

$$
U=RZ,\qquad R^\top R=I,\qquad
f_R(u,x)=f(R^\top u,x).
$$

则在对应的物理输入点上：

$$
H_{UU}^{(R)}=RH_{ZZ}R^\top,\qquad
H_{UX}^{(R)}=RH_{ZX},\qquad
H_{XX}^{(R)}=H_{XX}.
\tag{5}
$$

积分时，$U$ 的参考分布必须是原 $Z$ 分布经 $R$ 推前后的分布（或使用相同物理样本及其旋转坐标）。否则不同 $R$ 可能在比较不同的输入区域。

对每个点，以下式子成立：

$$
\|RH_{ZZ}R^\top\|_F^2=\|H_{ZZ}\|_F^2,\qquad
\|RH_{ZX}\|_F^2=\|H_{ZX}\|_F^2.
\tag{6}
$$

因此 **UU 或 UX 全区块平方和对正交旋转不变**；单独最小化它们无法找出旋转。应考察逐元素能量的 **稀疏性、集中度、零模式和分组模式**，例如对非负能量向量 $v(R)=\{M_{ij}(R)\}$ 计算 tail-$L_2$、top-$k$ 质量、熵或凹惩罚 $\sum\log(\varepsilon+M_{ij})$。它们是从 $M$ 派生的**候选损失**，不是本指标的定义。

**重要反例：** $f(a,b)=ab$ 的 Hessian 为 $\begin{pmatrix}0&1\\1&0\end{pmatrix}$；旋转 45° 后 $f=(u_1^2-u_2^2)/2$、Hessian 变成对角矩阵。仅最小化 UU 非对角能量会偏好旋转后坐标，但原函数的双线性结构未必更复杂。因此要联合考察对角自身曲率、UX 作用模式等。

**尺度问题：** 若 $W=\mu+D W_{\mathrm{std}}$、$D=\operatorname{diag}(\sigma_1,\ldots,\sigma_d)$，则

$$
H_{\mathrm{std}}(w_{\mathrm{std}})=D H_{\mathrm{physical}}(w)D,
\quad
M^{\mathrm{std}}_{ij}=\sigma_i^2\sigma_j^2M^{\mathrm{physical}}_{ij}.
\tag{7}
$$

标准化坐标里的正交旋转与原始坐标里的正交旋转并不等价。记录均值、标准差、旋转顺序和全部逆变换。

---

# Part II. 仅凭输入–输出矩阵推断 $M_{ij}$ 的方法综述

## 7. 问题分类与通用估计模板

设唯一可用信息是 $W\in\mathbb R^{N\times d}$、$Y\in\mathbb R^N$ 或 $Y\in\{0,1\}^N$、被试/组 ID（如果存在）。没有真函数、oracle 参数或可额外调用的生成器。

最一般的 **plug-in 路线** 是：

$$
\mathcal D_{\mathrm{fit}}\longrightarrow\widehat f
\longrightarrow
\widehat H_{ij}(w)
\longrightarrow
\widehat M_{ij}^{\mathrm{plug}}
=\frac{\sum_{n\in\mathcal E}\omega_n[\partial_{ij}\widehat f(w_n)]^2}
{\sum_{n\in\mathcal E}\omega_n},
\tag{8}
$$

其中 $\mathcal E$ 是评估点集合，最好与训练样本分开，权重 $\omega_n$ 实现指定 $P_W$。严格说数据中只有带噪声 Y 时，$\widehat f$ 的二阶导数一般存在系统误差。

如果 $\widehat H_{ij}=H_{ij}+b_{ij}+\xi_{ij}$，那么

$$
\mathbb E[\widehat H_{ij}^2]
=(H_{ij}+b_{ij})^2+\mathrm{Var}(\xi_{ij}),
\tag{9}
$$

因此 **“先估计导数再平方”通常有平方偏差和方差膨胀**。良好的 Y 预测不保证 $\widehat M$ 准确；尤其高频、局部和边界曲率可能被平滑惩罚抹掉。

### 7.1 方法路线总表

| ID | 算法家族 | 只给固定 W,Y 可用？ | 导数从何而来 | 主要优点 | 主要风险 |
|---|---|---|---|---|---|
| A1 | 局部二次 / 更高阶多项式回归 | 是 | 局部拟合的二次项 | 不假设全局解析式；局部直观 | 高维邻域稀疏、带宽敏感 |
| A2 | 局部似然 / 局部 logistic 回归 | 是（choice） | 局部 logit 的二次项 | 直接针对 Bernoulli 的 logit Hessian | 分离、平滑和边界问题 |
| A3 | Smoothing spline / tensor-product GAM / SS-ANOVA | 是 | 可微基函数解析导数 | 可正则化、可稀疏分解 | 基函数数目、交互截断偏差 |
| A4 | 正交级数 / 多项式 sieve / PCE | 是 | 多项式解析导数 | 易生成矩阵与二次型 | 高阶多项式爆炸、分布错配 |
| A5 | Kernel Ridge Regression (KRR) | 是 | 核的混合二阶导数 | 多维导数已有理论 | 核与正则化参数敏感 |
| A6 | Gaussian Process regression / classification | 是 | 核导数的后验分布 | 有不确定性量化 | 高计算开销、先验敏感 |
| A7 | Smooth MLP + 自动微分 | 是 | $\nabla^2\widehat f_\theta$ | 适合大量样本、高维非线性 | 函数拟合好不等于曲率拟合好 |
| A8 | 直接二次泛函估计 / cross-fit sieve | 是，需额外假设 | 用基函数的导数 Gram 矩阵直接估 $M$ | 不必逐点评价全部 Hessian | 光滑度、截断和分布估计难 |
| A9 | 弱形式 / RKHS 导数估计 | 是，需权重/边界假设 | 线性泛函、核表示或积分分部 | 能避免部分不稳定逐点导数 | 边界项、输入密度估计 |
| A10 | 差分 / 局部配对估计 | 仅密集、成对或近网格数据才可用 | 有限差分 | 简单、易检验 | 普通稀疏随机设计不能直接用 |
| A11 | 平滑版本的 MARS / 交互样条 | 是 | 分段多项式导数 | 自适应低阶交互 | 线性 hinge 的二阶导数几乎处处为 0 |
| A12 | 多点模型查询的随机扰动导数估计 | **否，仅固定数据不够** | 新调用 $f(w+\delta)$ 等 | 有专门误差理论 | 必须另有可查询函数，不是盲矩阵估计 |

表中“是”只表示无需已知真函数即可**定义并尝试**估计，不表示所有 $P_W$、样本量、维数和噪声分布下均具有可靠一致性。多数文献首先研究函数或逐点导数，并不直接证明 $\widehat M_{ij}$ 的收敛率。

## 8. 各种算法：计算原理、公式和适用条件

### A1. 局部二次多项式回归（Local polynomial Hessian）

在 $w_0$ 附近解局部加权最小二乘：

$$
(\hat a,\hat b,\hat B)
=\arg\min_{a,b,B=B^\top}
\sum_n K_h(w_n-w_0)
\left[y_n-a-b^\top(w_n-w_0)
-\tfrac12(w_n-w_0)^\top B(w_n-w_0)\right]^2.
\tag{10}
$$

使用 $\hat H(w_0)=\hat B$，再用式 (8) 汇总。对于 $d=9$，**仅完整二次多项式就有 $1+9+45=55$ 个局部系数**；三阶局部多项式需要更多数据。估计导数通常需要单独选择带宽，不可以仅凭预测 MSE 选平滑强度。[R4,R5]

- 优点：没有全局 CPT 或神经网络结构假设，可以看局部结构。
- 风险：各方向尺度、边界校正、带宽、局部设计条件数、重复主体 Z 与自相关；高维下常需稀疏交互或局部降维。
- 重要区别：De Brabanter et al. [R5] 研究非参数导数估计，不等于直接保证本项目全维 Hessian 能量的精度。

### A2. 局部 likelihood / 局部 logistic 模型

仅有 choice $y_n\in\{0,1\}$ 时，用局部二次 **logit** $\eta_n(a,b,B)$，优化：

$$
\arg\min_{a,b,B}\sum_n K_h(w_n-w_0)
\left[\log(1+e^{\eta_n})-y_n\eta_n\right]
+\lambda_{\mathrm{reg}}\mathcal P(B).
\tag{11}
$$

这里 $\hat B$ 直接估计 $\nabla^2\operatorname{logit}q(w_0)$，而非 $H_q$。局部 Bernoulli MLE 在稀疏区域可能接近完全分离；须有适当正则化、有效样本量限制和选择概率检查。此方法属于局部多项式和局部似然文献的合理延伸，**不是在此声称存在已验证的项目专用 estimator**。[R4,R5]

### A3. Smoothing splines / tensor-product GAM / SS-ANOVA

采用平滑基函数：

$$
\widehat f(w)=\sum_{k=1}^K\hat\beta_k\phi_k(w),
\qquad
\widehat H_{ij}(w)=\sum_k\hat\beta_k\,\partial_{ij}\phi_k(w).
\tag{12}
$$

对连续响应可以用平方损失；对 choice 可对 logistic 线性预测器拟合 penalized likelihood。二阶导数惩罚与 Sobolev 粗糙度有直接联系；tensor-product smooths 允许在非同量纲输入上建立主效应和交互项。[R3,R7,R8]

- 适用：较低内禀维度、预计交互作用稀疏、需要可解释项与可调平滑度。
- 风险：没有包含的高阶交互项永远不可能被发现；平滑度通常按预测或似然选择，不保证最优二阶导数；需要检查张量基数和边界。
- 额外思路：比较对各轴施加不同 smoothing penalty 的结果，但不要把惩罚强弱误读为真实曲率能量。

### A4. 正交多项式 / sieve / Polynomial Chaos Expansion (PCE)

把函数表示为：

$$
\widehat f(w)=\sum_{k=1}^K\hat\beta_k\phi_k(w),
\quad
G_{ij}[k,l]=
\mathbb E_{P_W}[
\partial_{ij}\phi_k(W)\partial_{ij}\phi_l(W)].
\tag{13}
$$

于是：

$$
\boxed{\widehat M_{ij}^{\mathrm{sieve}}=\hat\beta^\top\hat G_{ij}\hat\beta.}
\tag{14}
$$

$G_{ij}$ 可从已知参考分布解析积分，或由未参与拟合的 $W$ 评估点近似。PCE 尤其适合独立、已知边缘分布且比较光滑的输入，正交多项式的导数可解析求出。[R10]

- 对 $d=9$，以总次数不超过 4 的全多项式基为例：$\binom{9+4}{4}=715$ 项；正则化、稀疏筛选或交互阶数限制很重要。
- 适用：低/中度多项式复杂度、变量间平滑性较高、希望对 $M$ 使用精确矩阵代数。
- 风险：CR3 的复杂非多项式结构可能需要很高阶；固定小阶数的预测能力好并不保证导数能量被恢复。

### A5. Kernel Ridge Regression (KRR)

对足够光滑的核 $k$ 拟合：

$$
\widehat f(w)=\sum_{n=1}^{N}\alpha_n k(w,w_n),
\quad
\widehat H_{ij}(w)=
\sum_n\alpha_n\partial_{ij}^{(w)}k(w,w_n).
\tag{15}
$$

令 $Q_{ij}[n,m]=\mathbb E_{P_W}[
\partial_{ij}k(W,w_n)\partial_{ij}k(W,w_m)]$，
则 $\hat M_{ij}=\alpha^\top Q_{ij}\alpha$。

Liu & Li (2023) [R6] 直接研究多维支持上的任意混合偏导数 plug-in KRR 估计，属于本项目最直接、最应优先实验的方案之一。但其逐点导数误差界不自动给出二次泛函的最优误差界。

- 优点：数学结构清晰、导数可解析、相对神经网络更便于控制 smoothness。
- 风险：RBF 核带宽、核矩阵规模、数值条件数、boundary bias；选 kernel/hyperparameter 时需要 derivative-aware diagnostics。

### A6. Gaussian Process (GP)

对足够光滑的核和合适的回归/分类似然，GP 可推断每个点的导数后验：

$$
H_{ij}(w)\mid\mathcal D,\mathcal H
\quad\leadsto\quad
\mu_{ij}(w),\;v_{ij}(w).
$$

若这个条件边缘分布具有有限二阶矩，则：

$$
\mathbb E[H_{ij}(w)^2\mid\mathcal D,\mathcal H]
=\mu_{ij}(w)^2+v_{ij}(w).
\tag{16}
$$

积分后可得到 **posterior expected curvature energy**；也可单独报告 plug-in $\int \mu_{ij}^2\,dP_W$。两者是不同统计量；式 (16) 包含 posterior uncertainty，并**不是自动去偏的真实频率学能量估计量**。[R9]

- 连续高斯回归通常较直接；二元 choice 的 GP classification 后验一般非高斯，需 Laplace/EP/变分或采样等近似。
- 协方差核需要足够的可微性；估计 Hessian 的方差比估计函数值要求更高。
- 优点：诊断估计不确定性；缺点：大样本核计算和模型先验成本较高。

### A7. Smooth neural network + automatic differentiation

仅用训练集拟合 $\widehat f_\theta(W)$；对其做精确自动微分：

$$
\widehat H_{ij}(w)
=\partial_{ij}\widehat f_\theta(w),\qquad
\widehat M_{ij}=\frac1{|\mathcal E|}
\sum_{w\in\mathcal E}\widehat H_{ij}(w)^2.
\tag{17}
$$

- 必须使用足够光滑的激活（例如 tanh、SiLU、softplus）才能得到有意义的经典二阶导数。
- **普通 ReLU MLP 是分段仿射函数，其经典 Hessian 几乎处处为 0，不能直接用它评估曲率能量。**
- 架构、weight decay、平滑正则化、ensemble、输入尺度、模型容量会影响 Hessian；decoder 的预测 $R^2$ 高不代表导数质量好。
- 若比较不同 $R$，优先在同一个冻结的可微代理上做坐标变换并解析变换 Hessian；如果每个 $R$ 重新拟合网络，那么 loss 混入了重新训练误差。
- 本方法为自动微分与光滑代理模型的自然组合；**不要误称神经网络对任意目标 $M$ 已有普遍的统计一致性保证**。

### A8. 直接二次泛函估计 / split-sample 或 cross-fit sieve

这是最接近指标本身的数学统计路线。[R1]

有可微基 $\phi$、$f\approx\beta^\top\phi$ 时定义 $G_{ij}$ 如式 (13)，目标近似为：

$$
M_{ij}\approx \beta^\top G_{ij}\beta.
$$

在互不重叠的训练子样本上获得 $\hat\beta_A,\hat\beta_B$，使用：

$$
\boxed{
\hat M_{ij}^{\mathrm{split}}
=\hat\beta_A^\top\hat G_{ij}\hat\beta_B
}
\tag{18}
$$

再在不同 split 上平均，并用独立样本估计 $\hat G_{ij}$。这避免同一系数估计误差在平方时直接出现全部自身方差项；但**并不消除基截断误差、正则化偏差或依赖结构造成的偏差**。

单个 split cross-product 可以是负数，所以必要时的非负化须标注为后处理。对未加独立性或有效平滑度假设的多维非参数问题不能宣称它无偏或达到参数速率。

可进一步研究投影 / U-statistic 型直接二次泛函估计；是否优于 plug-in 取决于光滑度、维数、噪声与目标积分测度。[R1]

### A9. Weak derivatives / RKHS / integration-by-parts 方法

如果 $P_W$ 有足够光滑的密度 $p_W$、边界条件允许，则对测试函数 $\psi$ 可用积分分部得到：

$$
\int\psi(w)\partial_{ij}f(w)\,dw
=\int f(w)\partial_{ij}\psi(w)\,dw
+\text{边界项},
\tag{19}
$$

从而用观测值间接估计某些导数投影，而不必在每个点直接求导。配合正交基、RKHS 正则化或弱形式矩条件，可建立导数的投影估计，再构造式 (13) 的能量。

这不是一个无需假设的“直接读出 $M$”公式：**必须处理边界项**；对于带权 $dP_W=p_Wdw$，积分分部会涉及 $p_W$ 的导数，相关输入时更复杂。它主要是可探索的 estimator 设计路线，需就具体设定另行证明适用性。[R3,R6]

### A10. 近邻差分 / 配对设计 / 网格求导

当数据按规则网格采样或真的存在 $w\pm h e_i\pm h e_j$ 的可比较邻点，才可以用：

$$
\widehat{\partial_{ij}f}(w)
\approx\frac{
Y(w+h e_i+h e_j)-Y(w+h e_i-h e_j)
-Y(w-h e_i+h e_j)+Y(w-h e_i-h e_j)
}{4h^2}
\quad (i\ne j).
\tag{20}
$$

对角项使用对应的二阶中心差分。choice 噪声会在小 $h$ 下被放大；应对条件均值做平滑或有大量重复测量，不能对普通随机高维 W–Y 数据直接应用。CR4 的每试次随机 X **不等于**具有这种精确配对的 factorial grid。差分技术可作为合成的受控 oracle 实验，但不是盲矩阵的默认方案。[R4]

### A11. MARS / 稀疏低阶交互样条

使用自适应选择的分段多项式与低阶交互来拟合输出，可显式控制最高交互阶数。[R12]

**限制：** 单个 piecewise-linear hinge 主效应的自身二阶导数几乎处处为零（折点处不可微）；hinge 的乘积交互项仍可能具有非零混合二阶导数。因此这种非光滑基可能扭曲想研究的连续曲率结构，不能简单把整个 MARS Hessian 说成恒为零。若要用该家族，采用更光滑的 cubic variants，并报告由 knot/平滑方式引起的结构偏差。MARS 在这里是待测试的低阶结构化代理，而不是已经证明有效的 Hessian estimator。

### A12. 需要额外查询函数的随机扰动 / 多点方法（分开存放）

Lamboni (2024) [R11] 研究由多个随机扰动的 **模型函数值查询** 构建混合偏导估计，并用于敏感性分析。该方法很相关，但需要能评价 $f(w+\delta)$ 等新点，不属于“只给一次固定观测矩阵 $W,Y$ 且不允许再调用真函数”的同一信息约束。

同理，已知解析式的自动微分、直接调用冻结真 decoder 或生成新的 oracle 标签，必须视为单独的 **oracle / query-access benchmark**，不能在 blind WY 排名中与 A1–A9 平等比较。

## 9. 只比较预测误差是不够的：推荐评估协议

**Ground truth 仅用于最终评价，而不用于 blind 估计或超参数选择。**

1. **固定目标：** 明确 $f$ 是 logit、概率还是其它连续输出；固定 $P_W$、坐标尺度、样本划分、积分权重。
2. **构造 oracle：** 仅在合成模型有真函数时，先用解析 Hessian + 足够多的独立积分点算 $M_{ij}^{\mathrm{oracle}}$；对 CR4/CR5 遵守各自 X 分布。
3. **blind 拟合：** 只给每种算法 $W_{\mathrm{train}},Y_{\mathrm{train}}$；由主体分组的 validation 选择超参数，不向方法泄露真 T/Hessian。
4. **共同评估点：** 全算法在同一套参考 $W_{\mathrm{eval}}$ 估计 $\hat M$，避免采样误差伪装为算法差异。若选用训练之外的外部分布，必须检查 extrapolation/support。
5. **原始矩阵准确度：** 记录 $\|\hat M-M\|_F$、逐元素绝对和相对误差、$\hat D=\sqrt{\hat M}$、非零排序、对结构零单元的误报率。接近零的真实元素不能用相对误差除以零。
6. **稀疏性保真：** 比较每行 top-$k$、tail-$L_2$、熵或归一化能量分布；矩阵整体相关高不等于零模式恢复得好。
7. **rotation test：** 固定被试、固定物理函数与评估点，扫描二维 Givens 及一般 $k$ 维正交旋转；比较 oracle loss 曲线和 blind loss 曲线、排序一致性、最优解的角度误差；特别检查 UU 的 45° 双线性反例。
8. **统计不确定性：** 对被试层做 bootstrap / grouped cross-fitting；不要把同一被试重复试次误认为独立个体 Z。报告种子和重复次数。
9. **超参数稳定性：** 测试 spline df、核带宽、局部多项式带宽、多项式阶数、MLP 平滑、样本数与每人试次数；对 derivative metric 而不是仅 Y 预测作 sensitivity analysis。
10. **复杂度和内存：** 记录 wall time、GPU/CPU 与峰值内存；在 $d=k+p$ 下完整 Hessian 是 $d\times d$，每种方法都必须给出可扩展方案。

### 9.1 首轮推荐优先级

| 优先级 | 方法 | 主要原因 |
|---|---|---|
| P0 | A1 局部多项式 | 可解释的非参数基线，检验局部可辨识性 |
| P0 | A4 正交多项式 / sparse sieve | 二阶导数和积分 Gram 可解析化；可设计 cross-fit |
| P0 | A5 KRR | 多维混合偏导的专门理论支撑 [R6] |
| P1 | A3 tensor GAM / SS-ANOVA | 检查可分性、低阶交互和局部光滑结构 |
| P1 | A6 GP | 为导数不确定性和模型偏差提供对照 |
| P1 | A7 smooth MLP | 衡量灵活黑盒代理在二阶导数目标上的成败 |
| Research | A8 直接二次泛函 | 值得发展，但需对具体高维/重复测量形式进行证明 |
| 非 blind | A12 函数查询型随机扰动 | 只能作为外部参照，不能算仅凭一次 WY 观测的推断 |

**特别建议：** 先验证 $M_{ij}$ 的估计是否可靠，再验证它被稀疏化后是否真的识别真实参数轴。这是两个逻辑上独立的检验；某一种 estimator 在矩阵层面恢复得准，不意味着由其构造的 rotation loss 一定合理。

## 10. 与 mod-rot-4 当前数据接口的衔接

根据项目根 README，目前 S1 保存真 T、原始 X、logit、p、choice；S5 提取每个被试固定潜变量 Z；S8/S8_2 在不同坐标规范下搜索监督旋转，并有 S9/S9_2 的 Givens sweep。

未来独立实验应 **只读取数据，不污染 S1–S9**。建议输入协议：

| 字段 | 用途 |
|---|---|
| $Z_s$ 或 $Z_{z,s}$ | 被试级潜变量；保存其均值、标准差和坐标版本 |
| $X_{s,t}$ | 试次物理输入或项目指定 X 标准化版本 |
| $Y_{s,t}$ | 选定的 choice / p / logit 之一；blind 模式只见被批准的一种 |
| subject_index, trial_index | 正确 split、bootstrap 和复现 |
| rotation $R$ | 记录对潜变量的正交变换；X 固定不转 |
| train/validation/evaluation splits | 防止同被试跨集合泄漏 |
| evaluation distribution $P_W$ | 明确是原数据经验分布、条件分布还是外部参考分布 |

建议输出逐方法保存：$\hat M^{UU}$、$\hat M^{UX}$、$\hat M^{XX}$，$\hat D$，选择的超参数、坐标规范、积分/评估点指纹、训练主体分组、估计不确定性、失败原因以及与 oracle 的**事后**对比。

不要把 **同一真实函数下改变的输入分布**（CR4/CR5）与 **在同一输入分布下只改变旋转坐标** 混为一谈。

---

# References / 参考文献

以下引用均为相关数学与统计方法的基础或专门研究。**“用于本项目的 $M_{ij}$”是所述方法的应用/延伸；不等于原论文已经验证 mod-rot-4 的盲旋转效果。**

- **[R1] Efromovich, S., & Samarov, A. (2000).** Adaptive estimation of the integral of squared regression derivatives. *Scandinavian Journal of Statistics, 27*(2), 335–351. [JSTOR](https://www.jstor.org/stable/4616607). **直接二次泛函**：积分平方回归导数与估计率。
- **[R2] Roustant, O., Fruth, J., Iooss, B., & Kuhnt, S. (2014).** Crossed-derivative based sensitivity measures for interaction screening. *Mathematics and Computers in Simulation, 105*, 105–118. [DOI](https://doi.org/10.1016/j.matcom.2014.05.005). **Crossed-DGSM**：混合二阶导数平方能量的解释。
- **[R3] Wahba, G. (1990).** *Spline Models for Observational Data*. SIAM. [Publisher](https://epubs.siam.org/doi/book/10.1137/1.9781611970128). **Sobolev/样条**：平滑度与二阶导数惩罚。
- **[R4] Stone, C. J. (1982).** Optimal global rates of convergence for nonparametric regression. *The Annals of Statistics, 10*(4), 1040–1053. [PDF](https://websites.umich.edu/~jizhu/jizhu/wuke/Stone-AoS82.pdf). **非参数估计率背景**，不是本指标的专门估计器。
- **[R5] De Brabanter, K., De Brabanter, J., De Moor, B., & Gijbels, I. (2013).** Derivative estimation with local polynomial fitting. *Journal of Machine Learning Research, 14*, 281–301. [Full text](https://jmlr.csail.mit.edu/papers/volume14/debrabanter13a-deleted/debrabanter13a.pdf). **局部导数估计**。
- **[R6] Liu, Z., & Li, M. (2023).** On the estimation of derivatives using plug-in kernel ridge regression estimators. *Journal of Machine Learning Research, 24*(266), 1–37. [JMLR](https://jmlr.org/papers/v24/21-1110.html). **KRR 多维混合偏导**。
- **[R7] Eilers, P. H. C., & Marx, B. D. (1996).** Flexible smoothing with B-splines and penalties. *Statistical Science, 11*(2), 89–121. [DOI](https://doi.org/10.1214/ss/1038425655). **P-splines**。
- **[R8] Wood, S. N. (2017).** *Generalized Additive Models: An Introduction with R*, 2nd ed. Chapman & Hall/CRC. [Publisher](https://www.routledge.com/Generalized-Additive-Models-An-lntroduction-with-R-Second-Edition/Wood/p/book/9781498728331). **GAM / tensor smooths**。
- **[R9] De Lozzo, M., & Marrel, A. (2016).** Estimation of the derivative-based global sensitivity measures using a Gaussian process metamodel. *SIAM/ASA Journal on Uncertainty Quantification, 4*(1), 708–738. [DOI](https://doi.org/10.1137/15M1013377). **GP 代理与导数型敏感性估计**。
- **[R10] Sudret, B., & Mai, C. V. (2015).** Computing derivative-based global sensitivity measures using polynomial chaos expansions. *Reliability Engineering & System Safety, 134*, 241–250. [arXiv](https://arxiv.org/abs/1405.5740). **PCE + 导数敏感性指标**。
- **[R11] Lamboni, M. (2024).** Optimal estimators of cross-partial derivatives and surrogates of functions. *Stats, 7*(3), 697–718. [DOI](https://doi.org/10.3390/stats7030042). **可查询函数的随机扰动导数代理**；不能直接当作固定噪声 WY 的估计器。
- **[R12] Friedman, J. H. (1991).** Multivariate adaptive regression splines. *The Annals of Statistics, 19*(1), 1–67（含讨论见当期后续页）. [DOI](https://doi.org/10.1214/aos/1176347963). **稀疏自适应交互样条**。

**文献到任务的映射：** 若问“这种量是否有数学先例”，看 [R1–R3]；若问“仅有 noisy WY 怎么估计”，看 [R5–R9] 并研究 [R1] 的直接二次泛函；若问“交互意义和可解释性”，看 [R2,R9,R10]；若问“需要额外 f 查询的导数方法”，看 [R11]。

**版本说明：** 本文只定义目标及综述方法，不宣称任何盲估计方法已经在 C1/C2/C3/CR3/CR4/CR5 上成功恢复最优旋转。后续实现应保留 oracle–blind 信息隔离和逐模型验证。
