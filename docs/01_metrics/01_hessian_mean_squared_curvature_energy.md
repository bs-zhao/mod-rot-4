# 均平方 Hessian 曲率能量

**研究对象：** 均平方（平均平方）Hessian 曲率能量；逐元素平均平方 Hessian（elementwise expected squared Hessian / mean-squared second-derivative energy）

**核心指标：**

$$
\boxed{M_{ij}(f;P_W)=\mathbb E_{W\sim P_W}\!\left[
\left(\frac{\partial^2 f(W)}{\partial W_i\partial W_j}\right)^2
\right]}
$$

**文档定位：** 研究备忘录 / 文献综述 / 后续盲矩阵实验的设计依据；**不是已实现的 S1–S9 模块**。这里的 *infer* 现分为三种信息条件：①已有训练好的 MLP；②仅有 WY，先拟合近似函数；③仅有 WY，直接构造能量统计量，不拟合任何函数。任何类别都不得将真实生成解析函数或 oracle Hessian 当成盲估计输入。

**项目关联：** [项目总览](../../README.md)；[模型定义与 S1–S2](../../models/README.md)；[NP、S5 的 Z 和 S8/S9 旋转](../../nn/README.md)。

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

## 4. W 的连续变量如何进入平均：正态分布积分与固定观测点平均

本研究有三个不同层次的问题：（i）Hessian 如何取得（已有 MLP、从 WY 拟合函数、仅用 WY 直接估计）；（ii）**在哪些 W 上对 Hessian 平方取平均**；（iii）怎样把得到的能量矩阵转化为简单性指标。此节专门定义第二个层次。关键是：**W 的每个维度都可以是连续型数值变量；是否为它指定概率分布，是独立的统计选择。**

对同一个可微函数 $f$，平均平方曲率能量由函数和研究者选择的评价测度 $Q_W$ 共同决定：

$$
\boxed{M_{ij}(f;Q_W)=\int_\Omega[\partial_{ij}f(w)]^2\,Q_W(dw).}
$$

### 4.1 第一种：分布建模的连续变量（Distribution-modeled continuous inputs）

假设一组连续潜变量 $Z\in\mathbb R^k$ 服从正态：

$$
Z\sim\mathcal N_k(\mu_Z,\Sigma_Z).
$$

可再区分：①分布参数由理论/模拟设计**已知**；②**假设正态族**，由被试级观测 Z 估计 $\widehat\mu_Z,\widehat\Sigma_Z$。例如后者定义：

$$
\boxed{
M_{ij}^{\rm Gaussian}
=\int H_{ij}(z)^2\phi_k(z;\widehat\mu_Z,\widehat\Sigma_Z)\,dz.
}
\tag{4a}
$$

原观测 Z 的作用是估计参数，之后可从拟合分布生成更多 Z 积分点。但增加积分点只降低 Monte Carlo 误差，**不增加训练数据对未知真函数导数的约束**。

还要区分 **多元联合正态** $\mathcal N(\widehat\mu,\widehat\Sigma)$ 与 **各维独立正态** $\prod_j\mathcal N(\widehat\mu_j,\widehat\sigma_j^2)$；只使用各维均值与标准差，等于额外丢弃相关性。

### 4.2 第二种：固定设计/给定观测点上的连续变量（Fixed-design continuous inputs）

例如 CR3/CR5 已有 500 个完整刺激向量：

$$
\mathcal X=\{x_1,\ldots,x_{500}\},\qquad x_t\in\mathbb R^p.
$$

这些 X 的各维仍然是连续数值尺度，**不是 500 个类别构成的离散型变量**。研究者不假设这 500 题来自正态、均匀或任何已知/未知生成分布；只决定在这 500 个给定位置上计算函数指标。

等权设计平均：

$$
\boxed{M_{ij}^{\rm fixed}=\frac1{500}\sum_{t=1}^{500}H_{ij}(x_t)^2.}
\tag{4b}
$$

也可选择预先指定的设计权重 $a_t\ge0,\sum_ta_t=1$，计算 $\sum_ta_tH_{ij}(x_t)^2$。数学记号 $Q_X^{\rm fixed}=\sum_ta_t\delta_{x_t}$ **仅仅表示有限加权求和**；它不是“X 物理上只能取这些值”或“断言 X 的总体抽样分布是离散的”。

**完整向量是最小设计单位：** 500 个六维刺激向量是 500 个已确定的组合，不允许把不同题目的六列独立重排或进行笛卡尔乘积，然后仍称之为原题库平均。

如果已知可微 decoder $f_\theta(Z,X)$，在这些固定点处依然能求 X 方向的一阶/二阶导数。若**仅有 WY** 且没有函数/光滑性假设，有限的固定点一般无法识别周围连续空间的真导数；不要混淆**积分的点是固定的**与**被微分的变量是离散的**。

### 4.3 混合使用：对 Z 正态积分，对 X 保留固定 500 题

令 $W=(Z,X)$，选择：

- Z：正态分布建模（使用已知或从被试 Z 拟合的联合 $\mu_Z,\Sigma_Z$）。
- X：CR3/CR5 固定的 $K=500$ 个连续刺激向量；不对 X 假设任何生成分布。

如果**明确指定每个 Z 与每一道 X 进行全组合**作为评价标准，那么：

$$
\boxed{
M_{ij}^{\rm hybrid}
=\frac1K\sum_{t=1}^{K}
\int_{\mathbb R^k}H_{ij}(z,x_t)^2
\phi_k(z;\widehat\mu_Z,\widehat\Sigma_Z)\,dz.
}
\tag{4c}
$$

从正态分布抽取 $L$ 个积分点 $z^{(1)},\ldots,z^{(L)}$，近似为：

$$
\boxed{
\widehat M_{ij}^{\rm hybrid}
=\frac1{KL}\sum_{\ell=1}^{L}\sum_{t=1}^{K}
H_{ij}(z^{(\ell)},x_t)^2.
}
\tag{4d}
$$

这可称为 **Gaussian latent integration + fixed continuous-design averaging**。本式在 Z 上积分、在 X 上对确定设计求和，**并不把 X 声称为离散随机变量**。但“全组合”是主动选择的**乘积评价规则**，不是数据已证明 Z 与 X 独立。若研究目的是保持同被试的真实 Z–X 配对，应改为观测联合配对平均；二者是不同的估计目标。

### 4.4 与原来三类 Hessian 获取方式的关系

| Hessian 获取的信息条件 | 正态等分布积分 | 固定连续点平均 | Z 正态 × X 固定点 |
|---|---|---|---|
| **1. 已有冻结 MLP** | 可对新 Z 求网络导数，检查尾部支持 | 对原观测点求网络导数 | **可直接计算，推荐首轮比较** |
| **2. 只有 WY，先拟合近似函数** | 代理函数可对新点求导，但可能外推 | 在给定点对代理函数求导 | 可计算，但误差取决于函数拟合与支持 |
| **3. 只有 WY，直接统计能量** | 须满足特定 U-statistic/弱形式的分布条件 | 仅凭有限点一般不能无假设识别连续导数 | 新生成的 Z 没有真实 Y，不能直接套用所有方法 |

因此 **Hessian 取得方式**和 **W 的平均规则**是不同分类维度；前者三类，后者可以在不同连续变量块上指定不同规则。进一步得到的 Tail/Hoyer/Entropy 等简单性指标又是第三个层次。

### 4.5 旋转、输入支撑和权重不变的要求

如果 $U=RZ,\ R^\top R=I$，并要求在**同一个输入总体**上比较旋转，那么高斯的推前分布必须是：

$$
\boxed{Z\sim\mathcal N(\mu,\Sigma)
\ \Longrightarrow\
U\sim\mathcal N(R\mu,R\Sigma R^\top).}
\tag{4e}
$$

不能每旋转一次就重新将各 U 轴拟合为**相互独立**的高斯；这同时改变了评价分布。X 的原 500 个固定完整向量应保持不变。

**训练边界：** 无界高斯 Z 的尾部可能超出网络训练支撑；对训练好的 NN 可以数学上计算导数，但真函数导数不一定可识别。需要报告 OOD/尾部比例，或预先指定截断分布（注意这是新 $Q_W$）。固定连续点不要求有自然分布，也不自动保证点与网络训练支持吻合。

**设计权重：** 需预先说明每道题等权、每行观测等权，还是每个被试先等权再对其试次平均。权重选择属于评价测度的定义，不属于函数/导数估计器本身。


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

# Part II. 三大类：已有 MLP、WY 拟合近似函数、WY 直接估计能量

本部分严格按照**是否拥有训练好的网络**、**仅有 WY 时是否还需要拟合函数**来分类。三类共享 Part I 的指标定义，但信息条件和估计误差不同：

| 大类 | 输入与访问权限 | 要不要先拟合函数？ | 实际对象 |
|---|---|---|---|
| **1. 基于已有 MLP** | 已训练好的、可前向查询/可微分的 $\widehat f_\theta(W)$ | B1–B10 **不需要重新拟合**（B11/B12 是可选训练扩展） | 网络本身的 $\mathbb E[(\partial_{ij}\widehat f_\theta)^2]$；真函数仍需验证 |
| **2. 仅有 WY，先拟合近似函数** | 固定的 $W,Y$，无现成可调用网络 | **需要**：局部曲面、样条、GP、KRR、从头训练 MLP、sieve 等 | 通过拟合函数的导数间接估计条件均值函数的曲率 |
| **3. 仅有 WY，直接算能量** | 固定的 $W,Y$，无现成网络 | **严格不拟合任何 $\widehat f$ 或函数系数** | 基于成对样本、已知测度/测试函数或严格匹配设计的二次泛函 |

**界线：** “不显式输出函数曲线”并不等于“没有拟合函数”。拟合 $\widehat\beta$ 构造 $\widehat f=\sum_k\widehat\beta_k\phi_k$，即使直接计算 $\widehat\beta^\top G\widehat\beta$，仍属于**第二类**。若用独立观测对直接形成 U-statistic，不拟合 $\widehat f$ 或 $\widehat\beta$，才是**第三类**。

这三类都必须说明输出是连续条件均值、二元选择概率还是 logit。固定样本 $Y\in\{0,1\}$ 对应可推断的基本目标是 $q(W)=P(Y=1\mid W)$；logit Hessian 是不同的非线性目标。数据区域存在边界和稀疏联合支持时，不能把函数在训练域外可计算等同于真实函数导数可识别。

## 7. 三类共同的比较规则与项目接口

### 7.1 通用误差验证 / oracle-blind 隔离

**合成数据的 oracle 导数与解析函数只用于最终评价；第二类和第三类不允许读取 oracle 来拟合或挑选方法。**

1. **固定目标：** 明确 $f$ 是 logit、概率还是其它连续输出；固定 $P_W$、坐标尺度、样本划分、积分权重。
2. **构造 oracle：** 仅在合成模型有真函数时，先用解析 Hessian + 足够多的独立积分点算 $M_{ij}^{\mathrm{oracle}}$；对 CR4/CR5 遵守各自 X 分布。
3. **严格区分路径：** 第一类冻结已有 MLP；第二类只能用 WY 拟合代理；第三类只能计算预先规定的直接统计量，不能训练函数/基函数系数。所有方法都不向 blind 路线泄露真 T/Hessian。
4. **共同评估点：** 全算法在同一套参考 $W_{\mathrm{eval}}$ 估计 $\hat M$，避免采样误差伪装为算法差异。若选用训练之外的外部分布，必须检查 extrapolation/support。
5. **原始矩阵准确度：** 记录 $\|\hat M-M\|_F$、逐元素绝对和相对误差、$\hat D=\sqrt{\max(\hat M,0)}$（若直接统计有负值，另留原始结果）、非零排序、对结构零单元的误报率。接近零的真实元素不能用相对误差除以零。
6. **稀疏性保真：** 比较每行 top-$k$、tail-$L_2$、熵或归一化能量分布；矩阵整体相关高不等于零模式恢复得好。
7. **rotation test：** 固定被试、固定物理函数与评估点，扫描二维 Givens 及一般 $k$ 维正交旋转；比较 oracle loss 曲线和 blind loss 曲线、排序一致性、最优解的角度误差；特别检查 UU 的 45° 双线性反例。
8. **统计不确定性：** 对被试层做 bootstrap / grouped cross-fitting；不要把同一被试重复试次误认为独立个体 Z。报告种子和重复次数。
9. **超参数稳定性：** 测试 spline df、核带宽、局部多项式带宽、多项式阶数、MLP 平滑、样本数与每人试次数；对 derivative metric 而不是仅 Y 预测作 sensitivity analysis。
10. **复杂度和内存：** 记录 wall time、GPU/CPU 与峰值内存；在 $d=k+p$ 下完整 Hessian 是 $d\times d$，每种方法都必须给出可扩展方案。

### 7.2 项目数据接口

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

## 8. 第一大类：基于已有训练好的 MLP / 神经网络

这一类的优势是**不需要重新拟合 $W\to Y$ 函数**，而是利用已训练的 MLP 作为已知的可微/可查询映射；B1–B6 是计算 Hessian 的方法，B7–B9 选择积分测度和边界，B10–B12 处理稳定性或额外训练。原 A12 需额外查询函数，已并入此类，而**不属于只有固定 WY 的情况**。

> **第一类：已有训练好的 MLP / decoder。** B1–B10 是在冻结网络上推断其自身的平均平方输入 Hessian 能量；B11/B12 需要额外训练，作为明显标识的扩展。不允许用真实生成解析式替代已训练模型。

### 8.1 首先区分两个统计目标

对冻结网络，定义：

$$
\boxed{
M^{\rm NN}_{ij}(P_W)
=\mathbb E_{W\sim P_W}
\left[\bigl(\partial_{ij}\widehat f_\theta(W)\bigr)^2\right].
}
\tag{B1}
$$

未知真实函数的指标则是：

$$
M^\star_{ij}(P_W)
=\mathbb E_{W\sim P_W}
\left[\bigl(\partial_{ij}f^\star(W)\bigr)^2\right].
\tag{B2}
$$

它们不自动相等。已训练网络允许对 $\widehat f_\theta$ 精确自动求导（至浮点精度），但**网络导数是否逼近真实函数导数**仍是需要单独检验的统计问题。估计误差至少可分解为：

$$
\widehat M^{\rm NN}_{ij}-M^\star_{ij}
=
\underbrace{\widehat M^{\rm NN}_{ij}-M^{\rm NN}_{ij}}_{\text{数值导数/积分误差}}
+
\underbrace{M^{\rm NN}_{ij}-M^\star_{ij}}_{\text{网络二阶函数逼近误差}}.
\tag{B3}
$$

**输出语义：** 如果网络输出两类 logits $a_0(W),a_1(W)$，二元 logit 是 $\eta_\theta=a_1-a_0$，不是单独的 $a_1$。选择概率是 $q_\theta=\sigma(a_1-a_0)$，其 Hessian 与 logit Hessian 不一样。最终采样的 0/1 choice 不属于平滑函数，不能直接求连续 Hessian。先固定 output_kind（logit/probability）、eval mode、latent 表示和网络 stochastic 状态。

### 8.2 方法地图：三层而非一条路线

| 层次 | 方法 | 网络要求 | 是否重训 | 核心输出与适用范围 |
|---|---|---|---|---|
| 导数获取 **B1** | 完整自动微分 Hessian（AD） | 可用 autograd | 否 | 精确得到完整 $H_\theta(W)$；本项目首选 |
| 导数获取 **B2** | Hessian–vector product / 逐列 | 可用输入梯度 | 否 | $H_\theta v$、$H e_j$，只算目标列/区块 |
| 导数获取 **B3** | Hutchinson 型随机 probing | HVP | 否 | 直接估行平方能量与总能量；高维更有价值 |
| 导数获取 **B4** | MLP 二阶链式法则传播 | 可读网络层/权重 | 否 | 逐层解析计算 Hessian，可独立验证 AD |
| 导数获取 **B5** | 前向有限差分 | 只需前向查询 | 否 | 用差分计算二阶导数近似 |
| 导数获取 **B6** | 直接平均差分商平方 | 只需前向查询 | 否 | 不保存 $H$，直接估 $\mathbb E[H_{ij}^2]$ |
| 期望/支持 **B7** | 经验积分（empirical W） | 上述任意求导法 | 否 | 在真实 $(Z_s,X_{s,t})$ 输入点计算期望 |
| 期望/支持 **B8** | 有界 MC / scrambled QMC | 明确合法联合 $P_W$ | 否 | 用已知设计分布或可信参考分布数值积分 |
| 期望/支持 **B9** | Boundary-stratified integration | 训练支持信息 | 否 | 分别汇总内部、边界、可疑 OOD 区域 |
| 可靠性 **B10** | 多网络/checkpoint ensemble | 多个已训练网络 | 已有则否 | 网络曲率分歧、边界敏感性 |
| 可选改造 **B11** | 局部平滑代理 / 蒸馏 | 网络可查询 | 是 | 训练第二个代理，附加新的函数拟合偏差 |
| 可选改造 **B12** | Sobolev/导数增强训练 | 可重训且有额外约束 | 是 | 改善导数质量，不属于冻结网络直接估计 |

B1–B6 是**导数获取层**；B7–B9 是**积分测度和边界层**；B10–B12 是**模型可信度/改造层**。例如 B1+B7 与 B1+B8+B9 是两套完整而不同的估计方案，不应把 B7、B8 误称为替代自动微分的“求导算法”。

### 8.3 B1：冻结网络的完整自动微分

在给定积分点 $w_n$ 上计算：

$$
H_\theta(w)=\nabla_w^2\widehat f_\theta(w),\qquad
\boxed{\widehat M_{ij}^{\rm AD}
=\frac{\sum_n\omega_n[H_\theta(w_n)]_{ij}^2}
{\sum_n\omega_n}.}
\tag{B4}
$$

对的是**网络输入 W 的 Hessian**，不是网络权重的 Hessian，也不是训练损失对权重的 Hessian。PyTorch 使用 torch.func.hessian 或 jacfwd(jacrev)；函数应返回**单个样本的标量 logit**，再批量映射多个点。[N1]

对于 $d\approx9$–$12$，完整矩阵通常可负担，是最适合与 oracle 和其它估计器对比的高精度计算基线。检查矩阵对称性、有限值、输入列顺序、Float32/Float64 敏感性，并用 B5 在部分内部点核验导数。

**激活函数问题：** 对纯 ReLU 分段仿射 MLP，经典 Hessian 几乎处处为零，即使预测准确也无法表示平滑真函数的曲率；GELU、SiLU、tanh、softplus 等允许非零、连续的二阶导数。但“网络光滑”仍不等于“网络已学到真曲率”。

### 8.4 B2/B3：HVP、随机投影与无需完整 H 的直接能量

对任意方向 $v$：

$$
\boxed{H_\theta(w)v
=\nabla_w\left[\nabla_w\widehat f_\theta(w)^\top v\right].}
\tag{B5}
$$

使用 $v=e_j$ 得第 $j$ 列，从而仅计算想要的 UU/UX 部分；这是 Hessian–vector product（Pearlmutter, 1994）[N2]。

如果 $v$ 是独立 Rademacher 探针，$\mathbb E_v[vv^\top]=I$，则：

$$
\boxed{\mathbb E_v[(Hv)_i^2]=\sum_jH_{ij}^2},
\qquad
\boxed{\mathbb E_v[\|Hv\|_2^2]=\|H\|_F^2}.
\tag{B6}
$$

可直接估整行/区块曲率能量。但**单个** $(Hv)_iv_j$ 的平方不等于 $H_{ij}^2$ 的无偏估计。若确实需要逐元素平方，用两个条件独立的探针 $v,v'$：

$$
\boxed{
\mathbb E_{v,v'}\left[v_j(Hv)_i\,v'_j(Hv')_i\right]=H_{ij}^2.
}
\tag{B7}
$$

这是探针恒等式的代数推论，不是宣称 Hutchinson 原论文给出了本项目的逐元素指标；有限次结果可能为负且方差较大，不可静默 clip，须随 probe 数和 seed 报告收敛。[N2,N3]

对 UX 的行总能量，可只在 X 子空间取 probe；但行总和不等于完整逐元素 $M^{UX}$。对于本项目的低输入维数，B1 一般更简洁可靠。

### 8.5 B4：显式传播网络层的 Hessian

对 $a=A h+b$，$h_{\rm new}=\phi(a)$：

$$
J_a=A J_h,\quad H_{a,r}=\sum_sA_{rs}H_{h,s},
$$

$$
\boxed{
H_{{\rm new},r}
=\phi''(a_r)J_{a,r}J_{a,r}^\top
+\phi'(a_r)H_{a,r}.
}
\tag{B8}
$$

逐层递推得到输出 Hessian，不用知道真实生成公式。适合作为 GELU MLP 的数学独立校验。若包含 attention、LayerNorm、mask、softmax 等算子，需要额外推导对应导数；不能直接把简单 MLP 公式用于整个 NP encoder。

### 8.6 B5/B6：只有 forward API 时的数值差分与直接能量

对 $i\ne j$ 用混合二阶差分：

$$
\widehat H_{ij,h}(w)=
\frac{
\widehat f_\theta(w+h_ie_i+h_je_j)
-\widehat f_\theta(w+h_ie_i-h_je_j)
-\widehat f_\theta(w-h_ie_i+h_je_j)
+\widehat f_\theta(w-h_ie_i-h_je_j)
}{4h_ih_j}.
\tag{B9}
$$

对角项：

$$
\widehat H_{ii,h}(w)=
\frac{\widehat f_\theta(w+h_ie_i)-2\widehat f_\theta(w)+\widehat f_\theta(w-h_ie_i)}{h_i^2}.
\tag{B10}
$$

**B5** 保存每点的数值 Hessian，再平方积分；**B6** 不显式存全矩阵，而直接在选定元素上平均 $\widehat H_{ij,h}^2$。两者都存在步长截断偏差和浮点相消。中心差分在足够光滑、stencil 完全合法的内部点通常具 $O(h^2)$ 截断阶，但 $h$ 过小会放大数值误差。必须对多尺度 $h$ 与网络输出精度做敏感性检查。

若差分点超出训练支持，其数值可能精准反映 NN 的**外推**却未必反映真实生成函数。边界处应优先使用 AD 计算网络导数；如只允许 forward，应区分：(i) 仅评估完整中心 stencil 留在域内的内点，(ii) 采用单侧差分/局部拟合并接受不同误差阶，(iii) 明确标记外推差分。**把越界输入直接裁剪回边界再代入中心差分公式，不是有效的无偏边界修正。**

### 8.7 B7/B8：经验点与已知分布积分

**B7：经验 W 积分（首选支持内 baseline）。** 对原始合法的被试–试次配对 $(Z_s,X_{s,t})$ 直接求网络 Hessian：

$$
\boxed{
\widehat M^{\rm emp,NN}_{ij}
=
\frac{\sum_{s,t}\omega_{s,t}
[\partial_{ij}\widehat f_\theta(Z_s,X_{s,t})]^2}
{\sum_{s,t}\omega_{s,t}}.
}
\tag{B11}
$$

每条观测等权时 $\omega_{s,t}=1$；每被试等权时 $\omega_{s,t}=1/n_s$。一个被试有 500 个试次仍只有一个 Z 位置，bootstrap 应按主体分组。这个估计刻画**观测经验输入分布**的网络曲率，不自动等于任何独立高斯参考分布下的能量。

**B8：已知目标 $P_W$ 的 MC/QMC。** 若已明确合法的联合分布 $P_{\rm target}$：

$$
\widehat M_{ij}^{\rm MC}
=\frac1K\sum_{k=1}^K
[\partial_{ij}\widehat f_\theta(W^{(k)})]^2,\quad
W^{(k)}\sim P_{\rm target}.
\tag{B12}
$$

也可用 scrambled Sobol QMC 在定义清楚的物理支持域采样后正确变换分布；这改善的是**积分点选择**，不是导数计算，优势依赖积分函数和有效维数。[N4] **均值/方差近似标准正态**并不等于潜变量 Z 的全部联合分布是无界独立高斯；随意在尾部或不同 Z 轴间拼接样本，可能走到网络完全没训练过的位置。

对截断高斯应按**正确归一化后的截断分布**采样；将越界值 clip 到边界会产生点质量、改变 $P_W$，不能当作原期望。对于需要重新定义的外部参考分布，应在结果中明确标识，而不是混写成训练分布期望。

### 8.8 B9：训练数据有边界——最需要单独评估的一部分

神经网络可以在训练支持外输出有数值的函数和导数，但：

$$
\boxed{
\nabla^2\widehat f_\theta(w)\text{ 可由 AD 算出}
\ \not\Rightarrow\
\nabla^2 f^\star(w)\text{ 在该位置有充分统计支持}.
}
\tag{B13}
$$

**边界类型与本项目对应关系：**

| 输入类型 | 真实支持/边界 | 推荐方法 | 不恰当做法 |
|---|---|---|---|
| **CR4 的连续 X** | 金额 $U(5,200)$，概率 $U(0.05,0.95)$，6 维有界物理设计 | 在合法物理分布采样，使用 checkpoint 中 TRAIN-only 的 X scaler 再进入 decoder | 在任意标准化盒内放大范围并称为原 CR4 分布 |
| **CR3/CR5 的共享 X bank** | 有限 500 个题目，$P_X$ 为离散经验测度 | 对固定 500 个 X 求平均，按真实题库权重汇总 | 在连续盒中新抽题，然后称为 CR5 原设计期望 |
| **个体潜变量 Z** | 每被试只有一条冻结 Z；可能相关、非高斯、近似低维流形 | 首先从观测 Z 经验抽样/真实组合；必要时拟合联合分布并检测 OOD | 只看各维 mean/SD，就假定独立正态全空间均可靠 |
| **联合 W=(Z,X)** | $Z_s$ 共享于该主体各试次，X 可能受分组/实验设计约束 | 首先使用原真实主体–试次配对，外部积分必须定义联合参考分布 | 随机打乱 W 各列或独立组合极端值，忽视联合支持 |

CR4 设计上的 X 是逐试次独立抽样；但**经 encoder 从某主体实际 $(X,Y)$ 提取的 Z** 在有限样本中可能与其观测 X 相关。因此把 $P_ZP_X$ 当积分分布，需要标明这是新的理论/参考测度，不能自动等同于经验 $P_{Z,X}$。在 CR5，离散 X 上可以求 NN 对连续 X 的微分，但真实函数在题库之外的变化**无法仅由这 500 个固定 X 识别**；网络曲率反映其学习到的插值/外推结构。

**分层边界诊断。** 若目标 X 的真实支持是盒 $\Omega_X=\prod_a[a_a,b_a]$，用相对于各边界的归一化距离：

$$
d_{\partial\Omega}(x)
=\min_a\left\{
\frac{x_a-a_a}{b_a-a_a},
\frac{b_a-x_a}{b_a-a_a}
\right\}.
\tag{B14}
$$

预先选 $\delta>0$，划分 $\Omega_{\rm in}=\{d_{\partial\Omega}\ge\delta\}$ 和 $\Omega_{\rm bdry}=\{0\le d_{\partial\Omega}<\delta\}$；输出内部与边界的条件能量及其质量比例：

$$
\begin{aligned}
M^{\rm interior}_{ij}&=\mathbb E[H_{ij}^2\mid X\in\Omega_{\rm in}],\\
M^{\rm boundary}_{ij}&=\mathbb E[H_{ij}^2\mid X\in\Omega_{\rm bdry}],\\
\boxed{
M^{\rm full}_{ij}
&=\pi_{\rm in}M^{\rm interior}_{ij}
+\pi_{\rm bdry}M^{\rm boundary}_{ij}.
}
\end{aligned}
\tag{B15}
$$

**删掉边界点后得到的内部平均不是原来的完整 $P_W$ 期望。** 对 $d_X$ 维独立均匀盒，内部 $P$ 质量是 $(1-2\delta)^{d_X}$（$\delta\in[0,1/2]$），所以六维 CR4 中边界层可能覆盖很大比例，即使 $\delta$ 看似不大。边界层应单独报告而不是隐藏。

若联合 Z/X 支持不是矩形，不能对其简单采用坐标 min–max 作为真实可信域。可以补充 kNN 距离、局部密度、条件约束、到已观测流形的距离等**诊断指标**，但它们也需阈值校准、不能保证真实导数正确。AD 在 NN 数学上求边界处导数，不需要先从边界两侧取观测；**这种网络导数可计算性**并不能弥补真实函数一侧无观测数据所造成的不可识别性。[N7,N8]

### 8.9 B10–B12：不确定性、蒸馏和导数增强

**B10：固定评估点上的多网络 ensemble。** 对 $B$ 个独立训练 checkpoint 分别求 $M^{(b)}$，输出：

$$
\bar M_{ij}=\frac1B\sum_bM^{(b)}_{ij},\quad
V^{\rm model}_{ij}=\frac1{B-1}\sum_b(M^{(b)}_{ij}-\bar M_{ij})^2.
\tag{B16}
$$

比较各模型边界距离与 Hessian 能量的关系。注意平均各模型的 $H^2$ **不等于**先平均各模型函数、再对平均函数求 Hessian 平方。ensemble 分歧可诊断部分不确定性，但如果所有模型一致地错误外推，低分歧也不能证明导数可靠。[N9]

**B11：支持内局部代理蒸馏。** 若不能高效求完整 NN Hessian，可在可信支持区域查询 teacher NN 的输出（可选导数），拟合更易解析积分的样条、多项式或小型光滑代理。它会增加新的拟合误差，需要与 B1 对照，不能称作对原网络完全精确的微分。

**B12：允许重训时的 Sobolev/derivative-enhanced training。** 若真正有额外可监督的梯度、导数约束或解析 toy oracle，可把输出误差与导数误差共同加入训练，改善 curvature fidelity；Czarnecki et al. (2017) 是相关经典研究。[N5] 但用原 NN 可能错误的自身导数作为唯一 teacher，并不能创造额外真实信息；高频/局部曲率还可能受到网络频率偏差影响。[N6]

### 8.10 mod-rot-4 的冻结 NP decoder 接口、旋转及评估推荐

基于仓库当前源代码：

- [离散 NP decoder](../../nn/src/modrot4_nn/net/discrete.py) 是 **Linear + GELU MLP**，输出两个类别 logits，实际拼接顺序为 **[X, Z]**。外部研究可统一记 $W=(Z,X)$，但执行计算前一定要重排到 NN 的真实输入位置，并用 class-1 minus class-0 logits 作为二元选择 logit。
- [训练预处理](../../nn/src/modrot4_nn/net/preprocessing.py) 与 [nn/README.md](../../nn/README.md) 规定 X 的标准化只在 TRAIN 被试拟合，并为 encoder/decoder 共用。求导应固定原始物理 X 还是网络标准化 X，必要时按 Part I 式 (7) 转换 Hessian。
- S5 的 Z 是每被试冻结的 posterior mean。研究 decoder 的函数结构时，要把 Z 作为 decoder 的自由输入变量，不要把 encoder 的 $\partial Z/\partial X$ 混入。S8 和 S8_2 两种 Z 坐标规范并不一样；S8_2 全局标准化涉及 TEST，属于 transductive 约定，需标明。
- 计算输入 Hessian 时保持 decoder 冻结、eval mode，**但不能在输入导数计算路径上关闭 autograd 或 detach 输入**。目标 $P_W$、Z 版本、X scaler、checkpoint sha、评估点集合均需固定。

对潜变量正交旋转 $U=RZ$ 且保持同一物理函数、同一积分点时：

$$
H^{(R)}_{UU}=R H_{ZZ}R^\top,\quad
H^{(R)}_{UX}=R H_{ZX},\quad
H^{(R)}_{XX}=H_{XX}.
\tag{B17}
$$

建议**先在合法原始 $(Z,X)$ 上求 Hessian，再做坐标变换**，避免每换一个 $R$ 就对旋转坐标按各维 min–max/正态重新抽样并走到支持外。Part III §17.1 已解释 $M(R)$ 不能通过 $RMR^\top$ 得到，而需要在固定点评估 Hessian 或保存交叉二阶矩张量。

**推荐首轮实验：**

| 目的 | 方法套餐 | 必须记录 |
|---|---|---|
| 冻结网络自身曲率 baseline | **B1 + B7** | UU/UX/XX 全矩阵、真实评估点与主体权重 |
| 连续设计 X 的分布期望 | **B1 + B8 + B9** | 物理范围、标准化、QMC/MC seed、内部/边界两套指标 |
| 高维只计算选定子块 | **B2 + B7/B8** | 方向和列索引、与 B1 抽检结果 |
| 高维只看行/整体能量 | **B3 + B7/B8** | probe 次数、方差、逐元素估计负值 |
| 只能查询 forward | **B5/B6 + B7 + B9** | 步长扫描、stencil 越界率、单侧方案 |
| 对网络曲率不放心 | **B10 + B9** | checkpoint 间分歧、边界距离/OOD 分层 |
| 需要另外改进导数代理 | **B11/B12（另立研究分支）** | 训练来源、teacher 偏差、额外导数监督 |
| 对很多 R 搜索相同曲率损失 | **B1/B2 + 固定点 + 交叉二阶矩** | 同一物理函数/评估点/参考分布、旋转不变量检查 |

建议新实验输出：checkpoint 与数据集 hash、decoder 输入列顺序、output kind、Z 版本与均值/SD、X scaler、P_W、被试/试次权重、support kind、boundary delta、OOD diagnostics、求导方法与精度、$M^{UU}$/$M^{UX}$/$M^{XX}$、内部/边界完整三套 $M$、积分 seed 与样本数、稳定性/旋转元数据。以上均为设计建议，**不意味着现有 S1–S9 代码已经实现 B 类模块**。

### 8.11 需要额外模型函数查询的随机扰动（原 A12）

Lamboni (2024) [R11] 研究由多个随机扰动的 **模型函数值查询** 构建混合偏导估计，并用于敏感性分析。该方法很相关，但需要能评价 $f(w+\delta)$ 等新点，不属于“只给一次固定观测矩阵 $W,Y$ 且不允许再调用真函数”的同一信息约束。

同理，已知解析式的自动微分、直接调用冻结真 decoder 或生成新的 oracle 标签，必须视为单独的 **oracle / query-access benchmark**，不能在 blind WY 排名中与 A1–A9 平等比较。

### 8.12 神经网络曲率推断专题文献（与原 A 类文献分开）

- **[N1] PyTorch Contributors.** [Jacobians, Hessians, hvp, vhp, and more: composing function transforms](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html). 标准自动微分、Hessian、HVP 实现参考。
- **[N2] Pearlmutter, B. A. (1994).** Fast exact multiplication by the Hessian. *Neural Computation, 6*(1), 147–160. [DOI](https://doi.org/10.1162/neco.1994.6.1.147). HVP 的基础。
- **[N3] Hutchinson, M. F. (1990).** A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics – Simulation and Computation, 19*(2), 433–450. [DOI](https://doi.org/10.1080/03610919008812866). 随机探针思想；式 (B7) 是本节给出的代数恒等式。
- **[N4] Owen, A. B.** [Practical quasi-Monte Carlo integration](https://artowen.su.domains/mc/practicalqmc.pdf). 数值积分与 scrambled QMC；[SciPy Sobol](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)。
- **[N5] Czarnecki, W. M., et al. (2017).** Sobolev training for neural networks. *NeurIPS*. [Proceedings](https://papers.nips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html). 导数增强训练。
- **[N6] Tancik, M., et al. (2020).** Fourier features let networks learn high frequency functions in low dimensional domains. *NeurIPS*. [Proceedings](https://papers.nips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html). 神经网络的频率偏差。
- **[N7] Schwinn, L., et al. (2021).** Identifying untrustworthy predictions in neural networks by geometric gradient analysis. *UAI / PMLR 161*, 854–864. [PMLR](https://proceedings.mlr.press/v161/schwinn21a.html). 梯度几何与不可信预测的诊断。
- **[N8] Extrapolating from neural network models: a cautionary tale (2021).** [DOI](https://doi.org/10.1088/1361-6471/abf08a). 训练域外外推的风险。
- **[N9] Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).** Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*. [Proceedings](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html). 多网络不确定性。

## 9. 第二大类：只有 WY，先拟合一个近似函数

**信息条件：** 只有 $(W,Y)$，不知道真实 $f^\star$、没有预训练 MLP。先拟合可以微分的代理函数 $\widehat f$，再通过代数二阶导数、自动微分或数值差分和期望积分获得：

$$
\widehat M_{ij}^{\mathrm{plug}}
=\frac{\sum_{n\in\mathcal E}\omega_n[\partial_{ij}\widehat f(W_n)]^2}
{\sum_{n\in\mathcal E}\omega_n}.
\tag{8}
$$

若 $\widehat H_{ij}=H_{ij}+b_{ij}+\xi_{ij}$ 且 $\mathbb E[\xi_{ij}]=0$：

$$
\mathbb E[\widehat H_{ij}^2]=(H_{ij}+b_{ij})^2+\operatorname{Var}(\xi_{ij}).
\tag{9}
$$

因此**准确预测 Y 并不充分保证准确估计二阶曲率**；平滑偏差、拟合误差、高维局部数据不足和边界同样重要。

| 方法 | 需要先拟合的对象 | 数值估计途径 |
|---|---|---|
| A1/A2 局部多项式 / local logistic | 每个局部位置的二次多项式或局部 logit | 局部二次系数 $\to$ 积分 |
| A3 tensor-product spline / GAM | 样条基函数系数 | 求导 $\to$ 积分 |
| A4 多项式 sieve / PCE | 全局投影系数 | 可微基函数和导数 Gram 矩阵 |
| A5 Kernel Ridge | 核展开系数 | 对核函数求混合二阶导数 |
| A6 Gaussian Process | 后验函数 | 导数后验均值/方差 |
| A7 Smooth MLP | **从 WY 新训练一个 MLP** | 对新网络作 AD；不同于第一类已有冻结 MLP |
| A8 cross-fit sieve | 两个数据子集拟合的 $\widehat\beta_A,\widehat\beta_B$ | 二次型 $\widehat\beta_A^\top G\widehat\beta_B$ |
| A11 MARS / 交互样条 | 交互样条项与系数 | 平滑版本的二阶导数 |

### 9.1 各算法的公式、优缺点

#### A1. 局部二次多项式回归（Local polynomial Hessian）

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

#### A2. 局部 likelihood / 局部 logistic 模型

仅有 choice $y_n\in\{0,1\}$ 时，用局部二次 **logit** $\eta_n(a,b,B)$，优化：

$$
\arg\min_{a,b,B}\sum_n K_h(w_n-w_0)
\left[\log(1+e^{\eta_n})-y_n\eta_n\right]
+\lambda_{\mathrm{reg}}\mathcal P(B).
\tag{11}
$$

这里 $\hat B$ 直接估计 $\nabla^2\operatorname{logit}q(w_0)$，而非 $H_q$。局部 Bernoulli MLE 在稀疏区域可能接近完全分离；须有适当正则化、有效样本量限制和选择概率检查。此方法属于局部多项式和局部似然文献的合理延伸，**不是在此声称存在已验证的项目专用 estimator**。[R4,R5]

#### A3. Smoothing splines / tensor-product GAM / SS-ANOVA

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

#### A4. 正交多项式 / sieve / Polynomial Chaos Expansion (PCE)

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

#### A5. Kernel Ridge Regression (KRR)

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

#### A6. Gaussian Process (GP)

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

#### A7. Smooth neural network + automatic differentiation

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

#### A8. Cross-fit sieve / 基函数系数拟合后计算二次型（**归第二类**）

该版本虽然不逐点输出 Hessian，但仍**用 WY 拟合 $\hat\beta$ 对应的函数投影**，故归入第二类。[R1]

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

#### A11. MARS / 稀疏低阶交互样条

使用自适应选择的分段多项式与低阶交互来拟合输出，可显式控制最高交互阶数。[R12]

**限制：** 单个 piecewise-linear hinge 主效应的自身二阶导数几乎处处为零（折点处不可微）；hinge 的乘积交互项仍可能具有非零混合二阶导数。因此这种非光滑基可能扭曲想研究的连续曲率结构，不能简单把整个 MARS Hessian 说成恒为零。若要用该家族，采用更光滑的 cubic variants，并报告由 knot/平滑方式引起的结构偏差。MARS 在这里是待测试的低阶结构化代理，而不是已经证明有效的 Hessian estimator。

### 9.2 A8 与真正“不拟合函数”的区别

在这里的 A8，虽然可以只计算 $\widehat\beta_A^\top G\widehat\beta_B$ 而不保存逐点 Hessian，但 $\widehat\beta$ 仍是由 WY 拟合的**函数展开系数**。所以它仍属于第二类。

如果不拟合 $\widehat\beta$，直接用独立样本对 $Y_nY_m$ 与预先确定的核来构造无对角的 U-statistic，则属于下面的第三类 D1/D3；二者的偏差、方差以及数学假设必须分别研究。

## 10. 第三大类：只有 WY，直接算平均平方曲率能量，也不拟合任何函数

**严格定义：** 只从已给定的 $(W_n,Y_n)$ 构造样本统计量；不训练/拟合 $\widehat f$、不拟合 $\widehat\beta$、不逐点重建 Hessian。可以使用**事先固定**的基函数、测试函数、核以及已知的输入测度；这仍然有正则性、截断阶数与设计要求。

**前提性限制：** 在完全无平滑性假设、无支持条件的有限 WY 中，真实完整的 $M_{ij}=\mathbb E[(\partial_{ij}f)^2]$ 一般不可识别。第三类方法主要直接估计**二阶导数能量的有限投影/正则化版本**，或真实配对设计下的差分曲率平方。不能承诺“只凭任意矩阵就精确得到真 Hessian 能量”。

| ID | 直接方法 | 是否拟合函数？ | 直接输出与限制 |
|---|---|---|---|
| D1 | 正交级数的 quadratic-functional U-statistic | **完全不拟合** | 投影函数二阶导数的能量；已知 $P_W$、独立观测对 |
| D2 | Weak derivatives + 积分分部后的导数矩估计 | **完全不拟合**（若输入密度及 score 已知） | 导数自身在已知测试函数空间的投影能量 |
| D3 | 预定核的 pairwise quadratic U-statistic | **完全不拟合** | 直接 $Y^\top K(W,W)Y$ 去对角型统计量 |
| D4 | 严格重复的网格有限差分与双重复乘积 | **完全不拟合** | 差分导数平方；需要配对与独立重复/已知噪声方差 |

### D1. Orthogonal-series quadratic U-statistic（首选直接统计方案）

设 $(W_n,Y_n)$ 是**独立样本单位**，$Y=f(W)+\varepsilon$ 且 $\mathbb E[\varepsilon\mid W]=0$，已知输入分布 $P_W$ 及其正交归一基 $\{\phi_k\}$：

$$
\mathbb E[\phi_k(W)\phi_\ell(W)]=\mathbf1[k=\ell],
\quad
\beta_k=\mathbb E[Y\phi_k(W)].
\tag{D1}
$$

这里只在理论中定义 $\beta_k$，**计算中不拟合或输出 $\widehat\beta$**。设 $f_K=\sum_{k=1}^K\beta_k\phi_k$；其二阶曲率能量为：

$$
M^{(K)}_{ij}
=\boldsymbol\beta^\top G_{ij}\boldsymbol\beta,\quad
G_{ij}[k,\ell]=\mathbb E[\partial_{ij}\phi_k(W)\partial_{ij}\phi_\ell(W)].
\tag{D2}
$$

直接用所有独立观测对：

$$
\boxed{
\widehat M^{(K)}_{ij,U}
=\frac{1}{N(N-1)}\sum_{n\ne m}
Y_nY_m\sum_{k,\ell=1}^{K}
G_{ij}[k,\ell]\phi_k(W_n)\phi_\ell(W_m).
}
\tag{D3}
$$

独立、已知 $P_W$、正确条件均值及有限二阶矩下，其期望是**投影目标** $M^{(K)}_{ij}$，不一定是完整 $M_{ij}$；$K$ 截断、正交系选择、二阶导数放大噪声和高维数据稀疏仍决定统计效率。对于同被试多试次，不能简单将主体内两行视为独立 $n\ne m$，应使用跨独立被试的配对或发展明确的 cluster 修正。[R1]

对二元 choice，此方法直接估计 $q(W)=P(Y=1\mid W)$ 的投影能量，**不直接估计 logit Hessian 能量**；非线性 logit 变换需要新的估计设计或额外模型结构。

### D2. Weak-form / integration by parts 直接导数投影

选择 $L_2(P_W)$ 下正交归一测试函数 $\psi_k$；假设 $P_W$ 具有已知正的可微密度 $p(w)$，且函数、测试函数满足积分分部所需的**全部边界条件**：

$$
\alpha_{ij,k}=\mathbb E[\partial_{ij}f(W)\psi_k(W)]
=\mathbb E\left[Y\,\frac{\partial_{ij}(\psi_k p)(W)}{p(W)}\right].
\tag{D4}
$$

记 $r_{ij,k}(w)=\partial_{ij}(\psi_kp)(w)/p(w)$。因为 $\psi_k$ 正交，二阶导数在其前 $K$ 个测试函数上的平方投影满足：

$$
M_{ij,\mathrm{weak}}^{(K)}=\sum_{k=1}^K\alpha_{ij,k}^2\ \le\ M_{ij}.
\tag{D5}
$$

不用拟合函数，直接从独立样本对构造：

$$
\boxed{
\widehat M_{ij,\mathrm{weak}}^{(K)}
=\frac1{N(N-1)}\sum_{n\ne m}
Y_nY_m\sum_{k=1}^K r_{ij,k}(W_n)r_{ij,k}(W_m).
}
\tag{D6}
$$

若**还需从 WY 或 W 另拟合密度/score**，严格说已另拟合一个函数，不应再声称“完全不拟合任何函数”；它只能标记为直接曲率泛函 + 额外密度模型。特别地，CR4 有界均匀 X 在盒边界的密度并不自然为 0，积分分部的边界项**不能自动忽略**；若测试函数只在内部有效，那么估计的是有限测试空间覆盖的投影而非无条件全域能量。[R1,R3]

### D3. 直接 quadratic-kernel U-statistic 的矩阵计算

给定事先定义、无需拟合的二元核 $K_{ij}(w,w')$：

$$
\boxed{
\widehat M^{\mathrm{quad}}_{ij}
=\frac1{N(N-1)}\sum_{n\ne m}
Y_nY_m\,K_{ij}(W_n,W_m).
}
\tag{D7}
$$

D1 的核可设 $K_{ij}(w,w')=\sum_{k,\ell}G_{ij}[k,\ell]\phi_k(w)\phi_\ell(w')$；D2 可设 $K_{ij}=\sum_kr_{ij,k}(w)r_{ij,k}(w')$。这是一种**计算组织方式**，不代表任意选定核都能无偏估计完整目标；核的截断/平滑决定 estimand。剔除对角项可避免最直接的平方噪声膨胀，但不能消除投影截断偏差。计算成本可通过分块或低秩核降低。[R1]

### D4. 有真实配对和重复响应的差分能量

若输入确实构成 $w\pm h_ie_i\pm h_je_j$ 的精确匹配网格，混合二阶差分商为：

$$
\widehat D_{ij,h}(w)
=\frac{Y_{++}-Y_{+-}-Y_{-+}+Y_{--}}{4h_ih_j},\quad i\ne j.
\tag{D8}
$$

若上述四点各有两套**条件独立响应重复** $r=1,2$，直接使用交叉乘积：

$$
\boxed{
\widehat M^{\mathrm{diff}}_{ij}
=\frac1K\sum_{k=1}^K
\widehat D^{(1)}_{ij,h}(w_k)
\widehat D^{(2)}_{ij,h}(w_k).
}
\tag{D9}
$$

其条件期望是**有限步长差分近似的平方**；在平滑性与 $h\to0$ 的合理极限下才趋向真实二阶能量。单次 $\widehat D^2$ 带有额外观测噪声方差；若噪声方差已知，也可先推导去偏估计。对角项换成二阶中心差分。

**CR4 的独立随机连续 X 和 CR5 的有限 500-bank 一般不提供所需的严格配对网格。** 随机稀疏邻居不能直接当成等步长的差分角点；否则相当于隐藏的局部拟合/插值步骤，已改变信息与统计假设。[R4,R5]

### 10.1 第三类为什么不能无条件替代第二类？

- **维数与截断。** 9–10 维全变量正交级数阶数爆炸，且二阶导数对高频估计更敏感；直接估计整体泛函可能高效，但需要具体平滑度/维数条件，而不是天生免疫维数灾难。
- **边界/密度。** 弱形式 D2 要正确处理支持域边界项；已知平滑密度不是所有情形都成立，固定离散 X 时由 WY **无法无模型假设识别连续 X 方向的真实局部导数**。
- **独立性与方差。** 同被试重复试次共享 Z；独立观测对不能无条件跨所有 trial 拼接。直接 U-stat 有时出现负值，虽然真实 $M\ge0$，要保留原始估计而非静默 clip。
- **输出语义。** 如果只观测 choice，则 D1–D4 针对条件概率 $q$ 的导数；不能声称自动得到 logit Hessian。
- **近似目标诚实性。** D1 是投影函数的 Hessian 能量，D2 是 Hessian 自身的有限测试空间投影能量；**它们不是同一个有限维近似，也都不是未经证明的完整 $M$**。

### 10.2 建议三类方法的首轮实验顺序

| 顺序 | 方法 | 实验目标 |
|---|---|---|
| P0 | **第一类 B1+B7** 冻结 GELU decoder AD + 经验支持积分 | 网络自身的完整 $M^{UU},M^{UX},M^{XX}$ 基准 |
| P0 | **第一类 B9** 连续/离散 X 的边界与支持诊断 | 区分 CR4 合法盒、CR5 固定题库、Z 联合支持 |
| P0 | **第二类 A1/A4/A5** 局部多项式、sieve、KRR | 仅 WY 时的函数拟合 + 二阶导数保真 |
| P1 | **第二类 A3/A6/A7** GAM、GP、从 WY 训练 smooth MLP | 比较模型平滑偏差和预测/导数准确性的差距 |
| Research | **第三类 D1/D3** 直接 U-stat / quadratic-kernel | 已知连续分布与独立样本单位条件下直接估能量 |
| Research | **第三类 D2** weak-form | 先解决有界 X 的边界和已知密度要求 |
| Design-only | **第三类 D4** 配对差分 | 仅在真正网格/重复设计满足时使用 |

**对旋转的共同原则：** 三类方法都应在固定物理输入支持、同一参考测度、同一坐标/输出语义下比较。先评价 $\widehat M_{ij}$ 是否接近对应的 oracle $M_{ij}$，再评价 Part III 的稀疏性损失能否恢复参数轴；两道检验不可混为一谈。


---

# Part III. 基于平均平方曲率能量可以进一步计算的指标清单

**研究起点：** 已经从解析式、oracle 数值积分或仅凭输入–输出矩阵得到逐元素二阶曲率能量矩阵：

$
\boxed{M_{ij}=\mathbb E_{W\sim P_W}\left[\left(\frac{\partial^2f(W)}{\partial W_i\partial W_j}\right)^2\right]\ge0,\quad M=M^\top.}
\tag{21}
$

目标是进一步把这个矩阵汇总为**有含义、可比较、可复现的标量指标**：总曲率大小、单元稀疏性、能量集中度、参数轴分工、矩阵有效秩，以及能够用于搜索旋转 $R$ 的候选 loss。

**注意两层目标：** （i）描述当前坐标下的曲率矩阵；（ii）寻找有利于生成函数简洁表达的参数坐标。二者不能混为一谈。Shannon entropy、Gini、HHI、Hoyer、effective rank 等是成熟通用的统计/矩阵指标，但把它们应用于 $M$ 并不构成“必然恢复真实参数轴”的定理；下面专门设计的自曲率 tail、UX 分组损失等都需在 C1/C2/C3/CR3/CR4/CR5 上实验验证。

## 11. 统一记号与避免错误汇总的原则

取 $W=(U,X)$、$U\in\mathbb R^k$、$X\in\mathbb R^p$，划分：

$
M=\begin{pmatrix}M^{UU}&M^{UX}\\(M^{UX})^\top&M^{XX}\end{pmatrix},
\qquad M^{UU}\in\mathbb R_+^{k\times k},\
M^{UX}\in\mathbb R_+^{k\times p}.
\tag{22}
$

**11.1 对称矩阵是否重复计数。** 对任意 $i\ne j$，$M_{ij}=M_{ji}$。如果研究总 Frobenius 曲率，必须双计数非对角项：

$
\sum_{ij}M_{ij}=\sum_iM_{ii}+2\sum_{i<j}M_{ij}.
$

但如果问“有多少种独立变量对发挥作用”，应优先用半向量化 $\mathrm{vech}(M)$（仅 $i\le j$），每个无序变量对计数一次。对于矩形 $M^{UX}$，直接使用全部 $kp$ 个元素，不应再重复加入 $M^{XU}$。在 entropy、Hoyer、Gini、top-$k$ 等计算中，**重复计数约定会实质改变数值**，必须记录为方法配置。

**11.2 非负能量向量与归一化。** 由指定矩阵区块和计数规则提取向量 $v=(v_1,\ldots,v_n)\ge0$，令

$
S=\sum_{e=1}^n v_e,\qquad p_e=\frac{v_e}{S}\ (S>0),\qquad
v_{(1)}\ge v_{(2)}\ge\cdots\ge v_{(n)}.
\tag{23}
$

在 $S=0$ 时输出独立状态 **all-zero**：原始总能量为 0，但归一化熵、Gini、HHI、Hoyer 等不定义；$n=1$ 时 Hoyer 分母也为 0。统一定义 $0\log0:=0$。

**11.3 元素非负不等于矩阵半正定。** $M=\mathbb E[H\odot H]$ 非负、对称，**通常不保证半正定**，不等于 $\mathbb E[H^\top H]$。做有效秩/谱指标时优先用奇异值 SVD，而不是把 $M$ 的特征值强行当概率。

**11.4 尺度、Y 类型和输入测度固定。** $M_{ij}$ 的量纲是 $f^2/(W_i^2W_j^2)$。若 $W=\mu+D W_{\mathrm{std}}$，则 $M^{\mathrm{std}}_{ij}=D_{ii}^2D_{jj}^2M^{\mathrm{physical}}_{ij}$。必须标明 logit 还是概率输出、X 和 Z 的变换、期望分布 $P_W$，以及不同试次/被试是否等权。不可对带量纲、尺度不同的 $M$ 不作说明就比较稀疏性。

## 12. 第一类：总能量、对角和非对角比例（E01–E10）

记 $d=\dim W$，$E=\sum_{ij}M_{ij}$：

| ID | 指标 | 数学公式 | 含义 / 在正交旋转中的地位 |
|---|---|---|---|
| E01 | Total Hessian energy | $E=\sum_{ij}M_{ij}=\mathbb E\|H\|_F^2$ | 总二阶曲率；**整体正交旋转不变** |
| E02 | Mean cell energy | $E/d^2$ | 平均有序单元能量；跨维数时需注意计数规则 |
| E03 | Integrated Hessian norm | $\sqrt{E}$ | Hessian 的积分 $L_2$ 范数；旋转不变 |
| E04 | Self-curvature energy | $E_{\mathrm{diag}}=\sum_iM_{ii}=\operatorname{tr}(M)$ | 轴自身曲率总量；随旋转可变 |
| E05 | Mixed-curvature energy | $E_{\mathrm{off}}=2\sum_{i<j}M_{ij}$ | 混合二阶作用总量；随旋转可变 |
| E06 | Self fraction | $r_{\mathrm{diag}}=E_{\mathrm{diag}}/E$ | 自曲率占比，**不能直接断言越大越好** |
| E07 | Mixed fraction | $r_{\mathrm{off}}=E_{\mathrm{off}}/E$ | 混合曲率占比，**不能直接断言越小越真** |
| E08 | Maximum energy | $\max_{ij}M_{ij}$ | 最强二阶作用 |
| E09 | Median / quantile | $\mathrm{median}(v)$、$Q_q(v)$ | 典型与尾部作用量 |
| E10 | Frobenius norm of M | $\|M\|_F=(\sum_{ij}M_{ij}^2)^{1/2}$ | 能量矩阵自身的 $L_2$，**不是** $\mathbb E\|H\|_F^2$；可随旋转变化 |

分块总量：

$
\begin{aligned}
E_{UU}&=\sum_{a,b}M^{UU}_{ab},&
E_{UX}&=\sum_{a,j}M^{UX}_{aj},&
E_{XX}&=\sum_{i,j}M^{XX}_{ij},\\
E_{\mathrm{all}}&=E_{UU}+2E_{UX}+E_{XX}.
\end{aligned}
\tag{24}
$

**只对 $U$ 做正交旋转并固定 $X$ 与输入分布时，$E_{UU}$、$E_{UX}$、$E_{XX}$ 分别恒定。** 因此只用区块总能量或三者的比例不能选出 $R$。可以变化的是 UU 对角/非对角拆分，以及 UX 每行和每个元素的分布。

## 13. 第二类：稀疏性、集中度、有效作用项数（S01–S19）

以下使用 §11.2 的 $v$、$p$ 与从大到小排序。无额外说明时，**越小代表越集中**适用于 entropy/effective number/tail，**越大代表越集中**适用于 HHI/Gini/Hoyer/top-$k$。

### 13.1 信息论与不均匀程度

| ID | 指标 | 公式 | 含义 / 优化方向 |
|---|---|---|---|
| S01 | Shannon entropy | $H_1=-\sum_ep_e\log p_e$ | 小：集中于较少作用项 |
| S02 | Shannon effective number | $N_1=\exp(H_1)$ | 小：有效作用项数少 |
| S03 | HHI / Simpson concentration | $\mathrm{HHI}=\sum_ep_e^2$ | 大：少数大值主导 |
| S04 | Inverse participation ratio | $N_2=1/\sum_ep_e^2$ | 小：有效主导项数少 |
| S05 | Rényi entropy | $H_\alpha=\frac{\log\sum_ep_e^\alpha}{1-\alpha}$，$\alpha>0,\alpha\ne1$ | 不同 $\alpha$ 调整对小项的关注 |
| S06 | Rényi effective number | $N_\alpha=(\sum_ep_e^\alpha)^{1/(1-\alpha)}$ | 有效项数；包含 $N_1$ 的连续极限 |
| S07 | Normalized Shannon entropy | $H_1/\log n$，$n>1$ | 归一化均匀度，取值 $[0,1]$ |
| S08 | Gini coefficient | $\frac{\sum_{a,b}|v_a-v_b|}{2n\sum_ev_e}$ | 大：能量分布不均匀 |
| S09 | Hoyer sparsity | $\frac{\sqrt n-\|v\|_1/\|v\|_2}{\sqrt n-1}$，$n>1$ | 大：更稀疏；标准来源 [R13] |

如果能量恰好平均分配到 $m$ 个非零项，其他项为零，则 $N_1=N_2=m$；但在一般不均匀分布下两个“有效数量”不同。所有指标只说明能量集中性，不自动说明正确的生成参数表示。

### 13.2 头部/尾部与可微近似

| ID | 指标 | 公式 | 含义 / 优化方向 |
|---|---|---|---|
| S10 | Top-$k$ mass | $\mathrm{Top}_k=\sum_{e\le k}v_{(e)}/S$ | 大：前 $k$ 项覆盖更多能量 |
| S11 | Tail-$k$ mass | $\mathrm{Tail}_k=\sum_{e>k}v_{(e)}/S$ | 小：能量集中在前 $k$ 项 |
| S12 | Absolute tail sum | $\sum_{e>k}v_{(e)}$ | 小：残余原始曲率能量更小 |
| S13 | Tail-$L_2$ on M | $(\sum_{e>k}v_{(e)}^2)^{1/2}$ | 小：**平方的是已积分的 $M_{ij}$** |
| S14 | Concave power penalty | $\sum_e v_e^\alpha$，$0<\alpha<1$ | 固定 $S$ 时，小：鼓励集中 |
| S15 | Log-sum penalty | $\sum_e\log(v_e+\epsilon)$，$\epsilon>0$ | 固定 $S$、固定 $n$ 时，小：鼓励集中 |
| S16 | Absolute threshold support | $\sum_e{\bf1}(v_e>\delta)$ | 小：超过绝对阈值的项少 |
| S17 | Relative threshold support | $\sum_e{\bf1}(p_e>\delta_p)$ | 小：超过相对份额阈值的项少 |
| S18 | Coverage count | $K_c=\min\{k:\sum_{e\le k}p_{(e)}\ge c\}$ | 小：较少项可覆盖 $c$ 比例 |
| S19 | $L_2/L_1$ concentration | $\|v\|_2/\|v\|_1$ | 大：更集中；与 HHI 的平方根一致 |

**S11 vs S13：** S11/S12 累积的是各项 $M_{ij}=\mathbb E[H_{ij}^2]$，而 S13 中有 $(M_{ij})^2$，不能把 S13 写作 $\mathbb E[H_{ij}^4]$：一般 $(\mathbb E[H_{ij}^2])^2\ne\mathbb E[H_{ij}^4]$。

任何带有 $k,\delta,\delta_p,c,\alpha,\epsilon$ 的指标都要先记录超参数，不可观察 oracle $R$ 之后再反向挑选对该模型最有利的值。

### 13.3 “有没有”：软支持度（Soft Support / Soft-$L_0$）（S20–S22）

**动机。** 如果关心的不是交互作用的强弱，而是稀疏性——“有没有”，思路应该改变：**不再直接把 Hessian 的大小当成简单性，而是把每个 Hessian 元素看作一条潜在的连接，判断这条连接是否存在。** 也就是把连续的能量矩阵转换成一张近似的 0–1 结构矩阵（Soft Support Matrix），然后最小化非零连接的数量。这比直接最小化平均绝对曲率或平均平方曲率更接近“用尽可能少的非零二阶项描述函数”这一目标（见项目文档 03 的简单性原则）。

理想情形下想定义的是 Support（非零结构），而不是 Magnitude（强度）：

$
S_{ij}=\begin{cases}0,&H_{ij}(W)\equiv0\\ 1,&H_{ij}(W)\not\equiv0\end{cases}
$

- $H_{ij}\equiv0$：这两个变量之间不存在对应的二阶作用；
- $H_{ij}\not\equiv0$：存在二阶作用，不管强度是 0.1 还是 100。

但严格的 0–1 指标不连续，难以用于优化旋转矩阵 $R$（S16、S17 的硬阈值计数同理）。因此需要一个连续近似。**平均平方 Hessian 能量 $M_{ij}$ 只作中间统计量；最终损失应尽可能近似非零连接的数量。**

#### S20. Soft-$L_0$（单一阈值 $\epsilon$）

对每个元素的均平方能量 $M_{ij}(R)=\mathbb E_W[H^{(R)}_{ij}(W)^2]$，用饱和函数映射到 $[0,1]$：

$
\boxed{S^{(\epsilon)}_{ij}(R)=\frac{M_{ij}(R)}{M_{ij}(R)+\epsilon^2}}
\qquad\Longrightarrow\qquad
\boxed{L_{\mathrm{support}}(R)=\sum_{i\le j}S^{(\epsilon)}_{ij}(R)}
\tag{S20}
$

$\epsilon>0$ 是“曲率小到什么程度算没有”的尺度参数。三条性质：

$
M_{ij}=0\Rightarrow S_{ij}=0,\qquad
M_{ij}\gg\epsilon^2\Rightarrow S_{ij}\approx1,\qquad
M_{ij}\ll\epsilon^2\Rightarrow S_{ij}\approx0.
$

当 $\sqrt{M_{ij}}=\epsilon$ 时恰好 $S_{ij}=0.5$，所以 $\epsilon$ 决定连续曲率到近似 0–1 的转换尺度。$\epsilon=0.10$ 时（曲率已标准化）：

| RMS 曲率 $\sqrt{M_{ij}}$ | 0.00 | 0.01 | 0.03 | 0.10 | 0.30 | 1.00 | 3.00 |
|---|---|---|---|---|---|---|---|
| Soft support | 0.000 | 0.010 | 0.083 | 0.500 | 0.900 | 0.990 | 0.999 |

对称 Hessian 只统计上三角，避免重复计数（见 §11.1）；$UX$ 块是矩形，取全部 $kp$ 个元素。**统计哪一组独立元素 $\mathcal I$ 是方法配置的一部分，必须记录**：UU 取上三角（含对角），UX 取全部。

**与强度类指标的区别：** 设两个旋转得到下面的 RMS 矩阵（只看独立元素，即上三角）。

| | RMS 矩阵 | 独立元素的总平方能量 | $L_{\mathrm{support}}$（$\epsilon$ 很小） |
|---|---|---|---|
| 旋转 A | $\begin{pmatrix}0&2&0\\2&0&0\\0&0&0\end{pmatrix}$：一个非零交互项 | $2^2=4$ | $\approx1$ |
| 旋转 B | $\begin{pmatrix}\sqrt2&0&0\\0&\sqrt2&0\\0&0&0\end{pmatrix}$：两个非零自曲率项 | $2+2=4$ | $\approx2$ |

两者的总平方能量相等，最小化总体均方 Hessian 能量无法区分它们；Soft-$L_0$ 偏好 A，因为它问的是“需要多少个非零二阶项来描述函数”，而不是交互作用有多强。（注意计数约定：若 A 的对称位置双计数，A 的矩阵总能量是 8，而 B 是 4。比较时必须使用同一约定，见 §11.1。）

#### 整条连接是否存在，还是只在部分试次存在？（方法 A / 方法 B）

两个问题并不相同：

$
\boxed{S^{A}_{ij}=\frac{\mathbb E[H_{ij}^2]}{\mathbb E[H_{ij}^2]+\epsilon^2}}
\qquad\qquad
\boxed{S^{B}_{ij}=\mathbb E\!\left[\frac{H_{ij}(W)^2}{H_{ij}(W)^2+\epsilon^2}\right]}
$

| 特点 | 方法 A：先求期望，再近似二值化 | 方法 B：先近似二值化，再求期望 |
|---|---|---|
| 含义 | 这个 Hessian 元素作为函数，是否总体上存在非零作用 | 在多大比例的输入条件下，该元素有明显的非零作用 |
| 关注 | 函数是否存在该二阶作用 | 该二阶作用在多少输入点活跃 |
| 少量极大曲率 | 可能接近 1 | 仍可能较小 |
| 结构识别 | 更合适 | 次选 |
| 条件性作用 | 不能直接反映出现比例 | 更合适 |
| 旋转计算 | 可以预计算二阶矩张量（§17.1） | 通常需要逐点计算 |

**S20 默认用方法 A。** 但对 CR5 的 gain-only、loss-only、gamma-neutral 等试次子集，方法 B 值得另外测试，因为某些二阶作用可能只在特定刺激条件中出现。

#### 标准化：先统一量纲，再对区块做相对尺度归一化

不能直接拿原始 Hessian 比较，因为 $\lambda,\gamma,\tau$ 与 $X$ 的量纲不一致（§11.4）。步骤：

1. 对真实参数 $T$（或 $Z$）先标准化，再旋转；$X$ 的导数也使用固定的标准化尺度。CR4 与 CR5 采用相同的尺度定义，以便比较。
2. 计算每个旋转的 RMS：$D_{ij}(R)=\sqrt{\mathbb E[H^{(R)}_{ij}(W)^2]}$。
3. 对每个区块（UU、UX 分别）作相对尺度归一化：

$
\widetilde M_{ij}(R)=\frac{M_{ij}(R)}{E_{\mathrm{blk}}},\qquad
E_{\mathrm{blk}}=\mathbb E\big[\|H_{\mathrm{blk}}(W)\|_F^2\big]=\sum_{(i,j)\in\mathrm{blk}}M_{ij}.
$

由 §17，在固定输入分布且只对 $U$ 作正交旋转时，$E_{UU}$、$E_{UX}$ 各自不随 $R$ 变化，所以分母不随旋转变化，不会因为旋转本身改变归一化标准。

4. 用 $\widetilde M_{ij}$ 代替 $M_{ij}$ 代入 S20：

$
\boxed{L(R)=\sum_{(i,j)\in\mathcal I}\frac{\widetilde M_{ij}(R)}{\widetilde M_{ij}(R)+\epsilon^2}}
$

#### 为什么 $\epsilon$ 不能与 $R$ 一起优化

若损失为 $L(R,\epsilon)=\sum_{i\le j}\frac{M_{ij}(R)}{M_{ij}(R)+\epsilon^2}$ 并允许同时优化 $R$ 和 $\epsilon$，只要存在正的 $M_{ij}$，就有 $\partial L/\partial\epsilon<0$；算法会不断增大 $\epsilon$，当 $\epsilon\to\infty$ 时 $L\to0$，与旋转无关。所以 $\epsilon$ 必须固定、约束，或者改变指标定义。区分两件事：

- $\epsilon$：定义什么程度的二阶作用可以视为不存在；
- $R$：真正要优化的旋转，使非零二阶作用尽可能少。

三种选择 $\epsilon$ 的方法：

| 方法 | 原理 | 评价 |
|---|---|---|
| 固定阈值 | 例如 $\epsilon=0.05$ | 最简单，但可能任意 |
| 相对阈值 | 先按区块总能量标准化（见上） | 推荐作为基础 |
| 多阈值积分 | 在多个 $\epsilon$ 上评估并整合稀疏性 | 最值得研究 |

**固定标准化阈值的扫描。** 在标准化之后系统测试 $\epsilon\in\{0.01,0.03,0.05,0.1,0.2\}$。这些只是候选值，不是经过证明的最优值。如果真实参数方向在所有这些阈值下都表现良好，说明指标有一定稳健性；如果只在某个极其精确的阈值下成功，就值得怀疑。

#### S21. 多阈值 Soft-$L_0$

不选择单个 $\epsilon$，而在一个范围内对 Soft-$L_0$ 按 $\log\epsilon$ 平均：

$
\overline S_{ij}(R)=\frac{1}{\log(\epsilon_{\max}/\epsilon_{\min})}\int_{\epsilon_{\min}}^{\epsilon_{\max}}\frac{M_{ij}(R)}{M_{ij}(R)+\epsilon^2}\,d\log\epsilon .
$

该积分有解析解。令 $m=M_{ij}(R)$：

$
\boxed{\overline S_{ij}(R)=1-\frac{\log\!\left(\dfrac{m+\epsilon_{\max}^2}{m+\epsilon_{\min}^2}\right)}{2\log(\epsilon_{\max}/\epsilon_{\min})}}
\tag{S21}
$

仍满足 $\overline S_{ij}(0)=0$ 与 $\lim_{m\to\infty}\overline S_{ij}(m)=1$。优势：不必把“有没有”的判定完全押在单一阈值上。局限：仍需规定 $\epsilon_{\min},\epsilon_{\max}$，并没有彻底消除超参数。它依然可微，可用于优化 $R$。最值得优先尝试。

#### S22. 完全不用 $\epsilon$：有效支持度（Effective Support）

令 $m_k$ 表示展平后的各个 Hessian 平方曲率能量：

$
N_{\mathrm{eff}}=\frac{(\sum_km_k)^2}{\sum_km_k^2}
$

等价于 S04（$N_2=1/\sum p_e^2$）。只有一个元素非零时为 1；五个完全相等的非零元素时为 5；但若五个元素强度差距很大，它也可能接近 1。因此它衡量**有效能量集中度**，并不等价于真实非零元素数量。作无需 $\epsilon$ 的对照指标。

#### $\epsilon$ 过小或过大的取舍

对 $S(m)=m/(m+\epsilon^2)$，$\dfrac{dS}{dm}=\dfrac{\epsilon^2}{(m+\epsilon^2)^2}$。

- 当所有非零的 $m$ 都远大于 $\epsilon^2$ 时，Soft-$L_0$ 接近 1，梯度趋于平坦。
- 当所有 $m$ 都远小于 $\epsilon^2$ 时，$S(m)\approx m/\epsilon^2$，退化为近似线性的曲率能量指标；若把整个 UU 或 UX 块的能量相加，损失就接近旋转不变量（§17），无法选择 $R$。

所以不能只选择非常小或非常大的阈值。

#### 一个必须承认的数学事实

在 CR4 这类解析函数中，一个 Hessian 元素通常要么恒等于 0，要么在连续输入分布下几乎处处非零。严格的 0–1 支持度会使优化目标极不连续。因此，Soft-$L_0$ **不是数学上精确的结构计数**，而是在指定曲率尺度下的近似结构计数。

#### Oracle 层面的实验设计与旋转优化

1. **预计算 Hessian 二阶矩张量**（原 §17.1）：$K^{UU}_{abcd}=\mathbb E[H_{ab}H_{cd}]$；UX 块对每个刺激坐标 $\ell$：$K^{UX,\ell}_{ab}=\mathbb E[H_{a\ell}H_{b\ell}]$。对原始解析函数只需积分一次。
2. **用张量运算快速计算任意旋转的 $M(R)$：** $M_{ij}(R)=\sum_{abcd}R_{ia}R_{jb}R_{ic}R_{jd}K_{abcd}$，不必每次优化都重新积分。
3. **应用 S20 或 S21**，得到稀疏性损失 $L(R)$。
4. **多初始点优化。** $R$ 用平面旋转（Givens）参数化；例如 3 维（CR4/CR5 的 3 个参数）：$R(\theta_1,\theta_2,\theta_3)=R_{12}(\theta_1)R_{13}(\theta_2)R_{23}(\theta_3)$，一般 $k$ 维有 $k(k-1)/2$ 个角。用 L-BFGS-B 等优化器，多个随机起点以降低局部最优风险：$R^\ast=\arg\min_{R^\top R=I}L_{\mathrm{support}}(R)$。
5. **与真实参数方向比较**，允许参数轴的排列与正负号变化。
6. **扫描多个阈值**（如 $\epsilon=0.01,0.03,0.1$），观察真实坐标是否在一系列阈值下都保持优势，而非只在一个精心挑选的阈值下成功。

#### 三类指标的优先级与原则

| 指标 | 目的 | 优先级 |
|---|---|---|
| Soft-$L_0$（S20/S21） | 近似计算非零 Hessian 元素的数量 | 最高 |
| $L_p$，$0<p<1$（S14） | 用非凸准范数鼓励稀疏性 | 第二 |
| Hoyer / Gini / effective support（S08、S09、S22） | 衡量曲率能量集中于多少元素 | 第三 |

三者并不等价：Hoyer 与 Gini 更强调**能量集中**，Soft-$L_0$ 更接近“有没有”。

对 CR4/CR5，应把 **logit Hessian** 放在第一优先级：概率输出的 sigmoid 链式法则（式 (3)）会引入大量额外的非零交叉项，可能掩盖生成模型本身的简单结构。

**原则：不能为了让 CR4/CR5 的真实参数旋转获胜，反过来挑选最有利的 $\epsilon$。** 应事先固定阈值选择规则，并检查该规则能否在 C1、C2、C3、CR3、CR4、CR5 上共同成立。

## 14. 第三类：逐轴、变量对、分组与交互结构（G01–G18）

以 $B=M^{UX}\in\mathbb R_+^{k\times p}$ 为例，定义：

$
r_i=\sum_{a=1}^{p}B_{ia},\quad c_a=\sum_{i=1}^{k}B_{ia},\quad
S_B=\sum_ir_i=\sum_ac_a.
\tag{25}
$

| ID | 指标 | 公式 / 操作 | 解释 |
|---|---|---|---|
| G01 | Axis self-curvature | $a_i=M^{UU}_{ii}$ | 潜变量轴自身非线性 |
| G02 | Axis UU energy | $\sum_jM^{UU}_{ij}$ | 该轴与所有参数的二阶作用（含自己） |
| G03 | Axis UX energy | $r_i$ | 该轴与全部刺激的二阶作用 |
| G04 | Stimulus UX energy | $c_a$ | 一个刺激属性对应的总体 UX 作用 |
| G05 | UX axis share | $r_i/S_B$ | UX 能量如何分配给各潜变量轴 |
| G06 | UX feature share | $c_a/S_B$ | UX 能量如何分配给各刺激变量 |
| G07 | UX row entropy | $-\sum_a(B_{ia}/r_i)\log(B_{ia}/r_i)$ | 每个参数轴的刺激作用是否分散 |
| G08 | UX row effective number | $\exp(\mathrm{RowEntropy}_i)$ | 每个参数轴有效作用于多少刺激维度 |
| G09 | UX row tail-$k$ | 对每行 $B_{i:}$ 算 S11 | 一条轴是否只主要作用于少数刺激维度 |
| G10 | UX row Hoyer | 对 $B_{i:}$ 算 S09 | 逐轴的作用稀疏性 |
| G11 | UX column entropy / Hoyer | 对每列 $B_{:a}$ 算 S01/S09 | 每个刺激是否仅受少数轴影响 |
| G12 | Row-total concentration | 对 $(r_1,\ldots,r_k)$ 算 S01/S03/S09 | 所有 UX 能量是否集中在少数潜变量轴 |
| G13 | Active-axis count | $\sum_i{\bf1}(r_i>\delta)$ | 有明显刺激交互的轴数 |
| G14 | Predefined group energy | $B_{ig}=\sum_{a\in\mathcal G_g}B_{ia}$ | 参数轴 × 刺激属性组的曲率能量 |
| G15 | Group-block sparsity | 对 $\mathrm{vec}([B_{ig}])$ 算 S 类指标 | 是否只有少量轴–刺激组发挥作用 |
| G16 | Row-pattern cosine | $\cos(B_{i:},B_{j:})$ | 两个潜变量轴的刺激作用模式是否相似 |
| G17 | Row-pattern distinctness | $\frac{2}{k(k-1)}\sum_{i<j}(1-\cos(B_{i:},B_{j:}))$ | 轴分工的区分度，**未必越大越好** |
| G18 | Conditional-group loss | $M^{(s)}_{ij}=\mathbb E[H_{ij}^2\mid S=s]$，再算 $L(M^{(s)})$ | 将同一简单性规则应用于多个试次类别 |

**遇到 $r_i=0$：** 保存“inactive axis”标记；该轴的归一化行熵、Hoyer、cosine 不能按普通活跃行解释。聚合 row sparsity 时需事先规定活跃轴等权、总能量加权，或把无效轴单独作为一个目标。

若比较不同试次类型，可预先选择公共指标函数 $\ell$ 与非负权重 $\omega_s$：

$
\boxed{L_{\mathrm{group}}(R)=\sum_s\omega_s\,\ell(M^{(s)}(R)).}
\tag{26}
$

因为 $\ell$ 通常非线性，一般有 $\sum_s\omega_s\ell(M^{(s)})\ne \ell(\sum_s\omega_sM^{(s)})$。这对应**对所有试次类别使用相同的简单性原则，再联合约束旋转**，而不是为每类试次设计不同规则。组可重叠时须注意，简单按 $P(S=s)$ 加权并不自动重构整体分布。

对于 CR4/CR5，可按收益、损失、概率属性（也可细分选项 A/B）事先确定 stimulus groups；CR3/CR5 固定题库还可以用 gain-only、loss-only、gamma-neutral 等条件子集。

## 15. 第四类：奇异值、低秩与有效维数（R01–R08）

对任意选定的矩阵子块 $B$（例如 $M^{UU}$ 或矩形 $M^{UX}$）做 SVD，令 $\sigma_1\ge\cdots\ge\sigma_r\ge0$，$s_j=\sigma_j/\sum_{\ell}\sigma_\ell$：

| ID | 指标 | 公式 | 解释 / 注意 |
|---|---|---|---|
| R01 | Matrix rank | $\operatorname{rank}(B)$ | 精确秩，受数值噪声影响大 |
| R02 | Effective rank | $\exp(-\sum_js_j\log s_j)$ | 基于奇异值熵的有效秩 [R14] |
| R03 | Stable rank | $\|B\|_F^2/\|B\|_2^2=\sum_j\sigma_j^2/\sigma_1^2$ | 奇异模式集中程度 |
| R04 | Nuclear norm | $\|B\|_*=\sum_j\sigma_j$ | 常用于矩阵低秩正则化，依赖尺度 |
| R05 | Top singular share | $\sigma_1/\sum_j\sigma_j$ | 最大奇异方向的相对份额 |
| R06 | Top-$k$ spectral energy | $\sum_{j\le k}\sigma_j^2/\sum_j\sigma_j^2$ | 前 $k$ 个模式解释的 Frobenius 能量 |
| R07 | Best rank-$k$ tail | $(\sum_{j>k}\sigma_j^2)^{1/2}$ | 最佳秩-$k$ 近似的 Frobenius 残差 |
| R08 | Thresholded rank | $\#\{j:\sigma_j/\sigma_1>\delta_\sigma\}$ | 依赖阈值的近似秩 |

**不要把能量矩阵的低秩理解为生成函数只有这么多个真实参数。** 这个矩阵是在 Hessian **逐元素平方后再取期望**得到的；谱结构只是能量模式的线性代数特征，是否反映真实函数可分性需要单独证明。

## 16. 第五类：直接用于旋转的候选简单性损失（L01–L08）

这些指标是从前四组进一步**设计的研究假设**。不是所有小 loss 都代表简单，也不是每种真生成模型都必须偏好其原始参数轴。

### L01. 自曲率尾部：寻找自身线性轴（强烈推荐）

令 $a_i(R)=M^{UU}_{ii}(R)$，按降序 $a_{(1)}\ge\cdots\ge a_{(k)}$：

$
\boxed{L_{\mathrm{self},q}(R)=\sum_{i=q+1}^{k}a_{(i)}(R).}
\tag{27}
$

意思是：允许最多 $q$ 个轴有自身非线性，希望其它轴的二阶曲率能量为零。

对于 CR4/CR5 的真实三参数 **logit** 函数，$\partial_{\lambda\lambda}f=\partial_{\tau\tau}f=0$，所以在真实物理参数坐标下 $L_{\mathrm{self},1}(I)=0$。但**真轴未必是唯一零损失解**；未经对齐的 NP latent $Z$ 也不自动继承这一性质。

### L02. 自曲率集中度

$
L_{\mathrm{self,log}}(R)=\sum_i\log(\epsilon+a_i(R)),\qquad
L_{\mathrm{self,entropy}}(R)=H_1\bigl(a_i(R)/\sum_ja_j(R)\bigr).
\tag{28}
$

前者需固定 $\epsilon$，后者需单独记录总自身曲率 $E_{\mathrm{diag}}(R)$，避免对角总和变化被归一化隐藏。

### L03. UU 非对角能量（必须包括负面对照）

$
L_{UU,\mathrm{off}}(R)=2\sum_{i<j}M^{UU}_{ij}(R).
\tag{29}
$

$f(a,b)=ab$ 的 Hessian 纯非对角，但旋转 45° 后 $f=(u_1^2-u_2^2)/2$，Hessian 纯对角，式 (29) 会奖励旋转后的坐标。故**“最小化混合二阶作用”不能作为唯一简单性准则**。

### L04. UX 全局集中度

对 $v(R)=\mathrm{vec}(M^{UX}(R))$ 使用 S01/S03/S09/S11/S14/S15 等，特别是：

$
L_{UX,\mathrm{tail},q}(R)=\frac{\sum_{e>q}v_{(e)}(R)}{\sum_ev_e(R)}.
\tag{30}
$

由于 $E_{UX}$ 在正交旋转下不变，而 UX 各元素的能量分布可变，此项确实可以区分旋转；但是否喜欢真轴必须依赖 oracle landscape 检验。

### L05. UX 逐轴稀疏性

$
L_{UX,\mathrm{rows}}(R)=\sum_{i:r_i(R)>0}\omega_i\,\mathrm{Tail}_{q_i}(M^{UX}_{i:}(R)).
\tag{31}
$

可换 row entropy / Hoyer（Hoyer 越大越稀疏；作为 loss 可取 $1-\mathrm{Hoyer}$），并需明确 inactive row 处理规则及 $\omega_i$。

### L06. 刺激组聚合稀疏性

将 X 属性组 $\mathcal G_g$ 固定：

$
B_{ig}(R)=\sum_{a\in\mathcal G_g}M^{UX}_{ia}(R),\qquad
L_{\mathrm{block}}(R)=\mathrm{Tail}_q(\mathrm{vec}(B(R))).
\tag{32}
$

使模型在“参数 × 刺激类别”层面只有少量关键作用，而不是强制每个单一原始刺激坐标都稀疏。

### L07. 多试次组共同约束

$
L_{\mathrm{multi}}(R)=\sum_s\omega_s\ell(M^{(s)}(R)).
\tag{33}
$

同一 $\ell$ 对各组完全相同。各组权重、样本覆盖、重叠规则及等权还是概率加权都必须明确。

### L08. 多目标 / Pareto

定义 $\mathbf L(R)=(L_{\mathrm{self}},L_{UX,\mathrm{rows}},L_{UU,\mathrm{off}},L_{\mathrm{block}},\ldots)$，先独立比较这些分量的解析 oracle landscape、轴识别性和假极值，再考虑基于固定规则的加权/字典序/Pareto。不要只因某个 CR4 旋转角成功就反向调节指标权重。

### L09. 软支持度损失（Soft-$L_0$ support）

在 §13.3 标准化（UU、UX 各按区块总能量归一化）之后：

$
\boxed{L_{\mathrm{support}}(R)=\sum_{(i,j)\in\mathcal I}S_{ij}(R),}
$

其中 $S_{ij}$ 取 S20（固定 $\epsilon$，须事先固定并扫描 $\epsilon\in\{0.01,0.03,0.05,0.1,0.2\}$ 检查稳健性）或 S21（多阈值）；$\mathcal I$ 对 UU 取上三角，对 UX 取全部元素。意思是：直接近似“非零二阶作用项的个数”，而不是它们的强度。$\epsilon$ 不与 $R$ 一起优化；不得观察 oracle $R$ 之后挑选对真实坐标最有利的 $\epsilon$。

## 17. 旋转不变量与一个关键的数学限制

在固定物理函数、固定分布和固定 X 的前提下，$U=RZ$、$R^\top R=I$：

$
H_{UU}^{(R)}=RH_{ZZ}R^\top,\qquad
H_{UX}^{(R)}=RH_{ZX},\qquad H_{XX}^{(R)}=H_{XX}.
\tag{34}
$

| 汇总量 | 随 R 变化？ | 可单独搜索 R？ |
|---|---|---|
| $E_{UU},E_{UX},E_{XX}$ | **不变** | 否 |
| 这三个区块总量之间的比例 | **不变** | 否 |
| $E^{UU}_{\mathrm{diag}}$ 与 $E^{UU}_{\mathrm{off}}$ | 一般变化，但和固定 | 可作为候选 |
| $M^{UX}$ 的 Shannon、Hoyer、tail | 一般变化 | 可作为候选 |
| UX 行能量 $r_i$ 与行 sparsity | 一般变化 | 可作为候选 |
| UX **列能量** $c_a=\sum_iM^{UX}_{ia}$ | **不变** | 否 |
| $M^{XX}$ 全矩阵及其所有指标 | **不变** | 否 |
| $M^{UX}$ 的 effective rank | 不保证不变；也不保证偏好真轴 | 仅候选 |

### 17.1 仅有 M 不能准确推导 M(R)

不可以直接用 $RMR^\top$ 代替旋转后的平均平方 Hessian，因为旋转先对 Hessian 元素进行线性组合，**随后平方将产生交叉乘积**。

对 UU，必须知道四阶下标（但只是 Hessian 的二阶矩）张量：

$
C^{UU}_{abcd}=\mathbb E[H^{UU}_{ab}(W)H^{UU}_{cd}(W)],
$

$
\boxed{
M^{UU}_{ij}(R)=\sum_{a,b,c,d}R_{ia}R_{jb}R_{ic}R_{jd}\,C^{UU}_{abcd}.
}
\tag{35}
$

对 UX 的每个固定刺激坐标 $\ell$：

$
C^{UX,\ell}_{ab}=\mathbb E[H^{UX}_{a\ell}(W)H^{UX}_{b\ell}(W)],
$

$
\boxed{
M^{UX}_{i\ell}(R)=\sum_{a,b}R_{ia}R_{ib}\,C^{UX,\ell}_{ab}.
}
\tag{36}
$

原始 $M$ 只有 $\mathbb E[H_{ab}^2]$，缺失 $a,b$ 与 $c,d$ 不同的交叉二阶矩。要快速、精确比较多个旋转，应保留每个评估点的 Hessian、足够完整的 $C$ 张量，或在每次旋转后重新估计。若 Hessian 仅从 WY 估计，$C$ 本身也有估计误差。

## 18. 优先指标清单与未来输出协议

| 输入矩阵 / 子向量 | 第一批重点指标 | 首要检验 |
|---|---|---|
| UU 对角 | E04、S02、S09、S11、L01 | 恢复线性轴，是否有其它零 loss 旋转 |
| UU 非对角上三角 | E05、S02、S11、L03 | 45° 双线性反例是否失败 |
| UX 全矩阵 | E_UX、S02、S03、S09、S11、L04 | 全局作用稀疏性 |
| UX 各行 | G03、G07–G10、G12、L05 | 每轴分工是否明确 |
| UX 属性组 | G14、G15、L06 | 结构在收益/损失/概率组是否集中 |
| 条件试次组 | G18、L07 | 不同 stimulus subsets 能否联合约束 |
| UX 全矩阵 / UU 上三角（“有没有”） | S20、S21、L09；对照 S14、S22 | 在多个 $\epsilon$ 下真实坐标是否保持最优；规则能否在 C1/C2/C3/CR3/CR4/CR5 上共同成立 |
| UX 谱 | R02、R03、R06 | 补充低维模式诊断，不作为主要 loss |
| 全矩阵 | E01、E03 | 旋转不变性 sanity check |

**推荐输出而非新代码要求：** 保存每个模型、Y 类型、Hessian estimator 和 R 对应的 (i) 未截断 $M^{UU},M^{UX},M^{XX}$；(ii) RMS $D_{ij}=\sqrt{M_{ij}}$；(iii) 每个指标的值、计数约定、超参数、minimize/maximize 方向、valid/all-zero 标志；(iv) $P_W$ / 坐标版本 / subject split / seed；(v) oracle–blind 的事后比较。若直接二次泛函估计出现负的原始 $\widehat M$（估计误差导致），应单独保留原始估计，不可静默 clip 后假装是严格非负量。

**推荐实验顺序：** 先固定真实解析函数与积分点，对所有候选 loss 做一维 Givens 和一般 $k$ 维正交旋转 landscape；用 CR4/CR5 的物理真参数坐标（允许置换和符号等价）检验 oracle 目标，再以只有 $(W,Y)$ 的方法估计 $\widehat M$，比较估计矩阵与旋转损失是否仍保真。别把“某指标描述能量分布很漂亮”误认为“它必然恢复真实参数”。


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
- **[R13] Hoyer, P. O. (2004).** Non-negative matrix factorization with sparseness constraints. *Journal of Machine Learning Research, 5*, 1457–1469. [JMLR](https://www.jmlr.org/papers/v5/hoyer04a.html). **Hoyer 基于 L1/L2 的稀疏性指标**。
- **[R14] Roy, O., & Vetterli, M. (2007).** The effective rank: A measure of effective dimensionality. *15th European Signal Processing Conference (EUSIPCO)*. [Archive](https://zenodo.org/records/40328). **基于奇异值熵的有效秩**。

**文献到任务的映射：** 若问“这种量是否有数学先例”，看 [R1–R3]；若问“仅有 noisy WY 怎么估计”，看 [R5–R9] 并研究 [R1] 的直接二次泛函；若问“交互意义和可解释性”，看 [R2,R9,R10]；若问“需要额外 f 查询的导数方法”，看 [R11]。

**版本说明：** 本文只定义目标及综述方法，不宣称任何盲估计方法已经在 C1/C2/C3/CR3/CR4/CR5 上成功恢复最优旋转。后续实现应保留 oracle–blind 信息隔离和逐模型验证。
