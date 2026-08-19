# 2026-08-18 Fuzzy Dynamics 实验笔记

本文记录今天两次实验暴露出的建模、训练与评价问题。程序崩溃、Slurm、依赖和内存等运行问题不计入。

## 两次实验

| 实验 | 目录 | 目的与结论 |
|---|---|---|
| E0：问题实验 | `fuzzy_dynamics_20260818_065222_694752` | 流程跑通，但 state loss 失衡且 projector 出现坍缩，整体指标具有误导性 |
| E1：两项修复实验 | `fuzzy_dynamics_20260818_203127_038416` | 使用冻结共享 PCA projector 和 block-balanced loss；单步完整 state 预测明显改善，但 rollout 与语义分化仍未解决 |

两次实验都使用7,232条 SOCRATES、category-level split、968条验证数据和 seed 42。

### 指标如何理解

- `R² = 1`：预测与目标完全一致。
- `R² = 0`：只达到“对所有样本预测目标均值”的基线。
- `R² < 0`：比预测均值还差，不能视为学到了该目标。
- Overall R²：把全部坐标的残差平方和与目标方差汇总，目标方差大的 block 权重更高。
- Component Macro R²：先分别计算 `z/concept/belief/uncertainty` 的 R²，再对四项等权平均，更适合判断完整 state 是否都可预测。
- Rollout R²：从真实初始状态出发，后续反复使用模型自己的预测状态；它衡量递归稳定性，不等同于 one-step R²。

## 总体对比

| 指标 | E0 问题实验 | E1 修复实验 | 说明 |
|---|---:|---:|---|
| Overall one-step R² | 0.608 | 0.710 | 两次均为方差加权，只能作辅助指标 |
| z R² | -0.851 | 0.800 | 从不如均值基线变为有效预测 |
| concept R² | 0.630 | 0.648 | 保持稳定并略有改善 |
| belief R² | -0.430 | 0.680 | 从不如均值基线变为有效预测 |
| uncertainty R² | -0.229 | 0.451 | 转正，但仍是最弱分量 |
| Component Macro R² | -0.220 | 0.645 | 两项修复的主要成功指标 |
| Semantic diagnostic macro AUROC | 0.503 | 0.572 | 有改善，但不是独立语义证据 |
| Membership entropy | 1.572 | 1.495 | 分化增强，但仍偏高熵 |
| Effective modes | 4.816 | 4.458 | 仍接近五种模式共同参与 |
| Rollout macro R² | -231.01 | -240.90 | 没有改善，递归稳定性仍失败 |
| Best epoch | 16 | 24 | 两次都在50 epoch以前开始过拟合 |

### 对两次实验的总体判断

E0 是一次“流程成功、科学结论失败”的实验：训练和评价都完成了，但高 overall R² 主要来自 concept，`z/belief/uncertainty` 没有学好，projector 还提供了表示坍缩的捷径。

E1 修复了两个基础问题后，四个 state block 的 one-step R² 全部转正，因此可以支持“完整 state 的局部变化具有可预测性”。但 E1 的完整 rollout 更快发散，五个 mode 也仍缺少独立语义证据，所以还不能支持“已经学到了稳定且可解释的五模态推理动力系统”。

### 一个重要的实验限制：E1 同时修改了两个变量

E1 同时做了：

1. trainable projector → 共享冻结 PCA projector；
2. concatenated MSE → block-balanced normalized loss。

因此，E0→E1 的提升证明“两项修改合在一起有效”，但不能严格判断 Macro R² 的提升分别有多少来自 projector、多少来自 loss。现有方差恢复直接支持 projector 修复了 collapse；四个 block 的共同改善支持 balanced loss 有效，但这仍不是完整的单因素消融。若论文需要归因，应补 `PCA + 原始MSE` 和 `trainable projector + balanced loss` 两个中间实验。

## 1. Overall R² 掩盖了 state block 失衡

**对比现象：** E0 的 overall R² 为 0.608，看似成功，但只有 concept R² 为正，另外三个 block 及 Macro R² 都为负。E1 使用 block-balanced loss 后，四个 block 全部转正，Macro R² 从 -0.220 提升到 0.645。

**说明的问题：** E0 没有学会完整 state dynamics，只学好了 concept change。负 R² 表示 `z`、`belief` 和 `uncertainty` 甚至不如直接预测验证集均值。因此，E0 不能支持“模型能预测完整 reasoning state”，最多只能支持“concept transition 可预测”。

**原因：** `s=[z;c;b;u]` 的维度和变化尺度不同，concept 占 E0 总 target sum of squares 约97.4%。拼接后的 MSE和overall R²几乎由concept决定，训练也可以只降低concept error来获得较好的总loss。

**处理与结论：** 改成逐坐标标准化、四个 block 等权的 dynamics loss；以 component Macro R² 为主要完整状态指标，overall R² 仅作辅助。E1 证明该修复有效。

## 2. Trainable projector 产生了表示坍缩

**对比现象：** E0 中 `z` 和 `belief` 的 target variance 只有 `1.92e-6` 和 `1.57e-6`；E1 中增至0.214和0.702，对应 R² 从 -0.851/-0.430 变为0.800/0.680。

**说明的问题：** E0 中很小的 MSE 不是高精度，而是 projector 把 state 压成近似常数，导致目标变化本身接近零。原来的 `z` 和 `belief` dynamics 因此缺少实际信息。

**原因：** projector 一边定义

\[
\Delta z_l=P_h(h_{l+1})-P_h(h_l),
\]

一边接受同一个 dynamics loss 的梯度，存在 `P_h(h)≈constant` 的捷径；目标坐标还会随训练移动。

**处理与结论：** E1 只用 training split、跨所有层抽样拟合一套共享 PCA projector，随后冻结并保存到 checkpoint。方差和 R² 的同时恢复说明坍缩基本消除，相邻层也处于同一坐标系。

## 3. 单步预测成功不等于动力系统稳定

**对比现象：** 两次实验的 rollout 都失败，而且 E1 虽然 one-step 明显更好，递归表现反而更差：

| Horizon | E0 R² | E1 R² |
|---:|---:|---:|
| 1 | 0.914 | 0.681 |
| 2 | 0.689 | 0.523 |
| 3 | 0.681 | 0.310 |
| 4 | 0.241 | 0.033 |
| 5 | 0.423 | -0.742 |
| 6 | -0.935 | -3.064 |
| 8 | -0.862 | -17.85 |
| 32 | -12.25 | -329.62 |

E0 连续保持正 R² 到 horizon 5，E1 只到 horizon 4。两者的 state coordinate 不同，因此 raw rollout MSE 不可横向比较；但各自坐标内 R² 随 horizon 快速转负，足以说明递归失败。

**说明的问题：** 两项修复解决了“给定真实当前状态，能否预测下一步”，没有解决“模型能否根据自己的预测持续演化”。因此当前结果只能证明局部单步拟合，不能证明获得了稳定的 reasoning dynamical system。

**原因：** one-step 训练始终输入真实 `s_l`；rollout 从第二步开始输入预测状态，早期小误差会把后续输入推离训练分布并逐层放大。

**处理与待验证：** 已实现可选 `H=4` multi-step loss，从随机层开始连续预测、不 detach，并约束 horizon 2--4。下一轮比较 `R²@4/@8/@32` 及连续正 R² 的最大 horizon。相对于 E1，目标先从4提高到8以上，同时不能显著牺牲 one-step Macro R²。

## 4. 共享 dynamics 缺少 layer position

**两次实验的共同现象：** E0 和 E1 的五个 `F_k` 都不知道当前位于哪个 Transformer block。分层结果为：

| One-step layer R² 汇总 | E0 | E1 |
|---|---:|---:|
| 32层 Macro | -0.680 | 0.123 |
| 0--8层 Macro | -0.148 | 0.470 |
| 9--17层 Macro | 0.134 | 0.427 |
| 18--29层 Macro | -1.774 | -0.418 |
| 30--31层 Macro | -0.165 | 0.434 |
| R²为负的层数 | 19/32 | 12/32 |

E1 已显著改善分层预测，但18--29层仍是唯一平均 R² 为负的连续区域。

**说明的问题：** 当前共享 `F_k` 隐含假设各层遵循相同动力规律。E1 的 overall R² 为0.710，却仍有12层 R² 为负，说明总体好成绩没有覆盖所有深度。18--29层持续较差与“缺少层位置信息”一致，但目前只是动机，不是因果证明；必须通过 layer-condition 消融验证。

**原因：** Transformer block 参数随深度变化，而 state/attention/MLP 投影并不能唯一标识层位置。

**处理与待验证：** 已实现可选 `τ_l=l/(L-1)` 并加入每个 `F_k`；membership 保持不变以隔离变量。下一轮重点比较 layer-macro R² 和第18--29层。

## 5. 五个 mode 仍未形成清晰分工

**对比现象：** E1 的 membership entropy 从 E0 的1.572降至1.495，effective modes 从4.816降至4.458；轨迹更有变化，但仍接近每一步同时混合五个模式。

五分类均匀分布的最大熵为 `log(5)=1.609`。因此 E0 的1.572非常接近均匀分配；E1 虽下降，但4.458个effective modes意味着典型 transition 仍同时依赖约4--5个局部系统。

**说明的问题：** E1 的单步预测改善不能证明五个 `F_k` 已分别对应五种 reasoning semantics。由于最终只监督加权和

\[
\widehat{\Delta s_l}=\sum_k\mu_{l,k}F_k(\cdot),
\]

不同 `F_k` 可以互相补偿；它们仍可能只是五个共同拟合总变化的 ensemble 成员，而不是五种可辨识机制。

**原因：** 当前 entropy 项用于避免过早 hard assignment，会鼓励较高熵；全局 balance 只能防止所有样本坍缩到一个 mode，不能保证单个 transition 具有明确主导 mode。

**处理状态：** 暂不修改，先完成 layer condition 和 rollout loss 的受控实验，避免同时引入第三个变量。

## 6. Semantic alignment 改善，但证据不独立

**对比现象：** diagnostic macro AUROC 从 E0 的0.503提高到 E1 的0.572。Concept Composition 从0.450升至0.604，Prediction Refinement 从0.441升至0.579，但 Hop Transition 从0.617降至0.534；另外两种 mode 没有事件定义。

**说明的问题：** AUROC 0.5约等于随机排序。E0 的 macro 0.503基本没有总体语义区分能力；E1 的0.572说明出现弱正信号，但不同 mode 一升一降且只覆盖3/5模式，尚不能证明五个 mode 具有预设语义。尤其不能用单个 `Hop Transition` 的偶然高值代表整个体系成功。

**原因：** 当前事件由 bridge/answer probability 和 uncertainty 自动构造，并与训练 prior 重叠；同源 proxy 容易高估语义一致性，也不能充当独立验证。

**结论：** 这些 AUROC 只能作为诊断。论文强结论需要五种 mode 的独立事件定义、held-out category，并最好加入人工标注。

## 7. 当前 `P_c` 稳定，但还不是语义 concept projector

**对比现象：** E0 的 learned projector 容易坍缩；E1 改用共享冻结 PCA 后，concept R² 保持在0.648，但 `P_c` 实际是共享 PCA 的第65--96维。

**说明的问题：** E1 证明这个低维子空间适合稳定预测 concept-block transition，不等于证明每个坐标或整个子空间天然表示 intermediate concept。高 concept R²回答的是“这个向量的变化能否预测”，不是“这个向量语义是否正确”。Linear bridge probe 只能检测 bridge 信息是否存在，不能把冻结 `P_c` 塑造成语义空间。

**原因：** PCA 按方差选择方向，不按 bridge entity、relation composition 或推理语义选择方向。

**结论：** 当前 `c_l` 应称为“经过 concept probe 检查的冻结候选子空间”。未来需将 PCA 与有防坍缩约束的共享 learned projector、跨层对齐 SAE 做消融。

## 8. 两次实验的 raw MSE 不可直接比较

**对比现象：** E1 one-step raw MSE 为0.101，高于 E0 的0.00112，但 E1 的四个 component R² 全部显著更好。

**说明的问题：** 如果只看 MSE，会错误地把修复实验判断为退步。E0 的小 MSE部分来自低方差/坍缩坐标；E1 在白化后的单位方差坐标中预测更困难，但相对于目标方差解释了更多变化。因此不能把不同 state coordinate 下的绝对误差直接排名。

**原因：** E1 的 PCA 白化改变了 `z/concept/belief` 的单位与方差；E0 和 E1 的 MSE 不处于同一坐标尺度。

**结论：** 跨表示比较使用 component/macro R²、normalized block loss、semantic metrics 和 rollout horizon；raw MSE 仅用于同一 state 定义内部。

## 9. 更多 epoch 不能解释当前失败

**对比现象：** E0 最佳 validation loss 为0.00359（epoch 16），到 epoch 50恶化为0.00908；E1 最佳为0.28687（epoch 24），到 epoch 50为0.30122。两者后期 training loss 都继续下降。

**说明的问题：** 当前失败不是简单的“训练轮数不够”。延长到更多 epoch 更可能扩大 train-validation gap，也没有目标去直接修复 rollout。

**原因：** 已出现过拟合；此外原 checkpoint selection 主要依据 teacher-forced validation objective，而不是完整递归稳定性。

**处理与结论：** 启用 rollout loss 后，短程 rollout loss 同时进入 validation 和 `best.pt` 选择。完整32步 rollout 仍单独报告，避免仅凭 total loss 判断动力稳定性。

## 10. 尚无证据表明 MLP 太小

**两次实验的共同现象：** 每个 `F_k` 只有一层宽度128的 hidden layer，单个约2.9万--4.2万参数，五个合计约16.7万。尽管规模保守，training loss 仍可继续下降；只改变 loss 和 projector 后，Macro R² 已从 -0.220 提升到0.645。

**说明的问题：** 主要瓶颈首先来自目标失衡和表示坍缩，而不是网络容量。直接增大 MLP 不能解决 mode 不可辨识或 rollout 发散，还可能加重过拟合。

**结论：** 先完成 objective ablation；之后再在相同设置下比较 hidden width 64/128/256/512。

## 下一轮受控实验

固定 cache、split、seed、projector 和其他超参数：

| 实验 | Layer condition | H=4 rollout loss |
|---|---:|---:|
| Baseline | 关闭 | 关闭 |
| Layer | 开启 | 关闭 |
| Rollout | 关闭 | 开启 |
| Layer + Rollout | 开启 | 开启 |

先运行 seed 42 pilot，再对有效配置补 seed 43、44。
