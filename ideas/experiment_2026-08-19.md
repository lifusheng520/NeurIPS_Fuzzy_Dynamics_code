# 2026-08-19 实验与修改记录

## 1. 本记录的范围

本记录汇总 2026-08-19 完成的正式实验：

$$
4\text{ 个配置}\times 3\text{ 个 seed}=12\text{ 次实验}.
$$

这些结果全部来自当时的 **operation-conditioned FRDS**：递归状态使用模型预测的
$\hat s_l$，但每层 attention/MLP operation 仍来自真实 LLM 缓存：

$$
\hat s_{l+1}
=
\hat s_l+F(\hat s_l,a_l^{\mathrm{LLM}},m_l^{\mathrm{LLM}}).
$$

因此，本文档中的 rollout 指标都是 conditional rollout，不能解释为脱离原始
LLM 的 autonomous simulation。

2026-08-20 新增的 operation predictor 和 autonomous rollout 尚未产生服务器实验
结果，不属于本文档的数值。

---

## 2. 为这批实验做的修改

### 2.1 训练轮数从 50 提高到 100

前一批 seed 42 试验中，Layer 和 Rollout 配置的 best epoch 接近 50，说明可能被
原来的训练上限截断。因此四个正式配置统一改为：

```json
"epochs": 100
```

本轮 12 次实验均实际训练了 100 epochs，并根据 validation loss 保存 `best.pt`。

### 2.2 Layer-conditioned dynamics

对每个 transition 定义归一化层位置：

$$
\tau_l=\frac{l}{L-1}.
$$

Layer 配置把 $\tau_l$ 加入五个 local dynamics 的输入：

$$
F_k(s_l,a_l,m_l)
\longrightarrow
F_k(s_l,a_l,m_l,\tau_l).
$$

对应开关：

```json
"use_layer_condition": true
```

### 2.3 H=4 multi-step rollout loss

Rollout-H4 配置从真实窗口起点 $s_l$ 出发，连续递归预测四步，并对
$h=2,3,4$ 的状态误差取平均：

$$
\mathcal L_{\mathrm{roll}}
=
\frac{1}{3}\sum_{h=2}^{4}
D(\hat s_{l+h},s_{l+h}).
$$

配置为：

```json
"rollout_weight": 0.25,
"rollout_horizon": 4
```

递归中没有 detach，因此后续 horizon 的误差能够反向约束前面的预测。

### 2.4 四状态块平衡的标准化重建损失

状态由四部分组成：

$$
s=[z;c;b;u].
$$

训练损失不再让高维或高方差状态块支配结果，而是分别按训练集 delta scale
标准化，再对四个状态块等权平均：

$$
\mathcal L_{\mathrm{dyn}}
=
\frac14\sum_{q\in\{z,c,b,u\}}
\operatorname{MSE}
\left(
\frac{\widehat{\Delta s}^{(q)}-\Delta s^{(q)}}{\sigma_q}
\right).
$$

隐藏状态和 belief 使用训练 split、跨所有层拟合的一套共享 frozen PCA 坐标系，
保证 $s_{l+1}-s_l$ 在同一坐标系中有意义。

### 2.5 完善评估输出

本轮正式记录：

- one-step MSE、MAE、cosine similarity、overall $R^2$；
- $z$/concept/belief/uncertainty 四块 $R^2$ 及 component macro $R^2$；
- 32 个 transition 的 per-layer $R^2$；
- early 0--8、middle 9--17、late 18--29、final 30--31 分段 $R^2$；
- conditional rollout MSE、final-state MSE、per-horizon $R^2$；
- rollout component macro $R^2$；
- 从 horizon 1 开始连续保持正 $R^2$ 的最大 horizon；
- 三个当前可定义 mode 的 diagnostic AUROC/AP。

---

## 3. 正式实验设置

四个配置：

| 配置 | Layer condition | H=4 rollout loss |
|---|---:|---:|
| Base | 否 | 否 |
| Layer | 是 | 否 |
| Rollout-H4 | 否 | 是，权重 0.25 |
| Layer+Rollout-H4 | 是 | 是，权重 0.25 |

共同设置：

- activation cache：`cache/socrates_llama3_8b_tuned_full.pt`；
- LLM：Meta-Llama-3-8B；
- 数据：SOCRATES，共 7,232 条 trajectory；
- seeds：42、43、44；
- epochs：100；
- batch size：16；
- learning rate：$10^{-3}$；
- weight decay：$10^{-4}$；
- gradient clipping：1.0；
- category-level validation split；
- 每个配置在同一个 seed 下使用相同的验证样本集合。

正式运行目录与 best epoch：

| Seed | 配置 | Experiment ID | Best epoch |
|---:|---|---|---:|
| 42 | Base | `20260819_183338_421695` | 66 |
| 42 | Layer | `20260819_185121_050733` | 93 |
| 42 | Rollout-H4 | `20260819_185415_807872` | 89 |
| 42 | Layer+Rollout-H4 | `20260819_191154_145111` | 98 |
| 43 | Base | `20260819_193905_050686` | 91 |
| 43 | Layer | `20260819_193911_669122` | 94 |
| 43 | Rollout-H4 | `20260819_195840_946201` | 96 |
| 43 | Layer+Rollout-H4 | `20260819_195920_444161` | 98 |
| 44 | Base | `20260819_202613_781722` | 89 |
| 44 | Layer | `20260819_202617_174249` | 88 |
| 44 | Rollout-H4 | `20260819_204526_868768` | 99 |
| 44 | Layer+Rollout-H4 | `20260819_204549_677442` | 95 |

另有两个 50-epoch Base 旧任务、一个重复的 100-epoch Base，以及一个被取消且没有
metrics 的 array 任务；它们没有进入下面的正式统计。

---

## 4. 三个 seed 的主要结果

以下均为 mean $\pm$ sample standard deviation，$n=3$。

| 配置 | One-step MSE $\downarrow$ | One-step overall $R^2$ $\uparrow$ | Component macro $R^2$ $\uparrow$ | Rollout MSE $\downarrow$ | Final-state MSE $\downarrow$ | Rollout macro $R^2$ $\uparrow$ |
|---|---:|---:|---:|---:|---:|---:|
| Base | $0.053\pm0.028$ | $0.854\pm0.086$ | $0.802\pm0.116$ | $389.786\pm633.286$ | $1713.506\pm2713.119$ | $-363.995\pm597.598$ |
| Layer | $\mathbf{0.052\pm0.025}$ | $\mathbf{0.858\pm0.077}$ | $\mathbf{0.804\pm0.117}$ | $68.826\pm78.912$ | $412.815\pm465.357$ | $-58.689\pm61.620$ |
| Rollout-H4 | $0.072\pm0.035$ | $0.802\pm0.108$ | $0.763\pm0.151$ | $\mathbf{0.788\pm0.148}$ | $\mathbf{4.872\pm0.794}$ | $\mathbf{0.206\pm0.245}$ |
| Layer+Rollout-H4 | $0.066\pm0.031$ | $0.819\pm0.095$ | $0.782\pm0.124$ | $0.878\pm0.160$ | $4.923\pm0.978$ | $0.140\pm0.269$ |

### 4.1 单步重建最好的是 Layer

Layer 配置得到：

$$
R^2_{\mathrm{one-step}}=0.858\pm0.077,
$$

$$
R^2_{\mathrm{component\ macro}}=0.804\pm0.117,
$$

$$
\mathrm{MSE}_{\mathrm{one-step}}=0.052\pm0.025.
$$

但是它相对 Base 的提升很小：

$$
0.858-0.854=0.004.
$$

因此 layer condition 对单步预测只有轻微平均收益，不能描述为大幅提升。

### 4.2 完整 rollout 最好的是 Rollout-H4

Rollout-H4 得到：

$$
R^2_{\mathrm{rollout\ macro}}=0.206\pm0.245,
$$

$$
\mathrm{MSE}_{\mathrm{rollout}}=0.788\pm0.148.
$$

与 Base 相比，rollout loss 将递归 MSE 从数百量级降低到 1 以下，说明 H=4
训练显著改善了递归稳定性。

Layer+Rollout-H4 没有进一步提升完整 rollout：

$$
0.140<0.206.
$$

---

## 5. Rollout 的 seed-wise 结果

### 5.1 完整 rollout component macro $R^2$

| 配置 | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| Base | $-1053.886$ | $-31.740$ | $-6.359$ |
| Layer | $-39.059$ | $-127.733$ | $-9.277$ |
| Rollout-H4 | $-0.076$ | $\mathbf{0.323}$ | $\mathbf{0.371}$ |
| Layer+Rollout-H4 | $-0.170$ | $0.283$ | $0.306$ |

最好的一次是 Rollout-H4、seed 44：

$$
R^2_{\mathrm{rollout\ macro}}=0.371.
$$

它仍然低于预先希望达到的：

$$
R^2_{\mathrm{rollout\ macro}}\ge0.5.
$$

因此当前结果不能支持“完整 32 层轨迹已经被准确重建”。

### 5.2 连续正 $R^2$ horizon

| 配置 | Seed 42 | Seed 43 | Seed 44 | 平均 |
|---|---:|---:|---:|---:|
| Rollout-H4 | 6 | 22 | 17 | 15.0 |
| Layer+Rollout-H4 | 6 | 20 | 17 | 14.3 |

所以多步预测已经不能称为“完全无效”。更准确的结论是：

> H=4 rollout training 在部分 held-out category 上可保持约 17--22 层的正
> $R^2$，但还不能稳定覆盖全部 32 层。

---

## 6. 状态组件结果

### 6.1 One-step component $R^2$

| 配置 | $z$ | Concept | Belief | Uncertainty |
|---|---:|---:|---:|---:|
| Base | $0.903\pm0.072$ | $0.828\pm0.124$ | $0.826\pm0.074$ | $0.653\pm0.226$ |
| Layer | $\mathbf{0.915\pm0.048}$ | $\mathbf{0.842\pm0.102}$ | $0.822\pm0.080$ | $0.639\pm0.251$ |
| Rollout-H4 | $0.903\pm0.047$ | $0.821\pm0.105$ | $0.718\pm0.135$ | $0.609\pm0.326$ |
| Layer+Rollout-H4 | $0.900\pm0.057$ | $0.837\pm0.099$ | $0.746\pm0.106$ | $0.646\pm0.252$ |

单步四个状态块总体都能得到正且较高的 $R^2$。Layer 主要改善了 $z$ 和 concept，
但没有改善 belief/uncertainty。

### 6.2 Full rollout component $R^2$

| 配置 | $z$ | Concept | Belief | Uncertainty |
|---|---:|---:|---:|---:|
| Rollout-H4 | $0.141\pm0.288$ | $-0.156\pm0.300$ | $0.370\pm0.237$ | $0.469\pm0.293$ |
| Layer+Rollout-H4 | $0.000\pm0.287$ | $-0.341\pm0.420$ | $0.400\pm0.161$ | $0.500\pm0.263$ |

完整 rollout 的正 macro 主要由 belief 和 uncertainty 支撑；concept trajectory
仍未重建成功，$z$ 也很弱。因此不能把正 macro $R^2$ 解释为所有 reasoning-state
component 都已经学会。

---

## 7. 不同深度区间的 one-step 结果

| 配置 | Early 0--8 | Middle 9--17 | Late 18--29 | Final 30--31 |
|---|---:|---:|---:|---:|
| Base | $0.730\pm0.135$ | $0.673\pm0.146$ | $0.281\pm0.444$ | $0.722\pm0.183$ |
| Layer | $0.715\pm0.151$ | $0.671\pm0.136$ | $0.261\pm0.468$ | $0.749\pm0.138$ |
| Rollout-H4 | $0.574\pm0.238$ | $0.521\pm0.202$ | $-0.026\pm0.702$ | $0.688\pm0.159$ |
| Layer+Rollout-H4 | $0.609\pm0.196$ | $0.559\pm0.173$ | $0.068\pm0.616$ | $0.674\pm0.195$ |

“只有前中层有效”并不是跨 seed 都成立。以 Layer 的 late 18--29 为例：

- seed 42：$-0.274$；
- seed 43：$0.466$；
- seed 44：$0.591$。

因此更准确的表述是：

> 晚层 reconstruction 有明显的 held-out category 依赖性，seed 42 对应类别较难，
> seed 43/44 的晚层已经得到正 $R^2$。

---

## 8. Diagnostic semantic alignment

当前只有 Concept Composition、Prediction Refinement 和 Hop Transition 三个 mode
具有自动事件定义。三模式 diagnostic macro 结果为：

| 配置 | Diagnostic macro AUROC | Diagnostic macro AP |
|---|---:|---:|
| Base | $\mathbf{0.603\pm0.060}$ | $\mathbf{0.231\pm0.130}$ |
| Layer | $0.578\pm0.019$ | $0.198\pm0.090$ |
| Rollout-H4 | $0.557\pm0.032$ | $0.185\pm0.089$ |
| Layer+Rollout-H4 | $0.532\pm0.033$ | $0.159\pm0.073$ |

这些事件与训练 prior 同源，不是 independent labels；Knowledge Enrichment 和
Information Routing 也没有独立事件定义。因此：

$$
\text{strict five-mode macro AUROC/AP}=\texttt{null}.
$$

这部分只能称为 diagnostic semantic alignment，不能作为五个 fuzzy mode 具有独立
语义的最终证明。Rollout 配置提升稳定性的同时，diagnostic semantic alignment 反而
下降，显示 fidelity/stability 与当前 semantic specialization 之间存在 trade-off。

---

## 9. Seed 与 validation split 的解释限制

当前同一个 `seed` 同时控制：

1. 模型初始化；
2. PCA 状态采样；
3. category-level train/validation split。

三个 seed 的 validation size 分别为：

| Seed | Validation trajectories | 主要 held-out 大类别 |
|---:|---:|---|
| 42 | 968 | `person-year-masterschampion`（931 条） |
| 43 | 273 | `person-year-championsleaguecity`（196 条） |
| 44 | 1009 | `person-year-nobelchem`（853 条） |

因此上述标准差同时包含：

$$
\text{初始化波动}
+
\text{PCA 采样波动}
+
\text{held-out category 难度差异}.
$$

它不是固定测试集上的纯 random-initialization error bar。正式论文若要严格报告
mean $\pm$ std，应固定一份 category split，只改变 model seed；当前结果更适合解释
为三个不同 category holdout 上的重复实验。

同一个 seed 下四个配置的 validation index 排列顺序可能不同，但排序后的 index
集合一致，因此配置间比较仍使用相同验证样本。

---

## 10. 本轮最终结论

### 结论一：单步重建已经较强

最好的 Layer 配置达到：

$$
R^2_{\mathrm{one-step}}=0.858\pm0.077,
\qquad
R^2_{\mathrm{component\ macro}}=0.804\pm0.117.
$$

这支持“FRDS 能够重建 operation-conditioned one-step layer-wise state change”。

### 结论二：H=4 rollout loss 是最有效的稳定性修改

Rollout-H4 把递归 MSE 从数百量级压到 1 以下，并在 seed 43/44 得到正的完整
rollout macro $R^2$。它是本轮最适合作为 rollout 主模型的配置。

### 结论三：完整 reasoning trajectory 尚未学成

三 seed 平均完整 rollout macro $R^2$ 为：

$$
0.206\pm0.245,
$$

最好单次为：

$$
0.371<0.5.
$$

而且 concept rollout $R^2$ 仍为负。因此尚不能宣称成功重建完整 32 层推理过程。

### 结论四：当前 rollout 仍是 conditional，不是 autonomous

它仍然使用真实 LLM 的未来 attention/MLP operation。该实现问题在
`ideas/refine4.md` 中单独分析，并在 2026-08-20 的新代码中通过 operation predictor
和独立 `autonomous_rollout` 路径处理；新结构的效果必须通过下一轮实验重新验证。

---

## 11. 可用于论文的阶段性表述

> Across three runs, the layer-conditioned model achieves the strongest
> one-step reconstruction, with an overall $R^2$ of $0.858\pm0.077$ and a
> component-macro $R^2$ of $0.804\pm0.117$. Short-horizon rollout training
> substantially improves recursive stability: Rollout-H4 reduces the
> conditional full-rollout MSE from $389.786$ to $0.788$ and obtains a rollout
> component-macro $R^2$ of $0.206\pm0.245$. Nevertheless, the best individual
> full-rollout macro $R^2$ is $0.371$, and concept-trajectory reconstruction
> remains negative, indicating that faithful reconstruction of the complete
> depth-wise reasoning trajectory has not yet been achieved.
