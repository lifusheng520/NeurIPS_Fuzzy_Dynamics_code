
### 1. 单步预测为什么没有这个问题

对第 $l$ 层做单步预测时，代码输入：

$$
s_l^{\mathrm{LLM}},\quad
a_l^{\mathrm{LLM}},\quad
m_l^{\mathrm{LLM}},
$$

然后预测：

$$
\widehat{\Delta s_l}
=
F(s_l^{\mathrm{LLM}},a_l^{\mathrm{LLM}},m_l^{\mathrm{LLM}}),
$$

进而得到：

$$
\hat s_{l+1}
=
s_l^{\mathrm{LLM}}+\widehat{\Delta s_l}.
$$

这里三个输入都来自同一次真实 LLM 前向传播，它们彼此匹配，因此单步 $R^2$ 可以比较高。

### 2. 多步 rollout 从哪里开始不一致

第一步：

$$
\hat s_1
=
s_0^{\mathrm{LLM}}
+
F(s_0^{\mathrm{LLM}},a_0^{\mathrm{LLM}},m_0^{\mathrm{LLM}})
$$

还是正常的。

但是第二步开始，代码使用：

$$
\hat s_2
=
\hat s_1+
F(\hat s_1,a_1^{\mathrm{LLM}},m_1^{\mathrm{LLM}}).
$$

此时：

- $\hat s_1$ 是 FRDS 预测出来的状态；
- $a_1^{\mathrm{LLM}},m_1^{\mathrm{LLM}}$ 却是原始 LLM 在真实状态 $s_1^{\mathrm{LLM}}$ 上产生的操作。

如果第一步存在误差：

$$
\hat s_1\neq s_1^{\mathrm{LLM}},
$$

那么真实情况下，在 $\hat s_1$ 上产生的操作应该是：

$$
\hat a_1,\hat m_1,
$$

而不一定还是：

$$
a_1^{\mathrm{LLM}},m_1^{\mathrm{LLM}}.
$$

所以第二步实际组合成了：

$$
\underbrace{\hat s_1}_{\text{预测轨迹的状态}}
+
F\left(
\underbrace{\hat s_1}_{\text{预测轨迹}},
\underbrace{a_1^{\mathrm{LLM}},m_1^{\mathrm{LLM}}}_{\text{真实轨迹的操作}}
\right).
$$

这相当于把两条不同轨迹拼在一起。

### 3. 一个直观例子

可以把它理解成预测汽车运动：

- $s_l$：汽车当前的位置和速度；
- $a_l,m_l$：司机根据当前位置做出的方向盘和油门操作。

第一秒后，模型把汽车位置预测错了一点。第二秒却仍然使用“司机在真实位置上做出的方向盘操作”。

但如果汽车真的已经到了预测位置，司机本来可能会采用不同的方向盘角度。因此，“预测位置 + 真实路线上的方向盘”并不是一个自洽的运动过程。

### 4. 为什么它会导致 rollout 崩溃

当前模型训练时主要看到的是：

$$
F(s_l^{\mathrm{LLM}},a_l^{\mathrm{LLM}},m_l^{\mathrm{LLM}}).
$$

但 rollout 时看到的是：

$$
F(\hat s_l,a_l^{\mathrm{LLM}},m_l^{\mathrm{LLM}}).
$$

随着误差累积，$\hat s_l$ 会越来越偏离真实状态，形成模型训练时没见过的输入组合。于是：

$$
\text{状态误差}
\rightarrow
\text{不匹配的输入组合}
\rightarrow
\text{更大的预测误差}
\rightarrow
\text{继续累积}.
$$

这可以解释为什么你的单步结果不错，而完整 32 步 rollout 的 $R^2$ 极低甚至为负。

但它不是唯一原因，下面这些也会引起误差累积：

- 共享 dynamics 无法描述不同深度的变化；
- PCA 状态不满足 Markov 性；
- 模型没有接受足够长的 free-running 训练；
- 晚层状态变化较小、信噪比较低；
- dynamics 本身缺乏稳定性约束。

### 5. 当前结果到底能证明什么

当前 rollout 更准确的名字是：

> operation-conditioned rollout  
> 给定真实 LLM 操作序列条件下的状态重建

它回答的是：

> 如果提前提供真实 LLM 每层产生的 attention/MLP 信号，FRDS 能否沿深度方向重建状态？

它不能证明：

> FRDS 可以脱离原始 LLM，独立生成完整的内部推理轨迹。

而且即使作为 conditional rollout，目前预测状态和真实操作之间仍有分布错配。

### 6. 真正闭合的 surrogate 应该怎么做

需要让操作也由 FRDS 根据当前预测状态产生：

$$
(\hat a_l,\hat m_l)
=
G(\hat s_l,l,\text{context}),
$$

然后：

$$
\hat s_{l+1}
=
\hat s_l+
F(\hat s_l,\hat a_l,\hat m_l,l).
$$

这样整个过程才闭合：

$$
\hat s_l
\longrightarrow
(\hat a_l,\hat m_l)
\longrightarrow
\hat s_{l+1}.
$$

另一种方案是完全去掉 $a_l,m_l$，直接学习：

$$
\hat s_{l+1}
=
\hat s_l+F(\hat s_l,l,\text{history}),
$$

但这可能要求状态包含更多信息，否则仅靠当前低维状态无法预测下一层。

所以一句话总结：

> 现在的代码是在“用 FRDS 预测状态，但用真实 LLM 提供后续操作”；它属于有真实操作辅助的条件重建，不是能够独立运行的完整 surrogate dynamics。