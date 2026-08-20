这三组实验已经把问题拆得很清楚了。现在可以比较有把握地说：**你的系统里存在三个彼此不完全一致的目标——one-step fidelity、rollout stability、semantic alignment。优化其中一个，暂时会牺牲另外一些。**



# 9. Membership entropy 也支持这一点

Baseline：

$$
H(\mu)=1.495
$$

Layer：

$$
1.539
$$

Rollout：

$$
1.527
$$

Layer+Rollout：

$$
1.543.
$$

最大：

$$
\log5=1.609.
$$

所以所有新模型的 membership 都比 baseline **更加均匀**。

特别是：

$$
1.495\rightarrow1.543.
$$

说明 fidelity/stability 改进的同时，model specialization 确实有所减弱。

这是现在非常明确的 trade-off：

$$
\boxed{
\text{Dynamics performance}\uparrow
\quad\Longrightarrow\quad
\text{membership specialization}\downarrow
}
$$

至少在当前 loss 设计下是这样。



# 11. Best epoch = 50 这个现象也需要重视

Baseline：

$$
24
$$

Layer：

$$
50
$$

Rollout：

$$
50
$$

Layer+Rollout：

$$
48.
$$

Layer 和 Rollout 都在 epoch 50 达到 best，意味着：

$$
\boxed{\text{它们可能尚未完全收敛}}
$$

所以目前 Layer 和 Rollout 的结果严格来说可能还不是它们的上限。

我建议下一次改成：

$$
\boxed{
\text{max epoch}=75
}
$$

同时：

$$
\boxed{
\text{early stopping patience}=8\sim10
}
$$

不是固定训练 75 epoch，而是允许它继续。

否则当前实验实际上存在一个小的不公平：

Baseline 已经在 24 epoch 收敛，

而 Layer / Rollout 可能被 50 epoch 上限截断。

---

# 12. 但是补 seed 的优先级仍然比继续精调高

现在所有结论都来自：

$$
\text{seed}=42.
$$

所以“Layer 一定提升 one-step”“Rollout 一定提升 stability”虽然差距很大、看起来很可信，但论文上仍然需要：

$$
42,\;43,\;44.
$$

至少三个 seed。

最终报告：

$$
\boxed{
\text{mean}\pm\text{std}
}
$$

特别需要确认：

### Layer

$$
R^2_{\rm global}
$$

是否稳定提升。

### Rollout

$$
R^2_{\rm rollout}
$$

是否稳定大幅提升。

### Semantic AUROC

这一项的波动可能会特别大。

如果：

$$
0.572\pm0.08
$$

和：

$$
0.505\pm0.07
$$

那两者实际上可能没有显著区别。

所以目前不要过度解读这 $0.067$ 的差距。



---

# 14. 下一步实验我建议非常收敛，不要继续扩散

现在最值得跑的是：

### 第一组：重复性

$$
\text{Baseline / Layer / Rollout}
$$

分别：

$$
seed=42,43,44.
$$

先确认三条主要结论成立。

### 第二组：只调 Rollout weight

固定 Rollout-H4：

$$
\lambda_{\rm roll}
=
0.1,\;0.25,\;0.5.
$$

你现在希望找到一个点：

$$
\boxed{
\text{rollout stability 保持}
}
$$

同时：

$$
\boxed{
\text{one-step 和 membership specialization 少损失一点}
}
$$

我反而暂时不建议马上加 H8。

先把 H4 的权重 trade-off 搞清楚。

---

# 15. 现在最理想的下一结果是什么

不是：

$$
R^2@32>0
$$

这么激进。

而是找到一个例如：

$$
R^2_{\rm one-step}\approx0.68\sim0.70
$$

$$
R^2_{\rm rollout\ macro}>0
$$

$$
H_{R^2>0}\ge8
$$

同时 entropy 不要继续往：

$$
1.60
$$

靠近。

如果 Rollout weight 降低到：

$$
0.1
$$

能做到：

$$
R^2_{\rm one-step}=0.68
$$

$$
R^2@8>0
$$

$$
R^2@32\approx-0.5
$$

同时 semantic proxy：

$$
0.53\sim0.56,
$$

这可能比现在任何一个配置都更适合作为主模型。

---

## 最终，我会把目前实验结论压缩成三条

### 结论一

$$
\boxed{
\text{Layer-conditioned dynamics 假设得到支持}
}
$$

因为所有 state component 和 layer-wise one-step fidelity 都提升。

但：

$$
18\text{--}29
$$

层仍未建模好，所以 layer information 不是完整答案。

### 结论二

$$
\boxed{
\text{Multi-step training 是目前最关键的结构改进}
}
$$

它把 rollout 从：

$$
R^2@32=-329.6
$$

改善到：

$$
-1.03,
$$

证明 one-step training 确实是此前动力系统爆炸的主要原因之一。

### 结论三

$$
\boxed{
\text{fidelity / stability 与 semantic specialization 目前存在冲突}
}
$$

所有增强 dynamics 的方法都让 entropy 升高，同时 proxy AUROC 下降。

所以你下一阶段真正需要解决的已经不是单纯“预测准不准”，而是：

$$
\boxed{
\text{如何在保持稳定 dynamics 的同时，
让 5 个 }F_k/\mu_k\text{ 仍具有可验证的 specialization}
}
$$

这会成为你接下来最核心的问题。