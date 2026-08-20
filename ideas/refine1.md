根据这次结果和你原来的建模方案，我认为下一轮应该**先改模型目标和训练方式，再谈调参**。

你原本的系统是

$$
s_{i,l+1}
=
s_{i,l}
+
\sum_{k=1}^{5}
\mu_{i,l,k}F_k(s_{i,l}),
$$

而训练主要要求混合后的总变化重建真实 $\Delta s_{i,l}$。

这里实际上同时有三个不同目标：

1. **能不能预测 state change**
2. **递归以后动力学稳不稳定**
3. **5 个 $F_k$ 是否真的对应 5 种 reasoning semantics**

你目前只把第 1 个目标学得比较好，而且主要是 concept；第 2、3 个目标还没有真正被训练起来。

---

# 一、第一优先级：先解决 state 四个分量严重失衡

现在最明显的是：

$$
R^2_c=0.630
$$

但

$$
R^2_z=-0.851,\quad
R^2_b=-0.430,\quad
R^2_u=-0.229.
$$

所以你现在的

$$
\mathcal L_{\mathrm{dyn}}
=
\|\Delta s-\widehat{\Delta s}\|^2
$$

有很大问题。

因为你的 state 本身就是

$$
s=[z;c;b;u]
$$

四种性质完全不同的变量。

它们的：

- 维数不同；
- variance 不同；
- 数值范围不同；
- prediction difficulty 也不同。

直接 concat 后算一个 MSE，很容易出现：

> 模型只要把最容易、或者贡献总 variance 最大的 concept 拟合好，总 loss 和总体 $R^2$ 就会很好看。

这正是你现在发生的事情。

## 应该改成 block-balanced dynamics loss

分别计算：

$$
\mathcal L_z,\quad
\mathcal L_c,\quad
\mathcal L_b,\quad
\mathcal L_u.
$$

比如：

$$
\mathcal L_z
=
\frac{1}{d_z}
\left\|
\frac{
\Delta z-\widehat{\Delta z}
}{
\sigma_{\Delta z}
}
\right\|^2
$$

同样：

$$
\mathcal L_c,\mathcal L_b,\mathcal L_u.
$$

最终：

$$
\boxed{
\mathcal L_{\mathrm{state}}
=
\alpha_z\mathcal L_z
+
\alpha_c\mathcal L_c
+
\alpha_b\mathcal L_b
+
\alpha_u\mathcal L_u
}
$$

第一版直接：

$$
\alpha_z=\alpha_c=\alpha_b=\alpha_u=1.
$$

也就是说：

> **不是让每一个维度平权，而是先让四种 state component 平权。**

这一步我认为必须先做。

---

# 二、还有一个非常重要的 sanity check：你的 latent coordinate 必须跨层一致

你定义：

$$
z_{i,l}=P_h(h_{i,l})
$$

然后计算

$$
\Delta z_{i,l}=z_{i,l+1}-z_{i,l}.
$$

原方案只说 $P_h$ 是降维映射。

这里你一定检查代码：

> **是不是所有 layer 使用同一个 $P_h$？**

例如绝对不能：

$$
z_l=P_h^{(l)}(h_l)
$$

$$
z_{l+1}=P_h^{(l+1)}(h_{l+1})
$$

然后直接

$$
z_{l+1}-z_l.
$$

因为两个 PCA / projector 如果独立训练，它们的 coordinate system 不一样。

例如：

$$
z_l=[0.8,0.2]
$$

和

$$
z_{l+1}=[0.1,0.9]
$$

如果两个坐标轴本身定义就不一样，这个 difference 没有明确动力学意义。

所以：

$$
\boxed{
P_h,\;P_c,\;P_b
}
$$

最好都基于 **training set + all layers** 学一个共享 coordinate system。

尤其检查你现在：

- PCA 是不是 per-layer PCA；
- scaler 是不是 per-layer fit；
- concept projector 是否跨层共享；
- belief reduction 是否跨层共享。

这个问题甚至可能直接解释为什么 $z$ 的 $R^2=-0.85$。
