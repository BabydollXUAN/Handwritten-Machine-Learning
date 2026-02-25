# LoRA 面试速记（1页）

## 1. 一句话定义

LoRA（Low-Rank Adaptation）是参数高效微调：冻结原模型权重，只训练低秩增量矩阵。

## 2. 核心公式（必背）

```text
W' = W + ΔW
ΔW = (alpha / r) * B @ A
```

- `A`: `(r, in_features)`
- `B`: `(out_features, r)`
- `r`: rank（通常远小于输入/输出维度）

前向：

```text
y = xW^T + b + (alpha/r) * x(BA)^T
```

## 3. 为什么 LoRA 有效

- 微调时大模型主要需要小幅方向修正，不必更新完整大矩阵。
- 用低秩分解近似更新方向，参数更少、显存更省、训练更快。

## 4. 参数量对比（高频）

原线性层（不含 bias）：

```text
out_features * in_features
```

LoRA 可训练参数：

```text
r * in_features + out_features * r
= r * (in_features + out_features)
```

当 `r << min(in_features, out_features)` 时，参数量显著下降。

## 5. 初始化细节（常问）

标准做法：
- `A` 随机初始化
- `B` 初始化为 0

原因：训练开始时 `ΔW=0`，初始输出与基座模型一致，更稳定。

## 6. alpha 和 r 怎么理解

- 实际缩放是 `alpha / r`
- `r` 控制容量上限
- `alpha` 控制 LoRA 分支幅度

经验上：先定 `r`（如 4/8/16），再调 `alpha`（常设成 `r` 或 `2r`）。

## 7. LoRA 挂在哪些层

在 Transformer 中最常见：
- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`

也可挂到 MLP 层，但先从注意力投影层开始最常见。

## 8. 推理时怎么做

两种方式：
- 不合并：在线计算 `W + BA` 分支
- 合并：把 `ΔW` 并入 `W`（部署更简洁）

你仓库里的 `LoRA/model.py` 已提供 `merge_weights()`。

## 9. 高频快问快答

- 问：LoRA 会不会损失效果？
  - 答：在合适 `r` 下通常接近全量微调，且成本显著更低。

- 问：LoRA 和 Adapter 的区别？
  - 答：Adapter 是加新层，LoRA 是对原线性层做低秩增量。

- 问：为什么能省显存？
  - 答：只训练少量 LoRA 参数，优化器状态和梯度都更少。

## 10. 结合本仓库

- 手撕代码：`LoRA/model.py`
- 详细说明：`LoRA/README.md`
- 本页用途：面试前 3 分钟快速复盘
