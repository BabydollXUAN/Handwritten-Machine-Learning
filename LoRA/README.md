# LoRA 手写实现说明

本文档对应代码：`LoRA/model.py`

## 1. LoRA 是什么

LoRA（Low-Rank Adaptation）是一种参数高效微调方法。
核心思想：
- 冻结原始线性层权重 `W`
- 只训练一个低秩增量 `ΔW`

常见写法：

```text
W' = W + ΔW
ΔW = (alpha / r) * B @ A
```

其中：
- `A` 形状：`(r, in_features)`
- `B` 形状：`(out_features, r)`
- `r` 是低秩（通常远小于输入输出维度）

## 2. 前向计算（对应本代码）

`LoRALinear` 的前向是：

```text
y = xW^T + b + (alpha/r) * x(BA)^T
```

在 `model.py` 中分两段实现：
1. `base_out = F.linear(x, weight, bias)`
2. `lora_out = F.linear(F.linear(x, A), B)`

最终：`base_out + scaling * lora_out`。

## 3. 为什么初始化 B 为 0

代码里：
- `A` 随机初始化
- `B` 全 0 初始化

好处：训练开始时 `ΔW = 0`，模型初始输出等于基座模型输出，不会一上来就破坏原模型行为。

## 4. 你这份实现的关键点

- `freeze_base=True` 时：
  - `weight`、`bias` 不参与训练
  - 只训练 `lora_A`、`lora_B`

- `lora_parameters()`：
  - 返回 LoRA 需要优化的参数列表

- `merged_weight()`：
  - 计算融合后的等效权重 `W + (alpha/r) * (B @ A)`

- `merge_weights()`：
  - 把 LoRA 增量并入 `weight`，并将 `lora_B` 清零
  - 常用于推理前“合并权重”

## 5. 参数量为什么会减少

原线性层可训练参数（忽略 bias）：

```text
out_features * in_features
```

LoRA 可训练参数：

```text
r * in_features + out_features * r
= r * (in_features + out_features)
```

当 `r` 很小时，训练参数显著减少。

## 6. 运行方式

在项目根目录执行：

```powershell
python LoRA/model.py
```

会看到：
- 可训练参数数量
- 训练过程中的 loss
- `Final MSE`
- 合并后 `MSE after merge`

## 7. 常见调参建议

- `r`：越大表示容量越强，但参数更多
- `alpha`：影响 LoRA 分支幅度（通过 `alpha/r` 缩放）
- `dropout`：可用于抑制过拟合
- 学习率：LoRA 参数通常可用稍大的学习率

## 8. 可扩展方向

- 给 Transformer 的 `q_proj/k_proj/v_proj/o_proj` 接 LoRA
- 支持按层选择性挂载 LoRA
- 保存/加载 LoRA adapter（只存 A/B）
- 合并后导出推理模型
