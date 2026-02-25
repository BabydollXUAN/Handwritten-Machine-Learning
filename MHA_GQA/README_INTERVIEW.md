# MHA / GQA 极简面试版（1页）

## 1. 一句话定义

- MHA（Multi-Head Attention）：把 `Q/K/V` 切成多个头并行做注意力，再拼接。
- GQA（Grouped-Query Attention）：`Q` 头多，`K/V` 头少；通过共享/复制 KV 头降低开销。

## 2. 核心公式

```text
Attention(Q, K, V) = softmax((QK^T) / sqrt(d_h)) V
```

- `d_h`：单头维度（head_dim）
- 先算相关性，再归一化，再加权求和

## 3. MHA 流程（速背）

1. 线性映射得到 `Q, K, V`
2. 切头：`(B, L, D) -> (B, H, L, d_h)`
3. 计算分数：`QK^T / sqrt(d_h)`
4. 加 mask（可选）
5. `softmax`
6. 乘 `V`
7. 合并头并过输出线性层

## 4. GQA 和 MHA 区别（高频考点）

- MHA：`Hq = Hk = Hv`
- GQA：`Hq > Hk = Hv`

在你这份代码里：
- `Q` 用 `num_query_heads`
- `K/V` 用 `num_kv_heads`
- 把 `K/V` 沿 head 维复制到 `Hq`，再做标准 attention

## 5. 为什么要 GQA

- 降低参数量（KV 投影更小）
- 降低 KV cache 占用（推理更省显存）
- 通常在效果与速度间更均衡，常见于大模型推理优化

## 6. 关键约束（必须会说）

- MHA：`d_model % num_heads == 0`
- GQA：
  - `d_model % num_query_heads == 0`
  - `num_query_heads % num_kv_heads == 0`

## 7. Mask 要点

- 因果 mask：下三角，只看当前位置及之前
- padding mask：屏蔽补齐 token
- 你代码里语义：`True` 可见，`False` 屏蔽

## 8. 复杂度与瓶颈

- 主要瓶颈在注意力分数矩阵：`O(L^2)`
- MHA/GQA 都有 `L^2`，GQA主要优化 KV 侧开销，不改变主导阶

## 9. 面试快问快答

- 问：GQA是不是MQA？
  - 答：MQA是GQA特例，`num_kv_heads = 1`。

- 问：为什么要除 `sqrt(d_h)`？
  - 答：防止点积随维度变大导致 softmax 过饱和，稳定梯度。

- 问：GQA会影响效果吗？
  - 答：可能有轻微影响，但通常能换来更好的推理效率与显存收益。

## 10. 结合本仓库文件

- 代码：`MHA_GQA/model.py`
- 详细说明：`MHA_GQA/README.md`
- 本页用途：面试前 3-5 分钟速看
