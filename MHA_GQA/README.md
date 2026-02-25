# MHA / GQA 手写实现说明

本文档对应代码：`MHA_GQA/model.py`

## 1. 文件里实现了什么

- `MultiHeadAttentionScratch`：标准多头注意力（MHA）
- `GroupedQueryAttentionScratch`：分组查询注意力（GQA）
- 公共能力：
  - 支持 `attn_mask`
  - 支持 `is_causal=True` 因果掩码
  - 支持返回注意力权重 `need_weights=True`

实现是“手写 attention 逻辑”，没有使用 `nn.MultiheadAttention`。

## 2. MHA 和 GQA 的核心区别

- MHA：`Q/K/V` 头数相同
- GQA：`Q` 头数更多，`K/V` 头数更少

在本实现中，GQA 的做法是：

1. `Q` 按 `num_query_heads` 切头
2. `K/V` 按 `num_kv_heads` 切头
3. 将 `K/V` 头按 `repeat_factor = num_query_heads // num_kv_heads` 重复到和 `Q` 同头数
4. 再执行标准缩放点积注意力

这样可以减少 KV 投影参数量和缓存开销（尤其用于大模型推理）。

## 3. 张量维度流转

设：
- 批大小 `B`
- 序列长度 `L`
- 模型维度 `D`
- 头数 `H`
- 单头维度 `Dh`

### MHA

- 输入：`(B, L, D)`
- 线性投影后：`Q/K/V` 仍是 `(B, L, D)`
- 切头后：`(B, H, L, Dh)`，其中 `Dh = D / H`
- 注意力分数：`(B, H, L, L)`
- 输出合并：`(B, L, D)`

### GQA

- `Q` 切头：`(B, Hq, L, Dh)`
- `K/V` 切头：`(B, Hkv, L, Dh)`
- 重复 `K/V` 后：`(B, Hq, L, Dh)`
- 后续与 MHA 相同，输出 `(B, L, D)`

## 4. 关键公式

缩放点积注意力：

```text
Attention(Q, K, V) = softmax((QK^T) / sqrt(Dh)) V
```

代码中掩码规则：
- `True` 表示可见
- `False` 表示被屏蔽

被屏蔽位置会填充很小值（`-1e9`）后再做 `softmax`。

## 5. 代码结构对应

- `_normalize_attn_mask(...)`
  - 统一掩码到 `(B, 1, Q, K)` 布尔格式
- `_build_causal_mask(...)`
  - 构建下三角因果掩码
- `_scaled_dot_product_attention(...)`
  - 手写 `QK^T -> softmax -> @V`
- `MultiHeadAttentionScratch`
  - 标准 MHA 投影、切头、注意力、合并
- `GroupedQueryAttentionScratch`
  - GQA 的 Q/KV 不同头数与 KV 复制逻辑

## 6. 参数约束

### MHA

- `d_model % num_heads == 0`

### GQA

- `d_model % num_query_heads == 0`
- `num_query_heads % num_kv_heads == 0`

如果不满足会抛 `ValueError`。

## 7. 运行方式

在项目根目录执行：

```powershell
python MHA_GQA/model.py
```

预期输出会包含：
- `mha_out shape: (2, 6, 32)`
- `mha_attn shape: (2, 4, 6, 6)`
- `gqa_out shape: (2, 6, 32)`
- `gqa_attn shape: (2, 8, 6, 6)`

## 8. 可扩展方向

- 增加 `key_padding_mask` 支持
- 增加 KV cache（自回归推理常用）
- 增加 FlashAttention 路径（更高效）
- 增加多次单元测试（形状、掩码、数值稳定性）
