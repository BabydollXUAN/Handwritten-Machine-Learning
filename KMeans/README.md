# K-means 手写实现说明

本文档对应代码：`KMeans/model.py`

## 1. K-means 是什么

K-means 是一种无监督学习算法，用来把样本划分为 `K` 个簇（cluster）。
核心目标是：让每个样本尽量靠近自己所属簇的中心点（centroid）。

## 2. 优化目标

K-means 最小化的是簇内平方和（WCSS），代码里记为 `inertia_`：

```text
inertia = Σ ||x_i - c_{label_i}||^2
```

- `x_i`：第 `i` 个样本
- `c_{label_i}`：该样本所属簇的中心点

`inertia` 越小，说明聚类越紧凑（但不同 `K` 间不能直接简单比较）。

## 3. 算法流程（与你代码一致）

1. 随机初始化 `K` 个中心点
2. 计算每个样本到所有中心点的距离
3. 把样本分配给最近的中心点（得到标签 `labels`）
4. 对每个簇，重新计算中心点为该簇样本均值
5. 重复 2~4，直到：
   - 达到最大迭代次数 `max_iter`
   - 或中心点变化量 `shift <= tol`

## 4. `model.py` 结构说明

- `KMeansScratch.__init__`
  - `n_clusters`: 簇数量
  - `max_iter`: 最大迭代轮数
  - `tol`: 收敛阈值
  - `random_state`: 随机种子（便于复现）

- `_init_centroids(X)`
  - 从样本里随机选 `K` 个点作为初始中心

- `_euclidean_distances(X, centroids)`
  - 计算每个样本到每个中心的欧氏距离矩阵，形状是 `(n_samples, n_clusters)`

- `fit(X)`
  - 执行完整训练流程，最终得到：
    - `centroids_`：最终中心点
    - `labels_`：每个样本所属簇
    - `inertia_`：簇内平方和

- `predict(X)`
  - 对新样本分配最近簇标签

- `fit_predict(X)`
  - 训练并直接返回 `labels_`

## 5. 如何运行

在项目根目录执行：

```powershell
python KMeans/model.py
```

示例会：
- 构造三团二维模拟数据
- 训练 K-means
- 输出中心点、`inertia`、每个簇样本数量
- 对几个新点做聚类预测

## 6. 参数调优建议

- `n_clusters`
  - 业务先验已知类别数量时直接设定
  - 不确定时可尝试肘部法（Elbow）辅助选择

- `max_iter`
  - 一般 100~300 足够

- `tol`
  - 越小越严格，收敛更慢但更精细

- `random_state`
  - 固定后结果可复现

## 7. 常见问题

- 对特征尺度敏感：
  - 量纲差异很大时应先做标准化（如 z-score）

- 可能陷入局部最优：
  - 可多次不同初始化后选 `inertia` 最小的结果

- 空簇问题：
  - 当前实现中若某簇临时无样本，会保留旧中心点（避免崩溃）

## 8. 可扩展方向

- 加入 k-means++ 初始化
- 支持 `n_init` 多次重启取最优结果
- 记录每轮 `inertia` 变化用于可视化
- 增加二维散点图可视化聚类效果
