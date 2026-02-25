# 线性回归手写实现说明

本文档对应文件：`LinearRegression/model.py`

## 1. 线性回归是什么

线性回归用于拟合输入特征 `X` 和目标值 `y` 的线性关系。

单特征时可写成：

```text
y_hat = w * x + b
```

多特征时可写成：

```text
y_hat = Xw + b
```

其中：
- `w` 是权重（coef）
- `b` 是偏置（intercept）

## 2. 训练目标（损失函数）

代码里使用均方误差（MSE）来衡量预测误差：

```text
MSE = (1/m) * Σ(y_hat - y)^2
```

梯度下降里常用 `1/2 * MSE` 便于求导（`model.py` 中的 `loss` 打印就是这个形式）。

## 3. 参数怎么学出来

本实现支持两种方法：

1. `method="gd"`：梯度下降
- 初始化参数为 0
- 反复计算梯度并更新参数：

```text
theta = theta - learning_rate * gradient
```

2. `method="normal"`：正规方程
- 直接闭式解：

```text
theta = (X^T X)^(-1) X^T y
```

代码中用 `pinv`（伪逆）增强数值稳定性。

## 4. `model.py` 的结构说明

- `LinearRegressionScratch.__init__`
  - 配置学习率、迭代轮数、是否拟合截距、训练方法
- `_prepare_features`
  - 输入转 `numpy`，并在需要时给 `X` 前加一列 1（用于截距）
- `fit`
  - 训练入口，按 `method` 选择梯度下降或正规方程
- `_fit_gradient_descent`
  - 梯度下降核心循环
- `_fit_normal_equation`
  - 正规方程求解
- `predict`
  - 用训练好的参数预测
- `mse`
  - 计算均方误差
- `score`
  - 计算 `R^2` 决定系数

## 5. 如何运行

在项目根目录执行：

```powershell
python LinearRegression/model.py
```

程序会：
- 生成一组模拟数据 `y = 4 + 3x + noise`
- 训练模型
- 输出 `intercept`、`coef`、`mse`、`r2`
- 打印样例输入 `x=[0,1,2]` 的预测结果

## 6. 结果如何理解

- `intercept` 接近 4、`coef` 接近 3，说明模型学到了真实生成规则
- `mse` 越小越好
- `r2` 越接近 1 越好

## 7. 可扩展方向

- 加入特征标准化（提升梯度下降稳定性）
- 加入 L2 正则（Ridge）防止过拟合
- 支持 mini-batch / SGD
- 支持训练集-测试集划分与可视化
