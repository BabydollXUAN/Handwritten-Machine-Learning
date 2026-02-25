import numpy as np


class LinearRegressionScratch:
    """Linear regression implemented from scratch with NumPy."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        fit_intercept: bool = True,
        method: str = "gd",
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.method = method
        self.coef_ = None
        self.intercept_ = 0.0
        self.theta_ = None

    def _prepare_features(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1), dtype=float)
            X = np.hstack((ones, X))
        return X

    def fit(self, X, y, verbose: bool = False):
        X = self._prepare_features(X)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        if self.method == "gd":
            self.theta_ = self._fit_gradient_descent(X, y, verbose)
        elif self.method == "normal":
            self.theta_ = self._fit_normal_equation(X, y)
        else:
            raise ValueError("method must be 'gd' or 'normal'.")

        if self.fit_intercept:
            self.intercept_ = float(self.theta_[0])
            self.coef_ = self.theta_[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.theta_

        return self

    def _fit_gradient_descent(self, X, y, verbose: bool):
        m, n = X.shape
        theta = np.zeros(n, dtype=float)

        for epoch in range(self.epochs):
            y_pred = X @ theta
            error = y_pred - y
            gradient = (X.T @ error) / m
            theta -= self.learning_rate * gradient

            if verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                loss = np.mean(error ** 2) / 2
                print(f"epoch={epoch:4d}, loss={loss:.6f}")

        return theta

    def _fit_normal_equation(self, X, y):
        return np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, X):
        if self.theta_ is None:
            raise ValueError("Model is not fitted yet.")
        X = self._prepare_features(X)
        return X @ self.theta_

    def mse(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        return float(np.mean((y_pred - y) ** 2))

    def score(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot)


if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic data: y = 4 + 3x + noise
    X = 2 * np.random.rand(200, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(200) * 0.5

    model = LinearRegressionScratch(
        learning_rate=0.1,
        epochs=2000,
        fit_intercept=True,
        method="gd",
    )
    model.fit(X, y, verbose=True)

    print("\\nTrained parameters:")
    print(f"intercept: {model.intercept_:.4f}")
    print(f"coef: {model.coef_}")
    print(f"mse: {model.mse(X, y):.4f}")
    print(f"r2: {model.score(X, y):.4f}")

    sample = np.array([[0.0], [1.0], [2.0]])
    print("predictions for x=[0,1,2]:", model.predict(sample))
