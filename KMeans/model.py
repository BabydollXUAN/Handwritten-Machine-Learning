import numpy as np
from typing import Optional


class KMeansScratch:
    """K-means clustering implemented from scratch with NumPy."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    @staticmethod
    def _euclidean_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        # Shape: (n_samples, n_clusters)
        return np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters.")

        centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            distances = self._euclidean_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(axis=0)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift <= self.tol:
                break

        final_distances = self._euclidean_distances(X, centroids)
        final_labels = np.argmin(final_distances, axis=1)

        self.centroids_ = centroids
        self.labels_ = final_labels
        self.inertia_ = float(np.sum((X - centroids[final_labels]) ** 2))
        return self

    def predict(self, X):
        if self.centroids_ is None:
            raise ValueError("Model is not fitted yet.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        distances = self._euclidean_distances(X, self.centroids_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


if __name__ == "__main__":
    np.random.seed(42)

    # Build three synthetic 2D clusters.
    cluster_1 = np.random.randn(100, 2) + np.array([0, 0])
    cluster_2 = np.random.randn(100, 2) + np.array([5, 5])
    cluster_3 = np.random.randn(100, 2) + np.array([0, 6])
    X = np.vstack([cluster_1, cluster_2, cluster_3])

    model = KMeansScratch(n_clusters=3, max_iter=200, tol=1e-5, random_state=42)
    labels = model.fit_predict(X)

    print("Centroids:")
    print(model.centroids_)
    print("Inertia: {:.4f}".format(model.inertia_))

    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster sizes:", dict(zip(unique.tolist(), counts.tolist())))

    samples = np.array([[0, 0], [5, 5], [0, 6], [2, 2]])
    print("Predictions:", model.predict(samples))
