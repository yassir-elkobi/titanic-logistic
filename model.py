import numpy as np
from tqdm import tqdm


class LogisticRegression:
    """
    Manual logistic regression with:
      - batch gradient descent
      - L2 regularization
      - early stopping on validation loss
      - numerical stability tweaks
    """

    def __init__(self, learning_rate=0.01, max_iterations=1000, lambda_=0.0, tolerance=1e-4, patience=5, verbose=True):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.lambda_ = lambda_
        self.tolerance = tolerance
        self.patience = patience
        self.verbose = verbose

    @staticmethod
    def sigmoid(z):
        # prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_cost(self, X, y):
        m = y.shape[0]
        h = self.sigmoid(X.dot(self.weights))
        # cross-entropy + L2
        eps = 1e-15
        cost = -(1 / m) * (y.T.dot(np.log(h + eps)) + (1 - y).T.dot(np.log(1 - h + eps)))
        reg = (self.lambda_ / (2 * m)) * np.sum(self.weights[1:] ** 2)
        return float(cost + reg)

    def _compute_grad(self, X, y):
        m = y.shape[0]
        h = self.sigmoid(X.dot(self.weights))
        error = h - y
        grad = (1 / m) * (X.T.dot(error))
        # L2, skip bias
        grad[1:] += (self.lambda_ / m) * self.weights[1:]
        return grad

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        m, n = X_train.shape
        # init weights
        self.weights = np.zeros((n, 1))
        self.cost_history = []
        self.val_cost_history = []
        best_val_cost = float('inf')
        best_weights = None
        patience_ctr = 0
        use_early = X_val is not None and y_val is not None

        iterator = range(self.max_iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc="Training")

        for i in iterator:
            grad = self._compute_grad(X_train, y_train)
            self.weights -= self.learning_rate * grad
            train_cost = self._compute_cost(X_train, y_train)
            self.cost_history.append(train_cost)

            if use_early:
                val_cost = self._compute_cost(X_val, y_val)
                self.val_cost_history.append(val_cost)
                if val_cost + self.tolerance < best_val_cost:
                    best_val_cost = val_cost
                    best_weights = self.weights.copy()
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at iteration {i + 1}")
                    self.weights = best_weights
                    break

            if self.verbose and use_early:
                iterator.set_description(f"Train cost {train_cost:.4f}, val cost {val_cost:.4f}")
            elif self.verbose:
                iterator.set_description(f"Train cost {train_cost:.4f}")

        return self

    def predict_proba(self, X):
        return self.sigmoid(X.dot(self.weights))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
