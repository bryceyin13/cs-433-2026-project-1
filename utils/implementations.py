import numpy as np


def _mse(y, tx, w):
    """Return the mean squared error with the required factor 1/2."""
    error = y - tx @ w
    return error @ error / (2 * len(y))


def _sigmoid(z):
    """Evaluate 1 / (1 + exp(-z)) using a stable exponential."""
    return np.exp(-np.logaddexp(0, -z))


def _logistic_loss(y, tx, w):
    """Return the mean negative log likelihood, without a penalty."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Minimize MSE by full-batch gradient descent; return (w, MSE)."""
    w = initial_w.copy()
    for _ in range(max_iters):
        gradient = tx.T @ (tx @ w - y) / len(y)
        w = w - gamma * gradient
    return w, _mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Minimize MSE with one random sample per step; return (w, full MSE)."""
    w = initial_w.copy()
    for _ in range(max_iters):
        index = np.random.randint(len(y))
        gradient = tx[index] * (tx[index] @ w - y[index])
        w = w - gamma * gradient
    return w, _mse(y, tx, w)


def least_squares(y, tx):
    """Solve the least-squares normal equations; return (w, MSE)."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, _mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Minimize MSE + lambda_ * ||w||^2; return (w, unpenalized MSE)."""
    # The factor 2N follows from the 1/(2N) normalization of MSE.
    penalty = 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    w = np.linalg.solve(tx.T @ tx + penalty, tx.T @ y)
    return w, _mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Fit logistic regression by GD for y in {0, 1}; return (w, mean NLL)."""
    w = initial_w.copy()
    for _ in range(max_iters):
        gradient = tx.T @ (_sigmoid(tx @ w) - y) / len(y)
        w = w - gamma * gradient
    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Minimize mean NLL + lambda_ * ||w||^2; return (w, unpenalized NLL)."""
    w = initial_w.copy()
    for _ in range(max_iters):
        gradient = tx.T @ (_sigmoid(tx @ w) - y) / len(y) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w, _logistic_loss(y, tx, w)
