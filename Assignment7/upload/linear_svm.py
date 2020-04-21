import numpy as np


def svm_loss_vectorized(theta, X, y, C):
    delta = 1.0
    m = X.shape[0]
    h = X@theta
    hy = h - h[range(len(y)), y].reshape((-1, 1))
    hy[hy != 0] += delta
    cur = np.maximum(0.0, hy)
    J = np.sum(np.square(theta)) / 2 / m + C * np.sum(cur) / m
    cc = (hy > 0) * 1
    cc[range(len(y)), y] = -np.sum(cc, axis=1)
    dtheta = theta / m + C * X.T@cc / m

    return J, dtheta
