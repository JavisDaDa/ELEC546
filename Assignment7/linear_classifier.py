import numpy as np
from linear_svm import svm_loss_vectorized


class LinearClassifier(object):

    def __init__(self):
        self.theta = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200,
              verbose=False):
        m, d = X.shape
        K = np.max(y) + 1
        if self.theta is None:
            # lazily initialize theta
            if K == 2:
                self.theta = 0.001 * np.random.randn((d,))
            else:
                self.theta = 0.001 * np.random.randn(d, K)
        loss_history = []
        for it in range(num_iters):

            if batch_size < X.shape[0]:
                rand_idx = np.random.choice(m, batch_size)
                X_batch = X[rand_idx, :]
                y_batch = y[rand_idx]
            else:
                X_batch = X
                y_batch = y
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.theta = self.theta - learning_rate * grad
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return loss_history

    def predict(self, X):
        pass

    def loss(self, X, y, reg):
        pass


class LinearSVM(LinearClassifier):

    def loss(self, X, y, reg):
        return svm_loss_vectorized(self.theta, X, y, reg)

    def predict(self, X):
        h = X@self.theta
        y_pred = np.argmax(h, axis=1)
        return y_pred
