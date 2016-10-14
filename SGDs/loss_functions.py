import numpy as np
from abc import ABCMeta, abstractmethod

sigmoid = lambda x: 1. / (1. + np.exp(x))

def softmax(z):
    """
        softmax(z) = exp(z) / sum(exp(z[:,j])
    :param z: (n x c)
    :return: softmax of z (n x 1)
    """
    expz = np.exp(z)
    sm = None
    if np.ndim(z) == 1:
        sm = expz / sum(expz)
    else:
        sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
    return sm


def oneHotEncode(y):
    """
    one hot encoding: y = [0,1,2] => [[0,0,0],[0,1,0],[0,0,1]]
    :param y: labels
    :return: one hot encode (n x c)
    """
    d = y.shape[0]
    c = int(np.max(y)) + 1
    OHX = np.zeros((d, c))
    OHX[range(d), y] = 1
    return OHX


def oneHotDecode(y_mat):
    """
    one hot decoding: y = [[0,0,0],[0,1,0],[0,0,1]] => [0,1,2]
    :param y_mat: one-hot y matrix (n x c)
    :return: y (n x 1)
    """
    return np.argmax(y_mat, axis=1)


class Learner:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, w0):
        self._type = None
        self.w = w0

    @abstractmethod
    def compute_loss(self, X, y): pass

    @abstractmethod
    def compute_grad(self, X, y): pass

    def update(self, X, y, learning_rate):
        grad = self.compute_grad(X, y)
        self.w -= learning_rate * grad

    @abstractmethod
    def predict(self, X): pass


class LeastSquare(Learner):
    def __init__(self, w0):
        super(LeastSquare, self).__init__(w0=w0)
        self._type = 'regressor'

    def compute_loss(self, X, y):
        """
        loss function of least square
        :param X: data
        :param y: labels
        :return: loss
        """
        e = np.dot(X, self.w) - y
        loss = sum(e * e)
        return loss

    def compute_grad(self, X, y):
        """
        gradient of least square loss function
            l(X,y;w) = || Xw - y ||^2
            \nabla l = 2 X.T (Xw - y)
        :param X: data
        :param y: labels
        :param w: weight
        :return: grad (w size)
        """
        grad = 2 * np.dot(X.T, (np.dot(X, self.w) - y))
        return grad

    def predict(self, X):
        """
        predict the target
        :param X: data
        :return: target
        """
        return np.dot(X, self.w)


class Logistic(Learner):
    def __init__(self, w0):
        super(Logistic, self).__init__(w0=w0)
        self._type = 'classifier'

    def compute_loss(self, X, y):
        """
        loss function of logistic loss
            l(X,y;w) = -(y log h(x;w) + (1-y) log(1-h(x;w))
            h(x;w) = (1 + exp(-<x,w>)^{-1}
        :param X: data
        :param y: labels
        :param w: weights
        :return: grad (w size)
        """
        h = sigmoid(np.dot(X, self.w))
        loss = -(y * np.log(h) + (1 - y) * np.log(1 - h))
        return loss

    def compute_grad(self, X, y):
        """
        gradient of logistic loss
            l(X,y;w) = - (y log h(x;w) + (1-y) log(1-h(x;w))
            h(x;w) = (1 + exp(-<x,w>)^{-1}
            \nabla l = -(y - h(x;w)) x
        :param X: data
        :param y: labels
        :param w: weights
        :return: grad (w size)
        """
        grad = np.dot(X.T, sigmoid(np.dot(X, self.w)) - y)
        return grad

    def predict(self, X):
        """
        predict the target
        :param X: data
        :return: prediction
        """
        h = sigmoid(np.dot(X, self.w))
        pred = np.where(h > 0.5, 1, 0)
        return pred



class Softmax(Learner):
    def __init__(self, w0, oneHot):
        super(Softmax, self).__init__(w0=w0)
        self._type = 'classifier'
        self._oneHot = oneHot

    def compute_loss(self, X, y):
        """
        loss function of softmax
            P(y) = softmax(Wx) = exp(XW^T) / sum(exp(XW^T), along column)
            l(X,y;W) =
        :param X: data (n x d)
        :param y: labels (n x c)
        :param W: weight (c x d)
        :param oneHot: whether do one-hot encoding
        :return: grad (W size)
        """
        y_mat = oneHotEncode(y) if self._oneHot else y
        prob = self.compute_prob(X)
        loss = -np.sum(y_mat * np.log(prob))
        return loss

    def compute_grad(self, X, y):
        """
        gradient of softmax
            P(y) = softmax(Wx) = exp(XW^T) / sum(exp(XW^T), along column)
            l(X,y;W) =
        :param X: data (n x d)
        :param y: labels (n x c)
        :param W: weight (c x d)
        :param oneHot: whether do one-hot encoding
        :return: grad (W size)
        """
        y_mat = oneHotEncode(y) if self._oneHot else y
        prob = self.compute_prob(X)
        grad = (np.dot((prob - y_mat).T, X) if np.ndim(X) > 1
                 else np.outer(prob - y_mat, X))
        return grad

    def compute_prob(self, X):
        """
        compute probability of each class
        :param X: data
        :return: prob
        """
        scores = np.dot(X, self.w.T)
        prob = softmax(scores)
        return prob

    def predict(self, X):
        """
        predict the target
        :param X: data
        :return: target
        """
        prob = self.compute_prob(X)
        pred = np.argmax(prob, axis=1)
        return pred


def relative_error(y_esti, y_true):
    return np.linalg.norm(y_esti - y_true) / np.linalg.norm(y_true)


def accuracy(y_pred, y_true):
    no_cor = sum(y_pred == y_true)
    no_ttl = y_true.shape[0]
    return float(no_cor) / no_ttl
