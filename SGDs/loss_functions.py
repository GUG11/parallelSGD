import numpy as np


def sigmoid(a):
    """
        g(a) = 1 / (1 + exp(-a))
    :param a: scalar or vector (point-wise)
    :return: sigmoid value
    """
    sig = 1. / (1. + np.exp(a))
    return sig


def softmax(z):
    """
        softmax(z) = exp(z) / sum(exp(z[:,j])
    :param z: (n x c)
    :return: softmax of z (n x 1)
    """
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
    return sm


def oneHotEncode(y):
    """
    one hot encoding: y = [0,1,2] => [[0,0,0],[0,1,0],[0,0,1]]
    :param y: labels
    :return: one hot encode (n x c)
    """
    d = y.shape[0]
    c = np.max(y) + 1
    OHX = np.zeros((d, c))
    OHX[range(d), y] = 1
    return OHX


def grad_LS(X, y, w):
    """
    gradient of least square loss function
        l(X,y;w) = || Xw - y ||^2
        \nabla l = 2 X.T (Xw - y)
    :param X: data
    :param y: labels
    :param w: weight
    :return: grad (w size)
    """
    grad = 2 * np.dot(X.T, (np.dot(X, w) - y))
    return grad


def grad_logistic(X, y, w):
    """
    gradient of logistic loss
        l(X,y;w) = -log y log h(x;w) + (1-y) log(1-h(x;w))
        h(x;w) = (1 + exp(-<x,w>)^{-1}
        \nabla l = -(y - h(x;w)) x
    :param X: data
    :param y: labels
    :param w: weights
    :return: grad (w size)
    """
    grad = -np.dot(X.T, y - sigmoid(np.dot(X, w)))
    return grad


def grad_softmax(X, y, W):
    """
    gradient of softmax
        P(y) = softmax(Wx) = exp(XW^T) / sum(exp(XW^T), along column)
        l(X,y;W) =
    :param X: data (n x d)
    :param y: labels (n x c)
    :param W: weight (c x d)
    :return: grad (W size)
    """
    m = X.shape[0]
    y_mat = oneHotEncode(y)
    scores = np.dot(X, W.T)
    prob = softmax(scores)
    grad = (-1/m) * np.dot(X.T, (y_mat - prob))
    return grad