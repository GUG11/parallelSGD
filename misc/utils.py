import numpy as np

def xcorr(X):
    """
    cross correlation between arbitrary two lines of X
    :param X: data (n x d)
    :return: correlation matrix (n x n) [0]: absolute, [1]: relative
    """
    XXT = np.dot(X, X.T)
    xnorm = np.linalg.norm(X, 2, axis=1)
    Xnorm = np.outer(xnorm, xnorm)
    rcc = XXT / Xnorm
    return XXT, rcc