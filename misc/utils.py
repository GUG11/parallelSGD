import numpy as np

def xcorr(X):
    """
    cross correlation between arbitrary two lines of X
    :param X: data (n x d)
    :return: correlation matrix (n x n)
    """
    XXT = np.dot(X, X.T)
    xnorm = np.linalg.norm(X, 2, axis=1)
    Xnorm = np.outer(xnorm, xnorm)
    cc = XXT / Xnorm
    return cc