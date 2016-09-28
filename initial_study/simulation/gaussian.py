import numpy as np
from SGDs import sgd

if __name__ == '__main__':
    n, d = 500, 100
    np.random.seed(0)
    X = np.random.randn(n, d)
    w = np.random.uniform(low=0.0, high=1.0, size=(d,))
    y = np.dot(X, w)
    max_iter = 100

    #w1, objs1 = sgd.serial_sgd(X, y, gamma=0.0001, max_iter=1000, tol=1e-10)
    w2, objs2 = sgd.parallel_random_split(X, y, max_iter=max_iter, gamma=0.0001, P=8, tol=1e-3)
    w3, objs3 = sgd.parallel_correlation_split(X, y, max_iter=max_iter, gamma=0.0001, P=8, tol=1e-3)