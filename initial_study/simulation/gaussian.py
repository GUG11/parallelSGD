import numpy as np
from SGDs import sgd

if __name__ == '__main__':
    n, d = 50000, 10000
    np.random.seed(0)
    X = np.random.randn(n, d)
    w = np.random.uniform(low=0.0, high=1.0, size=(d,))
    y = np.dot(X, w)

    #w1, objs1 = sgd.serial_sgd(X, y, gamma=0.0001, max_iter=1000, tol=1e-10)
    w2, objs2 = sgd.parallel_random_split(X, y, max_iter=1000, gamma=0.0001, P=8, tol=1e-3)
