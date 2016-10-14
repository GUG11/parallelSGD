"""A very simple MNIST classifier"""
import numpy as np
from copy import deepcopy
import multiprocessing
from sharedmem import sharedmem
import time
from SGDs import sgd
import os
from SGDs.loss_functions import Softmax

# import data
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    mnist = input_data.read_data_sets(os.path.join('data', 'MNIST'), one_hot=True)
    n_train, d = mnist.train.images.shape
    n_validation = mnist.validation.images.shape[0]
    n_test = mnist.test.images.shape[0]

    n, c = 1000, 10
    np.random.seed(0)
    X, y = mnist.train.next_batch(n)
    w0 = np.zeros((c, d))
    # w0 = np.random.uniform(low=0.0, high=1.0, size=(c, d))
    max_iter = 100
    softmax_learner = Softmax(w0, oneHot=False)

    # w1, objs1 = sgd.serial_sgd(softmax_learner, X, y, w0=w0, gamma=0.01, max_iter=max_iter, tol=1e-10)
    sm1, objs2 = sgd.parallel_random_split(deepcopy(softmax_learner), X, y, max_iter=max_iter, gamma=0.01, P=8, tol=1e-3)
    sm2, objs3 = sgd.parallel_correlation_split(softmax_learner, X, y, max_iter=max_iter, gamma=0.01, P=8, tol=1e-3)
