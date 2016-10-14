"""A very simple MNIST classifier"""
import numpy as np
from copy import deepcopy
from SGDs import sgd
import os
from SGDs.loss_functions import Softmax, accuracy, oneHotDecode



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

    sm1, objs1, time_cost1 = sgd.serial_sgd(deepcopy(softmax_learner), X, y, gamma=0.5, max_iter=max_iter, tol=1e-10)
    print('train accuracy:%f, test accuracy:%f' % (accuracy(sm1.predict(X), oneHotDecode(y)),
        accuracy(sm1.predict(mnist.test.images), oneHotDecode(mnist.test.labels))))
    sm2, objs2, time_cost2 = sgd.parallel_random_split(deepcopy(softmax_learner), X, y, max_iter=max_iter, gamma=1, P=8, tol=1e-3)
    print('train accuracy:%f, test accuracy:%f' % (accuracy(sm2.predict(X), oneHotDecode(y)),
        accuracy(sm2.predict(mnist.test.images), oneHotDecode(mnist.test.labels))))
    sm3, objs3, time_cost3 = sgd.parallel_correlation_split(softmax_learner, X, y, max_iter=max_iter, gamma=1, P=8, tol=1e-3)
    print('train accuracy:%f, test accuracy:%f' % (accuracy(sm3.predict(X), oneHotDecode(y)),
        accuracy(sm3.predict(mnist.test.images), oneHotDecode(mnist.test.labels))))

    # draw
    save_path = os.path.join('results', 'real_data')
    # save variables

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.savez(os.path.join(save_path, 'mnist.dat'), losses=(objs1, objs2, objs3),
             time_costs=(time_cost1, time_cost2, time_cost3))
