"""A very simple MNIST classifier"""
import numpy as np
from copy import deepcopy
from SGDs import sgd, graph
import os
from SGDs.loss_functions import Softmax, accuracy, oneHotDecode
from misc import utils


# import data
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    mnist = input_data.read_data_sets(os.path.join('data', 'MNIST'), one_hot=True)
    n_train, d = mnist.train.images.shape
    n_validation = mnist.validation.images.shape[0]
    n_test = mnist.test.images.shape[0]

    n, c = 5000, 10
    np.random.seed(0)
    X, y = mnist.train.next_batch(n)
    w0 = np.zeros((c, d))
    # w0 = np.random.uniform(low=0.0, high=1.0, size=(c, d))
    max_iter = 100
    softmax_learner = Softmax(w0, oneHot=False)
    P = 2  # CPU cores

    # binary search find maximum learning rate
    gamma1, sm1, objs1, time_cost1 = sgd.max_learning_rate(sgd.serial_sgd, softmax_learner, 0.001, 5, 0.1,
        X=X, y=y, max_iter=max_iter, tol=1e-10)
    print('Serial SGD maximum learning rate=%f' % gamma1)
    print('train accuracy:%f, test accuracy:%f' % (accuracy(sm1.predict(X), oneHotDecode(y)),
        accuracy(sm1.predict(mnist.test.images), oneHotDecode(mnist.test.labels))))

    gamma2, sm2, objs2, time_cost2 = sgd.max_learning_rate(sgd.parallel_sgd, softmax_learner, 0.001, 5, 0.1,
                                   X=X, y=y, max_iter=max_iter, tol=1e-10, P=P)
    print('Parallel SGD (random split) maximum learning rate=%f' % gamma2)
    print('train accuracy:%f, test accuracy:%f' % (accuracy(sm2.predict(X), oneHotDecode(y)),
        accuracy(sm2.predict(mnist.test.images), oneHotDecode(mnist.test.labels))))

    # seq_par = sgd.split_data(X, P, 'cross-correlation')
    # gamma3 = sgd.max_learning_rate(sgd.parallel_sgd, softmax_learner, 0.001, 5, 0.1,
    #                                data_partition=seq_par, X=X, y=y, max_iter=20, tol=1e-10, P=P)
    # print ('Parallel SGD (random split) maximum learning rate=%f' % gamma3)
    # sm3, objs3, time_cost3 = sgd.parallel_sgd(softmax_learner, X, y, data_partition=seq_par, max_iter=max_iter, gamma=gamma3 * 0.8, P=P, tol=1e-3)
    # print('train accuracy:%f, test accuracy:%f' % (accuracy(sm3.predict(X), oneHotDecode(y)),
    #     accuracy(sm3.predict(mnist.test.images), oneHotDecode(mnist.test.labels))))


    max_deg = 100
    cc, _ = utils.xcorr(X)
    G = graph.gen_corr_graph(cc, max_deg=max_deg)
    cg = graph.ConflictGraph(G)
    gamma4, sm4, objs4, time_cost4 = sgd.max_learning_rate(sgd.parallel_sgd, softmax_learner, 0.001, 5, 0.1,
                                   data_partition=cg, X=X, y=y, max_iter=max_iter * max_deg, tol=1e-10, P=P)
    print ('Parallel SGD (CYCLADES) maximum learning rate=%f' % gamma4)
    objs4 = [objs4[i] for i in xrange(0, len(objs4), max_deg)]
    time_cost4 = [sum(time_cost4[i:i+max_deg]) for i in xrange(0, len(objs4), max_deg)]
    print('train accuracy:%f, test accuracy:%f' % (accuracy(sm4.predict(X), oneHotDecode(y)),
                                                   accuracy(sm4.predict(mnist.test.images), oneHotDecode(mnist.test.labels))))
    # draw
    save_path = os.path.join('results', 'real_data_%d' % n)
    # save variables

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.savez(os.path.join(save_path, 'mnist.dat'), learning_rate=(gamma1, gamma2, gamma4),
             losses=(objs1, objs2, objs4),
             time_costs=(time_cost1, time_cost2, time_cost4))
