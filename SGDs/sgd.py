import numpy as np
from copy import deepcopy
import multiprocessing
from sharedmem import sharedmem
import time
import graph
from misc import utils


def split_data(X, P, split_mode):
    """
    split data for parallel processing at random
    :param X: data
    :param P: number of cores
    :param split_mode: split mode
    :return: list of P sequences
    """
    if split_mode == 'random':
        n = X.shape[0]
        random_seq = np.random.permutation(n)
        seq_par = [random_seq[x::P] for x in range(P)]
    elif split_mode == 'cross-correlation':
        cc = utils.xcorr(np.abs(X))
        G = graph.gen_corr_graph(np.abs(cc))
        subGs = graph.split_evenly(G, P)
        seq_par = [x.nodes() for x in subGs]
    return seq_par


def sgd_one_update(learner, X, y, learning_rate, random_seq):
    """
    one step SGD, do not use np.dot(), which is automatically parallelized.
    :param learner: learner
    :param X: data (list)
    :param y: target (list)
    :param learning_rate: learning rate
    :param random_seq: update order
    :return: updated weight
    """
    for i in random_seq:
        learner.update(X[i], y[i], learning_rate)
    return learner.w


def serial_sgd(learner, X, y, gamma=0.0001, max_iter=100, tol=0.01):
    """
    serial SGD
        w <- w - gamma * dl / dw
    :param learner: learner
    :param X: data
    :param y: target
    :param gamma: relaxation factor
    :param w0: initial weights
    :param max_iter: maximum number of iterations
    :param tol: the tolerated relative error: ||Xw - y|| / ||y||
    :return: weight, [objs]: object function values in each iteration
    """
    n, d = X.shape
    objs = []
    print("Serial SGD: size (%d, %d)" % (n, d))
    for i in xrange(max_iter):
        t_start =time.time()
        random_seq = np.random.permutation(n)
        sgd_one_update(learner, X, y, gamma, random_seq)
        t_end = time.time()
        objs.append(learner.compute_loss(X, y))
        print("epoch: %d, obj = %f, time = %f" % (i, objs[i], t_end - t_start))
        # if objs[-1] / module_y < tol:
        #     break
    return learner, objs


def parallel_random_split(learner, X, y, gamma=0.0001, max_iter=100, tol=0.01, P=1):
    """
    parallel SGD
        split data into P subsets
        dispatch each set across cores from 1 to P
        SGD updates in each core
        collects updates in each core and average them
    :param learner: learner
    :param X: data
    :param y: target
    :param gamma: relaxation factor
    :param w0: initial weights
    :param max_iter: maximum number of iterations
    :param tol: the tolerated relative error: ||Xw - y|| / ||y||
    :return: weight, [objs]: object function values in each iteration
    :param P: number of cores
    :return: weights, [objs]: object function values in each iterations
    """
    n, d = X.shape
    objs = []
    # module_y = np.linalg.norm(y)
    # parallel
    P = min(P, multiprocessing.cpu_count())
    # create shared memory
    shared_X = sharedmem.copy(X)
    shared_y = sharedmem.copy(y)
    print("Parallel SGD: size (%d, %d), cores: %d" % (n, d, P))
    pool = multiprocessing.Pool(P)
    for i in xrange(max_iter):
        t_start = time.time()
        learners = [deepcopy(learner) for p in xrange(P)]
        seq_par = split_data(X, P, 'random')
        results = [pool.apply_async(sgd_one_update,
                                    args=(learners[p], shared_X, shared_y, gamma, seq_par[p]))
                    for p in xrange(P)]
        w_updates = np.array([res.get() for res in results])
        # average
        learner.w = np.average(w_updates, 0)
        t_end = time.time()
        objs.append(learner.compute_loss(X, y))
        print("epoch: %d, obj = %f, time = %f" % (i, objs[i], t_end - t_start))
        # if objs[-1] / module_y < tol:
        #     break
    return learner, objs


def parallel_correlation_split(learner, X, y, gamma=0.0001, max_iter=100, tol=0.01, P=1):
    """
    parallel SGD
        split data into P subsets
        dispatch each set across cores from 1 to P
        SGD updates in each core
        collects updates in each core and average them
    :param learner: learner
    :param X: data
    :param y: target
    :param gamma: relaxation factor
    :param w0: initial weights
    :param max_iter: maximum number of iterations
    :param tol: the tolerated relative error: ||Xw - y|| / ||y||
    :return: weight, [objs]: object function values in each iteration
    :param P: number of cores
    :return: weights, [objs]: object function values in each iterations
    """
    n, d = X.shape
    objs = []
    # parallel
    P = min(P, multiprocessing.cpu_count())
    seq_par = split_data(X, P, 'cross-correlation')
    # create shared memory
    shared_X = sharedmem.copy(X)
    shared_y = sharedmem.copy(y)
    print("Parallel SGD: size (%d, %d), cores: %d" % (n, d, P))
    pool = multiprocessing.Pool(P)
    for i in xrange(max_iter):
        t_start = time.time()
        learners = [deepcopy(learner) for p in xrange(P)]
        for p in xrange(P):
            np.random.shuffle(seq_par[p])
        results = [pool.apply_async(sgd_one_update,
                                    args=(learners[p], shared_X, shared_y, gamma, seq_par[p]))
                   for p in xrange(P)]
        w_updates = np.array([res.get() for res in results])
        # average
        learner.w = np.average(w_updates, 0)
        t_end = time.time()
        objs.append(learner.compute_loss(X, y))
        print("epoch: %d, obj = %f, time = %f" % (i, objs[i], t_end - t_start))
        # if objs[-1] / module_y < tol:
        #     break
    return learner, objs
