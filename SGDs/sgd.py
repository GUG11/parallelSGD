import numpy as np
from copy import deepcopy
import multiprocessing
from sharedmem import sharedmem
import time
import graph
from misc import utils
import loss_functions


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


def sgd_one_update(X, y, w0, gamma, grad_func, random_seq):
    """
    one step SGD, do not use np.dot(), which is automatically parallelized.
    :param X: data (list)
    :param y: target (list)
    :param w0: weights (list)
    :param grad_func: gradient function return the matrix with same size as w0
    :param gamma: relaxation factor
    :param random_seq: update order
    :return: updated weight
    """
    w = w0
    for i in random_seq:
        w -= 2 * gamma * grad_func(X[i], y[i], w)
    return w


def serial_sgd(X, y, gamma=0.0001, w0=None, grad_func=loss_functions.grad_LS, max_iter=100, tol=0.01):
    """
    serial SGD
        w <- w - gamma * dl / dw
    :param X: data
    :param y: target
    :param gamma: relaxation factor
    :param w0: initial weights
    :param grad_func: gradient function
    :param max_iter: maximum number of iterations
    :param tol: the tolerated relative error: ||Xw - y|| / ||y||
    :return: weight, [objs]: object function values in each iteration
    """
    n, d = X.shape
    w = w0 if w0 else np.zeros(d)
    objs = []
    module_y = np.linalg.norm(y)
    print("Serial SGD: size (%d, %d)" % (n, d))
    for i in xrange(max_iter):
        t_start =time.time()
        random_seq = np.random.permutation(n)
        w = sgd_one_update(X, y, w, gamma, grad_func, random_seq)
        t_end = time.time()
        objs.append(np.linalg.norm(np.dot(X, w) - y))
        print("epoch: %d, obj = %f, time = %f" % (i, objs[i], t_end - t_start))
        if objs[-1] / module_y < tol:
            break
    return w, objs


def parallel_random_split(X, y, gamma=0.0001, w0=None,grad_func=loss_functions.grad_LS, max_iter=100, tol=0.01, P=1):
    """
    parallel SGD
        split data into P subsets
        dispatch each set across cores from 1 to P
        SGD updates in each core
        collects updates in each core and average them
    :param X: data
    :param y: target
    :param gamma: relaxation factor
    :param w0: initial weights
    :param grad_func: gradient function
    :param max_iter: maximum number of iterations
    :param tol: the tolerated relative error: ||Xw - y|| / ||y||
    :return: weight, [objs]: object function values in each iteration
    :param P: number of cores
    :return: weights, [objs]: object function values in each iterations
    """
    n, d = X.shape
    w = w0 if w0 else np.zeros(d)
    objs = []
    module_y = np.linalg.norm(y)
    # parallel
    P = min(P, multiprocessing.cpu_count())
    # create shared memory
    shared_X = sharedmem.copy(X)
    shared_y = sharedmem.copy(y)
    print("Parallel SGD: size (%d, %d), cores: %d" % (n, d, P))
    pool = multiprocessing.Pool(P)
    for i in xrange(max_iter):
        t_start = time.time()
        w_par = [deepcopy(w) for x in xrange(P)]
        seq_par = split_data(X, P, 'random')
        results = [pool.apply_async(sgd_one_update,
                                    args=(shared_X, shared_y, w_par[p], gamma, grad_func, seq_par[p]))
                    for p in xrange(P)]
        w_updates = np.array([res.get() for res in results])
        # average
        w = np.average(w_updates, 0)
        t_end = time.time()
        objs.append(np.linalg.norm(np.dot(X, w) - y))
        print("epoch: %d, obj = %f, time = %f" % (i, objs[i], t_end - t_start))
        if objs[-1] / module_y < tol:
            break
    return w, objs


def parallel_correlation_split(X, y, gamma=0.0001, w0=None, grad_func=loss_functions.grad_LS, max_iter=100, tol=0.01, P=1):
    """
    parallel SGD
        split data into P subsets
        dispatch each set across cores from 1 to P
        SGD updates in each core
        collects updates in each core and average them
    :param X: data
    :param y: target
    :param gamma: relaxation factor
    :param w0: initial weights
    :param grad_func: gradient function
    :param max_iter: maximum number of iterations
    :param tol: the tolerated relative error: ||Xw - y|| / ||y||
    :return: weight, [objs]: object function values in each iteration
    :param P: number of cores
    :return: weights, [objs]: object function values in each iterations
    """
    n, d = X.shape
    w = w0 if w0 else np.zeros(d)
    objs = []
    module_y = np.linalg.norm(y)
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
        w_par = [deepcopy(w) for x in xrange(P)]
        for p in xrange(P):
            np.random.shuffle(seq_par[p])
        results = [pool.apply_async(sgd_one_update,
                                    args=(shared_X, shared_y, w_par[p], gamma, grad_func, seq_par[p]))
                   for p in xrange(P)]
        w_updates = np.array([res.get() for res in results])
        # average
        w = np.average(w_updates, 0)
        t_end = time.time()
        objs.append(np.linalg.norm(np.dot(X, w) - y))
        print("epoch: %d, obj = %f, time = %f" % (i, objs[i], t_end - t_start))
        if objs[-1] / module_y < tol:
            break
    return w, objs
