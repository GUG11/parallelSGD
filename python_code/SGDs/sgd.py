import numpy as np
from copy import deepcopy
import multiprocessing
from sharedmem import sharedmem
import time
import graph
import os
from misc import utils
import ctypes


class SGD_profile:
    """
    tracking SGD time and objective function
    """
    def __init__(self, T):
        self.T = T
        self.objs = []
        self.times = []


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
        cc, _ = utils.xcorr(np.abs(X))
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
    if np.any(np.isnan(learner.w)):
        raise RuntimeError('SGD diverges. learning rate=%f' % learning_rate)
    return learner.w


def serial_sgd(learner, X, y, w, gamma=0.0001, max_iter=100, tol=0.01,
        record_settings={'print_period': 100, 'save_period': 1}, S=None):
    """
    serial SGD
        w <- w - gamma * dl / dw
    :param learner: learner
    :param X: data
    :param y: target
    :param w: initial parameters of model (volatile)
    :param gamma: relaxation factor
    :param max_iter: maximum number of iterations
    :param tol: the tolerated relative error: ||Xw - y|| / ||y||
    :param S: set of indexes used (list)
    :return: weight, SGD_profiling
    """
    S = np.arange(X.shape[0]) if S is None else S
    if type(w) is str:
        w = sa.attach(w)
    n, d = len(S), X.shape[1]
    print_period = record_settings['print_period']
    sgd_prof = SGD_profile(record_settings['save_period'])
    print("Serial SGD: size (%d, %d)" % (n, d))
    for i in xrange(max_iter):
        t_start =time.time()
        si = S[np.random.randint(n)]
        learner.update(X[si], y[si], gamma, w)
        t_end = time.time()
        if i % sgd_prof.T == 0:
            sgd_prof.objs.append(learner.compute_loss(X, y, w))
        sgd_prof.times.append(t_end - t_start)
        if i % print_period == 0:
            print("pid: %d, epoch: %d, data index:%d, obj = %f, time = %f" %
                  (os.getpid(), i, si, sgd_prof.objs[i / sgd_prof.T], sum(sgd_prof.times[i-print_period:i])))
        # if objs[-1] / module_y < tol:
        #     break
    return w, sgd_prof


def parallel_sgd(learner, X, y, w, data_partition=None, gamma=0.0001, max_iter=100, tol=0.01,
                 record_settings={'print_period': 100, 'save_period': 1}, P=1):
    """
    parallel SGD
        split data into P subsets
        dispatch each set across cores from 1 to P
        SGD updates in each core
        collects updates in each core and average them
    :param learner: learner
    :param X: data
    :param y: target
    :param w: initial parameters of model 
    :param gamma: relaxation factor
    :param data_partition: groups of indexes of X to be assigned on different cores
    :param max_iter: maximum number of iterations
    :param tol: the tolerated relative error: ||Xw - y|| / ||y||
    :return: weight, [objs]: object function values in each iteration
    :param P: number of cores
    :return: weights, SGD profilings
    """
    n, d = X.shape
    # parallel
    P = min(P, multiprocessing.cpu_count())
    # create shared memory
    shared_X = sharedmem.copy(X)
    shared_y = sharedmem.copy(y)
    print("Parallel SGD: size (%d, %d), cores: %d" % (n, d, P))
    pool = multiprocessing.Pool(P)
    weights = [deepcopy(w) for p in xrange(P)]
    # partition data
    if data_partition is None:
        SS = split_data(X, P, 'random')
    elif isinstance(data_partition, graph.ConflictGraph):
        SS = data_partition.gen_partition(P)
    else:
        SS = [np.random.permutation(data_partition[p]) for
                    p in xrange(P)]
    results = [pool.apply_async(serial_sgd,
                   args=(learner, shared_X, shared_y, weights[p],
                    gamma, max_iter, tol, record_settings, SS[p]))
                for p in xrange(P)]
    sgd_profs = [res.get()[1] for res in results]
    w_updates = np.array([res.get()[0] for res in results])
    # average
    w = np.average(w_updates, 0)
    return w, sgd_profs


# def hogwild(learner, X, y, w, gamma=0.0001, max_iter=100, tol=0.01, 
#     record_settings={'print_period': 100, 'save_period': 1}, P=1):
#     """
#     Hogwild lock-free multithreading SGD
#     :param learner: learner
#     :param X: data
#     :param y: target
#     :param w: initial parameters of model (volatile)
#     :param gamma: relaxation factor
#     :param max_iter: maximum number of iterations
#     :param tol: the tolerated relative error: ||Xw - y|| / ||y||
#     :return: weight, [objs]: object function values in each iteration
#     :param P: number of cores
#     :return: weights, [objs]: object function values in each iterations, [times]: time consumption in each iteration
#     """
#     n, d = X.shape
#     objs, time_cost = [], []
#     # parallel
#     P = min(P, multiprocessing.cpu_count())
#     # create shared memory
#     shared_X = sharedmem.copy(X)
#     shared_y = sharedmem.copy(y)
#     print("Parallel SGD: size (%d, %d), cores: %d" % (n, d, P))
#     pool = multiprocessing.Pool(P)
#     shared_w = sa.create("weight", w.shape)
#     results = [pool.apply_async(serial_sgd,
#                     args=(learner, shared_X, shared_y, w, gamma, max_iter, tol, record_settings))
#                for p in xrange(P)]
#     sa.delete('weight')
#     return deepcopy(shared_w), 1#, objs, time_cost


def max_learning_rate(sgd_algo, learner, lo, hi, bin_tol, **args):
    """
    find maximum learning rate for sgd algorithm and learner
    :param sgd_algo: SGD algorithm
    :param learner: learner
    :param lo: lower bound
    :param hi: upper bound
    :param bin_tol: tolerance
    :param args: arguments for the SGD algorithm
    :return: maximum learning rate
    """
    trained_learner, objs, time_cost = None, None, None
    mid = hi
    while bin_tol < hi - lo or hi == mid:
        mid = (hi + lo) / 2
        print ('try learning rate: %f', mid)
        try:
            trained_learner, objs, time_cost = sgd_algo(deepcopy(learner), gamma=mid, **args)
            lo = mid
        except RuntimeError:
            print('learning rate %f too large' % mid)
            hi = mid
    return lo, trained_learner, objs, time_cost