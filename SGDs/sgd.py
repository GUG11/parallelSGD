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
    objs, time_cost = [], []
    print("Serial SGD: size (%d, %d)" % (n, d))
    for i in xrange(max_iter):
        t_start =time.time()
        random_seq = np.random.permutation(n)
        sgd_one_update(learner, X, y, gamma, random_seq)
        t_end = time.time()
        objs.append(learner.compute_loss(X, y))
        time_cost.append(t_end - t_start)
        print("epoch: %d, obj = %f, time = %f" % (i, objs[i], time_cost[i]))
        # if objs[-1] / module_y < tol:
        #     break
    return learner, objs, time_cost


def parallel_sgd(learner, X, y, data_partition=None, gamma=0.0001, max_iter=100, tol=0.01, P=1):
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
    :param data_partition: groups of indexes of X to be assigned on different cores
    :param max_iter: maximum number of iterations
    :param tol: the tolerated relative error: ||Xw - y|| / ||y||
    :return: weight, [objs]: object function values in each iteration
    :param P: number of cores
    :return: weights, [objs]: object function values in each iterations, [times]: time consumption in each iteration
    """
    n, d = X.shape
    objs, time_cost = [], []
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
        if data_partition is None:
            seq_par = split_data(X, P, 'random')
        elif isinstance(data_partition, graph.ConflictGraph):
            seq_par = data_partition.gen_partition(P)
        else:
            seq_par = [np.random.permutation(data_partition[p]) for
                       p in xrange(P)]
        results = [pool.apply_async(sgd_one_update,
                                    args=(learners[p], shared_X, shared_y, gamma, seq_par[p]))
                    for p in xrange(P)]
        w_updates = np.array([res.get() for res in results])
        # average
        learner.w = np.average(w_updates, 0)
        t_end = time.time()
        objs.append(learner.compute_loss(X, y))
        time_cost.append(t_end - t_start)
        print("epoch: %d, obj = %f, time = %f" % (i, objs[i], time_cost[i]))
        # if objs[-1] / module_y < tol:
        #     break
    return learner, objs, time_cost


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