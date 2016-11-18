import numpy as np
from misc import utils
from SGDs import sgd, loss_functions, graph
from copy import deepcopy

if __name__ == '__main__':
    n, d = 1000, 100
    np.random.seed(0)
    X = np.random.randn(n, d)
    w = np.random.uniform(low=0.0, high=1.0, size=(d,))
    y = np.dot(X, w)
    max_iter = 10000
    w0 = np.zeros(d)
    
    learner = loss_functions.LeastSquare()

    w1, sgd_prof1 = sgd.serial_sgd(learner, X, y,
        deepcopy(w0), gamma=0.0005, max_iter=max_iter, tol=1e-10)
    # w2, sgd_prof2 = sgd.hogwild(learner, X, y, deepcopy(w0), gamma=0.0005, max_iter=max_iter, P=8)
    # max_deg = 10
    # cc, ncc = utils.xcorr(X)
    # G = graph.gen_corr_graph(np.abs(ncc), max_deg=max_deg)
    # cg = graph.ConflictGraph(G)
    w3, sgd_prof3 = sgd.parallel_sgd(learner, X, y, w0, max_iter=max_iter, gamma=0.0005, P=8, tol=1e-3)
    print("Serial loss: %f. Aggregated parallel loss: %f" % 
        (learner.compute_loss(X, y, w1), learner.compute_loss(X, y, w3)))
    # learner4, sgd_prof4 = sgd.parallel_sgd(X, y, max_iter=max_iter, gamma=0.0001, P=8, tol=1e-3)