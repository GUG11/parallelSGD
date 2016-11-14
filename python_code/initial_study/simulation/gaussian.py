import numpy as np
from misc import utils
from SGDs import sgd, loss_functions, graph

if __name__ == '__main__':
    n, d = 1000, 100
    np.random.seed(0)
    X = np.random.randn(n, d)
    w = np.random.uniform(low=0.0, high=1.0, size=(d,))
    y = np.dot(X, w)
    max_iter = 10000
    w0 = np.zeros(d)

    # learner1, sgd_prof1 = sgd.serial_sgd(loss_functions.LeastSquare(w0), X, y, gamma=0.0005, max_iter=max_iter, tol=1e-10)
    learner2, sgd_prof2 = sgd.hogwild(loss_functions.LeastSquare(w0), X, y, gamma=0.0005, max_iter=max_iter, P=8)
    # max_deg = 10
    # cc, ncc = utils.xcorr(X)
    # G = graph.gen_corr_graph(np.abs(ncc), max_deg=max_deg)
    # cg = graph.ConflictGraph(G)
    # learner3, sgd_prof3 = sgd.parallel_sgd(loss_functions.LeastSquare(w0), X, y, max_iter=max_iter, gamma=0.0005, P=8, tol=1e-3)
    # print("Serial loss: %f. Aggregated parallel loss: %f" % 
    #     (learner1.compute_loss(X, y), learner3.compute_loss(X, y)))
    # learner4, sgd_prof4 = sgd.parallel_sgd(X, y, max_iter=max_iter, gamma=0.0001, P=8, tol=1e-3)