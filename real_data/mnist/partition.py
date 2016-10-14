import numpy as np
from misc.loader import MNIST
from misc import utils
import os
import matplotlib.pyplot as plt
import networkx as nx
from SGDs import graph

if __name__ == '__main__':
    mnist_path = os.path.join('data','MNIST')
    mnist_obj = MNIST(mnist_path)
    #train_data = mnist_obj.load_training()
    test_data = mnist_obj.load_testing()
    img = np.array(test_data[0])
    labels = np.array(test_data[1])

# only pick the first n data
    n = 1000
    img = img[:n,:]
    labels = labels[:n]

    cross_correlation = utils.xcorr(img)
    cc_sorted = np.sort(cross_correlation.ravel())

    plt.plot(cc_sorted, lw=2)
    plt.show()

    G = graph.gen_corr_graph(cross_correlation)
    subGs = graph.split_evenly(G, 8)
    for g in subGs:
        print g.nodes()
