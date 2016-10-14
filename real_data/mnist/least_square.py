import numpy as np
from misc.loader import MNIST
from misc import utils
import os
import matplotlib.pyplot as plt
import networkx as nx
from SGDs import graph, sgd

if __name__ == '__main__':
    mnist_path = os.path.join('data','MNIST')
    mnist_obj = MNIST(mnist_path)
    print('loading training data')
    train_data = mnist_obj.load_training()
    print('loading testing data')
    test_data = mnist_obj.load_testing()

    image_train = np.array(train_data[0])
    label_train = np.array(train_data[1])
    image_test = np.array(test_data[0])
    label_test = np.array(test_data[1])


    w1, objs1 = sgd.serial_sgd(image_train, label_train, gamma=0.0000001, max_iter=1000, tol=1e-10)

# only pick the first n data
"""
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
        """
