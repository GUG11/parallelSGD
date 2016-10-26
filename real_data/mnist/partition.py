import numpy as np
from misc import utils
import os
import matplotlib.pyplot as plt
import networkx as nx
from SGDs import graph
import time

# import data
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    mnist_path = os.path.join('data','MNIST')
    mnist = input_data.read_data_sets(mnist_path)
# only pick the first n data
    n = 5000
    img, labels = mnist.train.next_batch(n)

    cross_correlation, _ = utils.xcorr(img)
    # cc_sorted = np.sort(cross_correlation.ravel())
    #
    # plt.plot(cc_sorted, lw=2)
    # plt.show()

    # G = graph.gen_corr_graph(cross_correlation)
    # subGs = graph.split_evenly(G, 8)
    # for g in subGs:
    #     print g.nodes()
    #

    # limit the max degree of G
    max_deg = 50
    G = graph.gen_corr_graph(cross_correlation, max_deg=max_deg)

    degs = {k: v for k, v in G.degree().iteritems() if v > 0}
    print degs, max(list(degs.values()))

    # B = n / max_deg
    r = 20         # number of repetition
    # stats = {'mu': np.zeros(r), 'sigma': np.zeros(r)}
    # for x in xrange(r):
    #     samples = np.random.permutation(n)[:B]
    #     num_nodes = []
    #     subG = G.subgraph(samples)
    #     connected_comp = nx.connected_component_subgraphs(subG)
    #     print('connected components')
    #     for conncomp in connected_comp:
    #         print conncomp.nodes()
    #         num_nodes.append(len(conncomp.nodes()))
    #     stats['mu'][x] = np.mean(num_nodes)
    #     stats['sigma'][x] = np.std(num_nodes)
    # print stats['mu']
    # print stats['sigma']

    cg = graph.ConflictGraph(G)
    for x in xrange(r):
        seq_part = cg.gen_partition(P=8)
        print seq_part
