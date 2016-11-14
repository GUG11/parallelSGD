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
    n = 1000
    img, labels = mnist.train.next_batch(n)

    cross_correlation, cc_re = utils.xcorr(img)
    # hist, bin_edges = np.histogram(cross_correlation.ravel(), bins=200)

    # the histogram of the data
    fig = plt.figure(num=1, figsize=(20, 12))
    n, bins, patches = plt.hist(cc_re.ravel(), bins=200,
        normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('correlation', fontsize=20)
    plt.ylabel('frequency', fontsize=20)
    plt.tight_layout()
    plt.legend(fontsize=30)
    # plt.show()
    fig.savefig(os.path.join('results', 'real_data', 'mnist_ncc_hist.pdf'))
    # G = graph.gen_corr_graph(cross_correlation)
    # subGs = graph.split_evenly(G, 8)
    # for g in subGs:
    #     print g.nodes()


    # limit the max degree of G
    # max_deg = 100
    # G = graph.gen_corr_graph(cross_correlation, max_deg=max_deg)
    #
    # degs = {k: v for k, v in G.degree().iteritems() if v > 0}
    # print degs, max(list(degs.values()))
    #
    # B = n / max_deg
    # r = 20         # number of repetition
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
    #
    # cg = graph.ConflictGraph(G)
    # for x in xrange(r):
    #     seq_part = cg.gen_partition(P=8)
    #     print seq_part
