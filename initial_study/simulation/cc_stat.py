import numpy as np
from misc import utils
import matplotlib.pyplot as plt
import networkx as nx
from SGDs import graph
import os


def experiment1(n, d, repeat, threashold):
    """
    experiment repeat time on measuring the connected components of
    Gaussian random matrix
    :param n: number of data points
    :param d: dimensions
    :param repeat: number of repetition
    :param threashold: threshold of edges
    :return: statistic metrics
    """
    mu = np.zeros(repeat)
    sigma = np.zeros(repeat)
    min_size = np.zeros(repeat)
    max_size = np.zeros(repeat)
    num_groups = np.zeros(repeat)
    for x in xrange(repeat):
        print 'epoch %d' % x
        X = np.random.randn(n, d)
        cross_correlation = utils.xcorr(np.abs(X))
        G = graph.gen_corr_graph(cross_correlation, threashold)
        connected_comp = nx.connected_component_subgraphs(G)
        num_nodes = np.array([len(conncomp.nodes()) for conncomp in connected_comp])
        min_size[x] = min(num_nodes)
        max_size[x] = max(num_nodes)
        sigma[x] = np.std(num_nodes)
        mu[x] = np.mean(num_nodes)
        num_groups[x] = len(num_nodes)
    return mu, sigma, min_size, max_size, num_groups




if __name__ == '__main__':
    n, d = 500, 100
    X = np.random.randn(n, d)
    cc = utils.xcorr(np.abs(X))
    c_sort = np.sort(np.abs(cc.ravel()))
    threashold = 0.75
    repeat = 10
    mu, sigma, min_size, max_size, num_groups = experiment1(n, d, repeat, threashold)
    print mu
    print sigma
    print min_size
    print max_size
    print num_groups
    X = np.zeros((repeat, 5))
    X[:,0] = mu
    X[:,1] = sigma
    X[:,2] = min_size
    X[:,3] = max_size
    X[:,4] = num_groups
    print X

    # cc_vec = cc[np.triu_indices_from(cc, 1)]
    # cc_hist, cc_edges = np.histogram(cc_vec, bins=200)
    # plt.figure(num=1, figsize=(20, 12))
    # plt.bar(cc_edges[:-1], cc_hist, width=0.005)
    # plt.xlim([0, 1])
    # plt.xticks(fontsize=20); plt.yticks(fontsize=20)
    # plt.xlabel('correlation', fontsize=20)
    # plt.ylabel('number', fontsize=20)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(os.path.join('..', '..', 'results', 'simulations', 'gaussian_cor_dist.svg'))
