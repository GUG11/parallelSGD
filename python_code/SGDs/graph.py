import networkx as nx
import math
import numpy as np
from Queue import PriorityQueue
from copy import deepcopy
import sys


def gen_corr_graph(cross_corr, threshold=-0.01, max_deg=None):
    """
    generate a correlation graph from the cross correlation matrix
    :param cross_corr: cross correlation matrix (n x n)
    :param threshold: only keep edges whose cc is less than the threshold
    :param max_deg: maximum degrees of nodes
    :return: networkx graph
    """
    G = None
    if max_deg is not None:
        max_deg1 = sys.maxint
        lo, hi = np.min(cross_corr), np.max(cross_corr)
        tol_err = 1e-4
        print('search threshold. min correlation:%f, max correlation:%f' % (lo, hi))
        while max_deg < max_deg1 or tol_err < hi - lo:
            threshold = (hi + lo) / 2
            G, threshold = gen_corr_graph(cross_corr, threshold=threshold)
            max_deg1 = max(list(G.degree().values()))
            print('th = %f, max degree:%d, number of edges:%d' % (threshold, max_deg1, G.number_of_edges()))
            if max_deg1 < max_deg:     # threshold to high
                hi = threshold
            else:
                lo = threshold
    else:
        n = cross_corr.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in xrange(n):
            for j in xrange(i+1, n):
                if cross_corr[i][j] >= threshold:
                    G.add_edge(i, j, weight=cross_corr[i][j])
    return G, threshold


def split_evenly(G, P):
    """
    split graph G evenly to P subgraphs {S_1, ..., S_P}
    the idea optimization problem is:
        max \sum_{k=1}^P \sum_{vi,vj \in S_k} w_{ij}
        s.t. Card(S_k) = n / P
    Use a greedy algorithm to find an approximation (maybe? unproved)
    :param G: graph with weights
    :return: a list of subgraphs
    """
    n = nx.number_of_nodes(G)
    m = int(math.ceil(n / P))
    visited = [False] * n
    i = 0
    subGs = []
    for k in xrange(P):
        # find next unvisited node
        while visited[i]:
            i += 1
        subGs.append(maximum_span(G, i, m, visited))
    return subGs


def maximum_span(G, start, m, visited):
    """
    find maximum spanning tree of size m in G from start node
    :param G: graph
    :param start: starting node
    :param m: size of tree
    :param visited: visited node record
    :return: subgraphs
    """
    selected = []
    n = len(visited) - sum(visited)
    discovered = deepcopy(visited)
    pq = PriorityQueue(m * n)
    pq.put((1, start))
    discovered[start] = True
    for i in xrange(m):
        w, u = pq.get()
        selected.append(u)
        visited[u] = True
        for v in G.neighbors(u):
            if not discovered[v]:
                discovered[v] = True
                pq.put((1-G[u][v]['weight'], v))
    return G.subgraph(selected)


class ConflictGraph:
    def __init__(self, G):
        """
        construct the conflict subgraphs based on graph G
        :param G: graph
        """
        self._G = G
        self._n = G.number_of_nodes()
        self._max_deg = max(list(G.degree().values()))

    def _sample(self):
        """
        sample a B = n / max_deg -size subgraph
        :return: subgraph
        """
        B = self._n / self._max_deg
        samples = np.random.permutation(self._n)[:B]
        subG = self._G.subgraph(samples)
        return subG

    def gen_partition(self, P):
        """
        sample B subgraph from G,
        partition the conflict group of subG evenly to P sets
        :param P: number of sets (cores)
        :return: P lists
        """
        seq_part = [[] for p in xrange(P)]
        subG = self._sample()
        connected_comp = nx.connected_component_subgraphs(subG)
        connected_nodes = [x.nodes() for x in connected_comp]
        M = int(np.ceil(subG.number_of_nodes() / float(P)))
        p, count = 0, 0
        for i in xrange(len(connected_nodes)):
            seq_part[p] += connected_nodes[i]
            count += len(connected_nodes[i])
            if count >= M * (p+1):
                p += 1
        for i in xrange(len(seq_part)):
            seq_part[i] = np.array(seq_part[i])
            np.random.shuffle(seq_part[i])
        return seq_part

