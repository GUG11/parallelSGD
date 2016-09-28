import networkx as nx
import math
from Queue import PriorityQueue
from copy import deepcopy


def gen_corr_graph(cross_corr, threshold=-0.01):
    """
    generate a correlation graph from the cross correlation matrix
    :param cross_corr: cross correlation matrix (n x n)
    :param threshold: threshold [0, 1.0], only keep edges whose cc is less than the threshold
    :return: networkx graph
    """
    n = cross_corr.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in xrange(n):
        for j in xrange(i+1, n):
            if cross_corr[i][j] >= threshold:
                G.add_edge(i, j, weight=cross_corr[i][j])
    return G


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
