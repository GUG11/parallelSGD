import numpy as np
from misc import utils
import matplotlib.pyplot as plt
import networkx as nx
from SGDs import graph

if __name__ == '__main__':
    n, d = 200, 100
    X = np.random.randn(n, d)
    cc = utils.xcorr(X)
    c_sort = np.sort(np.abs(cc.ravel()))

    # plot correlations
    #plt.plot(c_sort, lw=2)
    #plt.show()

    # create a graph based on X
    threashold = 0.25
    pos = {i: (10 * np.cos(i * 2 * np.pi / n), 10 * np.sin(i * 2 * np.pi / n)) for i in range(n)}
    G = graph.gen_corr_graph(cc, threashold)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos=pos, with_labels=True, edge_color='b', width=2*weights)
    plt.show()
    connected_comp = nx.connected_component_subgraphs(G)
    for conncomp in connected_comp:
        print conncomp.nodes()

    # my algorithm
    G_full = graph.gen_corr_graph(np.abs(cc))
    subGs = graph.split_evenly(G_full, 8)
    for g in subGs:
        print g.nodes()
        print g.edges()