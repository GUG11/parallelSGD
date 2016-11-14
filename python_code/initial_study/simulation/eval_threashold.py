import numpy as np
from misc import utils
from SGDs import sgd, loss_functions, graph
import matplotlib.pyplot as plt
import matplotlib
import os


if __name__ == '__main__':
    n, d = 1000, 100
    np.random.seed(0)
    X = np.random.randn(n, d)
    w = np.random.uniform(low=0.0, high=1.0, size=(d,))
    y = np.dot(X, w)
    w0 = np.zeros(d)

    cc, ncc = utils.xcorr(X)
    max_degrees = range(0, n+1, 20)
    max_degrees_actual = []
    ths = []
    num_edges = []
    per_edges_used = []
    for max_deg in max_degrees:
        G, th = graph.gen_corr_graph(np.abs(ncc), max_deg=max_deg)
        ths.append(th)
        max_degrees_actual.append(max(list(G.degree().values())))
        num_edges.append(G.number_of_edges())
        per_edges_used.append(num_edges[-1] / float(n*(n-1)/2))
    print(max_degrees_actual)
    print(ths)
    print(num_edges)
    
    matplotlib.rcParams.update({'font.size': 30})
    fig = plt.figure(num=1, figsize=(20, 12))
    ax = fig.add_subplot(1,1,1)
    ax.plot(max_degrees_actual, ths, label='threshold', lw=2)
    ax.plot(max_degrees_actual, per_edges_used, label='percent of edges used', lw=2)
    utils.set_axis(ax, xlabel='max degree', ylabel=None, title='Tune max degree', 
        xticks=None, yticks=None, xlim=None, fontsize=30)
    plt.tight_layout()
    plt.show()
    save_dir = os.path.join('..', 'results', 'simulations', 'Gaussian',
        'n%d_d%d_th.pdf' % (n, d))
    fig.savefig(save_dir)