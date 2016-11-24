import numpy as np


def xcorr(X, Y=None):
    """
    cross correlation between data X ( and Y)
    :param X: data (n x d)
    :param Y: data (m x d) optional
    :return: correlation matrix (n x n) [0]: absolute, [1]: relative
    if Y is not None. cc (n x m)
    """
    if Y is None:
        Y = X
    XYT = np.dot(X, Y.T)
    xnorm = np.linalg.norm(X, 2, axis=1)
    ynorm = np.linalg.norm(Y, 2, axis=1)
    xy_norm = np.outer(xnorm, ynorm)
    rcc = XYT / xy_norm
    return XYT, rcc


def plot_hist(x, ax, num_bins=200, xlim=[0, 1], xlabel='data', title='title'):
    """
    my plot histogram function
    :param x: data
    :param num_bins: number of bins
    :return:
    """
    n, bins, patches = ax.hist(x, bins=num_bins,
        normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    set_axis(ax, xlabel, 'frequency', title, xticks=[np.min(x), np.max(x)],
        yticks=[0, np.max(n)], xlim=xlim, fontsize=30)


def set_axis(ax, xlabel=None, ylabel=None, title=None,
        xticks=None, yticks=None, xlim=None, fontsize=30):
    """
    set the outlook of an axis
    :param ax: axis
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontdict={'size': fontsize})
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontdict={'size': fontsize})
    if title is not None:
        ax.set_title(title, fontdict={'size': fontsize})
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(fontsize=fontsize)
