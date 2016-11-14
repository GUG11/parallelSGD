import numpy as np
from misc import utils
import os
import matplotlib.pyplot as plt
import networkx as nx
from SGDs import graph
import time
from images2gif import writeGif
import Image
from matplotlib import gridspec

# import data
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    mnist_path = os.path.join('data','MNIST')
    mnist = input_data.read_data_sets(mnist_path)
    # only pick the first n data
    n, R = 1000, 10
    imgs, labels = mnist.train.images, mnist.train.labels

    c = len(np.unique(labels))
    data_by_number = []

    for label in xrange(c):
        tmp = imgs[np.where(label == labels)[0], :]
        data_by_number.append(tmp[::R, :])

    # visualize MNIST
    # for label in xrange(c):
    #     frames = []
    #     for i in xrange(0, len(data_by_number[label]), 100):
    #         frames.append(Image.fromarray(np.uint8(255*data_by_number[label][i].reshape((28, 28)))))
    #         frames[-1].save(os.path.join('results', 'real_data', 'MNIST',
    #                 'sample_images', 'digit_%d_%i.jpg' % (label, i)))



    # compute correlation
    # cc, ncc = [], []

    fig = plt.figure(num=1, figsize=(20, 12))
    gs = gridspec.GridSpec(c, c)
    for i in xrange(c):
        # cc.append([])
        # ncc.append([])
        for j in xrange(c):
            print('digit %d - digit %d' % (i, j))
            cc0, ncc0 = utils.xcorr(data_by_number[i], data_by_number[j])
            # cc[-1].append(cc0)
            # ncc[-1].append(ncc0)

            ax = fig.add_subplot(gs[i, j])
            utils.plot_hist(ncc0.ravel(), ax, num_bins=200)

    plt.tight_layout()
    save_dir=os.path.join('results', 'real_data', 'MNIST', 'correlation_500',
                                      'mnist_ncc_hist_all.pdf')
    fig.savefig(save_dir)
    # # cross_correlation, cc_re = utils.xcorr(imgs)
    # hist, bin_edges = np.histogram(cross_correlation.ravel(), bins=200)

