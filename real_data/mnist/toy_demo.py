"""
display 6 data points and their correlation
"""

import numpy as np
from misc import utils
from SGDs import graph
import os



# import data
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    mnist = input_data.read_data_sets(os.path.join('data', 'MNIST'), one_hot=True)
    n = 6
    X, y = mnist.train.next_batch(n)
    C = utils.xcorr(X)
    print C