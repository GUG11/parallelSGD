"""A very simple MNIST classifier"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from copy import deepcopy
import multiprocessing
from sharedmem import sharedmem
import time


# import data
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import tensorflow as tf


class SoftmaxModel():
    def __init__(self, dim, num_classes):
        self.x = tf.placeholder(tf.float32, [None, dim])
        self.W = tf.Variable(tf.zeros([dim, num_classes]))
        self.b = tf.Variable(tf.zeros([num_classes]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        self.y_ = tf.placeholder(tf.float32, [None, 10])


def loss_function(y, y_):
    # Define loss
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    return loss


def sgd_learner(loss_func, learning_rate=0.5):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_func)


def train_serial(dataset, model, learner, num_iters, batch_size):
    if isinstance(dataset, DataSet):
        images, labels = dataset._images, dataset._labels
    elif isinstance(dataset, tuple):
        images, labels = dataset
    else:
        raise TypeError('dataset is neither MNIST DataSet nor tuple')

    n = images.shape[0]
    for i in range(num_iters):
        index = 0
        while (index < n):
            batch_xs = images[index:index+batch_size,:]
            batch_ys = labels[index:index+batch_size,:]
            index += batch_size;
            learner.run({model.x: batch_xs, model.y_:batch_ys})

        train_accuracy = accuracy.eval(feed_dict={
                    model.x: batch_xs, model.y_:batch_ys
                })
        print('step %d, training accuracy %g' % (i, train_accuracy))

    return sess.run(model.W), sess.run(model.b)



def train_parallel(dataset, model, learner, num_iters, batch_size, P):
    # P: number of cores
    images = dataset._images
    labels = dataset._labels
    n, d = images.shape
    P = min(P, multiprocessing.cpu_count())
    # create shared memory
    shared_X = sharedmem.copy(images)
    shared_y = sharedmem.copy(labels)
    print("Parallel SGD: size (%d, %d), cores: %d" % (n, d, P))
    pool = multiprocessing.Pool(P)
    for i in xrange(num_iters):
        t_start = time.time()
        random_seq = np.random.permutation(n)
        seq_par = [random_seq[x::P] for x in range(P)]

        results = [pool.apply_async(train_serial,
            args=((shared_X[seq_par[p], :], shared_y[seq_par[p]]), model, learner, 1, batch_size)) for p in xrange(P)]
        w_updates = np.array([res.get()[0] for res in results])
        b_updates = np.array([res.get()[1] for res in results])
        # average
        w_val = np.average(w_updates, 0)
        b_val = np.average(b_updates, axis=0)
        # update parameters
        model.W.assign(w_val)
        model.b.assign(b_val)
        t_end = time.time()
        #objs.append(np.linalg.norm(np.dot(X, w) - y))
        print("epoch: %d, obj = %f, time = %f" % (i, 0, t_end - t_start))
        #if objs[-1] / module_y < tol:
        #    break"""



flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

config = tf.ConfigProto(device_count={"CPU": 1},
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1)
sess = tf.InteractiveSession(config=config)

dim, num_classes = 784, 10
softmax_model = SoftmaxModel(dim, num_classes)
cross_entropy = loss_function(softmax_model.y, softmax_model.y_)
train_step = sgd_learner(cross_entropy, 0.5)
correct_prediction = tf.equal(tf.argmax(softmax_model.y, 1), tf.argmax(softmax_model.y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

writer = tf.train.SummaryWriter('tmp/', sess.graph)

# Train
tf.initialize_all_variables().run()
#W, b = train_serial(mnist.train, softmax_model, train_step, 20, 100);
train_parallel(mnist.train, softmax_model, train_step, 20, 100, 8);
# Test trained model
print(accuracy.eval({softmax_model.x: mnist.test.images, softmax_model.y_: mnist.test.labels}))
