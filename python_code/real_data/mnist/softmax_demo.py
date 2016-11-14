import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def oneHotIt(Y):
    d = Y.shape[0]
    c = np.max(Y) + 1
    OHX = np.zeros((c, d))
    OHX[Y, range(d)] = 1
    return OHX.T


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
    return sm


def getProbsAndPreds(someX, w):
    probs = softmax(np.dot(someX, w))
    preds = np.argmax(probs, axis=1)
    return probs, preds


def getLoss(w,x,y,lam):
    m = x.shape[0]
    y_mat = oneHotIt(y)
    scores = np.dot(x,w)
    prob = softmax(scores)
    loss = (-1/m) * np.sum(y_mat * np.log(prob)) + (lam/2) * np.sum(w*w)
    grad = (-1/m) * np.dot(x.T, (y_mat - prob)) + lam * w
    return loss, grad


def getAccuracy(someX, someY, w):
    prob, prede = getProbsAndPreds(someX, w)
    accuracy = sum(prede == someY) / float(len(someY))
    return accuracy


if __name__ == '__main__':
    mnist = input_data.read_data_sets(os.path.join('data', 'MNIST'))
    batch = mnist.train.next_batch(500)
    tb = mnist.train.next_batch(100)

    X, y = batch
    testX, testY = tb

    w = np.zeros([X.shape[1],len(np.unique(y))])
    lam = 1
    iterations = 1000
    learningRate = 1e-5
    losses = []
    for i in range(0,iterations):
        loss,grad = getLoss(w,X,y,lam)
        losses.append(loss)
        w = w - (learningRate * grad)
    print loss

    plt.plot(losses)
    plt.show()
    print 'Training Accuracy: ', getAccuracy(X,y, w)
    print 'Test Accuracy: ', getAccuracy(testX,testY, w)