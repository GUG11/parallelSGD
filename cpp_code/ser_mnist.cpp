#include <armadillo>
#include "sgd.h"
#include "io.h"

int main() {
    std::string filedir = "../data/MNIST", filename_i = "train-images-idx3-ubyte", filename_l = "train-labels-idx1-ubyte";
    arma::mat images;
    arma::mat labels;
    readMnistImages(filedir, filename_i, images);
    readMnistLabels(filedir, filename_l, labels);
    int n = 10000, d = images.n_rows, c = round(labels.max()) + 1;
    images = images.cols(0, n-1);
    labels = labels.cols(0, n-1);
    double learningRate=0.001, numIters=10000;
    SGDProfile sgdProfile;
    Softmax learner(arma::mat(c, d, arma::fill::zeros));
    serialSGD(&learner, images, labels, &sgdProfile, learningRate, numIters);
    return 0;
}
